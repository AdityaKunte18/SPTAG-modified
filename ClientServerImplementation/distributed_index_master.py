#!/usr/bin/env python3
"""Distributed SPTAG shard build master with centroid routing and rebalance."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


def _read_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_http_response(resp) -> dict:
    return json.loads(resp.read().decode("utf-8"))


def _http_json_post(url: str, payload: dict, timeout_s: int):
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Content-Length", str(len(raw)))
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return _read_json_http_response(r)
    except urllib.error.HTTPError as ex:
        body = ex.read()
        if body:
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                pass
        raise


def _http_binary_post(url: str, payload: bytes, timeout_s: int):
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header("Content-Length", str(len(payload)))
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return _read_json_http_response(r)
    except urllib.error.HTTPError as ex:
        body = ex.read()
        if body:
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                pass
        raise


def _http_json_get(url: str, timeout_s: int):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return _read_json_http_response(r)


def _post_with_retry(url: str, fn, retries: int, what: str):
    import socket
    import http.client

    err = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except (
            urllib.error.URLError,
            socket.timeout,
            TimeoutError,
            ConnectionResetError,
            BrokenPipeError,
            http.client.RemoteDisconnected,
        ) as ex:
            err = ex
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"{what} failed after retries: {err}")


def _send_batch_with_backpressure(
    *,
    url: str,
    payload: bytes,
    request_timeout: int,
    retries: int,
    what: str,
):
    total_rpc_roundtrip_s = 0.0
    total_backpressure_sleep_s = 0.0
    attempts = 0
    while True:
        t0 = time.time()
        resp = _post_with_retry(
            url,
            lambda p=payload: _http_binary_post(url, p, request_timeout),
            retries,
            what,
        )
        total_rpc_roundtrip_s += time.time() - t0
        attempts += 1
        if resp.get("queue_full", False):
            delay_ms = int(resp.get("retry_after_ms", 200))
            sleep_s = max(0.0, delay_ms / 1000.0)
            total_backpressure_sleep_s += sleep_s
            time.sleep(sleep_s)
            continue
        resp = dict(resp)
        resp["_rpc_roundtrip_s"] = total_rpc_roundtrip_s
        resp["_backpressure_sleep_s"] = total_backpressure_sleep_s
        resp["_attempts"] = attempts
        return resp


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        return


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    content_len = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_len) if content_len > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def _distance_matrix(vectors: np.ndarray, centroids: np.ndarray, metric: str) -> np.ndarray:
    if vectors.size == 0 or centroids.size == 0:
        return np.zeros((vectors.shape[0], centroids.shape[0]), dtype=np.float32)
    metric_name = metric.upper()
    if metric_name in {"COSINE", "COSINESIMILARITY", "COSINE_SIMILARITY"}:
        lhs = _normalize_rows(vectors.astype(np.float32, copy=False))
        rhs = _normalize_rows(centroids.astype(np.float32, copy=False))
        return 1.0 - (lhs @ rhs.T)
    diff = vectors[:, None, :] - centroids[None, :, :]
    return np.sum(diff * diff, axis=2)


def _seed_for_job(job_id: str, suffix: str = "") -> int:
    return zlib.crc32(f"{job_id}:{suffix}".encode("utf-8")) & 0xFFFFFFFF


def _optional_positive_int(value, name: str) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _optional_positive_float(value, name: str) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _build_params_payload(
    cef: int | None,
    max_check_for_refine_graph: int | None,
    graph_neighborhood_scale: float | None,
    tpt_number: int | None,
    tpt_leaf_size: int | None,
) -> dict:
    return {
        "cef": cef,
        "max_check_for_refine_graph": max_check_for_refine_graph,
        "graph_neighborhood_scale": graph_neighborhood_scale,
        "tpt_number": tpt_number,
        "tpt_leaf_size": tpt_leaf_size,
    }


def _kmeans_plus_plus(vectors: np.ndarray, k: int, metric: str, seed: int) -> np.ndarray:
    n = vectors.shape[0]
    if n == 0:
        return np.zeros((k, vectors.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    centroids = np.empty((k, vectors.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, n))
    centroids[0] = vectors[first]
    if k == 1:
        return centroids
    closest = _distance_matrix(vectors, centroids[:1], metric).reshape(-1)
    for idx in range(1, k):
        if np.allclose(closest.sum(), 0.0):
            pick = int(rng.integers(0, n))
        else:
            probs = closest / closest.sum()
            pick = int(rng.choice(n, p=probs))
        centroids[idx] = vectors[pick]
        newest = _distance_matrix(vectors, centroids[idx : idx + 1], metric).reshape(-1)
        closest = np.minimum(closest, newest)
    return centroids


def _balanced_assign(vectors: np.ndarray, centroids: np.ndarray, metric: str) -> np.ndarray:
    n, k = vectors.shape[0], centroids.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    capacity = max(1, int(math.ceil(float(n) / float(k))))
    dists = _distance_matrix(vectors, centroids, metric)
    order = np.argsort(dists.reshape(-1), kind="stable")
    assignments = np.full(n, -1, dtype=np.int32)
    counts = np.zeros(k, dtype=np.int32)

    for flat_idx in order.tolist():
        point_idx = flat_idx // k
        centroid_idx = flat_idx % k
        if assignments[point_idx] != -1:
            continue
        if counts[centroid_idx] >= capacity:
            continue
        assignments[point_idx] = centroid_idx
        counts[centroid_idx] += 1
        if np.all(assignments != -1):
            return assignments

    for point_idx in np.flatnonzero(assignments == -1).tolist():
        centroid_order = np.argsort(dists[point_idx], kind="stable")
        assigned = False
        for centroid_idx in centroid_order.tolist():
            if counts[centroid_idx] < capacity:
                assignments[point_idx] = centroid_idx
                counts[centroid_idx] += 1
                assigned = True
                break
        if not assigned:
            centroid_idx = int(np.argmin(counts))
            assignments[point_idx] = centroid_idx
            counts[centroid_idx] += 1
    return assignments


def _balanced_kmeans(vectors: np.ndarray, k: int, metric: str, seed: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    dim = vectors.shape[1] if vectors.ndim == 2 else 0
    if k <= 0:
        raise ValueError("k must be positive")
    if vectors.size == 0:
        return np.zeros((k, dim), dtype=np.float32), np.empty((0,), dtype=np.int32)

    if vectors.shape[0] < k:
        repeats = int(math.ceil(float(k) / float(vectors.shape[0])))
        padded = np.vstack([vectors] * repeats)[:k]
        return padded.astype(np.float32, copy=False), np.arange(vectors.shape[0], dtype=np.int32) % k

    centroids = _kmeans_plus_plus(vectors, k, metric, seed)
    assignments = np.full(vectors.shape[0], -1, dtype=np.int32)

    for _ in range(max_iter):
        new_assignments = _balanced_assign(vectors, centroids, metric)
        new_centroids = centroids.copy()
        for centroid_idx in range(k):
            members = vectors[new_assignments == centroid_idx]
            if members.size == 0:
                continue
            center = members.mean(axis=0, dtype=np.float32)
            if metric.upper() in {"COSINE", "COSINESIMILARITY", "COSINE_SIMILARITY"}:
                norm = np.linalg.norm(center)
                if norm > 1e-12:
                    center = center / norm
            new_centroids[centroid_idx] = center.astype(np.float32, copy=False)
        if np.array_equal(assignments, new_assignments):
            assignments = new_assignments
            centroids = new_centroids
            break
        assignments = new_assignments
        centroids = new_centroids
    return centroids.astype(np.float32, copy=False), assignments


@dataclass
class Worker:
    worker_id: int
    host: str
    port: int
    registered: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class ShardInfo:
    shard_id: int
    worker: Worker
    next_batch_id: int = 0
    vectors_forwarded: int = 0


@dataclass
class BufferedBatch:
    global_ids: np.ndarray
    vectors: np.ndarray
    normalized: bool


def _new_communication_summary() -> dict[str, float | int]:
    return {
        "init_worker_rpc_roundtrip_s": 0.0,
        "init_worker_comm_overhead_s": 0.0,
        "init_worker_count": 0,
        "ingest_worker_rpc_roundtrip_s": 0.0,
        "ingest_worker_rpc_roundtrip_max_s": 0.0,
        "ingest_worker_fanout_wall_time_s": 0.0,
        "ingest_worker_comm_overhead_s": 0.0,
        "ingest_worker_request_total_s": 0.0,
        "ingest_worker_body_read_time_s": 0.0,
        "ingest_worker_processing_after_read_s": 0.0,
        "ingest_worker_processing_after_read_max_s": 0.0,
        "ingest_worker_apply_time_s": 0.0,
        "ingest_worker_queue_wait_s": 0.0,
        "ingest_worker_backpressure_sleep_s": 0.0,
        "ingest_worker_batches": 0,
        "ingest_worker_vectors": 0,
        "checkpoint_worker_rpc_roundtrip_s": 0.0,
        "checkpoint_worker_comm_overhead_s": 0.0,
        "checkpoint_worker_count": 0,
        "finalize_worker_rpc_roundtrip_s": 0.0,
        "finalize_worker_comm_overhead_s": 0.0,
        "finalize_worker_count": 0,
    }


def _build_elapsed_s(job: "MasterJobState", now: float | None = None) -> float:
    ref = time.time() if now is None else now
    if job.build_pause_started_at is not None:
        ref = min(ref, job.build_pause_started_at)
    return max(0.0, ref - job.init_time - job.build_paused_time_s)


def _wall_elapsed_s(job: "MasterJobState", now: float | None = None) -> float:
    ref = time.time() if now is None else now
    return max(0.0, ref - job.init_time)


def _pause_build_timing(job: "MasterJobState") -> bool:
    if job.build_pause_started_at is not None:
        return False
    job.build_pause_started_at = time.time()
    return True


def _resume_build_timing(job: "MasterJobState") -> bool:
    if job.build_pause_started_at is None:
        return False
    job.build_paused_time_s += max(0.0, time.time() - job.build_pause_started_at)
    job.build_pause_started_at = None
    return True


@dataclass
class MasterJobState:
    job_id: str
    dim: int
    algo: str
    dist: str
    value_type: str
    output_dir: str
    threads: int
    cef: int | None
    max_check_for_refine_graph: int | None
    graph_neighborhood_scale: float | None
    tpt_number: int | None
    tpt_leaf_size: int | None
    with_meta_index: bool
    shards: list[ShardInfo] = field(default_factory=list)
    finalized: bool = False
    error: str | None = None
    init_time: float = 0.0
    build_paused_time_s: float = 0.0
    build_pause_started_at: float | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    total_vectors_ingested: int = 0

    routing_mode: str = "bootstrap"
    routing_epoch: int = 0
    worker_centroids: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    worker_counts: dict[int, int] = field(default_factory=dict)
    pending_new_workers: set[int] = field(default_factory=set)
    bootstrap_batches: list[BufferedBatch] = field(default_factory=list, repr=False)
    bootstrap_vector_count: int = 0
    reservoir_vectors: list[np.ndarray] = field(default_factory=list, repr=False)
    reservoir_seen: int = 0
    rebalance_thread: threading.Thread | None = field(default=None, repr=False, compare=False)
    centers_file: str | None = None
    activation_waiting_for_worker: bool = False
    checkpoint_history: list[dict] = field(default_factory=list, repr=False)
    rebalance_history: list[dict] = field(default_factory=list, repr=False)
    latest_checkpoint_dir: str | None = None
    latest_checkpoint_centers_file: str | None = None
    communication_summary: dict[str, float | int] = field(default_factory=_new_communication_summary, repr=False)


STATE: dict[str, MasterJobState] = {}
STATE_LOCK = threading.Lock()


def _worker_endpoint_key(host: str, port: int) -> str:
    return f"{host}:{port}"


def _parse_worker_spec(raw, default_id: int, field_name: str) -> Worker:
    if isinstance(raw, dict):
        try:
            host = str(raw["host"]).strip()
            port = int(raw["port"])
        except Exception as ex:
            raise ValueError(f"{field_name} contains invalid worker entry {raw!r}") from ex
        worker_id = int(raw.get("worker_id", default_id))
        return Worker(worker_id, host, port)

    token = str(raw).strip()
    if not token:
        raise ValueError(f"{field_name} contains an empty worker entry")
    if "@" in token:
        worker_id_raw, endpoint = token.split("@", 1)
        worker_id = int(worker_id_raw)
    else:
        worker_id = default_id
        endpoint = token
    host, port = endpoint.rsplit(":", 1)
    return Worker(worker_id, host, int(port))


def _resolve_workers(worker_file: str | None, inline_workers, field_name: str) -> list[Worker]:
    if worker_file:
        arr = _read_json_file(Path(worker_file))
        if not isinstance(arr, list):
            raise ValueError(f"{field_name} must point to a JSON array of workers")
        source = arr
    else:
        source = list(inline_workers or [])

    workers: list[Worker] = []
    seen_ids: set[int] = set()
    seen_endpoints: set[str] = set()
    for idx, raw in enumerate(source):
        worker = _parse_worker_spec(raw, idx, field_name)
        endpoint_key = _worker_endpoint_key(worker.host, worker.port)
        if worker.worker_id in seen_ids:
            raise ValueError(f"{field_name} repeats worker_id={worker.worker_id}")
        if endpoint_key in seen_endpoints:
            raise ValueError(f"{field_name} repeats worker endpoint {endpoint_key}")
        seen_ids.add(worker.worker_id)
        seen_endpoints.add(endpoint_key)
        workers.append(worker)
    workers.sort(key=lambda item: item.worker_id)
    return workers


def _snapshot_workers(server) -> list[Worker]:
    with server.worker_lock:
        workers = list(server.workers.values())
    workers.sort(key=lambda w: w.worker_id)
    return workers


def _snapshot_registered_workers(server) -> list[Worker]:
    with server.worker_lock:
        workers = [worker for worker in server.workers.values() if worker.registered]
    workers.sort(key=lambda w: w.worker_id)
    return workers


def _add_configured_worker(server, worker: Worker, *, registered: bool) -> Worker:
    endpoint_key = _worker_endpoint_key(worker.host, worker.port)
    with server.worker_lock:
        existing = server.workers.get(worker.worker_id)
        if existing is not None:
            if (existing.host, existing.port) != (worker.host, worker.port):
                raise ValueError(
                    f"worker_id={worker.worker_id} is already assigned to {existing.host}:{existing.port}"
                )
            existing.registered = existing.registered or registered
            server.worker_key_to_id[endpoint_key] = worker.worker_id
            return existing

        other_id = server.worker_key_to_id.get(endpoint_key)
        if other_id is not None and other_id != worker.worker_id:
            raise ValueError(f"worker endpoint {endpoint_key} is already assigned to worker_id={other_id}")

        worker.registered = registered
        server.workers[worker.worker_id] = worker
        server.worker_key_to_id[endpoint_key] = worker.worker_id
        server.next_worker_id = max(server.next_worker_id, worker.worker_id + 1)
        return worker


def _register_worker(server, host: str, port: int, cluster_worker_id: int | None = None) -> tuple[Worker, bool]:
    endpoint_key = _worker_endpoint_key(host, port)
    with server.worker_lock:
        worker: Worker | None = None
        if cluster_worker_id is not None:
            worker = server.workers.get(cluster_worker_id)
            if worker is not None and (worker.host, worker.port) != (host, port):
                raise ValueError(
                    f"cluster_worker_id={cluster_worker_id} expected {worker.host}:{worker.port}, got {host}:{port}"
                )
        else:
            worker_id = server.worker_key_to_id.get(endpoint_key)
            if worker_id is not None:
                worker = server.workers.get(worker_id)

        if worker is None:
            worker_id = int(cluster_worker_id) if cluster_worker_id is not None else server.next_worker_id
            while worker_id in server.workers:
                worker_id += 1
            worker = Worker(worker_id, host, port)
            server.workers[worker.worker_id] = worker
            server.next_worker_id = max(server.next_worker_id, worker.worker_id + 1)

        server.worker_key_to_id[endpoint_key] = worker.worker_id
        already_registered = worker.registered
        worker.host = host
        worker.port = port
        worker.registered = True
        return worker, already_registered


def _reservoir_matrix(job: MasterJobState) -> np.ndarray:
    if not job.reservoir_vectors:
        return np.zeros((0, job.dim), dtype=np.float32)
    return np.vstack(job.reservoir_vectors).astype(np.float32, copy=False)


def _update_reservoir(job: MasterJobState, vectors: np.ndarray, reservoir_size: int) -> None:
    for row in vectors:
        row_copy = np.ascontiguousarray(row, dtype=np.float32)
        if len(job.reservoir_vectors) < reservoir_size:
            job.reservoir_vectors.append(row_copy)
        else:
            replace_idx = np.random.randint(0, job.reservoir_seen + 1)
            if replace_idx < reservoir_size:
                job.reservoir_vectors[replace_idx] = row_copy
        job.reservoir_seen += 1


def _init_worker_for_job(
    *,
    job: MasterJobState,
    worker: Worker,
    request_timeout: int,
    retries: int,
) -> ShardInfo:
    save_dir = str(Path(job.output_dir) / f"index_shard{worker.worker_id}")
    worker_init = {
        "job_id": job.job_id,
        "shard_id": worker.worker_id,
        "algo": job.algo,
        "dist": job.dist,
        "value_type": job.value_type,
        "threads": job.threads,
        "cef": job.cef,
        "max_check_for_refine_graph": job.max_check_for_refine_graph,
        "graph_neighborhood_scale": job.graph_neighborhood_scale,
        "tpt_number": job.tpt_number,
        "tpt_leaf_size": job.tpt_leaf_size,
        "dim": job.dim,
        "save_dir": save_dir,
        "with_meta_index": job.with_meta_index,
    }
    url = f"{worker.base_url}/init"
    t0 = time.time()
    resp = _post_with_retry(
        url,
        lambda u=url, p=worker_init: _http_json_post(u, p, request_timeout),
        retries,
        f"init worker shard {worker.worker_id}",
    )
    if not resp.get("ok", False):
        raise RuntimeError(f"worker init failed shard {worker.worker_id}: {resp}")

    shard = ShardInfo(shard_id=worker.worker_id, worker=worker)
    job.shards.append(shard)
    job.shards.sort(key=lambda s: s.shard_id)
    rpc_roundtrip_s = time.time() - t0
    request_handling_s = float(resp.get("request_handling_time_s", 0.0))
    job.communication_summary["init_worker_rpc_roundtrip_s"] += rpc_roundtrip_s
    job.communication_summary["init_worker_comm_overhead_s"] += max(0.0, rpc_roundtrip_s - request_handling_s)
    job.communication_summary["init_worker_count"] += 1
    print(f"[init] shard={worker.worker_id} worker={worker.host}:{worker.port} save={save_dir}")
    return shard


def _ensure_initial_job_workers(server, job: MasterJobState) -> list[ShardInfo]:
    request_timeout = server.request_timeout
    retries = server.retries
    initialized_worker_ids = {shard.worker.worker_id for shard in job.shards}
    newly_initialized: list[ShardInfo] = []

    worker_ids = list(getattr(server, "initial_worker_ids", []))
    if not worker_ids:
        threshold_ids = set(getattr(server, "threshold_join_worker_ids", []))
        worker_ids = [
            worker.worker_id
            for worker in _snapshot_workers(server)
            if worker.worker_id not in threshold_ids
        ]

    with server.worker_lock:
        workers_by_id = {worker.worker_id: worker for worker in server.workers.values()}

    for worker_id in worker_ids:
        if worker_id in initialized_worker_ids:
            continue
        worker = workers_by_id.get(worker_id)
        if worker is None:
            raise RuntimeError(f"initial worker_id={worker_id} is not configured")
        shard = _init_worker_for_job(
            job=job,
            worker=worker,
            request_timeout=request_timeout,
            retries=retries,
        )
        newly_initialized.append(shard)
        initialized_worker_ids.add(worker.worker_id)

    _refresh_activation_wait_state(server, job)
    return newly_initialized


def _maybe_activate_threshold_worker(server, job: MasterJobState) -> bool:
    _refresh_activation_wait_state(server, job)
    if int(getattr(server, "activation_threshold_vectors", 0)) <= 0 and not getattr(server, "join_at_total_vectors", []):
        return False
    if job.routing_mode != "centroid":
        return False
    if job.pending_new_workers:
        return False
    if job.rebalance_thread is not None and job.rebalance_thread.is_alive():
        return False
    if not _activation_triggered(server, job):
        job.activation_waiting_for_worker = False
        return False

    next_candidate = _next_threshold_candidate_id(server, job)
    if next_candidate is None:
        job.activation_waiting_for_worker = False
        return False

    idle_ids = _idle_registered_threshold_worker_ids(server, job)
    if next_candidate not in idle_ids:
        job.activation_waiting_for_worker = True
        return False

    with server.worker_lock:
        worker = server.workers.get(next_candidate)
    if worker is None:
        raise RuntimeError(f"threshold worker_id={next_candidate} is not configured")

    current_join_milestone = _next_join_milestone(server, job)
    _init_worker_for_job(
        job=job,
        worker=worker,
        request_timeout=server.request_timeout,
        retries=server.retries,
    )
    job.pending_new_workers.add(worker.worker_id)
    job.activation_waiting_for_worker = False
    trigger_desc = (
        f"total_vectors={job.total_vectors_ingested} milestone={current_join_milestone}"
        if getattr(server, "join_at_total_vectors", [])
        else f"threshold_vectors={getattr(server, 'activation_threshold_vectors', 0)}"
    )
    print(f"[activate] worker_id={worker.worker_id} job={job.job_id} reason={trigger_desc}")

    if not _maybe_start_rebalance(server, job):
        raise RuntimeError(f"failed to start rebalance after activating worker_id={worker.worker_id}")
    return True


def _active_shards(job: MasterJobState) -> list[ShardInfo]:
    if job.worker_centroids:
        active_ids = set(job.worker_centroids.keys())
        return [shard for shard in job.shards if shard.worker.worker_id in active_ids]
    return list(job.shards)


def _update_worker_stats(job: MasterJobState, worker_id: int, vectors: np.ndarray) -> None:
    if vectors.size == 0:
        return
    count = int(job.worker_counts.get(worker_id, 0))
    batch_sum = vectors.sum(axis=0, dtype=np.float64)
    batch_count = int(vectors.shape[0])
    if count <= 0 or worker_id not in job.worker_centroids:
        job.worker_centroids[worker_id] = (batch_sum / float(batch_count)).astype(np.float32)
        job.worker_counts[worker_id] = batch_count
        return
    old_center = job.worker_centroids[worker_id].astype(np.float64)
    new_count = count + batch_count
    new_center = ((old_center * count) + batch_sum) / float(new_count)
    job.worker_centroids[worker_id] = new_center.astype(np.float32)
    job.worker_counts[worker_id] = new_count


def _initialized_worker_ids(job: MasterJobState) -> set[int]:
    return {shard.worker.worker_id for shard in job.shards}


def _active_worker_ids(job: MasterJobState) -> set[int]:
    if job.worker_centroids:
        return set(job.worker_centroids.keys())
    return _initialized_worker_ids(job)


def _threshold_breached_worker_ids(server, job: MasterJobState) -> list[int]:
    threshold = int(getattr(server, "activation_threshold_vectors", 0))
    if threshold <= 0:
        return []
    breached = [
        worker_id
        for worker_id in sorted(_active_worker_ids(job))
        if int(job.worker_counts.get(worker_id, 0)) >= threshold
    ]
    return breached


def _initialized_threshold_worker_count(server, job: MasterJobState) -> int:
    initialized_ids = _initialized_worker_ids(job)
    return sum(1 for worker_id in getattr(server, "threshold_join_worker_ids", []) if worker_id in initialized_ids)


def _next_join_milestone(server, job: MasterJobState) -> int | None:
    milestones = list(getattr(server, "join_at_total_vectors", []))
    idx = _initialized_threshold_worker_count(server, job)
    if idx < 0 or idx >= len(milestones):
        return None
    return int(milestones[idx])


def _join_milestone_reached(server, job: MasterJobState) -> bool:
    next_milestone = _next_join_milestone(server, job)
    if next_milestone is None:
        return False
    return int(job.total_vectors_ingested) >= int(next_milestone)


def _activation_triggered(server, job: MasterJobState) -> bool:
    if getattr(server, "join_at_total_vectors", []):
        return _join_milestone_reached(server, job)
    return bool(_threshold_breached_worker_ids(server, job))


def _next_threshold_candidate_id(server, job: MasterJobState) -> int | None:
    initialized_ids = _initialized_worker_ids(job)
    for worker_id in getattr(server, "threshold_join_worker_ids", []):
        if worker_id not in initialized_ids:
            return worker_id
    return None


def _idle_registered_threshold_worker_ids(server, job: MasterJobState) -> list[int]:
    initialized_ids = _initialized_worker_ids(job)
    idle_ids: list[int] = []
    with server.worker_lock:
        for worker_id in getattr(server, "threshold_join_worker_ids", []):
            if worker_id in initialized_ids:
                continue
            worker = server.workers.get(worker_id)
            if worker is not None and worker.registered:
                idle_ids.append(worker_id)
    return idle_ids


def _refresh_activation_wait_state(server, job: MasterJobState) -> None:
    next_candidate = _next_threshold_candidate_id(server, job)
    job.activation_waiting_for_worker = bool(
        _activation_triggered(server, job)
        and next_candidate is not None
        and not _idle_registered_threshold_worker_ids(server, job)
    )


def _route_vectors_to_workers(job: MasterJobState, vectors: np.ndarray, metric: str) -> np.ndarray:
    active = _active_shards(job)
    if len(active) <= 1:
        return np.zeros(vectors.shape[0], dtype=np.int32)
    ordered_centroids = np.vstack([job.worker_centroids[shard.worker.worker_id] for shard in active]).astype(np.float32)
    return np.argmin(_distance_matrix(vectors, ordered_centroids, metric), axis=1).astype(np.int32)


def _forward_sub_batch(
    *,
    server,
    job_id: str,
    dim: int,
    with_meta_index: bool,
    shard: ShardInfo,
    batch_id: int,
    vectors: np.ndarray,
    global_ids: np.ndarray,
    normalized: bool,
) -> dict:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    global_ids = np.ascontiguousarray(global_ids, dtype=np.int64)
    worker_meta = {
        "job_id": job_id,
        "shard_id": shard.shard_id,
        "batch_id": batch_id,
        "num": int(vectors.shape[0]),
        "dim": dim,
        "with_meta_index": with_meta_index,
        "normalized": normalized,
        "ids": global_ids.astype(np.int64).tolist(),
    }
    meta_raw = json.dumps(worker_meta).encode("utf-8")
    payload = struct.pack("<Q", len(meta_raw)) + meta_raw + vectors.tobytes()
    url = f"{shard.worker.base_url}/add_batch"
    resp = _send_batch_with_backpressure(
        url=url,
        payload=payload,
        request_timeout=server.request_timeout,
        retries=server.retries,
        what=f"add_batch shard {shard.shard_id} batch {batch_id}",
    )
    if not resp.get("ok", False):
        raise RuntimeError(f"worker add_batch failed shard={shard.shard_id} batch={batch_id}: {resp}")
    if server.debug:
        print(
            f"[debug] forwarded shard={shard.shard_id} batch={batch_id} "
            f"count={vectors.shape[0]} ids=[{int(global_ids[0])}..{int(global_ids[-1])}]"
        )
    rpc_roundtrip_s = float(resp.get("_rpc_roundtrip_s", 0.0))
    request_handling_s = float(resp.get("request_handling_time_s", 0.0))
    body_read_time_s = float(resp.get("body_read_time_s", 0.0))
    processing_after_read_s = float(resp.get("post_read_processing_s", request_handling_s))
    apply_time_s = float(resp.get("apply_time_s", 0.0))
    queue_wait_time_s = float(resp.get("queue_wait_time_s", 0.0))
    backpressure_sleep_s = float(resp.get("_backpressure_sleep_s", 0.0))
    return {
        "worker_id": int(shard.worker.worker_id),
        "shard_id": int(shard.shard_id),
        "batch_id": int(batch_id),
        "rpc_roundtrip_s": rpc_roundtrip_s,
        "request_handling_time_s": request_handling_s,
        "body_read_time_s": body_read_time_s,
        "processing_after_read_s": processing_after_read_s,
        "apply_time_s": apply_time_s,
        "queue_wait_time_s": queue_wait_time_s,
        "comm_overhead_s": max(0.0, rpc_roundtrip_s - processing_after_read_s),
        "backpressure_sleep_s": backpressure_sleep_s,
        "attempts": int(resp.get("_attempts", 1)),
        "vectors": int(vectors.shape[0]),
    }


def _forward_routed_batch(server, job: MasterJobState, vectors: np.ndarray, global_ids: np.ndarray, normalized: bool) -> dict:
    active = _active_shards(job)
    if not active:
        raise RuntimeError("no active shards available")
    assignments = _route_vectors_to_workers(job, vectors, job.dist)
    routed_batches = []
    for local_idx, shard in enumerate(active):
        rows = np.flatnonzero(assignments == local_idx)
        if rows.size == 0:
            continue
        routed_batches.append(
            {
                "shard": shard,
                "batch_id": int(shard.next_batch_id),
                "vectors": np.ascontiguousarray(vectors[rows], dtype=np.float32),
                "global_ids": np.ascontiguousarray(global_ids[rows], dtype=np.int64),
            }
        )
    if not routed_batches:
        return {
            "shards_touched": 0,
            "worker_rpc_roundtrip_s": 0.0,
            "worker_rpc_roundtrip_max_s": 0.0,
            "worker_fanout_wall_time_s": 0.0,
            "worker_request_handling_s": 0.0,
            "worker_body_read_time_s": 0.0,
            "worker_processing_after_read_s": 0.0,
            "worker_processing_after_read_max_s": 0.0,
            "worker_apply_time_s": 0.0,
            "worker_queue_wait_time_s": 0.0,
            "worker_comm_overhead_s": 0.0,
            "worker_backpressure_sleep_s": 0.0,
            "worker_attempts": 0,
        }

    touched = 0
    rpc_roundtrip_s = 0.0
    rpc_roundtrip_max_s = 0.0
    fanout_wall_time_s = 0.0
    request_handling_s = 0.0
    body_read_time_s = 0.0
    processing_after_read_s = 0.0
    processing_after_read_max_s = 0.0
    apply_time_s = 0.0
    queue_wait_time_s = 0.0
    comm_overhead_s = 0.0
    backpressure_sleep_s = 0.0
    attempts = 0
    forward_started_at = time.time()
    future_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(routed_batches)) as executor:
        for routed in routed_batches:
            future = executor.submit(
                _forward_sub_batch,
                server=server,
                job_id=job.job_id,
                dim=job.dim,
                with_meta_index=job.with_meta_index,
                shard=routed["shard"],
                batch_id=routed["batch_id"],
                vectors=routed["vectors"],
                global_ids=routed["global_ids"],
                normalized=normalized,
            )
            future_map[future] = routed

        forward_results = []
        first_error: Exception | None = None
        for future in concurrent.futures.as_completed(future_map):
            routed = future_map[future]
            try:
                metrics = future.result()
            except Exception as ex:
                if first_error is None:
                    first_error = ex
                continue
            forward_results.append((routed, metrics))

    fanout_wall_time_s = max(0.0, time.time() - forward_started_at)
    if first_error is not None:
        raise first_error

    for routed, metrics in forward_results:
        shard = routed["shard"]
        sub_vectors = routed["vectors"]
        touched += 1
        shard.next_batch_id += 1
        shard.vectors_forwarded += int(sub_vectors.shape[0])
        _update_worker_stats(job, shard.worker.worker_id, sub_vectors)
        rpc_roundtrip_s += float(metrics.get("rpc_roundtrip_s", 0.0))
        rpc_roundtrip_max_s = max(rpc_roundtrip_max_s, float(metrics.get("rpc_roundtrip_s", 0.0)))
        request_handling_s += float(metrics.get("request_handling_time_s", 0.0))
        body_read_time_s += float(metrics.get("body_read_time_s", 0.0))
        processing_after_read_s += float(metrics.get("processing_after_read_s", 0.0))
        processing_after_read_max_s = max(
            processing_after_read_max_s, float(metrics.get("processing_after_read_s", 0.0))
        )
        apply_time_s += float(metrics.get("apply_time_s", 0.0))
        queue_wait_time_s += float(metrics.get("queue_wait_time_s", 0.0))
        backpressure_sleep_s += float(metrics.get("backpressure_sleep_s", 0.0))
        attempts += int(metrics.get("attempts", 0))

    comm_overhead_s = max(0.0, fanout_wall_time_s - processing_after_read_max_s)
    job.communication_summary["ingest_worker_rpc_roundtrip_s"] += rpc_roundtrip_s
    job.communication_summary["ingest_worker_rpc_roundtrip_max_s"] += rpc_roundtrip_max_s
    job.communication_summary["ingest_worker_fanout_wall_time_s"] += fanout_wall_time_s
    job.communication_summary["ingest_worker_comm_overhead_s"] += comm_overhead_s
    job.communication_summary["ingest_worker_request_total_s"] += request_handling_s
    job.communication_summary["ingest_worker_body_read_time_s"] += body_read_time_s
    job.communication_summary["ingest_worker_processing_after_read_s"] += processing_after_read_s
    job.communication_summary["ingest_worker_processing_after_read_max_s"] += processing_after_read_max_s
    job.communication_summary["ingest_worker_apply_time_s"] += apply_time_s
    job.communication_summary["ingest_worker_queue_wait_s"] += queue_wait_time_s
    job.communication_summary["ingest_worker_backpressure_sleep_s"] += backpressure_sleep_s
    job.communication_summary["ingest_worker_batches"] += touched
    job.communication_summary["ingest_worker_vectors"] += int(global_ids.shape[0])
    return {
        "shards_touched": touched,
        "worker_rpc_roundtrip_s": rpc_roundtrip_s,
        "worker_rpc_roundtrip_max_s": rpc_roundtrip_max_s,
        "worker_fanout_wall_time_s": fanout_wall_time_s,
        "worker_request_handling_s": request_handling_s,
        "worker_body_read_time_s": body_read_time_s,
        "worker_processing_after_read_s": processing_after_read_s,
        "worker_processing_after_read_max_s": processing_after_read_max_s,
        "worker_apply_time_s": apply_time_s,
        "worker_queue_wait_time_s": queue_wait_time_s,
        "worker_comm_overhead_s": comm_overhead_s,
        "worker_backpressure_sleep_s": backpressure_sleep_s,
        "worker_attempts": attempts,
    }


def _bootstrap_if_ready(server, job: MasterJobState, *, force: bool) -> bool:
    if job.routing_mode != "bootstrap":
        return False
    active = list(job.shards)
    if len(active) <= 1:
        job.routing_mode = "centroid"
        return False
    if not force and job.bootstrap_vector_count < server.bootstrap_sample_size:
        return False
    if not job.bootstrap_batches:
        return False

    sample_vectors = np.vstack([batch.vectors for batch in job.bootstrap_batches]).astype(np.float32, copy=False)
    centroids, _ = _balanced_kmeans(
        sample_vectors,
        k=len(active),
        metric=job.dist,
        seed=_seed_for_job(job.job_id, f"bootstrap:{job.routing_epoch}"),
    )
    active.sort(key=lambda shard: shard.worker.worker_id)
    job.worker_centroids = {shard.worker.worker_id: centroids[idx] for idx, shard in enumerate(active)}
    job.worker_counts = {shard.worker.worker_id: 0 for shard in active}

    buffered = list(job.bootstrap_batches)
    job.bootstrap_batches.clear()
    job.bootstrap_vector_count = 0
    job.routing_mode = "centroid"

    for batch in buffered:
        _forward_routed_batch(server, job, batch.vectors, batch.global_ids, batch.normalized)
    _refresh_activation_wait_state(server, job)
    return True


def _wait_for_worker_drain(server, shard: ShardInfo, job_id: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    url = f"{shard.worker.base_url}/status?job_id={job_id}&shard_id={shard.shard_id}"
    while True:
        resp = _post_with_retry(
            url,
            lambda u=url: _http_json_get(u, server.request_timeout),
            server.retries,
            f"status shard {shard.shard_id}",
        )
        if not resp.get("ok", False):
            raise RuntimeError(f"status failed for shard {shard.shard_id}: {resp}")
        if resp.get("error"):
            raise RuntimeError(f"worker shard {shard.shard_id} error: {resp['error']}")
        if int(resp.get("queued", 0)) == 0 and int(resp.get("next_expected_batch_id", 0)) == int(resp.get("last_applied_batch_id", -1)) + 1:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"worker shard {shard.shard_id} did not drain before rebalance")
        time.sleep(0.25)


def _prepare_rebalance_plan(job: MasterJobState) -> tuple[list[int], list[int], np.ndarray, dict[int, int]]:
    pending_ids = sorted(job.pending_new_workers)
    if not pending_ids:
        raise RuntimeError("no pending workers to rebalance")
    existing_ids = sorted(job.worker_centroids.keys())
    target_worker_ids = existing_ids + pending_ids
    sample = _reservoir_matrix(job)
    if sample.shape[0] == 0:
        raise RuntimeError("cannot rebalance without any reservoir data")
    centroids, _ = _balanced_kmeans(
        sample,
        k=len(target_worker_ids),
        metric=job.dist,
        seed=_seed_for_job(job.job_id, f"rebalance:{job.routing_epoch}"),
    )

    unmatched = set(range(len(target_worker_ids)))
    centroid_to_worker: dict[int, int] = {}
    for worker_id in existing_ids:
        old_center = job.worker_centroids.get(worker_id)
        if old_center is None:
            centroid_idx = min(unmatched)
        else:
            remaining = sorted(unmatched)
            dist_rows = _distance_matrix(old_center.reshape(1, -1), centroids[remaining], job.dist).reshape(-1)
            centroid_idx = remaining[int(np.argmin(dist_rows))]
        centroid_to_worker[centroid_idx] = worker_id
        unmatched.remove(centroid_idx)

    for worker_id, centroid_idx in zip(sorted(pending_ids), sorted(unmatched)):
        centroid_to_worker[centroid_idx] = worker_id

    return existing_ids, target_worker_ids, centroids, centroid_to_worker


def _send_rebalance_phase(server, shard: ShardInfo, payload: dict, timeout_s: int) -> dict:
    url = f"{shard.worker.base_url}/rebalance"
    t0 = time.time()
    resp = _post_with_retry(
        url,
        lambda u=url, p=payload: _http_json_post(u, p, timeout_s),
        server.retries,
        f"rebalance shard {shard.shard_id} phase {payload['phase']}",
    )
    rpc_elapsed = time.time() - t0
    if not resp.get("ok", False):
        raise RuntimeError(f"rebalance phase {payload['phase']} failed for shard {shard.shard_id}: {resp}")
    resp = dict(resp)
    worker_phase_time = float(resp.get("phase_time_s", 0.0))
    resp["_master_rpc_roundtrip_s"] = rpc_elapsed
    resp["_master_comm_overhead_s"] = max(0.0, rpc_elapsed - worker_phase_time)
    return resp


def _run_rebalance(server, job_id: str, routing_epoch: int) -> None:
    with STATE_LOCK:
        job = STATE.get(job_id)
    if job is None:
        return

    try:
        rebalance_started = time.time()
        with job.lock:
            triggered_total_vectors = int(job.total_vectors_ingested)
            existing_ids, target_worker_ids, centroids, centroid_to_worker = _prepare_rebalance_plan(job)
            target_shards = [shard for shard in job.shards if shard.worker.worker_id in set(target_worker_ids)]
            target_shards.sort(key=lambda shard: shard.worker.worker_id)
            new_worker_ids = sorted(job.pending_new_workers)

        drain_started = time.time()
        for shard in target_shards:
            _wait_for_worker_drain(server, shard, job.job_id, server.rebalance_timeout)
        drain_elapsed = time.time() - drain_started

        workers_payload = [
            {
                "worker_id": shard.worker.worker_id,
                "base_url": shard.worker.base_url,
            }
            for shard in target_shards
        ]
        phase_payloads = []
        for shard in target_shards:
            assigned = None
            for centroid_idx, worker_id in centroid_to_worker.items():
                if worker_id == shard.worker.worker_id:
                    assigned = centroid_idx
                    break
            if assigned is None:
                raise RuntimeError(f"missing centroid assignment for worker {shard.worker.worker_id}")
            phase_payloads.append(
                (
                    shard,
                    {
                        "job_id": job.job_id,
                        "shard_id": shard.shard_id,
                        "routing_epoch": routing_epoch,
                        "phase": "prepare",
                        "centroids": centroids.tolist(),
                        "assigned_centroid_idx": int(assigned),
                        "centroid_to_worker": {str(k): int(v) for k, v in centroid_to_worker.items()},
                        "workers": workers_payload,
                    },
                )
            )

        prepare_started = time.time()
        prepare_results: dict[int, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(phase_payloads)) as executor:
            future_map = {
                executor.submit(_send_rebalance_phase, server, shard, payload, server.rebalance_timeout): shard.worker.worker_id
                for shard, payload in phase_payloads
            }
            for future in concurrent.futures.as_completed(future_map):
                worker_id = future_map[future]
                prepare_results[worker_id] = future.result()
        prepare_elapsed = time.time() - prepare_started

        migrate_payloads = [
            (
                shard,
                {
                    "job_id": job.job_id,
                    "shard_id": shard.shard_id,
                    "routing_epoch": routing_epoch,
                    "phase": "migrate",
                },
            )
            for shard in target_shards
        ]
        migrate_started = time.time()
        migrate_results: dict[int, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(migrate_payloads)) as executor:
            future_map = {
                executor.submit(_send_rebalance_phase, server, shard, payload, server.rebalance_timeout): shard.worker.worker_id
                for shard, payload in migrate_payloads
            }
            for future in concurrent.futures.as_completed(future_map):
                worker_id = future_map[future]
                migrate_results[worker_id] = future.result()
        migrate_elapsed = time.time() - migrate_started

        rebuild_payloads = [
            (
                shard,
                {
                    "job_id": job.job_id,
                    "shard_id": shard.shard_id,
                    "routing_epoch": routing_epoch,
                    "phase": "rebuild",
                },
            )
            for shard in target_shards
        ]
        rebuild_started = time.time()
        rebuild_results: dict[int, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(rebuild_payloads)) as executor:
            future_map = {
                executor.submit(_send_rebalance_phase, server, shard, payload, server.rebalance_timeout): shard.worker.worker_id
                for shard, payload in rebuild_payloads
            }
            for future in concurrent.futures.as_completed(future_map):
                worker_id = future_map[future]
                rebuild_results[worker_id] = future.result()
        rebuild_elapsed = time.time() - rebuild_started
        total_elapsed = time.time() - rebalance_started
        prepare_max_worker_phase_s = max((float(result.get("phase_time_s", 0.0)) for result in prepare_results.values()), default=0.0)
        migrate_max_worker_phase_s = max((float(result.get("phase_time_s", 0.0)) for result in migrate_results.values()), default=0.0)
        rebuild_max_worker_phase_s = max((float(result.get("phase_time_s", 0.0)) for result in rebuild_results.values()), default=0.0)
        prepare_master_phase_overhead_s = max(0.0, prepare_elapsed - prepare_max_worker_phase_s)
        migrate_master_phase_overhead_s = max(0.0, migrate_elapsed - migrate_max_worker_phase_s)
        rebuild_master_phase_overhead_s = max(0.0, rebuild_elapsed - rebuild_max_worker_phase_s)

        rebalance_record = {
            "routing_epoch": int(routing_epoch),
            "triggered_total_vectors": triggered_total_vectors,
            "existing_worker_ids": list(existing_ids),
            "new_worker_ids": list(new_worker_ids),
            "target_worker_ids": list(target_worker_ids),
            "drain_time_s": round(drain_elapsed, 6),
            "prepare_phase_s": round(prepare_elapsed, 6),
            "migrate_phase_s": round(migrate_elapsed, 6),
            "rebuild_phase_s": round(rebuild_elapsed, 6),
            "rebalance_total_s": round(total_elapsed, 6),
            "prepare_master_rpc_roundtrip_s": {
                str(worker_id): round(float(result.get("_master_rpc_roundtrip_s", 0.0)), 6)
                for worker_id, result in sorted(prepare_results.items())
            },
            "prepare_master_comm_overhead_s": {
                str(worker_id): round(float(result.get("_master_comm_overhead_s", 0.0)), 6)
                for worker_id, result in sorted(prepare_results.items())
            },
            "prepare_master_rpc_total_s": round(
                sum(float(result.get("_master_rpc_roundtrip_s", 0.0)) for result in prepare_results.values()), 6
            ),
            "prepare_master_comm_total_s": round(
                sum(float(result.get("_master_comm_overhead_s", 0.0)) for result in prepare_results.values()), 6
            ),
            "prepare_master_max_worker_phase_s": round(prepare_max_worker_phase_s, 6),
            "prepare_master_phase_overhead_s": round(prepare_master_phase_overhead_s, 6),
            "migrate_master_rpc_roundtrip_s": {
                str(worker_id): round(float(result.get("_master_rpc_roundtrip_s", 0.0)), 6)
                for worker_id, result in sorted(migrate_results.items())
            },
            "migrate_master_comm_overhead_s": {
                str(worker_id): round(float(result.get("_master_comm_overhead_s", 0.0)), 6)
                for worker_id, result in sorted(migrate_results.items())
            },
            "migrate_master_rpc_total_s": round(
                sum(float(result.get("_master_rpc_roundtrip_s", 0.0)) for result in migrate_results.values()), 6
            ),
            "migrate_master_comm_total_s": round(
                sum(float(result.get("_master_comm_overhead_s", 0.0)) for result in migrate_results.values()), 6
            ),
            "migrate_master_max_worker_phase_s": round(migrate_max_worker_phase_s, 6),
            "migrate_master_phase_overhead_s": round(migrate_master_phase_overhead_s, 6),
            "rebuild_master_rpc_roundtrip_s": {
                str(worker_id): round(float(result.get("_master_rpc_roundtrip_s", 0.0)), 6)
                for worker_id, result in sorted(rebuild_results.items())
            },
            "rebuild_master_comm_overhead_s": {
                str(worker_id): round(float(result.get("_master_comm_overhead_s", 0.0)), 6)
                for worker_id, result in sorted(rebuild_results.items())
            },
            "rebuild_master_rpc_total_s": round(
                sum(float(result.get("_master_rpc_roundtrip_s", 0.0)) for result in rebuild_results.values()), 6
            ),
            "rebuild_master_comm_total_s": round(
                sum(float(result.get("_master_comm_overhead_s", 0.0)) for result in rebuild_results.values()), 6
            ),
            "rebuild_master_max_worker_phase_s": round(rebuild_max_worker_phase_s, 6),
            "rebuild_master_phase_overhead_s": round(rebuild_master_phase_overhead_s, 6),
            "master_rebalance_overhead_s": round(
                prepare_master_phase_overhead_s + migrate_master_phase_overhead_s + rebuild_master_phase_overhead_s,
                6,
            ),
            "prepare_worker_phase_s": {
                str(worker_id): float(result.get("phase_time_s", 0.0))
                for worker_id, result in sorted(prepare_results.items())
            },
            "migrate_worker_phase_s": {
                str(worker_id): float(result.get("phase_time_s", 0.0))
                for worker_id, result in sorted(migrate_results.items())
            },
            "rebuild_worker_phase_s": {
                str(worker_id): float(result.get("phase_time_s", 0.0))
                for worker_id, result in sorted(rebuild_results.items())
            },
            "worker_rebalance_total_s": {
                str(worker_id): float(result.get("rebalance_total_time_s", 0.0))
                for worker_id, result in sorted(rebuild_results.items())
            },
            "worker_migrate_rpc_roundtrip_s": {
                str(worker_id): float(result.get("migrate_rpc_roundtrip_s", 0.0))
                for worker_id, result in sorted(migrate_results.items())
            },
            "worker_migrate_remote_apply_time_s": {
                str(worker_id): float(result.get("migrate_remote_apply_time_s", 0.0))
                for worker_id, result in sorted(migrate_results.items())
            },
            "worker_migrate_comm_time_s": {
                str(worker_id): float(result.get("migrate_comm_time_s", 0.0))
                for worker_id, result in sorted(migrate_results.items())
            },
            "worker_migrate_delete_time_s": {
                str(worker_id): float(result.get("migrate_delete_time_s", 0.0))
                for worker_id, result in sorted(migrate_results.items())
            },
            "worker_migrate_batches_sent": {
                str(worker_id): int(result.get("migrate_batches_sent", 0))
                for worker_id, result in sorted(migrate_results.items())
            },
            "worker_migrate_vectors_sent": {
                str(worker_id): int(result.get("moved_vectors", 0))
                for worker_id, result in sorted(migrate_results.items())
            },
        }

        with job.lock:
            job.worker_centroids = {
                worker_id: np.asarray(result["centroid"], dtype=np.float32)
                for worker_id, result in rebuild_results.items()
            }
            job.worker_counts = {
                worker_id: int(result.get("active_count", 0))
                for worker_id, result in rebuild_results.items()
            }
            for worker_id in target_worker_ids:
                job.pending_new_workers.discard(worker_id)
            job.routing_mode = "centroid"
            job.rebalance_thread = None
            job.rebalance_history.append(rebalance_record)
            _refresh_activation_wait_state(server, job)
            _maybe_activate_threshold_worker(server, job)

        print(
            f"[rebalance] epoch={routing_epoch} total_vectors={triggered_total_vectors} "
            f"workers={target_worker_ids} drain={drain_elapsed:.3f}s "
            f"prepare={prepare_elapsed:.3f}s migrate={migrate_elapsed:.3f}s "
            f"rebuild={rebuild_elapsed:.3f}s total={total_elapsed:.3f}s "
            f"master_overhead={rebalance_record['master_rebalance_overhead_s']:.3f}s "
            f"master_prepare_comm={rebalance_record['prepare_master_comm_total_s']:.3f}s "
            f"master_migrate_comm={rebalance_record['migrate_master_comm_total_s']:.3f}s "
            f"master_rebuild_comm={rebalance_record['rebuild_master_comm_total_s']:.3f}s"
        )

    except Exception as ex:
        with job.lock:
            job.error = str(ex)
            job.rebalance_thread = None
            job.rebalance_history.append(
                {
                    "routing_epoch": int(routing_epoch),
                    "triggered_total_vectors": int(job.total_vectors_ingested),
                    "error": str(ex),
                }
            )


def _maybe_start_rebalance(server, job: MasterJobState) -> bool:
    if not job.pending_new_workers:
        _refresh_activation_wait_state(server, job)
        return False
    if job.rebalance_thread is not None and job.rebalance_thread.is_alive():
        return True
    if not server.rebalance_enabled:
        return False
    if not job.worker_centroids:
        if len(job.shards) > 1:
            job.pending_new_workers.clear()
            job.routing_mode = "bootstrap"
        else:
            job.pending_new_workers.clear()
        _refresh_activation_wait_state(server, job)
        return False
    if _reservoir_matrix(job).shape[0] == 0:
        job.pending_new_workers.clear()
        job.worker_centroids.clear()
        job.worker_counts.clear()
        job.routing_mode = "bootstrap" if len(job.shards) > 1 else "centroid"
        _refresh_activation_wait_state(server, job)
        return False

    job.routing_mode = "rebalancing"
    job.activation_waiting_for_worker = False
    job.routing_epoch += 1
    worker_thread = threading.Thread(
        target=_run_rebalance,
        args=(server, job.job_id, job.routing_epoch),
        daemon=True,
        name=f"rebalance-{job.job_id}-{job.routing_epoch}",
    )
    job.rebalance_thread = worker_thread
    worker_thread.start()
    return True


def _write_centers_file(path: Path, centroids_by_worker: dict[int, np.ndarray], dim: int) -> None:
    ordered_ids = sorted(centroids_by_worker)
    matrix = np.vstack([centroids_by_worker[worker_id] for worker_id in ordered_ids]).astype(np.float32, copy=False)
    if matrix.ndim != 2:
        raise ValueError(f"invalid centers matrix shape: {matrix.shape}")
    if matrix.shape[1] != dim:
        raise ValueError(f"center dim mismatch: matrix has {matrix.shape[1]}, expected {dim}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", int(matrix.shape[0]), int(matrix.shape[1])))
        f.write(np.ascontiguousarray(matrix, dtype=np.float32).tobytes())


def _wait_until_job_stable_for_snapshot(server, job: MasterJobState) -> None:
    while True:
        with job.lock:
            if job.error is not None:
                raise RuntimeError(f"job error: {job.error}")
            if job.finalized:
                return
            if job.routing_mode == "bootstrap" and job.bootstrap_batches:
                _bootstrap_if_ready(server, job, force=True)
            _maybe_activate_threshold_worker(server, job)
            if job.pending_new_workers and job.routing_mode != "rebalancing":
                _maybe_start_rebalance(server, job)
            rebalance_thread = job.rebalance_thread

        if rebalance_thread is None:
            return
        rebalance_thread.join()


def _checkpoint_job(server, job: MasterJobState, checkpoint_id: str) -> dict:
    _wait_until_job_stable_for_snapshot(server, job)

    checkpoint_root = Path(job.output_dir) / "checkpoints" / checkpoint_id
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    with job.lock:
        active_shards = sorted(_active_shards(job), key=lambda item: item.worker.worker_id)
        checkpoint_timeout = getattr(server, "checkpoint_timeout", getattr(server, "finalize_timeout", 600))
        retries = server.retries
        shard_results = []

        for shard in active_shards:
            req = {
                "job_id": job.job_id,
                "shard_id": shard.shard_id,
                "checkpoint_dir": str(checkpoint_root / f"index_shard{shard.worker.worker_id}"),
            }
            url = f"{shard.worker.base_url}/checkpoint"
            try:
                t0 = time.time()
                resp = _post_with_retry(
                    url,
                    lambda u=url, p=req: _http_json_post(u, p, checkpoint_timeout),
                    retries,
                    f"checkpoint shard {shard.shard_id}",
                )
                rpc_roundtrip_s = time.time() - t0
                if not resp.get("ok", False):
                    raise RuntimeError(f"worker checkpoint failed shard {shard.shard_id}: {resp}")
            except Exception as ex:
                raise RuntimeError(f"worker checkpoint failed shard {shard.shard_id}: {ex}") from ex

            centroid = np.asarray(resp.get("centroid", np.zeros(job.dim, dtype=np.float32).tolist()), dtype=np.float32)
            job.worker_centroids[shard.worker.worker_id] = centroid
            job.worker_counts[shard.worker.worker_id] = int(resp.get("active_vectors", resp.get("vectors_ingested", 0)))
            request_handling_s = float(resp.get("request_handling_time_s", 0.0))
            job.communication_summary["checkpoint_worker_rpc_roundtrip_s"] += rpc_roundtrip_s
            job.communication_summary["checkpoint_worker_comm_overhead_s"] += max(0.0, rpc_roundtrip_s - request_handling_s)
            job.communication_summary["checkpoint_worker_count"] += 1
            shard_results.append(
                {
                    "shard_id": shard.shard_id,
                    "worker_id": shard.worker.worker_id,
                    "worker": f"{shard.worker.host}:{shard.worker.port}",
                    "active_vectors": resp.get("active_vectors", 0),
                    "checkpoint_dir": resp.get("checkpoint_dir", ""),
                    "checkpoint_time_s": resp.get("checkpoint_time_s", 0.0),
                    "add_time_s": resp.get("add_time_s", 0.0),
                    "elapsed_s": resp.get("elapsed_s", 0.0),
                    "rebalance_prepare_time_s": resp.get("rebalance_prepare_time_s", 0.0),
                    "rebalance_migrate_time_s": resp.get("rebalance_migrate_time_s", 0.0),
                    "rebalance_rebuild_time_s": resp.get("rebalance_rebuild_time_s", 0.0),
                    "rebalance_total_time_s": resp.get("rebalance_total_time_s", 0.0),
                }
            )
            print(
                f"[checkpoint] shard={shard.shard_id} dir={resp.get('checkpoint_dir')} "
                f"active={resp.get('active_vectors', 0)} "
                f"worker_checkpoint={resp.get('checkpoint_time_s', 0):.3f}s"
            )

        active_centroids = {
            shard.worker.worker_id: job.worker_centroids[shard.worker.worker_id]
            for shard in active_shards
            if shard.worker.worker_id in job.worker_centroids
        }
        centers_path = checkpoint_root / "centers"
        _write_centers_file(centers_path, active_centroids, job.dim)

        record = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_dir": str(checkpoint_root),
            "centers_file": str(centers_path),
            "total_vectors_ingested": int(job.total_vectors_ingested),
            "build_elapsed_s": _build_elapsed_s(job),
            "wall_elapsed_s": _wall_elapsed_s(job),
            "routing_epoch": int(job.routing_epoch),
            "active_workers": [shard.worker.worker_id for shard in active_shards],
            "shards": shard_results,
        }
        job.latest_checkpoint_dir = str(checkpoint_root)
        job.latest_checkpoint_centers_file = str(centers_path)
        job.checkpoint_history.append(record)
        return record


class MasterHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            return self._handle_status(parsed)
        _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path == "/register_worker":
            return self._handle_register_worker()
        if self.path == "/init":
            return self._handle_init()
        if self.path == "/add_batch":
            return self._handle_add_batch()
        if self.path == "/checkpoint":
            return self._handle_checkpoint()
        if self.path == "/build_timing":
            return self._handle_build_timing()
        if self.path == "/finalize":
            return self._handle_finalize()
        if self.path == "/shutdown":
            return self._handle_shutdown()
        _json_response(self, 404, {"ok": False, "error": "not found"})

    def _handle_register_worker(self):
        req = _read_json(self)
        try:
            host = str(req["host"]).strip()
            port = int(req["port"])
            cluster_worker_id = req.get("cluster_worker_id")
            if cluster_worker_id is not None:
                cluster_worker_id = int(cluster_worker_id)
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid worker payload: {ex}"})
            return

        if not host:
            _json_response(self, 400, {"ok": False, "error": "host is required"})
            return

        try:
            worker, already_registered = _register_worker(self.server, host, port, cluster_worker_id)
        except Exception as ex:
            _json_response(self, 409, {"ok": False, "error": str(ex)})
            return
        if not already_registered:
            print(f"[worker] registered worker_id={worker.worker_id} addr={worker.host}:{worker.port}")

        _json_response(
            self,
            200,
            {
                "ok": True,
                "worker_id": worker.worker_id,
                "host": worker.host,
                "port": worker.port,
                "already_registered": already_registered,
                "registered_workers": len(_snapshot_registered_workers(self.server)),
            },
        )

    def _handle_init(self):
        request_started = time.time()
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            algo = str(req.get("algo", "BKT"))
            dist = str(req.get("dist", "L2"))
            dim = int(req["dim"])
            output_dir = str(req["output_dir"])
            value_type = str(req.get("value_type", "Float"))
            threads = int(req.get("threads", 8))
            cef = _optional_positive_int(req.get("cef"), "cef")
            max_check_for_refine_graph = _optional_positive_int(
                req.get("max_check_for_refine_graph"), "max_check_for_refine_graph"
            )
            graph_neighborhood_scale = _optional_positive_float(
                req.get("graph_neighborhood_scale"), "graph_neighborhood_scale"
            )
            tpt_number = _optional_positive_int(req.get("tpt_number"), "tpt_number")
            tpt_leaf_size = _optional_positive_int(req.get("tpt_leaf_size"), "tpt_leaf_size")
            with_meta_index = bool(req.get("with_meta_index", False))
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid init payload: {ex}"})
            return

        if value_type not in {"Float", "UInt8"}:
            _json_response(self, 400, {"ok": False, "error": f"unsupported value_type: {value_type}"})
            return

        job = MasterJobState(
            job_id=job_id,
            dim=dim,
            algo=algo,
            dist=dist,
            value_type=value_type,
            output_dir=output_dir,
            threads=threads,
            cef=cef,
            max_check_for_refine_graph=max_check_for_refine_graph,
            graph_neighborhood_scale=graph_neighborhood_scale,
            tpt_number=tpt_number,
            tpt_leaf_size=tpt_leaf_size,
            with_meta_index=with_meta_index,
            init_time=time.time(),
        )

        try:
            with job.lock:
                _ensure_initial_job_workers(self.server, job)
                if not job.shards:
                    raise RuntimeError("no initial workers are configured for this job")
                job.routing_mode = "centroid" if len(job.shards) <= 1 else "bootstrap"
                _refresh_activation_wait_state(self.server, job)
        except Exception as ex:
            _json_response(self, 502, {"ok": False, "error": str(ex)})
            return

        with STATE_LOCK:
            STATE[job_id] = job

        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "num_shards": len(job.shards),
                "registered_workers": len(_snapshot_registered_workers(self.server)),
                "routing_mode": job.routing_mode,
                "request_total_s": max(0.0, time.time() - request_started),
            },
        )

    def _handle_add_batch(self):
        request_started = time.time()
        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len)
        if len(raw) < 8:
            _json_response(self, 400, {"ok": False, "error": "payload too small"})
            return

        meta_len = struct.unpack("<Q", raw[:8])[0]
        if len(raw) < 8 + meta_len:
            _json_response(self, 400, {"ok": False, "error": "invalid meta length"})
            return

        meta = json.loads(raw[8 : 8 + meta_len].decode("utf-8"))
        vec_bytes = raw[8 + meta_len :]

        try:
            job_id = str(meta["job_id"])
            global_offset = int(meta["global_offset"])
            num = int(meta["num"])
            dim = int(meta["dim"])
            with_meta_index = bool(meta.get("with_meta_index", False))
            normalized = bool(meta.get("normalized", False))
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid batch meta: {ex}"})
            return

        with STATE_LOCK:
            job = STATE.get(job_id)

        if job is None:
            _json_response(self, 404, {"ok": False, "error": "job not initialized"})
            return

        with job.lock:
            if job.finalized:
                _json_response(self, 409, {"ok": False, "error": "job already finalized"})
                return
            if job.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"job error: {job.error}"})
                return
            if dim != job.dim:
                _json_response(self, 400, {"ok": False, "error": f"dim mismatch: batch {dim}, job {job.dim}"})
                return
            if with_meta_index != job.with_meta_index:
                _json_response(self, 400, {"ok": False, "error": "with_meta_index changed within the same job"})
                return

            expected_bytes = num * dim * 4
            if len(vec_bytes) != expected_bytes:
                _json_response(
                    self,
                    400,
                    {"ok": False, "error": f"invalid vector payload bytes={len(vec_bytes)}, expected={expected_bytes}"},
                )
                return

            active = _active_shards(job)
            if not active:
                _json_response(
                    self,
                    429,
                    {"ok": False, "queue_full": True, "retry_after_ms": self.server.retry_after_ms, "error": "no workers registered yet"},
                )
                return

            vectors = np.frombuffer(vec_bytes, dtype=np.float32).reshape(num, dim)
            global_ids = (np.arange(num, dtype=np.int64) + global_offset).astype(np.int64, copy=False)
            _update_reservoir(job, vectors, self.server.reservoir_size)

            if job.routing_mode == "rebalancing":
                _json_response(
                    self,
                    429,
                    {"ok": False, "queue_full": True, "retry_after_ms": self.server.retry_after_ms, "error": "rebalance in progress"},
                )
                return

            if job.routing_mode == "bootstrap" and len(job.shards) > 1:
                job.bootstrap_batches.append(
                    BufferedBatch(
                        global_ids=np.ascontiguousarray(global_ids, dtype=np.int64),
                        vectors=np.ascontiguousarray(vectors, dtype=np.float32),
                        normalized=normalized,
                    )
                )
                job.bootstrap_vector_count += num
                try:
                    job.total_vectors_ingested += num
                    bootstrap_ready = _bootstrap_if_ready(self.server, job, force=False)
                    if bootstrap_ready:
                        _maybe_activate_threshold_worker(self.server, job)
                except Exception as ex:
                    job.error = str(ex)
                    _json_response(self, 502, {"ok": False, "error": str(ex)})
                    return
                if bootstrap_ready:
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "job_id": job_id,
                            "vectors_accepted": num,
                            "buffered": False,
                            "routing_mode": job.routing_mode,
                            "request_total_s": max(0.0, time.time() - request_started),
                        },
                    )
                    return
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "job_id": job_id,
                        "vectors_accepted": num,
                        "buffered": True,
                        "bootstrap_buffered_vectors": job.bootstrap_vector_count,
                        "request_total_s": max(0.0, time.time() - request_started),
                    },
                )
                return

            if job.pending_new_workers and _maybe_start_rebalance(self.server, job):
                _json_response(
                    self,
                    429,
                    {"ok": False, "queue_full": True, "retry_after_ms": self.server.retry_after_ms, "error": "rebalance started"},
                )
                return

            try:
                forward_metrics = _forward_routed_batch(self.server, job, vectors, global_ids, normalized)
                shards_touched = int(forward_metrics.get("shards_touched", 0))
                job.total_vectors_ingested += num
                _maybe_activate_threshold_worker(self.server, job)
            except Exception as ex:
                job.error = str(ex)
                print(f"[error] add_batch forward failed: {ex}")
                _json_response(self, 502, {"ok": False, "error": str(ex)})
                return

        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "vectors_accepted": num,
                "shards_touched": shards_touched,
                "routing_mode": job.routing_mode,
                "request_total_s": max(0.0, time.time() - request_started),
                "master_worker_rpc_roundtrip_s": float(forward_metrics.get("worker_rpc_roundtrip_s", 0.0)),
                "master_worker_rpc_roundtrip_max_s": float(forward_metrics.get("worker_rpc_roundtrip_max_s", 0.0)),
                "master_worker_fanout_wall_time_s": float(forward_metrics.get("worker_fanout_wall_time_s", 0.0)),
                "master_worker_request_handling_s": float(forward_metrics.get("worker_request_handling_s", 0.0)),
                "master_worker_body_read_time_s": float(forward_metrics.get("worker_body_read_time_s", 0.0)),
                "master_worker_processing_after_read_s": float(forward_metrics.get("worker_processing_after_read_s", 0.0)),
                "master_worker_processing_after_read_max_s": float(
                    forward_metrics.get("worker_processing_after_read_max_s", 0.0)
                ),
                "master_worker_apply_time_s": float(forward_metrics.get("worker_apply_time_s", 0.0)),
                "master_worker_queue_wait_time_s": float(forward_metrics.get("worker_queue_wait_time_s", 0.0)),
                "master_worker_comm_overhead_s": float(forward_metrics.get("worker_comm_overhead_s", 0.0)),
                "master_worker_backpressure_sleep_s": float(forward_metrics.get("worker_backpressure_sleep_s", 0.0)),
                "master_worker_attempts": int(forward_metrics.get("worker_attempts", 0)),
            },
        )

    def _handle_checkpoint(self):
        request_started = time.time()
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            checkpoint_id = str(req.get("checkpoint_id") or "")
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid checkpoint payload: {ex}"})
            return

        with STATE_LOCK:
            job = STATE.get(job_id)

        if job is None:
            _json_response(self, 404, {"ok": False, "error": "job not found"})
            return

        if not checkpoint_id:
            checkpoint_id = str(int(job.total_vectors_ingested))

        try:
            record = _checkpoint_job(self.server, job, checkpoint_id)
        except Exception as ex:
            with job.lock:
                job.error = str(ex)
            _json_response(self, 502, {"ok": False, "error": str(ex)})
            return

        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "request_total_s": max(0.0, time.time() - request_started),
                "communication_summary": dict(job.communication_summary),
                **record,
            },
        )

    def _handle_build_timing(self):
        request_started = time.time()
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            action = str(req["action"]).strip().lower()
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid build_timing payload: {ex}"})
            return

        if action not in {"pause", "resume"}:
            _json_response(self, 400, {"ok": False, "error": "action must be 'pause' or 'resume'"})
            return

        with STATE_LOCK:
            job = STATE.get(job_id)

        if job is None:
            _json_response(self, 404, {"ok": False, "error": "job not found"})
            return

        with job.lock:
            changed = _pause_build_timing(job) if action == "pause" else _resume_build_timing(job)
            payload = {
                "ok": True,
                "job_id": job_id,
                "action": action,
                "changed": changed,
                "build_elapsed_s": _build_elapsed_s(job),
                "wall_elapsed_s": _wall_elapsed_s(job),
                "build_timing_paused": bool(job.build_pause_started_at is not None),
                "build_paused_time_s": float(job.build_paused_time_s),
                "request_total_s": max(0.0, time.time() - request_started),
            }

        _json_response(self, 200, payload)

    def _handle_finalize(self):
        request_started = time.time()
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid finalize payload: {ex}"})
            return

        with STATE_LOCK:
            job = STATE.get(job_id)

        if job is None:
            _json_response(self, 404, {"ok": False, "error": "job not found"})
            return

        try:
            _wait_until_job_stable_for_snapshot(self.server, job)
        except Exception as ex:
            with job.lock:
                job.error = str(ex)
            _json_response(self, 502, {"ok": False, "error": str(ex)})
            return

        with job.lock:
            if job.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"job error: {job.error}"})
                return

            finalize_timeout = self.server.finalize_timeout
            retries = self.server.retries
            shard_results = []

            for shard in sorted(job.shards, key=lambda item: item.worker.worker_id):
                fin_req = {"job_id": job_id, "shard_id": shard.shard_id}
                url = f"{shard.worker.base_url}/finalize"
                try:
                    t0 = time.time()
                    resp = _post_with_retry(
                        url,
                        lambda u=url, p=fin_req: _http_json_post(u, p, finalize_timeout),
                        retries,
                        f"finalize shard {shard.shard_id}",
                    )
                    rpc_roundtrip_s = time.time() - t0
                    if not resp.get("ok", False):
                        raise RuntimeError(f"worker finalize failed shard {shard.shard_id}: {resp}")
                except Exception as ex:
                    err_msg = f"worker finalize failed shard {shard.shard_id}: {ex}"
                    print(f"[error] {err_msg}")
                    job.error = err_msg
                    _json_response(self, 502, {"ok": False, "error": err_msg})
                    return

                centroid = np.asarray(resp.get("centroid", np.zeros(job.dim, dtype=np.float32).tolist()), dtype=np.float32)
                job.worker_centroids[shard.worker.worker_id] = centroid
                job.worker_counts[shard.worker.worker_id] = int(resp.get("active_vectors", resp.get("vectors_ingested", 0)))
                request_handling_s = float(resp.get("request_handling_time_s", 0.0))
                job.communication_summary["finalize_worker_rpc_roundtrip_s"] += rpc_roundtrip_s
                job.communication_summary["finalize_worker_comm_overhead_s"] += max(0.0, rpc_roundtrip_s - request_handling_s)
                job.communication_summary["finalize_worker_count"] += 1

                shard_results.append(
                    {
                        "shard_id": shard.shard_id,
                        "worker_id": shard.worker.worker_id,
                        "worker": f"{shard.worker.host}:{shard.worker.port}",
                        "vectors_ingested": resp.get("vectors_ingested", 0),
                        "active_vectors": resp.get("active_vectors", 0),
                        "save_dir": resp.get("save_dir", ""),
                        "add_time_s": resp.get("add_time_s", 0.0),
                        "finalize_time_s": resp.get("finalize_time_s", 0.0),
                        "elapsed_s": resp.get("elapsed_s", 0.0),
                        "checkpoint_time_s": resp.get("checkpoint_time_s", 0.0),
                        "rebalance_prepare_time_s": resp.get("rebalance_prepare_time_s", 0.0),
                        "rebalance_migrate_time_s": resp.get("rebalance_migrate_time_s", 0.0),
                        "rebalance_rebuild_time_s": resp.get("rebalance_rebuild_time_s", 0.0),
                        "rebalance_total_time_s": resp.get("rebalance_total_time_s", 0.0),
                    }
                )
                print(
                    f"[finalize] shard={shard.shard_id} saved={resp.get('save_dir')} "
                    f"vectors={resp.get('vectors_ingested')} "
                    f"worker_add={resp.get('add_time_s', 0):.3f}s "
                    f"worker_finalize={resp.get('finalize_time_s', 0):.3f}s"
                )

            centers_path = Path(job.output_dir) / "centers"
            _write_centers_file(centers_path, job.worker_centroids, job.dim)
            job.centers_file = str(centers_path)
            job.finalized = True

        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "shards": shard_results,
                "centers_file": job.centers_file,
                "build_elapsed_s": _build_elapsed_s(job),
                "wall_elapsed_s": _wall_elapsed_s(job),
                "request_total_s": max(0.0, time.time() - request_started),
                "communication_summary": dict(job.communication_summary),
            },
        )

        if getattr(self.server, "exit_after_finalize", False):
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _handle_status(self, parsed):
        qs = parse_qs(parsed.query)
        job_id = qs.get("job_id", [None])[0]
        if job_id is None:
            _json_response(self, 400, {"ok": False, "error": "job_id is required"})
            return

        with STATE_LOCK:
            job = STATE.get(job_id)
            if job is None:
                _json_response(self, 404, {"ok": False, "error": "job not found"})
                return
        with job.lock:
            active_worker_ids = sorted(_active_worker_ids(job))
            idle_registered_worker_ids = _idle_registered_threshold_worker_ids(self.server, job)
            next_activation_candidate = _next_threshold_candidate_id(self.server, job)
            threshold_breached = bool(_threshold_breached_worker_ids(self.server, job))
            activation_triggered = bool(_activation_triggered(self.server, job))
            next_join_milestone = _next_join_milestone(self.server, job)
            shard_info = []
            for shard in sorted(job.shards, key=lambda item: item.worker.worker_id):
                shard_info.append(
                    {
                        "shard_id": shard.shard_id,
                        "worker_id": shard.worker.worker_id,
                        "worker": f"{shard.worker.host}:{shard.worker.port}",
                        "next_batch_id": shard.next_batch_id,
                        "vectors_forwarded": shard.vectors_forwarded,
                        "active": shard.worker.worker_id in active_worker_ids,
                        "pending_rebalance": shard.worker.worker_id in job.pending_new_workers,
                        "worker_count": int(job.worker_counts.get(shard.worker.worker_id, 0)),
                    }
                )
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "job_id": job_id,
                    "dim": job.dim,
                    "algo": job.algo,
                    "value_type": job.value_type,
                    "build_params": _build_params_payload(
                        job.cef,
                        job.max_check_for_refine_graph,
                        job.graph_neighborhood_scale,
                        job.tpt_number,
                        job.tpt_leaf_size,
                    ),
                    "registered_workers": len(_snapshot_registered_workers(self.server)),
                    "initialized_shards": len(job.shards),
                    "routing_mode": job.routing_mode,
                    "routing_epoch": job.routing_epoch,
                    "total_vectors_ingested": int(job.total_vectors_ingested),
                    "build_elapsed_s": _build_elapsed_s(job),
                    "wall_elapsed_s": _wall_elapsed_s(job),
                    "build_timing_paused": bool(job.build_pause_started_at is not None),
                    "build_paused_time_s": float(job.build_paused_time_s),
                    "pending_new_workers": sorted(job.pending_new_workers),
                    "bootstrap_vector_count": job.bootstrap_vector_count,
                    "reservoir_seen": job.reservoir_seen,
                    "activation_threshold_vectors": int(getattr(self.server, "activation_threshold_vectors", 0)),
                    "join_at_total_vectors": [int(value) for value in getattr(self.server, "join_at_total_vectors", [])],
                    "next_join_milestone": next_join_milestone,
                    "threshold_breached": threshold_breached,
                    "activation_triggered": activation_triggered,
                    "activation_waiting_for_worker": bool(job.activation_waiting_for_worker),
                    "next_activation_candidate": next_activation_candidate,
                    "active_workers": active_worker_ids,
                    "idle_registered_workers": idle_registered_worker_ids,
                    "finalized": job.finalized,
                    "error": job.error,
                    "centers_file": job.centers_file,
                    "latest_checkpoint_dir": job.latest_checkpoint_dir,
                    "latest_checkpoint_centers_file": job.latest_checkpoint_centers_file,
                    "checkpoint_history_size": len(job.checkpoint_history),
                    "last_checkpoint": job.checkpoint_history[-1] if job.checkpoint_history else None,
                    "rebalance_history_size": len(job.rebalance_history),
                    "last_rebalance": job.rebalance_history[-1] if job.rebalance_history else None,
                    "rebalance_history": list(job.rebalance_history),
                    "communication_summary": dict(job.communication_summary),
                    "shards": shard_info,
                },
            )

    def _handle_shutdown(self):
        _json_response(self, 200, {"ok": True, "message": "master shutting down"})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Distributed SPTAG shard build master with rebalance.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--workers_file", default=None, help='JSON array for initial workers: [{"worker_id":1,"host":"...","port":18080}, ...]')
    parser.add_argument("--workers", nargs="*", default=None, help="Inline initial worker list: worker_id@host:port ...")
    parser.add_argument(
        "--threshold_join_workers_file",
        default=None,
        help='JSON array for threshold-activated workers: [{"worker_id":3,"host":"...","port":18082}, ...]',
    )
    parser.add_argument(
        "--threshold_join_workers",
        nargs="*",
        default=None,
        help="Inline threshold-activated worker list: worker_id@host:port ...",
    )
    parser.add_argument(
        "--activation_threshold_vectors",
        type=int,
        default=0,
        help="Activate one idle threshold worker when any active worker reaches this many active vectors.",
    )
    parser.add_argument(
        "--join_at_total_vectors",
        default="",
        help="Comma-separated total-ingested milestones for activating threshold workers one at a time.",
    )
    parser.add_argument("--request_timeout", type=int, default=60)
    parser.add_argument("--finalize_timeout", type=int, default=600)
    parser.add_argument("--checkpoint_timeout", type=int, default=600)
    parser.add_argument("--rebalance_timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bootstrap_sample_size", type=int, default=50000)
    parser.add_argument("--reservoir_size", type=int, default=50000)
    parser.add_argument("--retry_after_ms", type=int, default=1000)
    parser.add_argument("--disable_rebalance", action="store_true")
    parser.add_argument(
        "--no_exit_after_finalize",
        action="store_true",
        help="Keep master (and workers) running after a successful /finalize.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MasterHandler)
    server.worker_lock = threading.Lock()
    server.workers = {}
    server.worker_key_to_id = {}
    server.next_worker_id = 0
    server.request_timeout = args.request_timeout
    server.finalize_timeout = args.finalize_timeout
    server.checkpoint_timeout = args.checkpoint_timeout
    server.rebalance_timeout = args.rebalance_timeout
    server.retries = args.retries
    server.debug = args.debug
    server.bootstrap_sample_size = max(1, args.bootstrap_sample_size)
    server.reservoir_size = max(1, args.reservoir_size)
    server.retry_after_ms = max(1, args.retry_after_ms)
    server.rebalance_enabled = not args.disable_rebalance
    server.exit_after_finalize = not args.no_exit_after_finalize
    server.activation_threshold_vectors = max(0, int(args.activation_threshold_vectors))
    join_at_total_vectors = []
    if args.join_at_total_vectors:
        for raw_value in str(args.join_at_total_vectors).split(","):
            token = raw_value.strip()
            if not token:
                continue
            try:
                parsed = int(token)
            except ValueError as ex:
                raise ValueError(f"invalid join_at_total_vectors entry: {token!r}") from ex
            if parsed <= 0:
                raise ValueError("join_at_total_vectors entries must be positive integers")
            join_at_total_vectors.append(parsed)
    server.join_at_total_vectors = sorted(join_at_total_vectors)

    initial_workers = _resolve_workers(args.workers_file, args.workers, "initial workers")
    threshold_workers = _resolve_workers(
        args.threshold_join_workers_file,
        args.threshold_join_workers,
        "threshold join workers",
    )
    initial_ids = {worker.worker_id for worker in initial_workers}
    threshold_ids = {worker.worker_id for worker in threshold_workers}
    overlap = sorted(initial_ids.intersection(threshold_ids))
    if overlap:
        raise ValueError(f"worker ids cannot appear in both initial and threshold worker sets: {overlap}")

    server.initial_worker_ids = sorted(initial_ids)
    server.threshold_join_worker_ids = [worker.worker_id for worker in threshold_workers]

    for worker in initial_workers:
        configured = _add_configured_worker(server, worker, registered=True)
        print(f"[worker] initial worker_id={configured.worker_id} addr={configured.host}:{configured.port}")
    for worker in threshold_workers:
        configured = _add_configured_worker(server, worker, registered=False)
        print(f"[worker] threshold worker_id={configured.worker_id} addr={configured.host}:{configured.port}")

    if server.activation_threshold_vectors > 0 and server.join_at_total_vectors:
        raise ValueError("activation_threshold_vectors and join_at_total_vectors are mutually exclusive")
    if (server.activation_threshold_vectors > 0 or server.join_at_total_vectors) and not server.threshold_join_worker_ids:
        raise ValueError("worker activation requires at least one threshold join worker")
    if (server.activation_threshold_vectors > 0 or server.join_at_total_vectors) and not server.rebalance_enabled:
        raise ValueError("worker activation requires rebalance to be enabled")

    configured_workers = _snapshot_workers(server)
    registered_workers = _snapshot_registered_workers(server)
    print(f"Master listening on {args.host}:{args.port} workers={len(configured_workers)}")
    print(
        f"  bootstrap_sample_size={server.bootstrap_sample_size} "
        f"reservoir_size={server.reservoir_size} "
        f"rebalance_enabled={server.rebalance_enabled} "
        f"activation_threshold_vectors={server.activation_threshold_vectors}"
    )
    if server.join_at_total_vectors:
        print(f"  join_at_total_vectors={server.join_at_total_vectors}")
    print(f"  initial_worker_ids={server.initial_worker_ids}")
    print(f"  threshold_join_worker_ids={server.threshold_join_worker_ids}")
    print(f"  registered_workers={len(registered_workers)}")
    for worker in configured_workers:
        status = "registered" if worker.registered else "idle"
        print(f"  worker[{worker.worker_id}]: {worker.host}:{worker.port} ({status})")
    server.serve_forever()


if __name__ == "__main__":
    main()
