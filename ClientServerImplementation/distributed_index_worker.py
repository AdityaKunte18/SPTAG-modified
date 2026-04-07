#!/usr/bin/env python3
"""Distributed SPTAG shard-build worker with rebalance support.

Endpoints:
- POST /init
- POST /add_batch
- POST /migrate_batch
- POST /rebalance        (phase = prepare | migrate | rebuild)
- POST /finalize
- GET  /status?job_id=...&shard_id=...
- POST /shutdown

Protocol notes:
- /init, /finalize, /rebalance use JSON bodies.
- /add_batch and /migrate_batch body format:
    [8-byte little-endian uint64 meta_len][meta_json][raw float32 bytes]
  where meta_json includes explicit vector ids.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlparse

import numpy as np
import SPTAG


def _job_key(job_id: str, shard_id: int) -> str:
    return f"{job_id}::{shard_id}"


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


def _post_with_retry(url: str, fn, retries: int, what: str):
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


def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    return url


def _default_advertise_host(bind_host: str) -> str:
    if bind_host in ("", "0.0.0.0", "::"):
        return socket.gethostname()
    return bind_host


def _register_with_master(
    master_url: str,
    advertise_host: str,
    port: int,
    timeout_s: int,
    retries: int,
    cluster_worker_id: int | None = None,
) -> dict:
    url = f"{_normalize_base_url(master_url)}/register_worker"
    payload = {"host": advertise_host, "port": port}
    if cluster_worker_id is not None:
        payload["cluster_worker_id"] = int(cluster_worker_id)
    return _post_with_retry(
        url,
        lambda u=url, p=payload: _http_json_post(u, p, timeout_s),
        retries,
        "register worker",
    )


def _meta_block_from_ids(ids: np.ndarray) -> bytes:
    return b"".join(f"{int(gid)}\n".encode("utf-8") for gid in ids)


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


def _make_index(algo: str, dist: str, dim: int, threads: int):
    index = SPTAG.AnnIndex(algo, "Float", dim)
    index.SetBuildParam("DistCalcMethod", dist, "Index")
    index.SetBuildParam("NumberOfThreads", str(threads), "Index")
    return index


@dataclass
class LocalShardStore:
    root_dir: Path
    dim: int
    vectors_path: Path
    ids_path: Path
    mask_path: Path
    vector_count: int = 0
    active_count: int = 0
    active_mask: bytearray = field(default_factory=bytearray)
    running_sum: np.ndarray | None = None

    def append(self, vectors: np.ndarray, global_ids: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        global_ids = np.ascontiguousarray(global_ids, dtype=np.int64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"invalid vectors shape {vectors.shape}, expected (*, {self.dim})")
        if vectors.shape[0] != global_ids.shape[0]:
            raise ValueError("vector/id count mismatch")

        num = vectors.shape[0]
        if num == 0:
            return

        with open(self.vectors_path, "ab") as f:
            f.write(vectors.tobytes())
        with open(self.ids_path, "ab") as f:
            f.write(global_ids.tobytes())
        with open(self.mask_path, "ab") as f:
            f.write(b"\x01" * num)

        self.vector_count += num
        self.active_count += num
        self.active_mask.extend(b"\x01" * num)
        self.running_sum += vectors.sum(axis=0, dtype=np.float64)

    def tombstone(self, positions: np.ndarray, vectors: np.ndarray) -> int:
        positions = np.ascontiguousarray(positions, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if positions.shape[0] != vectors.shape[0]:
            raise ValueError("tombstone positions/vector count mismatch")
        if positions.size == 0:
            return 0

        changed = 0
        with open(self.mask_path, "r+b") as f:
            for idx, pos in enumerate(positions.tolist()):
                if pos < 0 or pos >= self.vector_count:
                    continue
                if self.active_mask[pos] == 0:
                    continue
                self.active_mask[pos] = 0
                f.seek(pos)
                f.write(b"\x00")
                self.running_sum -= vectors[idx].astype(np.float64, copy=False)
                self.active_count -= 1
                changed += 1
        return changed

    def centroid(self) -> np.ndarray:
        if self.running_sum is None:
            self.running_sum = np.zeros(self.dim, dtype=np.float64)
        if self.active_count <= 0:
            return np.zeros(self.dim, dtype=np.float32)
        return (self.running_sum / float(self.active_count)).astype(np.float32)

    def iter_active_chunks(self, batch_size: int, limit: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        vector_count = self.vector_count
        total = vector_count if limit is None else min(limit, vector_count)
        if total <= 0:
            return
        vectors_mm = np.memmap(self.vectors_path, dtype=np.float32, mode="r", shape=(vector_count, self.dim))
        ids_mm = np.memmap(self.ids_path, dtype=np.int64, mode="r", shape=(vector_count,))
        # Rebalance can scan and receive migrated vectors concurrently. Snapshot
        # the active mask so appends do not resize a buffer that NumPy exported.
        mask = np.frombuffer(bytes(self.active_mask[:total]), dtype=np.uint8, count=total)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            local_mask = mask[start:end]
            active_local = np.flatnonzero(local_mask)
            if active_local.size == 0:
                continue
            positions = start + active_local
            vectors = np.asarray(vectors_mm[positions], dtype=np.float32)
            ids = np.asarray(ids_mm[positions], dtype=np.int64)
            yield positions.astype(np.int64), vectors, ids


def _create_local_store(save_dir: str, dim: int) -> LocalShardStore:
    root = Path(save_dir) / "_local_state"
    root.mkdir(parents=True, exist_ok=True)
    vectors_path = root / "vectors.f32"
    ids_path = root / "ids.i64"
    mask_path = root / "active.mask"
    for path in (vectors_path, ids_path, mask_path):
        with open(path, "wb"):
            pass
    return LocalShardStore(
        root_dir=root,
        dim=dim,
        vectors_path=vectors_path,
        ids_path=ids_path,
        mask_path=mask_path,
        running_sum=np.zeros(dim, dtype=np.float64),
    )


@dataclass
class RebalancePlan:
    routing_epoch: int
    centroids: np.ndarray
    assigned_centroid_idx: int
    centroid_to_worker: dict[int, int]
    worker_urls: dict[int, str]
    scan_limit: int


@dataclass
class JobState:
    job_id: str
    index: object
    algo: str
    dim: int
    dist: str
    threads: int
    save_dir: str
    shard_id: int
    with_meta_index: bool
    local_store: LocalShardStore

    first_batch_done: bool = False
    normalized_input: bool | None = None

    vectors_ingested: int = 0
    last_batch_id: int = -1
    next_expected_batch_id: int = 0

    finalized: bool = False
    init_time_s: float = 0.0
    add_time_s: float = 0.0
    finalize_time_s: float = 0.0
    checkpoint_time_s: float = 0.0

    q: "queue.Queue[tuple[int,np.ndarray,int,bool,bytes]]" | None = None
    stop_event: threading.Event | None = None
    consumer: threading.Thread | None = None

    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    error: str | None = None

    last_routing_epoch: int = 0
    rebalance_plan: RebalancePlan | None = None
    applied_migration_batches: set[tuple[int, int, int]] = field(default_factory=set)
    rebalance_started_at: float = 0.0
    last_rebalance_prepare_time_s: float = 0.0
    last_rebalance_migrate_time_s: float = 0.0
    last_rebalance_rebuild_time_s: float = 0.0
    last_rebalance_total_time_s: float = 0.0
    rebalance_history: list[dict] = field(default_factory=list, repr=False)


STATE: dict[str, JobState] = {}
STATE_LOCK = threading.Lock()


def _get_state(key: str) -> JobState | None:
    with STATE_LOCK:
        return STATE.get(key)


def _apply_vectors_to_index(st: JobState, vectors: np.ndarray, global_ids: np.ndarray, normalized: bool) -> float:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    global_ids = np.ascontiguousarray(global_ids, dtype=np.int64)
    if vectors.shape[0] != global_ids.shape[0]:
        raise ValueError("vector/id count mismatch")
    if vectors.size == 0:
        return 0.0

    metadata = _meta_block_from_ids(global_ids)
    t_add_start = time.time()
    if (not st.first_batch_done) and st.algo.upper() == "SPANN":
        ok = st.index.BuildWithMetaData(vectors.tobytes(), metadata, vectors.shape[0], st.with_meta_index, normalized)
    else:
        ok = st.index.AddWithMetaData(vectors.tobytes(), metadata, vectors.shape[0], st.with_meta_index, normalized)
    add_elapsed = time.time() - t_add_start
    if not ok:
        raise RuntimeError("SPTAG add/build with metadata failed")
    st.first_batch_done = True
    st.add_time_s += add_elapsed
    st.vectors_ingested = st.local_store.active_count
    return add_elapsed


def _delete_vectors_from_index(st: JobState, vectors: np.ndarray, global_ids: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    global_ids = np.ascontiguousarray(global_ids, dtype=np.int64)
    if vectors.shape[0] != global_ids.shape[0]:
        raise ValueError("vector/id count mismatch on delete")
    if vectors.size == 0:
        return

    metadata = _meta_block_from_ids(global_ids)
    delete_ok = False
    delete_error: Exception | None = None

    try:
        delete_ok = bool(st.index.DeleteByMetaData(metadata))
    except Exception as ex:
        delete_error = ex

    if not delete_ok:
        try:
            delete_ok = bool(st.index.Delete(vectors.tobytes(), vectors.shape[0]))
        except Exception as ex:
            delete_error = ex

    if not delete_ok:
        if delete_error is not None:
            raise RuntimeError(f"SPTAG delete failed: {delete_error}")
        raise RuntimeError("SPTAG delete failed")


def _rebuild_index_from_store(st: JobState, batch_size: int) -> tuple[int, np.ndarray]:
    new_index = _make_index(st.algo, st.dist, st.dim, st.threads)
    first_batch_done = False
    total = 0
    add_time_s = 0.0
    normalized = bool(st.normalized_input) if st.normalized_input is not None else False

    for _, vectors, global_ids in st.local_store.iter_active_chunks(batch_size=batch_size):
        metadata = _meta_block_from_ids(global_ids)
        t0 = time.time()
        if (not first_batch_done) and st.algo.upper() == "SPANN":
            ok = new_index.BuildWithMetaData(vectors.tobytes(), metadata, vectors.shape[0], st.with_meta_index, normalized)
        else:
            ok = new_index.AddWithMetaData(vectors.tobytes(), metadata, vectors.shape[0], st.with_meta_index, normalized)
        elapsed = time.time() - t0
        if not ok:
            raise RuntimeError("SPTAG rebuild failed while replaying active vectors")
        add_time_s += elapsed
        total += int(vectors.shape[0])
        first_batch_done = True

    if total > 0:
        t_upd = time.time()
        new_index.UpdateIndex()
        add_time_s += time.time() - t_upd

    st.index = new_index
    st.first_batch_done = first_batch_done
    st.vectors_ingested = st.local_store.active_count
    st.add_time_s = add_time_s
    return st.local_store.active_count, st.local_store.centroid()


def _save_index_snapshot(st: JobState, target_dir: str) -> tuple[float, np.ndarray, bool]:
    os.makedirs(target_dir, exist_ok=True)
    centroid = st.local_store.centroid()
    if st.local_store.active_count <= 0:
        return 0.0, centroid, True

    t0 = time.time()
    st.index.UpdateIndex()
    ok = st.index.Save(target_dir)
    elapsed = time.time() - t0
    if not ok:
        raise RuntimeError(f"failed to save index to {target_dir}")
    return elapsed, centroid, False


def _consumer_loop(key: str) -> None:
    while True:
        st = _get_state(key)
        if st is None:
            return

        if st.stop_event is not None and st.stop_event.is_set():
            return

        try:
            item = st.q.get(timeout=0.25)
        except queue.Empty:
            continue

        if item is None:
            st.q.task_done()
            return

        batch_id, global_ids, dim, normalized, vec_bytes = item

        st = _get_state(key)
        if st is None:
            return

        try:
            vectors = np.frombuffer(vec_bytes, dtype=np.float32).reshape(global_ids.shape[0], dim)
            with st.lock:
                if st.finalized or st.error is not None:
                    pass
                else:
                    st.local_store.append(vectors, global_ids)
                    _apply_vectors_to_index(st, vectors, global_ids, normalized)
                    st.last_batch_id = batch_id
        except Exception as ex:
            with st.lock:
                st.error = f"{type(ex).__name__}: {ex}"
        finally:
            st.q.task_done()


def _wait_for_queue_drain(st: JobState) -> None:
    if st.q is not None:
        st.q.join()


def _send_migrate_batch(
    *,
    st: JobState,
    destination_url: str,
    destination_shard_id: int,
    routing_epoch: int,
    source_worker_id: int,
    batch_seq: int,
    vectors: np.ndarray,
    global_ids: np.ndarray,
) -> dict:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    global_ids = np.ascontiguousarray(global_ids, dtype=np.int64)
    meta = {
        "job_id": st.job_id,
        "shard_id": int(destination_shard_id),
        "routing_epoch": routing_epoch,
        "source_worker_id": source_worker_id,
        "batch_seq": batch_seq,
        "num": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "normalized": bool(st.normalized_input) if st.normalized_input is not None else False,
        "ids": global_ids.astype(np.int64).tolist(),
    }
    meta_raw = json.dumps(meta).encode("utf-8")
    payload = struct.pack("<Q", len(meta_raw)) + meta_raw + vectors.tobytes()
    url = f"{destination_url}/migrate_batch"
    return _post_with_retry(
        url,
        lambda p=payload, u=url: _http_binary_post(u, p, st.server_peer_timeout),
        st.server_peer_retries,
        f"migrate batch {source_worker_id}->{destination_url}",
    )


class WorkerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/status":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return

        qs = parse_qs(parsed.query)
        job_id = qs.get("job_id", [None])[0]
        shard_raw = qs.get("shard_id", [None])[0]
        if job_id is None or shard_raw is None:
            _json_response(self, 400, {"ok": False, "error": "job_id and shard_id are required"})
            return
        shard_id = int(shard_raw)
        key = _job_key(job_id, shard_id)
        st = _get_state(key)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": "job not found"})
            return

        with st.lock:
            queued = st.q.qsize() if st.q is not None else 0
            qmax = st.q.maxsize if st.q is not None else 0
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "job_id": job_id,
                    "shard_id": shard_id,
                    "algo": st.algo,
                    "dim": st.dim,
                    "dist": st.dist,
                    "vectors_ingested": st.vectors_ingested,
                    "active_vectors": st.local_store.active_count,
                    "store_vectors_total": st.local_store.vector_count,
                    "last_applied_batch_id": st.last_batch_id,
                    "next_expected_batch_id": st.next_expected_batch_id,
                    "finalized": st.finalized,
                    "save_dir": st.save_dir,
                    "elapsed_s": max(0.0, time.time() - st.init_time_s),
                    "add_time_s": st.add_time_s,
                    "finalize_time_s": st.finalize_time_s,
                    "queued": queued,
                    "queue_max": qmax,
                    "error": st.error,
                    "routing_epoch": st.last_routing_epoch,
                    "rebalance_epoch": st.rebalance_plan.routing_epoch if st.rebalance_plan is not None else None,
                    "rebalance_history_size": len(st.rebalance_history),
                    "last_rebalance": st.rebalance_history[-1] if st.rebalance_history else None,
                    "rebalance_history": list(st.rebalance_history),
                },
            )

    def do_POST(self):
        if self.path == "/init":
            return self._handle_init()
        if self.path == "/add_batch":
            return self._handle_add_batch()
        if self.path == "/migrate_batch":
            return self._handle_migrate_batch()
        if self.path == "/rebalance":
            return self._handle_rebalance()
        if self.path == "/checkpoint":
            return self._handle_checkpoint()
        if self.path == "/finalize":
            return self._handle_finalize()
        if self.path == "/shutdown":
            return self._handle_shutdown()
        _json_response(self, 404, {"ok": False, "error": "not found"})

    def _handle_init(self):
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            shard_id = int(req["shard_id"])
            algo = str(req.get("algo", "BKT"))
            dim = int(req["dim"])
            dist = str(req.get("dist", "L2"))
            threads = int(req.get("threads", 8))
            save_dir = str(req["save_dir"])
            with_meta_index = bool(req.get("with_meta_index", False))
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid init payload: {ex}"})
            return

        key = _job_key(job_id, shard_id)
        qmax = getattr(self.server, "queue_max_batches", 3)

        stop_event = threading.Event()
        q = queue.Queue(maxsize=qmax)
        local_store = _create_local_store(save_dir, dim)
        st = JobState(
            job_id=job_id,
            index=_make_index(algo, dist, dim, threads),
            algo=algo,
            dim=dim,
            dist=dist,
            threads=threads,
            save_dir=save_dir,
            shard_id=shard_id,
            with_meta_index=with_meta_index,
            local_store=local_store,
            init_time_s=time.time(),
            q=q,
            stop_event=stop_event,
        )
        st.server_peer_timeout = getattr(self.server, "peer_request_timeout", 300)
        st.server_peer_retries = getattr(self.server, "peer_retries", 5)
        st.migrate_batch_size = getattr(self.server, "migrate_batch_size", 2048)
        st.rebuild_batch_size = getattr(self.server, "rebuild_batch_size", 2048)

        with STATE_LOCK:
            STATE[key] = st

        t = threading.Thread(target=_consumer_loop, args=(key,), daemon=True)
        st.consumer = t
        t.start()

        _json_response(self, 200, {"ok": True, "job_id": job_id, "shard_id": shard_id, "queue_max": qmax})

    def _handle_add_batch(self):
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
            shard_id = int(meta["shard_id"])
            batch_id = int(meta["batch_id"])
            num = int(meta["num"])
            dim = int(meta["dim"])
            normalized = bool(meta.get("normalized", False))
            global_ids = np.asarray(meta["ids"], dtype=np.int64)
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid batch meta: {ex}"})
            return

        key = _job_key(job_id, shard_id)
        st = _get_state(key)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": "job not initialized"})
            return

        if global_ids.shape[0] != num:
            _json_response(self, 400, {"ok": False, "error": "ids count does not match num"})
            return

        expected_bytes = num * dim * 4
        if len(vec_bytes) != expected_bytes:
            _json_response(
                self,
                400,
                {"ok": False, "error": f"invalid vector payload bytes={len(vec_bytes)}, expected={expected_bytes}"},
            )
            return

        with st.lock:
            if st.finalized:
                _json_response(self, 409, {"ok": False, "error": "job already finalized"})
                return
            if st.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
                return
            if dim != st.dim:
                _json_response(self, 400, {"ok": False, "error": f"dim mismatch: batch {dim}, job {st.dim}"})
                return
            if st.rebalance_plan is not None:
                _json_response(self, 429, {"ok": False, "queue_full": True, "retry_after_ms": 200})
                return
            if st.normalized_input is None:
                st.normalized_input = normalized
            elif st.normalized_input != normalized:
                _json_response(self, 400, {"ok": False, "error": "normalized flag changed within the same job"})
                return
            if batch_id <= st.last_batch_id:
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "duplicate": True,
                        "job_id": job_id,
                        "shard_id": shard_id,
                        "batch_id": batch_id,
                        "vectors_ingested": st.vectors_ingested,
                        "last_applied_batch_id": st.last_batch_id,
                    },
                )
                return
            if batch_id != st.next_expected_batch_id:
                _json_response(
                    self,
                    409,
                    {"ok": False, "error": f"out of order batch_id={batch_id}, expected {st.next_expected_batch_id}"},
                )
                return
            try:
                st.q.put_nowait((batch_id, global_ids, dim, normalized, vec_bytes))
            except queue.Full:
                _json_response(self, 429, {"ok": False, "queue_full": True, "retry_after_ms": 200})
                return
            st.next_expected_batch_id += 1
            queued = st.q.qsize()
            qmax = st.q.maxsize

        _json_response(
            self,
            202,
            {
                "ok": True,
                "accepted": True,
                "job_id": job_id,
                "shard_id": shard_id,
                "batch_id": batch_id,
                "queued": queued,
                "queue_max": qmax,
                "last_applied_batch_id": st.last_batch_id,
            },
        )

    def _handle_migrate_batch(self):
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
            routing_epoch = int(meta["routing_epoch"])
            source_worker_id = int(meta["source_worker_id"])
            batch_seq = int(meta["batch_seq"])
            num = int(meta["num"])
            dim = int(meta["dim"])
            normalized = bool(meta.get("normalized", False))
            global_ids = np.asarray(meta["ids"], dtype=np.int64)
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid migrate meta: {ex}"})
            return

        try:
            job_id = str(meta["job_id"])
            shard_id = int(meta["shard_id"])
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid migrate destination meta: {ex}"})
            return

        key = _job_key(job_id, shard_id)
        st = _get_state(key)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": "job not initialized"})
            return

        if global_ids.shape[0] != num:
            _json_response(self, 400, {"ok": False, "error": "ids count does not match num"})
            return

        expected_bytes = num * dim * 4
        if len(vec_bytes) != expected_bytes:
            _json_response(
                self,
                400,
                {"ok": False, "error": f"invalid vector payload bytes={len(vec_bytes)}, expected={expected_bytes}"},
            )
            return

        token = (routing_epoch, source_worker_id, batch_seq)
        vectors = np.frombuffer(vec_bytes, dtype=np.float32).reshape(num, dim)

        with st.lock:
            if st.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
                return
            if st.rebalance_plan is None or st.rebalance_plan.routing_epoch != routing_epoch:
                if routing_epoch <= st.last_routing_epoch and token in st.applied_migration_batches:
                    _json_response(self, 200, {"ok": True, "duplicate": True, "accepted": True})
                    return
                _json_response(self, 409, {"ok": False, "error": f"stale or unknown routing_epoch {routing_epoch}"})
                return
            if token in st.applied_migration_batches:
                _json_response(self, 200, {"ok": True, "duplicate": True, "accepted": True})
                return
            if st.normalized_input is None:
                st.normalized_input = normalized
            elif st.normalized_input != normalized:
                _json_response(self, 400, {"ok": False, "error": "normalized flag mismatch during migration"})
                return
            try:
                st.local_store.append(vectors, global_ids)
                _apply_vectors_to_index(st, vectors, global_ids, normalized)
                st.applied_migration_batches.add(token)
            except Exception as ex:
                st.error = f"{type(ex).__name__}: {ex}"
                _json_response(self, 500, {"ok": False, "error": st.error})
                return

        _json_response(self, 200, {"ok": True, "accepted": True, "duplicate": False, "count": num})

    def _handle_rebalance(self):
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            shard_id = int(req["shard_id"])
            routing_epoch = int(req["routing_epoch"])
            phase = str(req["phase"])
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid rebalance payload: {ex}"})
            return

        key = _job_key(job_id, shard_id)
        st = _get_state(key)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": "job not initialized"})
            return

        try:
            if phase == "prepare":
                return self._handle_rebalance_prepare(st, routing_epoch, req)
            if phase == "migrate":
                return self._handle_rebalance_migrate(st, routing_epoch)
            if phase == "rebuild":
                return self._handle_rebalance_rebuild(st, routing_epoch)
        except Exception as ex:
            with st.lock:
                st.error = f"{type(ex).__name__}: {ex}"
                st.rebalance_history.append(
                    {
                        "routing_epoch": int(routing_epoch),
                        "worker_id": int(st.shard_id),
                        "phase": str(phase),
                        "error": str(ex),
                    }
                )
            _json_response(self, 500, {"ok": False, "error": str(ex)})
            return

        _json_response(self, 400, {"ok": False, "error": f"unknown rebalance phase {phase}"})

    def _handle_rebalance_prepare(self, st: JobState, routing_epoch: int, req: dict):
        phase_t0 = time.time()
        centroids = np.asarray(req["centroids"], dtype=np.float32)
        assigned_centroid_idx = int(req["assigned_centroid_idx"])
        centroid_to_worker = {int(k): int(v) for k, v in req["centroid_to_worker"].items()}
        worker_urls = {int(item["worker_id"]): str(item["base_url"]) for item in req["workers"]}

        _wait_for_queue_drain(st)
        with st.lock:
            if st.error is not None:
                raise RuntimeError(st.error)
            if st.finalized:
                raise RuntimeError("job already finalized")
            if routing_epoch <= st.last_routing_epoch:
                elapsed = time.time() - phase_t0
                st.last_rebalance_prepare_time_s = elapsed
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "phase": "prepare",
                        "duplicate": True,
                        "active_count": st.local_store.active_count,
                        "phase_time_s": elapsed,
                        "rebalance_total_time_s": st.last_rebalance_total_time_s,
                    },
                )
                return
            st.rebalance_started_at = time.time()
            st.last_rebalance_prepare_time_s = 0.0
            st.last_rebalance_migrate_time_s = 0.0
            st.last_rebalance_rebuild_time_s = 0.0
            st.last_rebalance_total_time_s = 0.0
            st.rebalance_plan = RebalancePlan(
                routing_epoch=routing_epoch,
                centroids=centroids,
                assigned_centroid_idx=assigned_centroid_idx,
                centroid_to_worker=centroid_to_worker,
                worker_urls=worker_urls,
                scan_limit=st.local_store.vector_count,
            )
            elapsed = time.time() - phase_t0
            st.last_rebalance_prepare_time_s = elapsed

        print(
            f"[rebalance] shard={st.shard_id} epoch={routing_epoch} "
            f"phase=prepare active={st.local_store.active_count} elapsed={elapsed:.3f}s"
        )
        _json_response(
            self,
            200,
            {
                "ok": True,
                "phase": "prepare",
                "routing_epoch": routing_epoch,
                "scan_limit": st.local_store.vector_count,
                "active_count": st.local_store.active_count,
                "phase_time_s": elapsed,
            },
        )

    def _handle_rebalance_migrate(self, st: JobState, routing_epoch: int):
        phase_t0 = time.time()
        _wait_for_queue_drain(st)
        with st.lock:
            if st.error is not None:
                raise RuntimeError(st.error)
            plan = st.rebalance_plan
            if plan is None or plan.routing_epoch != routing_epoch:
                raise RuntimeError(f"rebalance plan for epoch {routing_epoch} is not prepared")
            centroids = plan.centroids.copy()
            assigned_centroid_idx = int(plan.assigned_centroid_idx)
            centroid_to_worker = dict(plan.centroid_to_worker)
            worker_urls = dict(plan.worker_urls)
            scan_limit = int(plan.scan_limit)
            migrate_batch_size = int(st.migrate_batch_size)

        batch_seq = 0
        moved_vectors = 0
        kept_vectors = 0
        metric = st.dist
        source_worker_id = st.shard_id
        pending: dict[int, dict[str, list]] = {}

        def flush_destination(dest_worker_id: int):
            nonlocal batch_seq, moved_vectors
            bucket = pending.get(dest_worker_id)
            if not bucket or not bucket["ids"]:
                return
            vectors = np.ascontiguousarray(np.vstack(bucket["vectors"]), dtype=np.float32)
            ids = np.ascontiguousarray(np.concatenate(bucket["ids"]), dtype=np.int64)
            positions = np.ascontiguousarray(np.concatenate(bucket["positions"]), dtype=np.int64)
            dest_url = worker_urls[dest_worker_id]
            resp = _send_migrate_batch(
                st=st,
                destination_url=dest_url,
                destination_shard_id=dest_worker_id,
                routing_epoch=routing_epoch,
                source_worker_id=source_worker_id,
                batch_seq=batch_seq,
                vectors=vectors,
                global_ids=ids,
            )
            if not resp.get("ok", False):
                raise RuntimeError(f"migrate_batch failed to worker {dest_worker_id}: {resp}")
            with st.lock:
                _delete_vectors_from_index(st, vectors, ids)
                st.local_store.tombstone(positions, vectors)
                st.vectors_ingested = st.local_store.active_count
            moved_vectors += int(vectors.shape[0])
            batch_seq += 1
            bucket["vectors"].clear()
            bucket["ids"].clear()
            bucket["positions"].clear()

        for positions, vectors, global_ids in st.local_store.iter_active_chunks(batch_size=migrate_batch_size, limit=scan_limit):
            dists = _distance_matrix(vectors, centroids, metric)
            nearest = np.argmin(dists, axis=1)
            keep_mask = nearest == assigned_centroid_idx
            kept_vectors += int(np.count_nonzero(keep_mask))
            move_rows = np.flatnonzero(~keep_mask)
            for row in move_rows.tolist():
                centroid_idx = int(nearest[row])
                dest_worker_id = int(centroid_to_worker[centroid_idx])
                if dest_worker_id == source_worker_id:
                    kept_vectors += 1
                    continue
                bucket = pending.setdefault(dest_worker_id, {"vectors": [], "ids": [], "positions": []})
                bucket["vectors"].append(vectors[row : row + 1])
                bucket["ids"].append(global_ids[row : row + 1])
                bucket["positions"].append(positions[row : row + 1])
                bucket_size = sum(chunk.shape[0] for chunk in bucket["ids"])
                if bucket_size >= migrate_batch_size:
                    flush_destination(dest_worker_id)

        for dest_worker_id in sorted(pending):
            flush_destination(dest_worker_id)

        with st.lock:
            st.vectors_ingested = st.local_store.active_count
            elapsed = time.time() - phase_t0
            st.last_rebalance_migrate_time_s = elapsed

        print(
            f"[rebalance] shard={st.shard_id} epoch={routing_epoch} "
            f"phase=migrate moved={moved_vectors} kept={kept_vectors} elapsed={elapsed:.3f}s"
        )
        _json_response(
            self,
            200,
            {
                "ok": True,
                "phase": "migrate",
                "routing_epoch": routing_epoch,
                "moved_vectors": moved_vectors,
                "kept_vectors": kept_vectors,
                "active_count": st.local_store.active_count,
                "phase_time_s": elapsed,
            },
        )

    def _handle_rebalance_rebuild(self, st: JobState, routing_epoch: int):
        phase_t0 = time.time()
        _wait_for_queue_drain(st)
        with st.lock:
            if st.error is not None:
                raise RuntimeError(st.error)
            plan = st.rebalance_plan
            if plan is None or plan.routing_epoch != routing_epoch:
                raise RuntimeError(f"rebalance plan for epoch {routing_epoch} is not prepared")
            batch_size = int(st.rebuild_batch_size)
            active_count, centroid = _rebuild_index_from_store(st, batch_size)
            st.last_routing_epoch = routing_epoch
            st.rebalance_plan = None
            elapsed = time.time() - phase_t0
            st.last_rebalance_rebuild_time_s = elapsed
            if st.rebalance_started_at > 0:
                st.last_rebalance_total_time_s = time.time() - st.rebalance_started_at
            else:
                st.last_rebalance_total_time_s = elapsed
            st.rebalance_history.append(
                {
                    "routing_epoch": int(routing_epoch),
                    "worker_id": int(st.shard_id),
                    "active_count": int(active_count),
                    "prepare_phase_s": round(st.last_rebalance_prepare_time_s, 6),
                    "migrate_phase_s": round(st.last_rebalance_migrate_time_s, 6),
                    "rebuild_phase_s": round(st.last_rebalance_rebuild_time_s, 6),
                    "rebalance_total_s": round(st.last_rebalance_total_time_s, 6),
                }
            )

        print(
            f"[rebalance] shard={st.shard_id} epoch={routing_epoch} "
            f"phase=rebuild active={active_count} elapsed={elapsed:.3f}s total={st.last_rebalance_total_time_s:.3f}s"
        )
        _json_response(
            self,
            200,
            {
                "ok": True,
                "phase": "rebuild",
                "routing_epoch": routing_epoch,
                "active_count": active_count,
                "centroid": centroid.tolist(),
                "phase_time_s": elapsed,
                "rebalance_total_time_s": st.last_rebalance_total_time_s,
            },
        )

    def _handle_checkpoint(self):
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            shard_id = int(req["shard_id"])
            checkpoint_dir = str(req["checkpoint_dir"])
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid checkpoint payload: {ex}"})
            return

        key = _job_key(job_id, shard_id)
        st = _get_state(key)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": "job not found"})
            return

        with st.lock:
            if st.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
                return
            if st.rebalance_plan is not None:
                _json_response(self, 409, {"ok": False, "error": "rebalance still in progress"})
                return

        _wait_for_queue_drain(st)
        with st.lock:
            if st.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
                return
            checkpoint_elapsed, centroid, empty = _save_index_snapshot(st, checkpoint_dir)
            st.checkpoint_time_s = checkpoint_elapsed

        print(
            f"[checkpoint] shard={shard_id} dir={checkpoint_dir} "
            f"active={st.local_store.active_count} elapsed={checkpoint_elapsed:.3f}s"
        )
        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "shard_id": shard_id,
                "checkpoint_dir": checkpoint_dir,
                "vectors_ingested": st.vectors_ingested,
                "active_vectors": st.local_store.active_count,
                "checkpoint_time_s": checkpoint_elapsed,
                "add_time_s": st.add_time_s,
                "elapsed_s": max(0.0, time.time() - st.init_time_s),
                "last_applied_batch_id": st.last_batch_id,
                "centroid": centroid.tolist(),
                "routing_epoch": st.last_routing_epoch,
                "empty": empty,
                "rebalance_prepare_time_s": st.last_rebalance_prepare_time_s,
                "rebalance_migrate_time_s": st.last_rebalance_migrate_time_s,
                "rebalance_rebuild_time_s": st.last_rebalance_rebuild_time_s,
                "rebalance_total_time_s": st.last_rebalance_total_time_s,
                "rebalance_history_size": len(st.rebalance_history),
                "last_rebalance": st.rebalance_history[-1] if st.rebalance_history else None,
                "rebalance_history": list(st.rebalance_history),
            },
        )

    def _handle_finalize(self):
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            shard_id = int(req["shard_id"])
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid finalize payload: {ex}"})
            return

        key = _job_key(job_id, shard_id)
        st = _get_state(key)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": "job not found"})
            return

        with st.lock:
            if st.finalized:
                _json_response(self, 200, {"ok": True, "already_finalized": True, "save_dir": st.save_dir})
                return
            if st.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
                return
            if st.rebalance_plan is not None:
                _json_response(self, 409, {"ok": False, "error": "rebalance still in progress"})
                return

        _wait_for_queue_drain(st)
        with st.lock:
            if st.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
                return
            if st.local_store.active_count <= 0:
                os.makedirs(st.save_dir, exist_ok=True)
                st.finalized = True
                st.finalize_time_s = 0.0
                centroid = st.local_store.centroid()
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "job_id": job_id,
                        "shard_id": shard_id,
                        "vectors_ingested": 0,
                        "active_vectors": 0,
                        "save_dir": st.save_dir,
                        "add_time_s": st.add_time_s,
                        "finalize_time_s": st.finalize_time_s,
                        "elapsed_s": max(0.0, time.time() - st.init_time_s),
                        "last_applied_batch_id": st.last_batch_id,
                        "centroid": centroid.tolist(),
                        "routing_epoch": st.last_routing_epoch,
                        "empty": True,
                        "rebalance_history_size": len(st.rebalance_history),
                        "last_rebalance": st.rebalance_history[-1] if st.rebalance_history else None,
                        "rebalance_history": list(st.rebalance_history),
                    },
                )
                if getattr(self.server, "exit_after_finalize", False):
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            fin_elapsed, centroid, _ = _save_index_snapshot(st, st.save_dir)
            st.finalized = True
            st.finalize_time_s = fin_elapsed

        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "shard_id": shard_id,
                "vectors_ingested": st.vectors_ingested,
                "active_vectors": st.local_store.active_count,
                "save_dir": st.save_dir,
                "add_time_s": st.add_time_s,
                "finalize_time_s": st.finalize_time_s,
                "elapsed_s": max(0.0, time.time() - st.init_time_s),
                "last_applied_batch_id": st.last_batch_id,
                "centroid": centroid.tolist(),
                "routing_epoch": st.last_routing_epoch,
                "checkpoint_time_s": st.checkpoint_time_s,
                "rebalance_prepare_time_s": st.last_rebalance_prepare_time_s,
                "rebalance_migrate_time_s": st.last_rebalance_migrate_time_s,
                "rebalance_rebuild_time_s": st.last_rebalance_rebuild_time_s,
                "rebalance_total_time_s": st.last_rebalance_total_time_s,
                "rebalance_history_size": len(st.rebalance_history),
                "last_rebalance": st.rebalance_history[-1] if st.rebalance_history else None,
                "rebalance_history": list(st.rebalance_history),
            },
        )

        if getattr(self.server, "exit_after_finalize", False):
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _handle_shutdown(self):
        _json_response(self, 200, {"ok": True, "message": "worker shutting down"})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="SPTAG distributed index build worker with rebalance support.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--master_url", default=None, help="Optional master URL for dynamic worker registration.")
    parser.add_argument("--advertise_host", default=None, help="Host/IP the master should use to reach this worker.")
    parser.add_argument("--cluster_worker_id", type=int, default=None, help="Stable cluster worker id used for master registration.")
    parser.add_argument("--register_timeout", type=int, default=30)
    parser.add_argument("--register_retries", type=int, default=5)
    parser.add_argument("--queue_max_batches", type=int, default=3, help="Max queued client batches per job.")
    parser.add_argument("--peer_request_timeout", type=int, default=300)
    parser.add_argument("--peer_retries", type=int, default=5)
    parser.add_argument("--migrate_batch_size", type=int, default=2048)
    parser.add_argument("--rebuild_batch_size", type=int, default=2048)
    parser.add_argument(
        "--no_exit_after_finalize",
        action="store_true",
        help="Keep worker running after a successful /finalize.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), WorkerHandler)
    server.exit_after_finalize = not args.no_exit_after_finalize
    server.queue_max_batches = args.queue_max_batches
    server.peer_request_timeout = args.peer_request_timeout
    server.peer_retries = args.peer_retries
    server.migrate_batch_size = args.migrate_batch_size
    server.rebuild_batch_size = args.rebuild_batch_size
    print(
        f"Worker listening on {args.host}:{args.port} queue_max_batches={args.queue_max_batches} "
        f"migrate_batch_size={args.migrate_batch_size}"
    )
    serve_thread = threading.Thread(target=server.serve_forever, name="worker-http")
    serve_thread.start()

    try:
        if args.master_url:
            advertise_host = args.advertise_host or _default_advertise_host(args.host)
            resp = _register_with_master(
                args.master_url,
                advertise_host,
                args.port,
                args.register_timeout,
                args.register_retries,
                args.cluster_worker_id,
            )
            if not resp.get("ok", False):
                raise RuntimeError(f"master registration failed: {resp}")
            print(
                f"[register] master={_normalize_base_url(args.master_url)} "
                f"worker_id={resp.get('worker_id')} advertised={advertise_host}:{args.port}"
            )

        serve_thread.join()
    except Exception:
        server.shutdown()
        serve_thread.join()
        raise
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
