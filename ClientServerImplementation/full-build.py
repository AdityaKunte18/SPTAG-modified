#!/usr/bin/env python3
"""Buffered full-build experiment for distributed SPTAG BKT.

This script provides three roles:

- worker: receive all vectors for a job, then build one SPTAG index at the end
- master: forward client batches to a single worker
- client: read a dataset in batches and send it through the master

The key difference from the existing incremental worker is that the worker does
not call AddWithMetaData for each batch. It buffers all vectors first, then
calls one BuildWithMetaData when all vectors have arrived.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

try:
    import SPTAG
except Exception:  # pragma: no cover - import is required only at runtime
    SPTAG = None


def _read_json_http_response(resp) -> dict:
    return json.loads(resp.read().decode("utf-8"))


def _http_json_post(url: str, payload: dict, timeout_s: int) -> dict:
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


def _http_binary_post(url: str, payload: bytes, timeout_s: int) -> dict:
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


def _normalize_value_type(value_type: str) -> str:
    value = str(value_type or "Float").strip()
    aliases = {"FLOAT": "Float", "UINT8": "UInt8"}
    normalized = aliases.get(value.upper())
    if normalized is None:
        raise ValueError(f"unsupported value_type: {value_type}")
    return normalized


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


def _vectors_for_index(value_type: str, vectors: np.ndarray) -> np.ndarray:
    normalized = _normalize_value_type(value_type)
    if normalized == "UInt8":
        if vectors.dtype == np.uint8:
            return np.ascontiguousarray(vectors)
        clipped = np.clip(np.rint(vectors), 0, 255)
        return np.ascontiguousarray(clipped.astype(np.uint8, copy=False))
    return np.ascontiguousarray(vectors.astype(np.float32, copy=False))


def _metadata_block(global_ids: np.ndarray) -> bytes:
    return b"".join(f"{int(gid)}\n".encode("utf-8") for gid in global_ids)


def bvecs_get_dim_and_total_points(path: str) -> tuple[int, int]:
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    raw_size = int(raw.size)
    if raw_size < 4:
        raise RuntimeError(f"File too small for bvecs: {path}")
    dim = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    if dim <= 0:
        raise RuntimeError(f"Invalid bvecs dim={dim}: {path}")
    rec = 4 + dim
    if raw_size % rec != 0:
        raise RuntimeError(f"Invalid bvecs size={raw_size}, rec={rec}")
    return dim, raw_size // rec


def read_bvecs_slice_u8(path: str, start: int, count: int) -> np.ndarray:
    dim, total = bvecs_get_dim_and_total_points(path)
    if start < 0 or count <= 0 or start + count > total:
        raise ValueError(f"Invalid bvecs slice [{start}, {start + count}) total={total}")
    rec = 4 + dim
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    mat = raw.reshape(total, rec)
    vecs_u8 = mat[start : start + count, 4:]
    return np.ascontiguousarray(vecs_u8)


def fvecs_get_dim_and_total_points(path: str) -> tuple[int, int]:
    raw = np.memmap(path, dtype=np.float32, mode="r")
    if raw.size == 0:
        raise RuntimeError(f"Empty fvecs: {path}")
    dim = int(raw.view(np.int32)[0])
    if dim <= 0:
        raise RuntimeError(f"Invalid fvecs dim={dim}: {path}")
    rec = dim + 1
    if raw.size % rec != 0:
        raise RuntimeError(f"Invalid fvecs size for dim={dim}: {path}")
    return dim, raw.size // rec


def read_fvecs_slice_f32(path: str, start: int, count: int) -> np.ndarray:
    dim, total = fvecs_get_dim_and_total_points(path)
    if start < 0 or count <= 0 or start + count > total:
        raise ValueError(f"Invalid fvecs slice [{start}, {start + count}) total={total}")
    raw = np.memmap(path, dtype=np.float32, mode="r")
    mat = raw.reshape(total, dim + 1)
    return np.ascontiguousarray(mat[start : start + count, 1:])


def _encode_batch(meta: dict[str, Any], vectors: np.ndarray) -> bytes:
    meta_raw = json.dumps(meta).encode("utf-8")
    header = struct.pack("<Q", len(meta_raw))
    return header + meta_raw + vectors.tobytes(order="C")


def _decode_batch(handler: BaseHTTPRequestHandler) -> tuple[dict, bytes]:
    content_len = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_len)
    if len(raw) < 8:
        raise RuntimeError("short binary payload")
    meta_len = struct.unpack("<Q", raw[:8])[0]
    if len(raw) < 8 + meta_len:
        raise RuntimeError("short metadata block")
    meta = json.loads(raw[8 : 8 + meta_len].decode("utf-8"))
    return meta, raw[8 + meta_len :]


def _memory_snapshot() -> dict:
    stats = {
        "rss_bytes": 0,
        "rss_mb": 0.0,
        "hwm_bytes": 0,
        "hwm_mb": 0.0,
        "vmsize_bytes": 0,
        "vmsize_mb": 0.0,
    }
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    stats["rss_bytes"] = kb * 1024
                    stats["rss_mb"] = round((kb * 1024) / (1024.0 * 1024.0), 3)
                elif line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    stats["hwm_bytes"] = kb * 1024
                    stats["hwm_mb"] = round((kb * 1024) / (1024.0 * 1024.0), 3)
                elif line.startswith("VmSize:"):
                    kb = int(line.split()[1])
                    stats["vmsize_bytes"] = kb * 1024
                    stats["vmsize_mb"] = round((kb * 1024) / (1024.0 * 1024.0), 3)
    except Exception:
        pass
    return stats


@dataclass
class BufferedJobState:
    job_id: str
    dim: int
    expected_count: int
    algo: str
    dist: str
    value_type: str
    wire_dtype: str
    threads: int
    with_meta_index: bool
    normalized_input: bool
    output_dir: str
    save_index: bool
    cef: int | None
    max_check_for_refine_graph: int | None
    graph_neighborhood_scale: float | None
    tpt_number: int | None
    tpt_leaf_size: int | None
    init_time_s: float = field(default_factory=time.time)
    receive_time_s: float = 0.0
    build_started: bool = False
    build_finished: bool = False
    error: str | None = None
    build_metrics: dict[str, Any] = field(default_factory=dict)
    buffered_count: int = 0
    last_global_offset: int = -1
    vectors: np.ndarray | None = None
    global_ids: np.ndarray | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    built_index: object | None = field(default=None, repr=False)

    def status_payload(self) -> dict:
        with self.lock:
            return {
                "ok": self.error is None,
                "job_id": self.job_id,
                "dim": self.dim,
                "expected_count": self.expected_count,
                "buffered_count": self.buffered_count,
                "build_started": self.build_started,
                "build_finished": self.build_finished,
                "error": self.error,
                "receive_time_s": round(float(self.receive_time_s), 6),
                "build_metrics": dict(self.build_metrics),
                "memory": _memory_snapshot(),
            }


class WorkerState:
    def __init__(self) -> None:
        self.jobs: dict[str, BufferedJobState] = {}
        self.lock = threading.Lock()

    def get(self, job_id: str) -> BufferedJobState | None:
        with self.lock:
            return self.jobs.get(job_id)

    def put(self, job: BufferedJobState) -> None:
        with self.lock:
            self.jobs[job.job_id] = job


WORKER_STATE = WorkerState()


def _make_index(
    algo: str,
    dist: str,
    dim: int,
    threads: int,
    value_type: str,
    cef: int | None,
    max_check_for_refine_graph: int | None,
    graph_neighborhood_scale: float | None,
    tpt_number: int | None,
    tpt_leaf_size: int | None,
):
    if SPTAG is None:
        raise RuntimeError("SPTAG Python module is not available")
    print(
        "[worker] creating AnnIndex "
        f"algo={algo} value_type={_normalize_value_type(value_type)} dim={dim} threads={threads}"
    )
    index = SPTAG.AnnIndex(algo, _normalize_value_type(value_type), dim)
    index.SetBuildParam("DistCalcMethod", dist, "Index")
    index.SetBuildParam("NumberOfThreads", str(int(threads)), "Index")
    if cef is not None:
        index.SetBuildParam("CEF", str(int(cef)), "Index")
    if max_check_for_refine_graph is not None:
        index.SetBuildParam("MaxCheckForRefineGraph", str(int(max_check_for_refine_graph)), "Index")
    if graph_neighborhood_scale is not None:
        index.SetBuildParam("GraphNeighborhoodScale", str(float(graph_neighborhood_scale)), "Index")
    if tpt_number is not None:
        index.SetBuildParam("TPTNumber", str(int(tpt_number)), "Index")
    if tpt_leaf_size is not None:
        index.SetBuildParam("TPTLeafSize", str(int(tpt_leaf_size)), "Index")
    return index


def _build_buffered_index(job: BufferedJobState) -> dict[str, Any]:
    with job.lock:
        if job.build_finished:
            return dict(job.build_metrics)
        if job.build_started:
            raise RuntimeError("build already in progress")
        if job.buffered_count != job.expected_count:
            raise RuntimeError(
                f"cannot build before all data arrives: buffered={job.buffered_count} expected={job.expected_count}"
            )
        job.build_started = True
        print(
            f"[worker] starting buffered build job_id={job.job_id} "
            f"buffered_count={job.buffered_count} expected_count={job.expected_count}"
        )

    total_build_stage_t0 = time.time()
    prep_t0 = time.time()
    vectors = job.vectors
    global_ids = job.global_ids
    if vectors is None or global_ids is None:
        raise RuntimeError("job buffers are missing")
    vectors_view = vectors[: job.expected_count]
    ids_view = global_ids[: job.expected_count]
    index_vectors = _vectors_for_index(job.value_type, vectors_view)
    metadata = _metadata_block(ids_view)
    prep_elapsed = time.time() - prep_t0

    index = _make_index(
        job.algo,
        job.dist,
        job.dim,
        job.threads,
        job.value_type,
        job.cef,
        job.max_check_for_refine_graph,
        job.graph_neighborhood_scale,
        job.tpt_number,
        job.tpt_leaf_size,
    )

    build_call_t0 = time.time()
    print(
        f"[worker] calling BuildWithMetaData job_id={job.job_id} "
        f"count={job.expected_count} dim={job.dim}"
    )
    ok = index.BuildWithMetaData(
        index_vectors.tobytes(order="C"),
        metadata,
        job.expected_count,
        job.with_meta_index,
        job.normalized_input,
    )
    build_call_elapsed = time.time() - build_call_t0
    if not ok:
        raise RuntimeError("SPTAG BuildWithMetaData failed")
    print(
        f"[worker] BuildWithMetaData completed job_id={job.job_id} "
        f"elapsed={build_call_elapsed:.6f}s"
    )

    update_t0 = time.time()
    print(f"[worker] calling UpdateIndex job_id={job.job_id}")
    index.UpdateIndex()
    update_elapsed = time.time() - update_t0
    print(f"[worker] UpdateIndex completed job_id={job.job_id} elapsed={update_elapsed:.6f}s")

    save_elapsed = 0.0
    saved = False
    if job.save_index:
        os.makedirs(job.output_dir, exist_ok=True)
        save_t0 = time.time()
        print(f"[worker] saving index job_id={job.job_id} output_dir={job.output_dir}")
        saved = bool(index.Save(job.output_dir))
        save_elapsed = time.time() - save_t0
        if not saved:
            raise RuntimeError(f"failed to save index to {job.output_dir}")
        print(f"[worker] save completed job_id={job.job_id} elapsed={save_elapsed:.6f}s")

    total_build_stage_elapsed = time.time() - total_build_stage_t0
    metrics = {
        "prep_time_s": round(float(prep_elapsed), 6),
        "build_call_time_s": round(float(build_call_elapsed), 6),
        "update_time_s": round(float(update_elapsed), 6),
        "native_build_time_s": round(float(build_call_elapsed + update_elapsed), 6),
        "save_time_s": round(float(save_elapsed), 6),
        "total_build_stage_s": round(float(total_build_stage_elapsed), 6),
        "saved": bool(saved),
        "output_dir": job.output_dir if job.save_index else "",
        "threads": int(job.threads),
        "value_type": job.value_type,
        "memory_after_build": _memory_snapshot(),
    }
    with job.lock:
        job.built_index = index
        job.build_metrics = dict(metrics)
        job.build_finished = True
    return metrics


class BufferedWorkerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/status":
            _json_response(self, 404, {"ok": False, "error": "unknown endpoint"})
            return
        params = parse_qs(parsed.query)
        job_id = params.get("job_id", [""])[0]
        if not job_id:
            _json_response(self, 400, {"ok": False, "error": "missing job_id"})
            return
        job = WORKER_STATE.get(job_id)
        if job is None:
            _json_response(self, 404, {"ok": False, "error": f"unknown job_id {job_id}"})
            return
        _json_response(self, 200, job.status_payload())

    def do_POST(self) -> None:
        try:
            if self.path == "/init":
                self._handle_init()
            elif self.path == "/add_batch":
                self._handle_add_batch()
            elif self.path == "/build":
                self._handle_build()
            elif self.path == "/shutdown":
                _json_response(self, 200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                _json_response(self, 404, {"ok": False, "error": "unknown endpoint"})
        except Exception as ex:
            _json_response(self, 500, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})

    def log_message(self, format: str, *args) -> None:
        return

    def _handle_init(self) -> None:
        req = _read_json(self)
        job_id = str(req["job_id"])
        dim = int(req["dim"])
        expected_count = int(req["expected_count"])
        wire_dtype = str(req["wire_dtype"])
        value_type = _normalize_value_type(req.get("value_type", "Float"))
        if wire_dtype not in {"uint8", "float32"}:
            raise ValueError(f"unsupported wire_dtype: {wire_dtype}")
        if expected_count <= 0 or dim <= 0:
            raise ValueError("expected_count and dim must be > 0")

        dtype = np.uint8 if wire_dtype == "uint8" else np.float32
        job = BufferedJobState(
            job_id=job_id,
            dim=dim,
            expected_count=expected_count,
            algo=str(req.get("algo", "BKT")),
            dist=str(req.get("dist", "L2")),
            value_type=value_type,
            wire_dtype=wire_dtype,
            threads=int(req.get("threads", max(1, os.cpu_count() or 1))),
            with_meta_index=bool(req.get("with_meta_index", False)),
            normalized_input=bool(req.get("normalized_input", False)),
            output_dir=str(req.get("output_dir", "")),
            save_index=bool(req.get("save_index", False)),
            cef=_optional_positive_int(req.get("cef"), "cef"),
            max_check_for_refine_graph=_optional_positive_int(
                req.get("max_check_for_refine_graph"), "max_check_for_refine_graph"
            ),
            graph_neighborhood_scale=_optional_positive_float(
                req.get("graph_neighborhood_scale"), "graph_neighborhood_scale"
            ),
            tpt_number=_optional_positive_int(req.get("tpt_number"), "tpt_number"),
            tpt_leaf_size=_optional_positive_int(req.get("tpt_leaf_size"), "tpt_leaf_size"),
            vectors=np.empty((expected_count, dim), dtype=dtype),
            global_ids=np.empty(expected_count, dtype=np.int64),
        )
        WORKER_STATE.put(job)
        print(
            f"[worker] initialized job_id={job_id} dim={dim} expected_count={expected_count} "
            f"wire_dtype={wire_dtype} value_type={value_type}"
        )
        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "dim": dim,
                "expected_count": expected_count,
                "wire_dtype": wire_dtype,
                "value_type": value_type,
            },
        )

    def _handle_add_batch(self) -> None:
        t0 = time.time()
        meta, vec_bytes = _decode_batch(self)
        job_id = str(meta["job_id"])
        start = int(meta["start"])
        global_start = int(meta.get("global_start", start))
        count = int(meta["count"])
        dim = int(meta["dim"])
        wire_dtype = str(meta["wire_dtype"])
        job = WORKER_STATE.get(job_id)
        if job is None:
            _json_response(self, 404, {"ok": False, "error": f"unknown job_id {job_id}"})
            return
        if dim != job.dim:
            raise ValueError(f"dim mismatch: {dim} vs {job.dim}")
        if wire_dtype != job.wire_dtype:
            raise ValueError(f"wire_dtype mismatch: {wire_dtype} vs {job.wire_dtype}")
        if count <= 0:
            raise ValueError("count must be > 0")
        end = start + count
        if start < 0 or end > job.expected_count:
            raise ValueError(f"invalid range [{start}, {end}) expected_count={job.expected_count}")
        dtype = np.uint8 if wire_dtype == "uint8" else np.float32
        expected_bytes = int(np.dtype(dtype).itemsize) * count * dim
        if len(vec_bytes) != expected_bytes:
            raise ValueError(f"payload size mismatch: got {len(vec_bytes)} expected {expected_bytes}")
        batch = np.frombuffer(vec_bytes, dtype=dtype).reshape(count, dim)

        with job.lock:
            if job.build_started:
                raise RuntimeError("cannot accept more batches after build started")
            if start != job.buffered_count:
                raise RuntimeError(f"expected next start {job.buffered_count}, got {start}")
            if job.vectors is None or job.global_ids is None:
                raise RuntimeError("job buffers not initialized")
            job.vectors[start:end] = batch
            job.global_ids[start:end] = np.arange(global_start, global_start + count, dtype=np.int64)
            job.buffered_count = end
            job.last_global_offset = end - 1
            job.receive_time_s += time.time() - t0
            print(
                f"[worker] buffered batch job_id={job_id} start={start} global_start={global_start} "
                f"count={count} buffered_count={job.buffered_count}/{job.expected_count}"
            )

        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "accepted": True,
                "start": start,
                "count": count,
                "buffered_count": end,
                "receive_time_s": round(float(job.receive_time_s), 6),
            },
        )

    def _handle_build(self) -> None:
        req = _read_json(self)
        job_id = str(req["job_id"])
        job = WORKER_STATE.get(job_id)
        if job is None:
            _json_response(self, 404, {"ok": False, "error": f"unknown job_id {job_id}"})
            return
        metrics = _build_buffered_index(job)
        print(f"[worker] build finished job_id={job_id} metrics={json.dumps(metrics, sort_keys=True)}")
        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "buffered_count": job.buffered_count,
                "expected_count": job.expected_count,
                "receive_time_s": round(float(job.receive_time_s), 6),
                "build_metrics": metrics,
            },
        )


@dataclass
class ForwardJobState:
    job_id: str
    worker_url: str
    init_payload: dict[str, Any]
    forwarded_vectors: int = 0
    forwarded_batches: int = 0
    forward_rpc_time_s: float = 0.0
    build_response: dict[str, Any] | None = None


class MasterState:
    def __init__(self, worker_url: str, request_timeout: int, retries: int) -> None:
        self.worker_url = _normalize_base_url(worker_url)
        self.request_timeout = request_timeout
        self.retries = retries
        self.jobs: dict[str, ForwardJobState] = {}
        self.lock = threading.Lock()

    def get(self, job_id: str) -> ForwardJobState | None:
        with self.lock:
            return self.jobs.get(job_id)

    def put(self, job: ForwardJobState) -> None:
        with self.lock:
            self.jobs[job.job_id] = job


class MasterHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/status":
            _json_response(self, 404, {"ok": False, "error": "unknown endpoint"})
            return
        params = parse_qs(parsed.query)
        job_id = params.get("job_id", [""])[0]
        if not job_id:
            _json_response(self, 400, {"ok": False, "error": "missing job_id"})
            return
        st = self.server.master_state.get(job_id)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": f"unknown job_id {job_id}"})
            return
        worker_status = _post_with_retry(
            f"{st.worker_url}/status?job_id={job_id}",
            lambda: json.loads(urllib.request.urlopen(f"{st.worker_url}/status?job_id={job_id}", timeout=self.server.master_state.request_timeout).read().decode("utf-8")),
            self.server.master_state.retries,
            "worker status",
        )
        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "forwarded_vectors": st.forwarded_vectors,
                "forwarded_batches": st.forwarded_batches,
                "forward_rpc_time_s": round(float(st.forward_rpc_time_s), 6),
                "worker_status": worker_status,
            },
        )

    def do_POST(self) -> None:
        try:
            if self.path == "/init":
                self._handle_init()
            elif self.path == "/add_batch":
                self._handle_add_batch()
            elif self.path == "/build":
                self._handle_build()
            elif self.path == "/shutdown":
                _json_response(self, 200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                _json_response(self, 404, {"ok": False, "error": "unknown endpoint"})
        except Exception as ex:
            _json_response(self, 500, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})

    def log_message(self, format: str, *args) -> None:
        return

    def _handle_init(self) -> None:
        req = _read_json(self)
        job_id = str(req["job_id"])
        master_state = self.server.master_state
        worker_url = master_state.worker_url
        t0 = time.time()
        resp = _post_with_retry(
            f"{worker_url}/init",
            lambda u=f"{worker_url}/init", p=req: _http_json_post(u, p, master_state.request_timeout),
            master_state.retries,
            "worker init",
        )
        rpc_elapsed = time.time() - t0
        if not resp.get("ok", False):
            raise RuntimeError(f"worker init failed: {resp}")
        master_state.put(
            ForwardJobState(
                job_id=job_id,
                worker_url=worker_url,
                init_payload=req,
                forward_rpc_time_s=rpc_elapsed,
            )
        )
        _json_response(self, 200, {"ok": True, "job_id": job_id, "worker_response": resp})

    def _handle_add_batch(self) -> None:
        master_state = self.server.master_state
        meta, vec_bytes = _decode_batch(self)
        job_id = str(meta["job_id"])
        st = master_state.get(job_id)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": f"unknown job_id {job_id}"})
            return
        payload = struct.pack("<Q", len(json.dumps(meta).encode("utf-8"))) + json.dumps(meta).encode("utf-8") + vec_bytes
        t0 = time.time()
        resp = _post_with_retry(
            f"{st.worker_url}/add_batch",
            lambda u=f"{st.worker_url}/add_batch", p=payload: _http_binary_post(u, p, master_state.request_timeout),
            master_state.retries,
            "worker add_batch",
        )
        rpc_elapsed = time.time() - t0
        if not resp.get("ok", False):
            raise RuntimeError(f"worker add_batch failed: {resp}")
        st.forwarded_vectors += int(meta["count"])
        st.forwarded_batches += 1
        st.forward_rpc_time_s += rpc_elapsed
        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "forwarded_vectors": st.forwarded_vectors,
                "forwarded_batches": st.forwarded_batches,
                "forward_rpc_time_s": round(float(st.forward_rpc_time_s), 6),
                "worker_response": resp,
            },
        )

    def _handle_build(self) -> None:
        req = _read_json(self)
        job_id = str(req["job_id"])
        master_state = self.server.master_state
        st = master_state.get(job_id)
        if st is None:
            _json_response(self, 404, {"ok": False, "error": f"unknown job_id {job_id}"})
            return
        t0 = time.time()
        resp = _post_with_retry(
            f"{st.worker_url}/build",
            lambda u=f"{st.worker_url}/build", p=req: _http_json_post(u, p, master_state.request_timeout),
            master_state.retries,
            "worker build",
        )
        rpc_elapsed = time.time() - t0
        if not resp.get("ok", False):
            raise RuntimeError(f"worker build failed: {resp}")
        st.forward_rpc_time_s += rpc_elapsed
        st.build_response = dict(resp)
        _json_response(
            self,
            200,
            {
                "ok": True,
                "job_id": job_id,
                "forwarded_vectors": st.forwarded_vectors,
                "forwarded_batches": st.forwarded_batches,
                "forward_rpc_time_s": round(float(st.forward_rpc_time_s), 6),
                "worker_build_response": resp,
            },
        )


def run_worker(args: argparse.Namespace) -> None:
    server = ThreadingHTTPServer((args.host, args.port), BufferedWorkerHandler)
    print(f"[worker] buffered full-build worker listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def run_master(args: argparse.Namespace) -> None:
    server = ThreadingHTTPServer((args.host, args.port), MasterHandler)
    server.master_state = MasterState(args.worker_url, args.request_timeout, args.retries)
    print(
        f"[master] buffered full-build master listening on {args.host}:{args.port} "
        f"forwarding to {_normalize_base_url(args.worker_url)}"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def run_client(args: argparse.Namespace) -> None:
    total_t0 = time.time()
    master_url = _normalize_base_url(args.master_url)
    base_path = Path(args.base_path)
    ext = base_path.suffix.lower()
    if ext not in {".bvecs", ".fvecs"}:
        raise ValueError("base_path must end with .bvecs or .fvecs")

    if ext == ".bvecs":
        dim, total_points = bvecs_get_dim_and_total_points(str(base_path))
        wire_dtype = "uint8"
    else:
        dim, total_points = fvecs_get_dim_and_total_points(str(base_path))
        wire_dtype = "float32"

    subset_start = int(args.start_point)
    if subset_start < 0 or subset_start >= total_points:
        raise ValueError(f"start_point out of range [0, {total_points - 1}]")
    subset_n = total_points - subset_start if args.max_points is None else int(args.max_points)
    if subset_n <= 0 or subset_start + subset_n > total_points:
        raise ValueError(f"invalid subset [{subset_start}, {subset_start + subset_n}) total={total_points}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if int(args.client_threads) != 1:
        raise ValueError("only sequential client sending is supported; set --client_threads=1")

    init_payload = {
        "job_id": args.job_id,
        "dim": dim,
        "expected_count": subset_n,
        "algo": args.algo,
        "dist": args.dist,
        "value_type": args.value_type,
        "threads": args.threads,
        "with_meta_index": bool(args.with_meta_index),
        "normalized_input": bool(args.normalized_input),
        "output_dir": args.output_dir,
        "save_index": bool(args.save_index),
        "wire_dtype": wire_dtype,
        **_build_params_payload(
            args.cef,
            args.max_check_for_refine_graph,
            args.graph_neighborhood_scale,
            args.tpt_number,
            args.tpt_leaf_size,
        ),
    }

    init_resp = _post_with_retry(
        f"{master_url}/init",
        lambda: _http_json_post(f"{master_url}/init", init_payload, args.request_timeout),
        args.retries,
        "master init",
    )
    if not init_resp.get("ok", False):
        raise RuntimeError(f"master init failed: {init_resp}")

    send_time_s = 0.0
    for local_start in range(0, subset_n, args.batch_size):
        count = min(args.batch_size, subset_n - local_start)
        global_start = subset_start + local_start
        if wire_dtype == "uint8":
            vectors = read_bvecs_slice_u8(str(base_path), global_start, count)
        else:
            vectors = read_fvecs_slice_f32(str(base_path), global_start, count)
        meta = {
            "job_id": args.job_id,
            "start": local_start,
            "global_start": global_start,
            "count": count,
            "dim": dim,
            "wire_dtype": wire_dtype,
        }
        payload = _encode_batch(meta, vectors)
        t0 = time.time()
        resp = _post_with_retry(
            f"{master_url}/add_batch",
            lambda p=payload: _http_binary_post(f"{master_url}/add_batch", p, args.request_timeout),
            args.retries,
            "master add_batch",
        )
        send_time_s += time.time() - t0
        if not resp.get("ok", False):
            raise RuntimeError(f"add_batch failed at offset={local_start}: {resp}")

    build_resp = _post_with_retry(
        f"{master_url}/build",
        lambda: _http_json_post(f"{master_url}/build", {"job_id": args.job_id}, args.build_timeout),
        args.retries,
        "master build",
    )
    if not build_resp.get("ok", False):
        raise RuntimeError(f"build failed: {build_resp}")
    total_elapsed = time.time() - total_t0
    result = {
        "ok": True,
        "job_id": args.job_id,
        "subset_start": subset_start,
        "subset_count": subset_n,
        "batch_size": args.batch_size,
        "wire_dtype": wire_dtype,
        "client_send_time_s": round(float(send_time_s), 6),
        "end_to_end_time_s": round(float(total_elapsed), 6),
        "master_response": build_resp,
    }
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buffered full-build SPTAG experiment harness.")
    sub = parser.add_subparsers(dest="role", required=True)

    worker = sub.add_parser("worker", help="Run the buffered worker")
    worker.add_argument("--host", default="0.0.0.0")
    worker.add_argument("--port", type=int, default=18080)

    master = sub.add_parser("master", help="Run the forwarding master")
    master.add_argument("--host", default="0.0.0.0")
    master.add_argument("--port", type=int, default=18079)
    master.add_argument("--worker_url", required=True)
    master.add_argument("--request_timeout", type=int, default=600)
    master.add_argument("--retries", type=int, default=5)

    client = sub.add_parser("client", help="Run the dataset sender")
    client.add_argument("--master_url", required=True)
    client.add_argument("--base_path", required=True)
    client.add_argument("--output_dir", required=True)
    client.add_argument("--job_id", default="buffered_full_build_job")
    client.add_argument("--start_point", type=int, default=0)
    client.add_argument("--max_points", type=int, default=1000000)
    client.add_argument("--batch_size", type=int, default=5000)
    client.add_argument("--algo", default="BKT", choices=["BKT", "KDT", "SPANN"])
    client.add_argument("--dist", default="L2", choices=["L2", "Cosine"])
    client.add_argument("--value_type", default="UInt8", choices=["Float", "UInt8"])
    client.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    client.add_argument("--client_threads", type=int, default=1)
    client.add_argument("--with_meta_index", action="store_true")
    client.add_argument("--normalized_input", action="store_true")
    client.add_argument("--save_index", action="store_true")
    client.add_argument("--request_timeout", type=int, default=600)
    client.add_argument("--build_timeout", type=int, default=7200)
    client.add_argument("--retries", type=int, default=5)
    client.add_argument("--cef", type=int, default=None)
    client.add_argument("--tpt_number", type=int, default=None)
    client.add_argument("--tpt_leaf_size", type=int, default=None)
    client.add_argument("--max_check_for_refine_graph", type=int, default=None)
    client.add_argument("--graph_neighborhood_scale", type=float, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.role == "worker":
        run_worker(args)
    elif args.role == "master":
        run_master(args)
    elif args.role == "client":
        run_client(args)
    else:
        raise ValueError(f"unknown role {args.role}")


if __name__ == "__main__":
    main()
