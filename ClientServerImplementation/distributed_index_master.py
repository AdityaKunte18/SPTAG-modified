# =========================
# master.py (HTTP server)
# =========================
#!/usr/bin/env python3
"""Distributed SPTAG shard build master (HTTP server).

Receives batches from a client, tracks workers that register dynamically,
splits each incoming client batch across the workers currently initialized
for the job, and forwards sub-batches to workers.

Endpoints:
- POST /register_worker — register a worker with the master
- POST /init            — initialize a job
- POST /add_batch       — receive a batch from client, split & forward to worker(s)
- POST /finalize        — finalize a job (drain workers, save indexes)
- GET  /status          — query job status
- POST /shutdown        — shut down the master server
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np


# ── HTTP helpers (for forwarding to workers) ────────────────────────────────

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
    while True:
        resp = _post_with_retry(
            url,
            lambda p=payload: _http_binary_post(url, p, request_timeout),
            retries,
            what,
        )
        if resp.get("queue_full", False):
            delay_ms = int(resp.get("retry_after_ms", 200))
            time.sleep(max(0.0, delay_ms / 1000.0))
            continue
        return resp


# ── HTTP response / request helpers (for serving clients) ───────────────────

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


# ── Shard computation ───────────────────────────────────────────────────────

def shard_bounds(n: int, num_shards: int, shard_id: int) -> Tuple[int, int]:
    base = n // num_shards
    rem = n % num_shards
    start = shard_id * base + (shard_id if shard_id < rem else rem)
    size = base + (1 if shard_id < rem else 0)
    return start, start + size


# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class Worker:
    worker_id: int
    host: str
    port: int

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
class MasterJobState:
    job_id: str
    dim: int
    algo: str
    dist: str
    output_dir: str
    threads: int
    with_meta_index: bool
    shards: List[ShardInfo] = field(default_factory=list)
    finalized: bool = False
    error: str | None = None
    init_time: float = 0.0
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)


STATE: dict[str, MasterJobState] = {}
STATE_LOCK = threading.Lock()


def _resolve_workers(args) -> List[Worker]:
    if args.workers_file:
        arr = _read_json_file(Path(args.workers_file))
        workers = [Worker(i, str(x["host"]), int(x["port"])) for i, x in enumerate(arr)]
    else:
        if not args.workers:
            return []
        workers = []
        for i, item in enumerate(args.workers):
            host, port = item.rsplit(":", 1)
            workers.append(Worker(i, host, int(port)))
    return workers


def _snapshot_registered_workers(server) -> List[Worker]:
    with server.worker_lock:
        workers = list(server.workers.values())
    workers.sort(key=lambda w: w.worker_id)
    return workers


def _register_worker(server, host: str, port: int) -> tuple[Worker, bool]:
    worker_key = f"{host}:{port}"
    with server.worker_lock:
        worker = server.workers.get(worker_key)
        if worker is not None:
            return worker, False
        worker = Worker(server.next_worker_id, host, port)
        server.workers[worker.key] = worker
        server.next_worker_id += 1
        return worker, True


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
        "threads": job.threads,
        "dim": job.dim,
        "save_dir": save_dir,
        "with_meta_index": job.with_meta_index,
    }
    url = f"{worker.base_url}/init"
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
    print(f"[init] shard={worker.worker_id} worker={worker.host}:{worker.port} save={save_dir}")
    return shard


def _ensure_job_workers(server, job: MasterJobState) -> List[ShardInfo]:
    request_timeout = server.request_timeout
    retries = server.retries
    registered_workers = _snapshot_registered_workers(server)
    initialized_worker_ids = {shard.worker.worker_id for shard in job.shards}

    for worker in registered_workers:
        if worker.worker_id in initialized_worker_ids:
            continue
        _init_worker_for_job(
            job=job,
            worker=worker,
            request_timeout=request_timeout,
            retries=retries,
        )
        initialized_worker_ids.add(worker.worker_id)

    return list(job.shards)


# ── Master HTTP handler ─────────────────────────────────────────────────────

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
        if self.path == "/finalize":
            return self._handle_finalize()
        if self.path == "/shutdown":
            return self._handle_shutdown()
        _json_response(self, 404, {"ok": False, "error": "not found"})

    # ── /register_worker ────────────────────────────────────────────────

    def _handle_register_worker(self):
        req = _read_json(self)
        try:
            host = str(req["host"]).strip()
            port = int(req["port"])
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid worker payload: {ex}"})
            return

        if not host:
            _json_response(self, 400, {"ok": False, "error": "host is required"})
            return

        worker, created = _register_worker(self.server, host, port)
        if created:
            print(f"[worker] registered worker_id={worker.worker_id} addr={worker.host}:{worker.port}")

        _json_response(
            self,
            200,
            {
                "ok": True,
                "worker_id": worker.worker_id,
                "host": worker.host,
                "port": worker.port,
                "already_registered": not created,
                "registered_workers": len(_snapshot_registered_workers(self.server)),
            },
        )

    # ── /init ───────────────────────────────────────────────────────────

    def _handle_init(self):
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            algo = str(req.get("algo", "BKT"))
            dist = str(req.get("dist", "L2"))
            dim = int(req["dim"])
            output_dir = str(req["output_dir"])
            threads = int(req.get("threads", 8))
            with_meta_index = bool(req.get("with_meta_index", False))
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid init payload: {ex}"})
            return

        job = MasterJobState(
            job_id=job_id,
            dim=dim,
            algo=algo,
            dist=dist,
            output_dir=output_dir,
            threads=threads,
            with_meta_index=with_meta_index,
            init_time=time.time(),
        )

        try:
            with job.lock:
                _ensure_job_workers(self.server, job)
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
            },
        )

    # ── /add_batch ──────────────────────────────────────────────────────

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

            expected_bytes = num * dim * 4
            if len(vec_bytes) != expected_bytes:
                _json_response(
                    self, 400,
                    {"ok": False, "error": f"invalid vector payload bytes={len(vec_bytes)}, expected={expected_bytes}"},
                )
                return

            try:
                active_shards = _ensure_job_workers(self.server, job)
            except Exception as ex:
                job.error = str(ex)
                _json_response(self, 502, {"ok": False, "error": str(ex)})
                return

            if not active_shards:
                _json_response(
                    self,
                    429,
                    {"ok": False, "queue_full": True, "retry_after_ms": 500, "error": "no workers registered yet"},
                )
                return

            # Deserialize vectors once, then split based on the worker count
            # that is active for this batch. When workers join, later batches
            # are rebalanced over the larger shard set automatically.
            vectors = np.frombuffer(vec_bytes, dtype=np.float32).reshape(num, dim)
            request_timeout = self.server.request_timeout
            retries = self.server.retries
            debug = self.server.debug
            num_shards = len(active_shards)

            def forward_sub_batch(shard: ShardInfo, sub_offset: int, local_begin: int, local_end: int):
                sub_num = local_end - local_begin
                sub_vectors = np.ascontiguousarray(vectors[local_begin:local_end], dtype=np.float32)

                batch_id = shard.next_batch_id

                worker_meta = {
                    "job_id": job_id,
                    "shard_id": shard.shard_id,
                    "batch_id": batch_id,
                    "global_offset": sub_offset,
                    "num": sub_num,
                    "dim": dim,
                    "with_meta_index": with_meta_index,
                    "normalized": normalized,
                }
                meta_raw = json.dumps(worker_meta).encode("utf-8")
                payload = struct.pack("<Q", len(meta_raw)) + meta_raw + sub_vectors.tobytes()

                url = f"{shard.worker.base_url}/add_batch"
                resp = _send_batch_with_backpressure(
                    url=url,
                    payload=payload,
                    request_timeout=request_timeout,
                    retries=retries,
                    what=f"add_batch shard {shard.shard_id} batch {batch_id}",
                )
                if not resp.get("ok", False):
                    raise RuntimeError(
                        f"worker add_batch failed shard={shard.shard_id} batch={batch_id}: {resp}"
                    )

                shard.next_batch_id += 1
                shard.vectors_forwarded += sub_num

                if debug:
                    print(
                        f"[debug] forwarded shard={shard.shard_id} batch={batch_id} "
                        f"offset={sub_offset} num={sub_num}"
                    )

            sub_batches = []
            for shard_index, shard in enumerate(active_shards):
                local_s, local_e = shard_bounds(num, num_shards, shard_index)
                if local_s >= local_e:
                    continue
                sub_offset = global_offset + local_s
                sub_batches.append((shard, sub_offset, local_s, local_e))

            try:
                if len(sub_batches) == 1:
                    forward_sub_batch(*sub_batches[0])
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sub_batches)) as executor:
                        futures = [executor.submit(forward_sub_batch, *sb) for sb in sub_batches]
                        for f in concurrent.futures.as_completed(futures):
                            f.result()
            except Exception as ex:
                job.error = str(ex)
                print(f"[error] add_batch forward failed: {ex}")
                _json_response(self, 502, {"ok": False, "error": str(ex)})
                return

        _json_response(self, 200, {
            "ok": True,
            "job_id": job_id,
            "vectors_accepted": num,
            "shards_touched": len(sub_batches),
        })

    # ── /finalize ───────────────────────────────────────────────────────

    def _handle_finalize(self):
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

        with job.lock:
            if job.finalized:
                _json_response(self, 200, {"ok": True, "already_finalized": True})
                return
            if job.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"job error: {job.error}"})
                return

            finalize_timeout = self.server.finalize_timeout
            retries = self.server.retries

            shard_results = []
            for shard in job.shards:
                fin_req = {"job_id": job_id, "shard_id": shard.shard_id}
                url = f"{shard.worker.base_url}/finalize"
                try:
                    resp = _post_with_retry(
                        url,
                        lambda u=url, p=fin_req: _http_json_post(u, p, finalize_timeout),
                        retries,
                        f"finalize shard {shard.shard_id}",
                    )
                    if not resp.get("ok", False):
                        err_msg = f"worker finalize failed shard {shard.shard_id}: {resp}"
                        print(f"[error] {err_msg}")
                        job.error = err_msg
                        _json_response(self, 502, {"ok": False, "error": err_msg})
                        return
                except Exception as ex:
                    err_msg = f"worker finalize failed shard {shard.shard_id}: {ex}"
                    print(f"[error] {err_msg}")
                    job.error = err_msg
                    _json_response(self, 502, {"ok": False, "error": err_msg})
                    return

                shard_results.append({
                    "shard_id": shard.shard_id,
                    "worker_id": shard.worker.worker_id,
                    "worker": f"{shard.worker.host}:{shard.worker.port}",
                    "vectors_ingested": resp.get("vectors_ingested", 0),
                    "save_dir": resp.get("save_dir", ""),
                    "add_time_s": resp.get("add_time_s", 0.0),
                    "finalize_time_s": resp.get("finalize_time_s", 0.0),
                    "elapsed_s": resp.get("elapsed_s", 0.0),
                })
                print(
                    f"[finalize] shard={shard.shard_id} saved={resp.get('save_dir')} "
                    f"vectors={resp.get('vectors_ingested')} "
                    f"worker_add={resp.get('add_time_s', 0):.3f}s "
                    f"worker_finalize={resp.get('finalize_time_s', 0):.3f}s"
                )

            job.finalized = True

        _json_response(self, 200, {"ok": True, "job_id": job_id, "shards": shard_results})

        if getattr(self.server, "exit_after_finalize", False):
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    # ── /status ─────────────────────────────────────────────────────────

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
            shard_info = []
            for s in job.shards:
                shard_info.append({
                    "shard_id": s.shard_id,
                    "worker_id": s.worker.worker_id,
                    "worker": f"{s.worker.host}:{s.worker.port}",
                    "next_batch_id": s.next_batch_id,
                    "vectors_forwarded": s.vectors_forwarded,
                })
            _json_response(self, 200, {
                "ok": True,
                "job_id": job_id,
                "dim": job.dim,
                "algo": job.algo,
                "registered_workers": len(_snapshot_registered_workers(self.server)),
                "initialized_shards": len(job.shards),
                "finalized": job.finalized,
                "error": job.error,
                "shards": shard_info,
            })

    # ── /shutdown ───────────────────────────────────────────────────────

    def _handle_shutdown(self):
        _json_response(self, 200, {"ok": True, "message": "master shutting down"})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distributed SPTAG shard build master (HTTP server).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--workers_file", default=None, help='JSON array: [{"host":"...","port":18080}, ...]')
    parser.add_argument("--workers", nargs="*", default=None, help="Inline worker list: host:port host:port ...")
    parser.add_argument("--request_timeout", type=int, default=60)
    parser.add_argument("--finalize_timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no_exit_after_finalize",
        action="store_true",
        help="Keep master (and workers) running after a successful /finalize.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MasterHandler)
    server.worker_lock = threading.Lock()
    server.workers = {}
    server.next_worker_id = 0
    server.request_timeout = args.request_timeout
    server.finalize_timeout = args.finalize_timeout
    server.retries = args.retries
    server.debug = args.debug
    server.exit_after_finalize = not args.no_exit_after_finalize

    workers = _resolve_workers(args)
    for worker in workers:
        registered, _ = _register_worker(server, worker.host, worker.port)
        print(f"[worker] configured worker_id={registered.worker_id} addr={registered.host}:{registered.port}")

    registered_workers = _snapshot_registered_workers(server)
    print(f"Master listening on {args.host}:{args.port} workers={len(registered_workers)}")
    for w in registered_workers:
        print(f"  worker[{w.worker_id}]: {w.host}:{w.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
