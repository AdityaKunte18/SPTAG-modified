# =========================
# worker.py (queue-based)
# =========================
#!/usr/bin/env python3
"""Distributed SPTAG shard-build worker (queue-based ingestion).

Endpoints:
- POST /init
- POST /add_batch
- POST /finalize
- GET  /status?job_id=...&shard_id=...
- POST /shutdown

Protocol notes:
- /init, /finalize use JSON body.
- /add_batch body format:
    [8-byte little-endian uint64 meta_len][meta_json][raw float32 bytes]
  where meta_json has:
    job_id, shard_id, batch_id, global_offset, num, dim, with_meta_index, normalized

Key change:
- /add_batch now enqueues and returns quickly.
- Background consumer thread per job applies AddWithMetaData/BuildWithMetaData.
- If queue is full, /add_batch returns 429 with {"queue_full": true, "retry_after_ms": ...}.
- /finalize waits for queue to drain before UpdateIndex/Save.
"""
from __future__ import annotations
import SPTAG
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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np



@dataclass
class JobState:
    index: object
    algo: str
    dim: int
    save_dir: str
    with_meta_index: bool

    first_batch_done: bool = False

    vectors_ingested: int = 0
    last_batch_id: int = -1          # last APPLIED batch id
    next_expected_batch_id: int = 0  # next batch id we will ACCEPT/QUEUE

    finalized: bool = False
    init_time_s: float = 0.0
    add_time_s: float = 0.0
    finalize_time_s: float = 0.0

    # Queue ingestion
    q: "queue.Queue[tuple[int,int,int,int,bool,bytes]]" | None = None
    stop_event: threading.Event | None = None
    consumer: threading.Thread | None = None

    # If consumer hits a fatal error, store it here
    error: str | None = None


STATE = {}
STATE_LOCK = threading.Lock()


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
        # Client disconnected before reading response.
        return


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    content_len = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_len) if content_len > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def _meta_block(global_begin: int, global_end: int) -> bytes:
    # Slightly less intermediate memory than building one massive Python string first.
    return b"".join(f"{gid}\n".encode("utf-8") for gid in range(global_begin, global_end))


def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    return url


def _default_advertise_host(bind_host: str) -> str:
    if bind_host in ("", "0.0.0.0", "::"):
        return socket.gethostname()
    return bind_host


def _register_with_master(master_url: str, advertise_host: str, port: int, timeout_s: int, retries: int) -> dict:
    url = f"{_normalize_base_url(master_url)}/register_worker"
    payload = {"host": advertise_host, "port": port}
    return _post_with_retry(
        url,
        lambda u=url, p=payload: _http_json_post(u, p, timeout_s),
        retries,
        "register worker",
    )


def _consumer_loop(key: str) -> None:
    """
    Applies queued batches to SPTAG for a given job key.
    Exits when stop_event is set or sentinel None is received.
    """
    while True:
        with STATE_LOCK:
            st = STATE.get(key)
        if st is None:
            return
        queue_ref = st.q

        # Stop requested?
        if st.stop_event is not None and st.stop_event.is_set():
            return

        try:
            item = st.q.get(timeout=0.25)
        except queue.Empty:
            continue

        if item is None:
            st.q.task_done()
            return

        batch_id, global_offset, num, dim, normalized, vec_bytes = item

        # If job got finalized while items remain, just drain.
        with STATE_LOCK:
            st = STATE.get(key)
            if st is None:
                # Job state disappeared after dequeue; acknowledge the work item
                # against the queue object we pulled it from.
                try:
                    if queue_ref is not None:
                        queue_ref.task_done()
                except Exception:
                    pass
                return
            if st.finalized:
                st.q.task_done()
                continue
            if st.error is not None:
                st.q.task_done()
                continue

        # Heavy work (no global lock)
        try:
            vectors = np.frombuffer(vec_bytes, dtype=np.float32).reshape(num, dim)
            metadata = _meta_block(global_offset, global_offset + num)

            t_add_start = time.time()
            if (not st.first_batch_done) and st.algo.upper() == "SPANN":
                ok = st.index.BuildWithMetaData(vectors.tobytes(), metadata, num, st.with_meta_index, normalized)
            else:
                ok = st.index.AddWithMetaData(vectors.tobytes(), metadata, num, st.with_meta_index, normalized)
            add_elapsed = time.time() - t_add_start

            if not ok:
                raise RuntimeError("SPTAG add/build with metadata failed")

            with STATE_LOCK:
                st.first_batch_done = True
                st.last_batch_id = batch_id
                st.vectors_ingested += num
                st.add_time_s += add_elapsed

        except Exception as ex:
            with STATE_LOCK:
                st.error = f"{type(ex).__name__}: {ex}"
        finally:
            st.q.task_done()


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

        with STATE_LOCK:
            st = STATE.get(key)
            if st is None:
                _json_response(self, 404, {"ok": False, "error": "job not found"})
                return
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
                    "vectors_ingested": st.vectors_ingested,
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
                },
            )

    def do_POST(self):
        if self.path == "/init":
            return self._handle_init()
        if self.path == "/add_batch":
            return self._handle_add_batch()
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

        index = SPTAG.AnnIndex(algo, "Float", dim)
        index.SetBuildParam("DistCalcMethod", dist, "Index")
        index.SetBuildParam("NumberOfThreads", str(threads), "Index")

        key = _job_key(job_id, shard_id)
        qmax = getattr(self.server, "queue_max_batches", 3)

        stop_event = threading.Event()
        q = queue.Queue(maxsize=qmax)

        st = JobState(
            index=index,
            algo=algo,
            dim=dim,
            save_dir=save_dir,
            with_meta_index=with_meta_index,
            init_time_s=time.time(),
            q=q,
            stop_event=stop_event,
        )

        with STATE_LOCK:
            # If re-init happens, overwrite old state (best effort)
            STATE[key] = st

        # Start consumer thread
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
            global_offset = int(meta["global_offset"])
            num = int(meta["num"])
            dim = int(meta["dim"])
            normalized = bool(meta.get("normalized", False))
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid batch meta: {ex}"})
            return

        key = _job_key(job_id, shard_id)
        with STATE_LOCK:
            st = STATE.get(key)

        if st is None:
            _json_response(self, 404, {"ok": False, "error": "job not initialized"})
            return
        if st.finalized:
            _json_response(self, 409, {"ok": False, "error": "job already finalized"})
            return
        if st.error is not None:
            _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
            return
        if dim != st.dim:
            _json_response(self, 400, {"ok": False, "error": f"dim mismatch: batch {dim}, job {st.dim}"})
            return

        expected_bytes = num * dim * 4
        if len(vec_bytes) != expected_bytes:
            _json_response(
                self,
                400,
                {"ok": False, "error": f"invalid vector payload bytes={len(vec_bytes)}, expected={expected_bytes}"},
            )
            return

        with STATE_LOCK:
            # Idempotency for already-applied
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

            # In-order acceptance (simple; master is sequential per shard anyway)
            if batch_id != st.next_expected_batch_id:
                _json_response(
                    self,
                    409,
                    {"ok": False, "error": f"out of order batch_id={batch_id}, expected {st.next_expected_batch_id}"},
                )
                return

            # Try enqueue without blocking long; if full, tell master to wait
            try:
                st.q.put_nowait((batch_id, global_offset, num, dim, normalized, vec_bytes))
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

    def _handle_finalize(self):
        req = _read_json(self)
        try:
            job_id = str(req["job_id"])
            shard_id = int(req["shard_id"])
        except Exception as ex:
            _json_response(self, 400, {"ok": False, "error": f"invalid finalize payload: {ex}"})
            return

        key = _job_key(job_id, shard_id)
        with STATE_LOCK:
            st = STATE.get(key)

        if st is None:
            _json_response(self, 404, {"ok": False, "error": "job not found"})
            return
        if st.finalized:
            _json_response(self, 200, {"ok": True, "already_finalized": True, "save_dir": st.save_dir})
            return
        if st.error is not None:
            _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
            return

        # Wait for queued batches to be applied
        if st.q is not None:
            st.q.join()

        # Re-check after drain
        with STATE_LOCK:
            st = STATE.get(key)
            if st is None:
                _json_response(self, 404, {"ok": False, "error": "job disappeared"})
                return
            if st.error is not None:
                _json_response(self, 500, {"ok": False, "error": f"worker error: {st.error}"})
                return

        t_fin_start = time.time()
        st.index.UpdateIndex()
        os.makedirs(st.save_dir, exist_ok=True)
        ok = st.index.Save(st.save_dir)
        fin_elapsed = time.time() - t_fin_start
        if not ok:
            _json_response(self, 500, {"ok": False, "error": f"failed to save index to {st.save_dir}"})
            return

        with STATE_LOCK:
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
                "save_dir": st.save_dir,
                "add_time_s": st.add_time_s,
                "finalize_time_s": st.finalize_time_s,
                "elapsed_s": max(0.0, time.time() - st.init_time_s),
                "last_applied_batch_id": st.last_batch_id,
            },
        )

        if getattr(self.server, "exit_after_finalize", False):
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _handle_shutdown(self):
        _json_response(self, 200, {"ok": True, "message": "worker shutting down"})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="SPTAG distributed index build worker (queue-based).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--master_url", default=None, help="Optional master URL for dynamic worker registration.")
    parser.add_argument("--advertise_host", default=None, help="Host/IP the master should use to reach this worker.")
    parser.add_argument("--register_timeout", type=int, default=30)
    parser.add_argument("--register_retries", type=int, default=5)
    parser.add_argument("--queue_max_batches", type=int, default=3, help="Max queued batches per job (backpressure).")
    parser.add_argument(
        "--no_exit_after_finalize",
        action="store_true",
        help="Keep worker running after a successful /finalize.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), WorkerHandler)
    server.exit_after_finalize = not args.no_exit_after_finalize
    server.queue_max_batches = args.queue_max_batches
    print(f"Worker listening on {args.host}:{args.port} queue_max_batches={args.queue_max_batches}")
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
