# =========================
# client.py
# =========================
#!/usr/bin/env python3
"""Client for distributed SPTAG shard build.

Reads vector data (.bvecs/.fvecs), streams batches to the master server,
which handles sharding and forwarding to workers.
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np


# ── HTTP helpers ────────────────────────────────────────────────────────────

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


# ── Vector file readers ─────────────────────────────────────────────────────

def bvecs_get_dim_and_total_points(path: str) -> Tuple[int, int]:
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


def read_bvecs_slice_memmap(path: str, start: int, count: int) -> np.ndarray:
    dim, total = bvecs_get_dim_and_total_points(path)
    if start < 0 or count <= 0 or start + count > total:
        raise ValueError(f"Invalid bvecs slice [{start}, {start+count}) total={total}")
    rec = 4 + dim
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    mat = raw.reshape(total, rec)
    vecs_u8 = mat[start : start + count, 4:]
    return np.ascontiguousarray(vecs_u8.astype(np.float32))


def read_fvecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f"Empty fvecs: {path}")
    dim = int(data.view(np.int32)[0])
    if dim <= 0:
        raise RuntimeError(f"Invalid fvecs dim={dim}: {path}")
    width = dim + 1
    if data.size % width != 0:
        raise RuntimeError(f"Invalid fvecs size for dim={dim}: {path}")
    mat = data.reshape(-1, width)
    return np.ascontiguousarray(mat[:, 1:].astype(np.float32))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    total_t0 = time.time()
    parser = argparse.ArgumentParser(description="Client for distributed SPTAG shard build.")
    parser.add_argument("--master_url", required=True, help="Master URL, e.g. http://10.0.0.1:19090")
    parser.add_argument("--base_path", required=True, help="Input .bvecs/.fvecs file path")
    parser.add_argument("--output_dir", required=True, help="Root dir for saved shard indexes (passed to master)")
    parser.add_argument("--job_id", default="job1")
    parser.add_argument("--start_point", type=int, default=0, help="Global start index (for subsets)")
    parser.add_argument("--max_points", type=int, default=None, help="Subset length from start_point")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--algo", default="BKT", choices=["BKT", "KDT", "SPANN"])
    parser.add_argument("--dist", default="L2", choices=["L2", "Cosine"])
    parser.add_argument("--value_type", default="Float", choices=["Float", "UInt8"])
    parser.add_argument("--cef", type=int, default=None, help="Optional SPTAG CEF build parameter.")
    parser.add_argument(
        "--max_check_for_refine_graph",
        type=int,
        default=None,
        help="Optional SPTAG MaxCheckForRefineGraph build parameter.",
    )
    parser.add_argument(
        "--graph_neighborhood_scale",
        type=float,
        default=None,
        help="Optional SPTAG GraphNeighborhoodScale build parameter.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Worker-side SPTAG build threads passed to /init.",
    )
    parser.add_argument(
        "--client_threads",
        type=int,
        default=1,
        help="Client sender threads. Only sequential mode is supported, so this must be 1.",
    )
    parser.add_argument("--with_meta_index", action="store_true")
    parser.add_argument("--request_timeout", type=int, default=60)
    parser.add_argument("--checkpoint_timeout", type=int, default=600)
    parser.add_argument("--finalize_timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--skip_init", action="store_true", help="Skip /init and append to an existing job")
    parser.add_argument(
        "--final_action",
        default="finalize",
        choices=["none", "checkpoint", "finalize"],
        help="Action to take after sending this phase's batches",
    )
    parser.add_argument("--checkpoint_id", default="", help="Checkpoint identifier when --final_action=checkpoint")
    parser.add_argument("--phase_label", default="", help="Optional label for logging this client phase")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if int(args.client_threads) != 1:
        raise ValueError("Only sequential client sending is supported; set --client_threads=1")
    if args.cef is not None and int(args.cef) <= 0:
        raise ValueError("--cef must be > 0")
    if args.max_check_for_refine_graph is not None and int(args.max_check_for_refine_graph) <= 0:
        raise ValueError("--max_check_for_refine_graph must be > 0")
    if args.graph_neighborhood_scale is not None and float(args.graph_neighborhood_scale) <= 0:
        raise ValueError("--graph_neighborhood_scale must be > 0")

    master_url = args.master_url.rstrip("/")
    if not master_url.startswith("http://") and not master_url.startswith("https://"):
        master_url = f"http://{master_url}"

    # ── Read file header ────────────────────────────────────────────────
    base_path = Path(args.base_path)
    ext = base_path.suffix.lower()
    if ext not in (".bvecs", ".fvecs"):
        raise ValueError("base_path must end with .bvecs or .fvecs")

    if ext == ".bvecs":
        dim, total_points = bvecs_get_dim_and_total_points(str(base_path))
        all_fvecs = None
    else:
        all_fvecs = read_fvecs(str(base_path))
        total_points = all_fvecs.shape[0]
        dim = all_fvecs.shape[1]

    # ── Compute subset ──────────────────────────────────────────────────
    subset_start = int(args.start_point)
    if subset_start < 0 or subset_start >= total_points:
        raise ValueError(f"start_point out of range [0, {total_points - 1}]")
    if args.max_points is None:
        subset_n = total_points - subset_start
    else:
        subset_n = min(int(args.max_points), total_points - subset_start)
        if subset_n <= 0:
            raise ValueError("max_points produced empty subset")
    subset_end = subset_start + subset_n

    print(f"Dataset: {base_path} (dim={dim}, total={total_points})")
    print(f"Subset: [{subset_start}, {subset_end}) size={subset_n}")
    print(
        f"Client send mode: sequential (client_threads=1, worker_threads={int(args.threads)}, "
        f"value_type={args.value_type})"
    )
    if args.phase_label:
        print(f"Phase: {args.phase_label}")

    # ── Init ────────────────────────────────────────────────────────────
    client_master_init_rpc_roundtrip_s = 0.0
    client_master_init_comm_overhead_s = 0.0
    client_master_add_rpc_roundtrip_s = 0.0
    client_master_add_comm_overhead_s = 0.0
    client_master_add_backpressure_sleep_s = 0.0
    client_master_checkpoint_rpc_roundtrip_s = 0.0
    client_master_checkpoint_comm_overhead_s = 0.0

    if args.skip_init:
        print(f"Master init skipped for existing job: {args.job_id}")
    else:
        init_payload = {
            "job_id": args.job_id,
            "algo": args.algo,
            "dist": args.dist,
            "value_type": args.value_type,
            "dim": dim,
            "output_dir": args.output_dir,
            "threads": args.threads,
            "with_meta_index": args.with_meta_index,
        }
        if args.cef is not None:
            init_payload["cef"] = int(args.cef)
        if args.max_check_for_refine_graph is not None:
            init_payload["max_check_for_refine_graph"] = int(args.max_check_for_refine_graph)
        if args.graph_neighborhood_scale is not None:
            init_payload["graph_neighborhood_scale"] = float(args.graph_neighborhood_scale)
        init_url = f"{master_url}/init"
        init_t0 = time.time()
        init_resp = _post_with_retry(
            init_url,
            lambda: _http_json_post(init_url, init_payload, args.request_timeout),
            args.retries,
            "init",
        )
        client_master_init_rpc_roundtrip_s = time.time() - init_t0
        if not init_resp.get("ok", False):
            raise RuntimeError(f"Master /init failed: {init_resp}")
        client_master_init_comm_overhead_s = max(
            0.0, client_master_init_rpc_roundtrip_s - float(init_resp.get("request_total_s", 0.0))
        )
        num_shards = init_resp.get("num_shards", "?")
        registered_workers = init_resp.get("registered_workers", num_shards)
        print(f"Master initialized: {num_shards} shard(s) active, {registered_workers} worker(s) registered")

    # ── Stream batches ──────────────────────────────────────────────────
    add_t0 = time.time()
    milestones = [25, 50, 75, 100]
    reached = set()
    batch_count = 0

    for off in range(0, subset_n, args.batch_size):
        n = min(args.batch_size, subset_n - off)
        global_offset = subset_start + off

        if ext == ".bvecs":
            vecs = read_bvecs_slice_memmap(str(base_path), global_offset, n)
        else:
            vecs = np.ascontiguousarray(all_fvecs[global_offset : global_offset + n], dtype=np.float32)

        meta = {
            "job_id": args.job_id,
            "global_offset": global_offset,
            "num": n,
            "dim": dim,
            "with_meta_index": args.with_meta_index,
            "normalized": False,
        }
        meta_raw = json.dumps(meta).encode("utf-8")
        payload = struct.pack("<Q", len(meta_raw)) + meta_raw + vecs.tobytes()

        url = f"{master_url}/add_batch"
        resp = _send_batch_with_backpressure(
            url=url,
            payload=payload,
            request_timeout=args.request_timeout,
            retries=args.retries,
            what=f"add_batch offset={global_offset} n={n}",
        )
        if not resp.get("ok", False):
            raise RuntimeError(f"add_batch failed at offset={global_offset}: {resp}")
        batch_rpc_roundtrip_s = float(resp.get("_rpc_roundtrip_s", 0.0))
        batch_backpressure_sleep_s = float(resp.get("_backpressure_sleep_s", 0.0))
        batch_master_request_total_s = float(resp.get("request_total_s", 0.0))
        client_master_add_rpc_roundtrip_s += batch_rpc_roundtrip_s
        client_master_add_backpressure_sleep_s += batch_backpressure_sleep_s
        client_master_add_comm_overhead_s += max(0.0, batch_rpc_roundtrip_s - batch_master_request_total_s)

        batch_count += 1
        sent = off + n
        pct = int((sent * 100) / subset_n) if subset_n > 0 else 100
        for m in milestones:
            if pct >= m and m not in reached:
                print(f"[progress] {m}% ({sent}/{subset_n})")
                reached.add(m)

        if args.debug:
            print(f"[debug] batch={batch_count - 1} global_offset={global_offset} num={n} sent={sent}/{subset_n}")

    add_elapsed = time.time() - add_t0
    qps = (subset_n / add_elapsed) if add_elapsed > 0 else 0.0
    print(f"[add done] vectors={subset_n} batches={batch_count} send_time={add_elapsed:.3f}s ({qps:.1f} vec/s)")

    # ── Final Action ────────────────────────────────────────────────────
    if args.final_action == "checkpoint":
        checkpoint_url = f"{master_url}/checkpoint"
        checkpoint_payload = {"job_id": args.job_id}
        if args.checkpoint_id:
            checkpoint_payload["checkpoint_id"] = args.checkpoint_id
        checkpoint_t0 = time.time()
        checkpoint_resp = _post_with_retry(
            checkpoint_url,
            lambda: _http_json_post(checkpoint_url, checkpoint_payload, args.checkpoint_timeout),
            args.retries,
            "checkpoint",
        )
        client_master_checkpoint_rpc_roundtrip_s = time.time() - checkpoint_t0
        if not checkpoint_resp.get("ok", False):
            raise RuntimeError(f"Master /checkpoint failed: {checkpoint_resp}")
        client_master_checkpoint_comm_overhead_s = max(
            0.0, client_master_checkpoint_rpc_roundtrip_s - float(checkpoint_resp.get("request_total_s", 0.0))
        )
        print(
            f"[checkpoint] id={checkpoint_resp.get('checkpoint_id')} "
            f"dir={checkpoint_resp.get('checkpoint_dir')} "
            f"centers={checkpoint_resp.get('centers_file')}"
        )
        for shard in checkpoint_resp.get("shards", []):
            print(
                f"[checkpoint] shard={shard.get('shard_id')} dir={shard.get('checkpoint_dir')} "
                f"active={shard.get('active_vectors')} "
                f"worker_checkpoint={shard.get('checkpoint_time_s', 0):.3f}s"
            )
    elif args.final_action == "finalize":
        finalize_url = f"{master_url}/finalize"
        finalize_payload = {"job_id": args.job_id}
        finalize_resp = _post_with_retry(
            finalize_url,
            lambda: _http_json_post(finalize_url, finalize_payload, args.finalize_timeout),
            args.retries,
            "finalize",
        )
        if not finalize_resp.get("ok", False):
            raise RuntimeError(f"Master /finalize failed: {finalize_resp}")

        shards = finalize_resp.get("shards", [])
        for s in shards:
            print(
                f"[finalize] shard={s.get('shard_id')} saved={s.get('save_dir')} "
                f"vectors={s.get('vectors_ingested')} "
                f"worker_add={s.get('add_time_s', 0):.3f}s "
                f"worker_finalize={s.get('finalize_time_s', 0):.3f}s"
            )
        print(f"[finalize] centers={finalize_resp.get('centers_file')}")
    else:
        print("[phase done] final action skipped")

    total_elapsed = time.time() - total_t0
    client_master_total_comm_overhead_s = (
        client_master_init_comm_overhead_s
        + client_master_add_comm_overhead_s
        + client_master_checkpoint_comm_overhead_s
    )
    print("Distributed build phase completed successfully.")
    print(f"[timing] total={total_elapsed:.3f}s add_phase={add_elapsed:.3f}s")
    print(
        f"[comm] client_master_init_rpc={client_master_init_rpc_roundtrip_s:.3f}s "
        f"client_master_init_comm={client_master_init_comm_overhead_s:.3f}s"
    )
    print(
        f"[comm] client_master_add_rpc={client_master_add_rpc_roundtrip_s:.3f}s "
        f"client_master_add_comm={client_master_add_comm_overhead_s:.3f}s "
        f"client_master_add_backpressure_sleep={client_master_add_backpressure_sleep_s:.3f}s"
    )
    if args.final_action == "checkpoint":
        print(
            f"[comm] client_master_checkpoint_rpc={client_master_checkpoint_rpc_roundtrip_s:.3f}s "
            f"client_master_checkpoint_comm={client_master_checkpoint_comm_overhead_s:.3f}s"
        )
    print(f"[comm] client_master_total_comm={client_master_total_comm_overhead_s:.3f}s")


if __name__ == "__main__":
    main()
