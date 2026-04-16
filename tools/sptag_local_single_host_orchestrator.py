#!/usr/bin/env python3
"""Local single-host runner for SPTAG experiments on Ubuntu.

This script is designed to live inside ``SPTAG-modified`` and run directly on
the Ubuntu machine. It avoids the SSH control plane entirely and launches all
services with local tmux sessions.

Default target:
- 10 workers
- 1,000,000 total vectors
- 100,000 target vectors per worker on average
- one host
- no rebalancing

The build flow is:
1. start workers and master locally
2. run one client phase that checkpoints at 1M
3. optionally run a local search sweep against that checkpoint

Use ``run-all`` for the common path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _repo_root(args: argparse.Namespace) -> Path:
    return _expand(args.repo)


def _release_dir(args: argparse.Namespace) -> Path:
    if args.release_dir:
        return _expand(args.release_dir)
    return _repo_root(args) / "Release"


def _log_dir(args: argparse.Namespace) -> Path:
    return _expand(args.log_dir)


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return _expand(args.output_dir)
    return _repo_root(args) / "index_build_output_sift1b_local"


def _artifacts_root(args: argparse.Namespace, run_id: str) -> Path:
    return _log_dir(args) / "local_runs" / run_id


def _env_exports(args: argparse.Namespace) -> str:
    release_dir = _release_dir(args)
    python_path = f"{release_dir}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    ld_library_path = f"{release_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return (
        f"export PYTHONPATH={shlex.quote(python_path)}; "
        f"export LD_LIBRARY_PATH={shlex.quote(ld_library_path)}; "
    )


def _run_shell(
    command: str,
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
        check=check,
    )


def _ensure_tmux() -> None:
    try:
        _run_shell("command -v tmux", check=True)
    except subprocess.CalledProcessError as ex:
        raise RuntimeError("tmux is required but was not found in PATH") from ex


def _tmux_has_session(session_name: str) -> bool:
    result = _run_shell(
        f"tmux has-session -t {shlex.quote(session_name)}",
        check=False,
    )
    return result.returncode == 0


def _tmux_launch(session_name: str, command: str) -> None:
    wrapped = shlex.quote(f"bash -lc {shlex.quote(command)}")
    if _tmux_has_session(session_name):
        tmux_cmd = f"tmux new-window -t {shlex.quote(session_name)} -d {wrapped}"
    else:
        tmux_cmd = f"tmux new-session -d -s {shlex.quote(session_name)} {wrapped}"
    _run_shell(tmux_cmd, check=True)


def _tmux_kill_prefix(prefix: str) -> None:
    result = _run_shell("tmux list-sessions -F '#S' 2>/dev/null || true", check=False)
    sessions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for session in sessions:
        if session == prefix or session.startswith(prefix + "-"):
            _run_shell(f"tmux kill-session -t {shlex.quote(session)}", check=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wall_time_snapshot() -> Dict[str, Any]:
    now = time.time()
    return {
        "epoch_s": round(now, 6),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    }


def _search_checkpoint_timing_payload(
    *,
    checkpoint_id: int,
    checkpoint_label: str,
    checkpoint_dir: Path,
    active_worker_ids: Sequence[int],
    parameter_set_count: int,
    search_repetitions: int,
    recorded_run_log_count: int,
    start_snapshot: Dict[str, Any],
    end_snapshot: Dict[str, Any],
    status: str,
    completed_parameter_set_count: int | None = None,
    failed_parameter_sets: Sequence[str] | None = None,
    failed_parameter_set: str | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "checkpoint_id": int(checkpoint_id),
        "checkpoint_label": str(checkpoint_label),
        "checkpoint_dir": str(checkpoint_dir),
        "active_worker_ids": [int(value) for value in active_worker_ids],
        "parameter_set_count": int(parameter_set_count),
        "service_group_count": 1,
        "search_repetitions": int(search_repetitions),
        "expected_run_count": int(parameter_set_count) * int(search_repetitions),
        "recorded_run_log_count": int(recorded_run_log_count),
        "status": str(status),
        "started_at_epoch_s": start_snapshot["epoch_s"],
        "started_at_utc": start_snapshot["utc"],
        "finished_at_epoch_s": end_snapshot["epoch_s"],
        "finished_at_utc": end_snapshot["utc"],
        "duration_seconds": round(
            max(0.0, float(end_snapshot["epoch_s"]) - float(start_snapshot["epoch_s"])),
            6,
        ),
    }
    if completed_parameter_set_count is not None:
        payload["completed_parameter_set_count"] = int(completed_parameter_set_count)
        payload["failed_parameter_set_count"] = max(
            0, int(parameter_set_count) - int(completed_parameter_set_count)
        )
    if failed_parameter_sets:
        payload["failed_parameter_sets"] = [str(value) for value in failed_parameter_sets]
    if failed_parameter_set:
        payload["failed_parameter_set"] = str(failed_parameter_set)
    if error:
        payload["error"] = str(error)
    return payload


def _run_logged(command: str, log_path: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(f"$ {command}\n\n")
        handle.flush()
        result = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {command}\nsee {log_path}")
    return result


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for {host}:{port}")


def _http_get_json(url: str, timeout_s: int) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=raw, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Content-Length", str(len(raw)))
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _phase_label(total_points: int) -> str:
    if total_points % 1_000_000_000 == 0:
        return f"{total_points // 1_000_000_000}B"
    if total_points % 1_000_000 == 0:
        return f"{total_points // 1_000_000}M"
    if total_points % 1_000 == 0:
        return f"{total_points // 1_000}K"
    return str(total_points)


def _int_csv_sequence(raw_value: str) -> List[int]:
    values: List[int] = []
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("phase/checkpoint values must be positive")
        values.append(value)
    return values


def _phase_plan(args: argparse.Namespace) -> tuple[List[int], List[int]]:
    total_points = int(args.max_points)
    if total_points <= 0:
        raise ValueError("--max-points must be positive")

    build_phases = sorted(set(_int_csv_sequence(args.build_phases)))
    search_checkpoints = sorted(set(_int_csv_sequence(args.search_checkpoints)))

    if not build_phases:
        build_phases = list(search_checkpoints)
    if not build_phases:
        build_phases = [total_points]
    if build_phases[-1] != total_points:
        build_phases.append(total_points)
    build_phases = sorted(set(build_phases))

    if any(value > total_points for value in build_phases):
        raise ValueError("build phases cannot exceed --max-points")
    if any(value not in build_phases for value in search_checkpoints):
        raise ValueError("every search checkpoint must also appear in build phases")

    return build_phases, search_checkpoints


def _workers(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [
        {
            "worker_id": worker_id,
            "host": "127.0.0.1",
            "port": int(args.worker_base_port) + worker_id - 1,
        }
        for worker_id in range(1, int(args.num_workers) + 1)
    ]


def _workers_file_path(args: argparse.Namespace) -> Path:
    return _repo_root(args) / "ClientServerImplementation" / "workers.json"


def _checkpoint_gt_filename(checkpoint_id: int) -> str:
    checkpoint_id = int(checkpoint_id)
    if checkpoint_id % 1_000_000 == 0:
        return f"idx_{checkpoint_id // 1_000_000}M.ivecs"
    if checkpoint_id % 1_000 == 0:
        return f"idx_{checkpoint_id // 1_000}K.ivecs"
    return f"idx_{checkpoint_id}.ivecs"


def _default_base_path(args: argparse.Namespace) -> Path:
    if args.base_path:
        return _expand(args.base_path)
    return _expand(args.data_root) / "bigann_base.bvecs"


def _default_query_path(args: argparse.Namespace) -> Path:
    if args.query_path:
        return _expand(args.query_path)
    return _expand(args.data_root) / "bigann_query.bvecs"


def _gt_path_mapping(raw_value: str) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for token in str(raw_value or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "invalid --search-gt-path-by-checkpoint entry. Expected checkpoint=path pairs."
            )
        key, value = token.split("=", 1)
        checkpoint = int(key.strip())
        if checkpoint <= 0:
            raise ValueError("checkpoint ids in --search-gt-path-by-checkpoint must be positive")
        if not value.strip():
            raise ValueError("checkpoint paths in --search-gt-path-by-checkpoint cannot be empty")
        mapping[checkpoint] = _expand(value.strip())
    return mapping


def _infer_gt_path_from_anchor(anchor: Path, checkpoint_id: int) -> Path:
    if anchor.suffix.lower() == ".ivecs":
        return anchor.with_name(_checkpoint_gt_filename(checkpoint_id))
    return anchor / "gnd" / _checkpoint_gt_filename(checkpoint_id)


def _gt_path_for_checkpoint(args: argparse.Namespace, checkpoint_id: int) -> Path:
    explicit_mapping = _gt_path_mapping(args.search_gt_path_by_checkpoint)
    if checkpoint_id in explicit_mapping:
        return explicit_mapping[checkpoint_id]

    search_checkpoints = _phase_plan(args)[1]
    if args.search_gt_path:
        explicit_path = _expand(args.search_gt_path)
        if len(search_checkpoints) <= 1:
            return explicit_path
        return _infer_gt_path_from_anchor(explicit_path, checkpoint_id)

    return _expand(args.data_root) / "gnd" / _checkpoint_gt_filename(checkpoint_id)


def _write_workers_file(args: argparse.Namespace) -> Path:
    path = _workers_file_path(args)
    _write_json(path, _workers(args))
    return path


def _build_worker_command(args: argparse.Namespace, worker: Dict[str, Any], log_path: Path) -> str:
    repo = _repo_root(args)
    exports = _env_exports(args)
    return (
        f"cd {shlex.quote(str(repo))} && "
        f"{exports}"
        f"{shlex.quote(args.python)} -u ClientServerImplementation/distributed_index_worker.py "
        f"--host 0.0.0.0 "
        f"--port {int(worker['port'])} "
        f"--queue_max_batches {int(args.queue_max_batches)} "
        f"--memory_log_interval_vectors {int(args.memory_log_interval_vectors)} "
        f"--no_exit_after_finalize "
        f"> {shlex.quote(str(log_path))} 2>&1"
    )


def _build_master_command(args: argparse.Namespace, log_path: Path) -> str:
    repo = _repo_root(args)
    workers_file = _workers_file_path(args)
    exports = _env_exports(args)
    command = (
        f"cd {shlex.quote(str(repo))} && "
        f"{exports}"
        f"{shlex.quote(args.python)} -u ClientServerImplementation/distributed_index_master.py "
        f"--host 0.0.0.0 "
        f"--port {int(args.master_port)} "
        f"--workers_file {shlex.quote(str(workers_file))} "
        f"--request_timeout {int(args.request_timeout)} "
        f"--checkpoint_timeout {int(args.checkpoint_timeout)} "
        f"--finalize_timeout {int(args.finalize_timeout)} "
        f"--rebalance_timeout {int(args.rebalance_timeout)} "
        f"--bootstrap_sample_size {int(args.bootstrap_sample_size)} "
        f"--reservoir_size {int(args.reservoir_size)} "
        f"--retry_after_ms {int(args.retry_after_ms)} "
        f"--retries {int(args.retries)} "
        f"--no_exit_after_finalize "
    )
    if args.disable_rebalance:
        command += "--disable_rebalance "
    command += f"> {shlex.quote(str(log_path))} 2>&1"
    return command


def _build_client_command(
    args: argparse.Namespace,
    *,
    start_point: int,
    max_points: int,
    checkpoint_id: int,
    final_action: str,
    skip_init: bool,
) -> str:
    repo = _repo_root(args)
    output_dir = _output_dir(args)
    exports = _env_exports(args)
    command = [
        f"cd {shlex.quote(str(repo))} &&",
        exports,
        "/usr/bin/time -p",
        shlex.quote(args.python),
        "-u",
        "ClientServerImplementation/client.py",
        "--master_url",
        shlex.quote(f"127.0.0.1:{int(args.master_port)}"),
        "--base_path",
        shlex.quote(str(_default_base_path(args))),
        "--output_dir",
        shlex.quote(str(output_dir)),
        "--job_id",
        shlex.quote(args.job_id),
        "--start_point",
        str(start_point),
        "--max_points",
        str(max_points),
        "--batch_size",
        str(int(args.batch_size)),
        "--algo",
        shlex.quote(args.algo),
        "--dist",
        shlex.quote(args.dist),
        "--value_type",
        shlex.quote(args.value_type),
        "--client_threads",
        str(int(args.client_threads)),
        "--threads",
        str(int(args.threads)),
        "--request_timeout",
        str(int(args.request_timeout)),
        "--checkpoint_timeout",
        str(int(args.checkpoint_timeout)),
        "--finalize_timeout",
        str(int(args.finalize_timeout)),
        "--retries",
        str(int(args.retries)),
        "--final_action",
        shlex.quote(final_action),
        "--checkpoint_id",
        shlex.quote(str(checkpoint_id)),
        "--phase_label",
        shlex.quote(_phase_label(checkpoint_id)),
    ]
    if args.cef is not None:
        command.extend(["--cef", str(int(args.cef))])
    if args.tpt_number is not None:
        command.extend(["--tpt_number", str(int(args.tpt_number))])
    if args.tpt_leaf_size is not None:
        command.extend(["--tpt_leaf_size", str(int(args.tpt_leaf_size))])
    if args.max_check_for_refine_graph is not None:
        command.extend(["--max_check_for_refine_graph", str(int(args.max_check_for_refine_graph))])
    if args.graph_neighborhood_scale is not None:
        command.extend(["--graph_neighborhood_scale", str(float(args.graph_neighborhood_scale))])
    if skip_init:
        command.append("--skip_init")
    return " ".join(command)


def _master_status(args: argparse.Namespace) -> Dict[str, Any]:
    payload = _http_get_json(
        f"http://127.0.0.1:{int(args.master_port)}/status?job_id={urllib.parse.quote(args.job_id, safe='')}",
        timeout_s=int(args.status_timeout),
    )
    if not payload.get("ok", False):
        raise RuntimeError(f"master status failed: {payload}")
    return payload


def _worker_status(args: argparse.Namespace, worker: Dict[str, Any]) -> Dict[str, Any]:
    payload = _http_get_json(
        (
            f"http://127.0.0.1:{int(worker['port'])}/status"
            f"?job_id={urllib.parse.quote(args.job_id, safe='')}&shard_id={int(worker['worker_id'])}"
        ),
        timeout_s=int(args.status_timeout),
    )
    if not payload.get("ok", False):
        raise RuntimeError(f"worker {worker['worker_id']} status failed: {payload}")
    return payload


def _print_status_summary(status: Dict[str, Any]) -> None:
    print("job_id:", status.get("job_id"))
    print("routing_mode:", status.get("routing_mode"))
    print("total_vectors_ingested:", status.get("total_vectors_ingested"))
    print("build_elapsed_s:", status.get("build_elapsed_s"))
    print("wall_elapsed_s:", status.get("wall_elapsed_s"))
    print("build_timing_paused:", status.get("build_timing_paused"))
    print("active_workers:", status.get("active_workers"))
    print("worker_counts:", status.get("worker_counts"))
    print("latest_checkpoint_dir:", status.get("latest_checkpoint_dir"))
    print("latest_checkpoint_centers_file:", status.get("latest_checkpoint_centers_file"))


def _phase_artifacts_dir(args: argparse.Namespace, run_id: str, phase_total: int) -> Path:
    return _artifacts_root(args, run_id) / f"build_{_phase_label(phase_total)}"


def _build_timing_action(args: argparse.Namespace, action: str, log_path: Path) -> Dict[str, Any]:
    payload = _http_post_json(
        f"http://127.0.0.1:{int(args.master_port)}/build_timing",
        {"job_id": args.job_id, "action": action},
        timeout_s=int(args.status_timeout),
    )
    _write_json(log_path, payload)
    if not payload.get("ok", False):
        raise RuntimeError(f"build_timing {action} failed: {payload}")
    return payload


def _start_build_services(args: argparse.Namespace) -> None:
    _ensure_tmux()
    _log_dir(args).mkdir(parents=True, exist_ok=True)
    _write_workers_file(args)
    session_name = args.session_name
    _tmux_kill_prefix(session_name)

    for worker in _workers(args):
        worker_log = _log_dir(args) / f"worker_{int(worker['worker_id']):02d}.log"
        _tmux_launch(session_name, _build_worker_command(args, worker, worker_log))
        time.sleep(float(args.between_workers_sec))

    if float(args.after_workers_sec) > 0:
        time.sleep(float(args.after_workers_sec))

    master_log = _log_dir(args) / "master.log"
    _tmux_launch(session_name, _build_master_command(args, master_log))
    _wait_for_port("127.0.0.1", int(args.master_port), timeout_s=float(args.master_start_timeout_sec))

    for worker in _workers(args):
        _wait_for_port("127.0.0.1", int(worker["port"]), timeout_s=float(args.worker_start_timeout_sec))

    if float(args.after_master_sec) > 0:
        time.sleep(float(args.after_master_sec))


def _stop_build_services(args: argparse.Namespace) -> None:
    _tmux_kill_prefix(args.session_name)


def _checkpoint_dir(args: argparse.Namespace, checkpoint_id: int) -> Path:
    return _output_dir(args) / "checkpoints" / str(checkpoint_id)


def _centers_path(args: argparse.Namespace, checkpoint_id: int) -> Path:
    return _checkpoint_dir(args, checkpoint_id) / "centers"


def _build_phase(
    args: argparse.Namespace,
    *,
    start_point: int,
    phase_total: int,
    run_id: str,
) -> Dict[str, Any]:
    artifacts = _phase_artifacts_dir(args, run_id, phase_total)
    artifacts.mkdir(parents=True, exist_ok=True)
    command = _build_client_command(
        args,
        start_point=start_point,
        max_points=int(phase_total - start_point),
        checkpoint_id=phase_total,
        final_action="checkpoint",
        skip_init=(start_point > 0),
    )
    build_log = artifacts / "client_phase.log"
    _run_logged(command, build_log, timeout=int(args.client_phase_timeout))

    status = _master_status(args)
    _write_json(artifacts / "master_status.json", status)

    worker_payload: Dict[str, Any] = {}
    for worker in _workers(args):
        try:
            payload = _worker_status(args, worker)
        except Exception as ex:
            payload = {"ok": False, "error": str(ex)}
        worker_payload[str(worker["worker_id"])] = payload
    _write_json(artifacts / "worker_statuses.json", worker_payload)
    return status


def _finalize(args: argparse.Namespace, run_id: str) -> Dict[str, Any]:
    artifacts = _artifacts_root(args, run_id)
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = _http_post_json(
        f"http://127.0.0.1:{int(args.master_port)}/finalize",
        {"job_id": args.job_id},
        timeout_s=int(args.finalize_timeout),
    )
    _write_json(artifacts / "finalize_response.json", payload)
    if not payload.get("ok", False):
        raise RuntimeError(f"finalize failed: {payload}")
    return payload


def _parse_positive_int_csv(raw_value: str) -> List[int]:
    values: List[int] = []
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("values must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one value is required")
    return values


def _expand_agg_topk_values(raw_value: str, active_worker_count: int) -> List[int]:
    token = str(raw_value).strip().lower()
    if token in {"all", "auto"}:
        return list(range(1, active_worker_count + 1))
    values = _parse_positive_int_csv(raw_value)
    if any(value > active_worker_count for value in values):
        raise ValueError(
            f"aggregator top_k cannot exceed active worker count ({active_worker_count})"
        )
    return values


def _search_session_name(args: argparse.Namespace, checkpoint_id: int) -> str:
    return f"{args.session_name}-search-{checkpoint_id}"


def _write_search_worker_ini(
    args: argparse.Namespace,
    *,
    worker_id: int,
    port: int,
    checkpoint_id: int,
) -> Path:
    ini_path = _release_dir(args) / f"generated_worker_{worker_id}_{checkpoint_id}.ini"
    index_folder = _checkpoint_dir(args, checkpoint_id) / f"index_shard{worker_id}"
    content = "\n".join(
        [
            "[Service]",
            "ListenAddr=0.0.0.0",
            f"ListenPort={port}",
            f"ThreadNumber={int(args.search_worker_threads)}",
            f"SocketThreadNumber={int(args.search_worker_socket_threads)}",
            "",
            "[QueryConfig]",
            f"DefaultMaxResultNumber={int(args.search_k)}",
            'DefaultSeparator=|',
            "",
            "[Index]",
            f"List={args.algo}",
            "",
            f"[Index_{args.algo}]",
            f"IndexFolder={index_folder}",
            "",
        ]
    )
    _write_text(ini_path, content)
    return ini_path


def _write_search_aggregator_ini(
    args: argparse.Namespace,
    *,
    checkpoint_id: int,
    active_workers: Sequence[Dict[str, Any]],
) -> Path:
    ini_path = _release_dir(args) / "Aggregator.ini"
    centers_path = _centers_path(args, checkpoint_id)
    lines = [
        "[Service]",
        "ListenAddr=0.0.0.0",
        f"ListenPort={int(args.search_master_port)}",
        f"SearchTimeout={int(args.search_timeout_ms)}",
        f"ThreadNumber={int(args.search_aggregator_threads)}",
        f"SocketThreadNumber={int(args.search_aggregator_socket_threads)}",
        f"TopK={len(active_workers)}",
        f"Centers={centers_path}",
        f"ValueType={args.search_value_type}",
        f"DistCalcMethod={args.dist}",
        "",
        "[Servers]",
        f"Number={len(active_workers)}",
        "",
    ]
    for index, worker in enumerate(active_workers):
        lines.extend(
            [
                f"[Server_{index}]",
                "Address=127.0.0.1",
                f"Port={int(args.search_worker_base_port) + int(worker['worker_id']) - 1}",
                "",
            ]
        )
    _write_text(ini_path, "\n".join(lines))
    return ini_path


def _build_search_worker_command(args: argparse.Namespace, ini_path: Path, log_path: Path) -> str:
    release_dir = _release_dir(args)
    exports = _env_exports(args)
    return (
        f"cd {shlex.quote(str(release_dir))} && "
        f"{exports}"
        f"./server -m socket -c {shlex.quote(str(ini_path))} "
        f"> {shlex.quote(str(log_path))} 2>&1"
    )


def _build_search_aggregator_command(args: argparse.Namespace, log_path: Path) -> str:
    release_dir = _release_dir(args)
    exports = _env_exports(args)
    return (
        f"cd {shlex.quote(str(release_dir))} && "
        f"{exports}"
        f"./aggregator > {shlex.quote(str(log_path))} 2>&1"
    )


def _build_search_client_command(
    args: argparse.Namespace,
    *,
    gt_path: Path,
    aggregator_top_k: int,
    max_check: int,
) -> str:
    release_dir = _release_dir(args)
    exports = _env_exports(args)
    return (
        f"cd {shlex.quote(str(release_dir))} && "
        f"{exports}"
        f"/usr/bin/time -p ./sift_eval_client "
        f"--query_path {shlex.quote(str(_default_query_path(args)))} "
        f"--gt_path {shlex.quote(str(gt_path))} "
        f"--value_type {shlex.quote(args.search_value_type)} "
        f"--K {int(args.search_k)} "
        f"--max_queries {int(args.search_max_queries)} "
        f"--num_threads {int(args.search_num_threads)} "
        f"--host 127.0.0.1 "
        f"--port {int(args.search_master_port)} "
        f"--aggregator_top_k {int(aggregator_top_k)} "
        f"--max_check {int(max_check)}"
    )


def _start_search_services(
    args: argparse.Namespace,
    *,
    checkpoint_id: int,
    active_workers: Sequence[Dict[str, Any]],
) -> None:
    session_name = _search_session_name(args, checkpoint_id)
    _tmux_kill_prefix(session_name)

    for worker in active_workers:
        worker_id = int(worker["worker_id"])
        port = int(args.search_worker_base_port) + worker_id - 1
        ini_path = _write_search_worker_ini(args, worker_id=worker_id, port=port, checkpoint_id=checkpoint_id)
        log_path = _log_dir(args) / f"search_worker_{worker_id}_{checkpoint_id}.log"
        _tmux_launch(session_name, _build_search_worker_command(args, ini_path, log_path))

    for worker in active_workers:
        port = int(args.search_worker_base_port) + int(worker["worker_id"]) - 1
        _wait_for_port("127.0.0.1", port, timeout_s=float(args.search_start_timeout_sec))

    if float(args.search_after_workers_sec) > 0:
        time.sleep(float(args.search_after_workers_sec))

    _write_search_aggregator_ini(args, checkpoint_id=checkpoint_id, active_workers=active_workers)
    aggregator_log = _log_dir(args) / f"aggregator_{checkpoint_id}.log"
    _tmux_launch(session_name, _build_search_aggregator_command(args, aggregator_log))

    _wait_for_port("127.0.0.1", int(args.search_master_port), timeout_s=float(args.search_start_timeout_sec))

    if float(args.search_after_master_sec) > 0:
        time.sleep(float(args.search_after_master_sec))


def _stop_search_services(args: argparse.Namespace, checkpoint_id: int) -> None:
    _tmux_kill_prefix(_search_session_name(args, checkpoint_id))


def _stop_all_search_services(args: argparse.Namespace) -> None:
    _tmux_kill_prefix(f"{args.session_name}-search")


def _run_search_sweep(args: argparse.Namespace, *, checkpoint_id: int, active_worker_ids: Sequence[int], run_id: str) -> None:
    gt_path = _gt_path_for_checkpoint(args, checkpoint_id)
    if not gt_path.is_file():
        raise FileNotFoundError(f"ground truth file not found for checkpoint {checkpoint_id}: {gt_path}")

    active_workers = [worker for worker in _workers(args) if int(worker["worker_id"]) in {int(value) for value in active_worker_ids}]
    if not active_workers:
        raise RuntimeError("no active workers available for search")

    agg_topk_values = _expand_agg_topk_values(args.search_agg_topk_values, len(active_workers))
    max_check_values = _parse_positive_int_csv(args.search_max_check_values)
    search_root = _artifacts_root(args, run_id) / f"search_{_phase_label(checkpoint_id)}"
    search_root.mkdir(parents=True, exist_ok=True)

    parameter_sets: List[Dict[str, Any]] = []
    for index, (agg_topk, max_check) in enumerate(itertools.product(agg_topk_values, max_check_values), start=1):
        parameter_sets.append(
            {
                "run_token": f"set{index:02d}",
                "folder_name": f"set_{index:02d}__agg_topk_{agg_topk}__maxcheck_{max_check}",
                "values": {
                    "aggregator.top_k": agg_topk,
                    "index.max_check": max_check,
                },
            }
        )

    _write_json(
        search_root / "sweep_manifest.json",
        {
            "checkpoint_id": checkpoint_id,
            "checkpoint_dir": str(_checkpoint_dir(args, checkpoint_id)),
            "centers_path": str(_centers_path(args, checkpoint_id)),
            "gt_path": str(gt_path),
            "active_worker_ids": [int(worker["worker_id"]) for worker in active_workers],
            "parameter_sets": parameter_sets,
        },
    )

    search_timing_start = _wall_time_snapshot()
    search_status = "success"
    search_error = None
    failed_parameter_set = None
    failed_parameter_sets: List[str] = []
    completed_parameter_set_count = 0
    try:
        _start_search_services(args, checkpoint_id=checkpoint_id, active_workers=active_workers)
        try:
            for parameter_set in parameter_sets:
                failed_parameter_set = parameter_set["run_token"]
                values = parameter_set["values"]
                run_dir = search_root / parameter_set["folder_name"]
                run_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    run_dir / "parameter_set.json",
                    {
                        **parameter_set,
                        "checkpoint_id": checkpoint_id,
                        "gt_path": str(gt_path),
                        "active_worker_ids": [int(worker["worker_id"]) for worker in active_workers],
                    },
                )

                parameter_status = "success"
                parameter_error = None
                completed_repetitions = 0
                for repetition in range(1, int(args.search_repetitions) + 1):
                    log_path = run_dir / f"run_{repetition:02d}.log"
                    command = _build_search_client_command(
                        args,
                        gt_path=gt_path,
                        aggregator_top_k=int(values["aggregator.top_k"]),
                        max_check=int(values["index.max_check"]),
                    )
                    try:
                        _run_logged(command, log_path, timeout=int(args.search_run_timeout))
                        completed_repetitions = repetition
                    except Exception as ex:
                        parameter_status = "failed"
                        parameter_error = str(ex)
                        if parameter_set["run_token"] not in failed_parameter_sets:
                            failed_parameter_sets.append(parameter_set["run_token"])
                        search_status = "completed_with_failures"
                        break
                _write_json(
                    run_dir / "parameter_set_status.json",
                    {
                        "checkpoint_id": checkpoint_id,
                        "run_token": parameter_set["run_token"],
                        "values": values,
                        "status": parameter_status,
                        "requested_repetitions": int(args.search_repetitions),
                        "completed_repetitions": int(completed_repetitions),
                        "error": parameter_error,
                    },
                )
                if parameter_status == "failed":
                    failed_parameter_set = None
                    continue
                completed_parameter_set_count += 1
                failed_parameter_set = None
        finally:
            _stop_search_services(args, checkpoint_id)
    except Exception as ex:
        search_status = "failed"
        search_error = str(ex)
        raise
    finally:
        search_timing_end = _wall_time_snapshot()
        _write_json(
            search_root / "checkpoint_timing.json",
            _search_checkpoint_timing_payload(
                checkpoint_id=checkpoint_id,
                checkpoint_label=_phase_label(checkpoint_id),
                checkpoint_dir=_checkpoint_dir(args, checkpoint_id),
                active_worker_ids=active_worker_ids,
                parameter_set_count=len(parameter_sets),
                search_repetitions=int(args.search_repetitions),
                recorded_run_log_count=len(list(search_root.rglob("run_*.log"))),
                start_snapshot=search_timing_start,
                end_snapshot=search_timing_end,
                status=search_status,
                completed_parameter_set_count=completed_parameter_set_count,
                failed_parameter_sets=failed_parameter_sets,
                failed_parameter_set=failed_parameter_set,
                error=search_error,
            ),
        )


def _run_all(args: argparse.Namespace) -> None:
    run_id = _timestamp()
    _start_build_services(args)
    status = _run_build_sequence(args, run_id=run_id, include_search=bool(args.run_search))
    if status is not None:
        _print_status_summary(status)

    if args.finalize_after_run:
        payload = _finalize(args, run_id)
        print(json.dumps(payload, indent=2, sort_keys=True))

    if args.stop_services_after_run:
        _stop_all_search_services(args)
        _stop_build_services(args)

    print("artifacts:", _artifacts_root(args, run_id))


def _status_command(args: argparse.Namespace) -> None:
    status = _master_status(args)
    _print_status_summary(status)


def _logs_command(args: argparse.Namespace) -> None:
    path = _log_dir(args) / args.log_file
    if not path.is_file():
        raise FileNotFoundError(f"log file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in lines[-int(args.lines):]:
        sys.stdout.write(line)


def _run_build_sequence(
    args: argparse.Namespace,
    *,
    run_id: str,
    include_search: bool,
) -> Dict[str, Any] | None:
    build_phases, search_checkpoints = _phase_plan(args)
    checkpoint_set = set(search_checkpoints)
    previous_total = int(getattr(args, "resume_from_point", 0) or 0)
    if previous_total < 0:
        raise ValueError("resume_from_point must be >= 0")
    last_status: Dict[str, Any] | None = None

    for phase_total in build_phases:
        if phase_total <= previous_total:
            continue
        status = _build_phase(
            args,
            start_point=previous_total,
            phase_total=phase_total,
            run_id=run_id,
        )
        last_status = status

        if include_search and phase_total in checkpoint_set:
            phase_dir = _phase_artifacts_dir(args, run_id, phase_total)
            pause_resp = _build_timing_action(
                args,
                "pause",
                phase_dir / "build_timing_pause_before_search.json",
            )
            paused = bool(pause_resp.get("build_timing_paused", False))
            try:
                active_worker_ids = [int(value) for value in status.get("active_workers", [])]
                _run_search_sweep(
                    args,
                    checkpoint_id=phase_total,
                    active_worker_ids=active_worker_ids,
                    run_id=run_id,
                )
            finally:
                if paused:
                    _build_timing_action(
                        args,
                        "resume",
                        phase_dir / "build_timing_resume_after_search.json",
                    )

        previous_total = phase_total

    return last_status


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", default="/home/akunte2/work/horizann/SPTAG-modified")
    parser.add_argument("--release-dir", default="")
    parser.add_argument("--data-root", default="/srv/local/data/anns_data/sift1b")
    parser.add_argument(
        "--base-path",
        default="",
        help="Optional explicit dataset base-vector path. Defaults to <data-root>/bigann_base.bvecs.",
    )
    parser.add_argument(
        "--query-path",
        default="",
        help="Optional explicit query-vector path. Defaults to <data-root>/bigann_query.bvecs.",
    )
    parser.add_argument("--python", default="python3")
    parser.add_argument("--session-name", default="sptag-local-single-host")
    parser.add_argument("--log-dir", default="~/logs/sptag-local-single-host")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--job-id", default="sptag_sift1b_local_single_host")
    parser.add_argument(
        "--resume-from-point",
        type=int,
        default=0,
        help="Resume build sequencing from an already-ingested global point count. Phases at or below this value are skipped.",
    )
    parser.add_argument("--master-port", type=int, default=18079)
    parser.add_argument("--worker-base-port", type=int, default=18080)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument(
        "--build-phases",
        default="",
        help="Comma-separated cumulative phase endpoints. --max-points is always included.",
    )
    parser.add_argument(
        "--search-checkpoints",
        default="",
        help="Comma-separated cumulative checkpoints at which to run search.",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--client-threads", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--algo", default="BKT")
    parser.add_argument("--dist", default="L2")
    parser.add_argument("--value-type", default="UInt8")
    parser.add_argument("--cef", type=int, default=150)
    parser.add_argument("--tpt-number", type=int, default=16)
    parser.add_argument("--tpt-leaf-size", type=int, default=4000)
    parser.add_argument("--max-check-for-refine-graph", type=int, default=4096)
    parser.add_argument("--graph-neighborhood-scale", type=float, default=1.0)
    parser.add_argument("--queue-max-batches", type=int, default=1)
    parser.add_argument("--memory-log-interval-vectors", type=int, default=100000)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--checkpoint-timeout", type=int, default=1800)
    parser.add_argument("--finalize-timeout", type=int, default=1800)
    parser.add_argument("--rebalance-timeout", type=int, default=1800)
    parser.add_argument("--status-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-after-ms", type=int, default=1000)
    parser.add_argument("--bootstrap-sample-size", type=int, default=5000)
    parser.add_argument("--reservoir-size", type=int, default=50000)
    parser.add_argument(
        "--enable-rebalance",
        dest="disable_rebalance",
        action="store_false",
        help="Enable rebalance in the local master. Disabled by default.",
    )
    parser.set_defaults(disable_rebalance=True)
    parser.add_argument("--between-workers-sec", type=float, default=0.5)
    parser.add_argument("--after-workers-sec", type=float, default=5.0)
    parser.add_argument("--after-master-sec", type=float, default=3.0)
    parser.add_argument("--worker-start-timeout-sec", type=float, default=30.0)
    parser.add_argument("--master-start-timeout-sec", type=float, default=30.0)
    parser.add_argument("--client-phase-timeout", type=int, default=86400)
    parser.add_argument("--finalize-after-run", action="store_true")
    parser.add_argument(
        "--stop-services-after-run",
        action="store_true",
        help="Stop build/search tmux sessions after a successful run-all or run-build completes.",
    )
    parser.add_argument("--run-search", action="store_true")
    parser.add_argument("--search-master-port", type=int, default=29200)
    parser.add_argument("--search-worker-base-port", type=int, default=29100)
    parser.add_argument(
        "--search-gt-path",
        default="",
        help="Optional default ground-truth path. For multi-checkpoint runs, sibling idx_<checkpoint>.ivecs files are inferred from this anchor.",
    )
    parser.add_argument(
        "--search-gt-path-by-checkpoint",
        default="",
        help="Optional comma-separated checkpoint=path overrides, for example 1000000=/path/idx_1M.ivecs,10000000=/path/idx_10M.ivecs",
    )
    parser.add_argument("--search-value-type", default="UInt8")
    parser.add_argument("--search-k", type=int, default=10)
    parser.add_argument("--search-max-queries", type=int, default=1000)
    parser.add_argument("--search-num-threads", type=int, default=16)
    parser.add_argument("--search-worker-threads", type=int, default=2)
    parser.add_argument("--search-worker-socket-threads", type=int, default=2)
    parser.add_argument("--search-aggregator-threads", type=int, default=8)
    parser.add_argument("--search-aggregator-socket-threads", type=int, default=8)
    parser.add_argument("--search-timeout-ms", type=int, default=5000)
    parser.add_argument("--search-after-workers-sec", type=float, default=5.0)
    parser.add_argument("--search-after-master-sec", type=float, default=3.0)
    parser.add_argument("--search-start-timeout-sec", type=float, default=30.0)
    parser.add_argument("--search-run-timeout", type=int, default=3600)
    parser.add_argument("--search-repetitions", type=int, default=3)
    parser.add_argument("--search-agg-topk-values", default="all")
    parser.add_argument("--search-max-check-values", default="4096,8192")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local single-host SPTAG orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all = subparsers.add_parser("run-all", help="Start services and run the 1M build phase")
    _add_shared_arguments(run_all)

    start = subparsers.add_parser("start-services", help="Start build master and workers")
    _add_shared_arguments(start)

    run_build = subparsers.add_parser("run-build", help="Run the 1M client build phase against running services")
    _add_shared_arguments(run_build)

    run_search = subparsers.add_parser("run-search", help="Run a search sweep against an existing 1M checkpoint")
    _add_shared_arguments(run_search)

    finalize = subparsers.add_parser("finalize", help="Finalize the current build job")
    _add_shared_arguments(finalize)

    status = subparsers.add_parser("status", help="Print master status")
    _add_shared_arguments(status)

    stop = subparsers.add_parser("stop", help="Kill build and search tmux sessions")
    _add_shared_arguments(stop)

    logs = subparsers.add_parser("logs", help="Tail one log file from the log directory")
    _add_shared_arguments(logs)
    logs.add_argument("--log-file", required=True, help="Example: master.log or worker_01.log")
    logs.add_argument("--lines", type=int, default=50)

    args = parser.parse_args()

    if args.command == "run-all":
        _run_all(args)
        return 0
    if args.command == "start-services":
        _start_build_services(args)
        return 0
    if args.command == "run-build":
        run_id = _timestamp()
        status = _run_build_sequence(args, run_id=run_id, include_search=bool(args.run_search))
        if status is not None:
            _print_status_summary(status)
        if args.finalize_after_run:
            payload = _finalize(args, run_id)
            print(json.dumps(payload, indent=2, sort_keys=True))
        if args.stop_services_after_run:
            _stop_all_search_services(args)
            _stop_build_services(args)
        return 0
    if args.command == "run-search":
        status = _master_status(args)
        active_worker_ids = [int(value) for value in status.get("active_workers", [])]
        if not active_worker_ids:
            active_worker_ids = [worker["worker_id"] for worker in _workers(args)]
        _run_search_sweep(args, checkpoint_id=int(args.max_points), active_worker_ids=active_worker_ids, run_id=_timestamp())
        return 0
    if args.command == "finalize":
        payload = _finalize(args, _timestamp())
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        _status_command(args)
        return 0
    if args.command == "stop":
        _stop_build_services(args)
        _stop_all_search_services(args)
        return 0
    if args.command == "logs":
        _logs_command(args)
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as ex:
        body = ex.read().decode("utf-8", errors="replace")
        print(body or str(ex), file=sys.stderr)
        raise
