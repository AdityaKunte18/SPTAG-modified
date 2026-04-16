#!/usr/bin/env python3
"""Export per-run SPTAG search sweep logs to CSV.

This script walks a downloaded milestone search directory and writes one CSV
file per `search_*` checkpoint. Each CSV contains one row per `run_*.log`
file with the exact headers requested by the user:

    run-id, top-k, maxCheck, Recall@10, total_time, QPS

Usage examples:
  python3 baseline/SPTAG/export_search_runs_csv.py \
      /path/to/milestone_root \
      --output-dir baseline/SPTAG/generated_search_csv/run_20260412_204937

  python3 baseline/SPTAG/export_search_runs_csv.py \
      /path/to/search_1M \
      --output-dir /tmp/search_csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


RECALL_RE = re.compile(r"Recall@10:\s*([0-9]+(?:\.[0-9]+)?)")
TOTAL_TIME_RE = re.compile(r"Total time:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
QPS_RE = re.compile(r"QPS:\s*([0-9]+(?:\.[0-9]+)?)")

CSV_HEADERS = ["run-id", "top-k", "maxCheck", "Recall@10", "total_time", "QPS"]


def _normalize_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _search_dirs(root: Path) -> List[Path]:
    if (root / "sweep_manifest.json").is_file():
        return [root]
    directories = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("search_") and (path / "sweep_manifest.json").is_file()
    )
    if directories:
        return directories
    raise FileNotFoundError(
        f"No sweep_manifest.json found under {root}. Provide a milestone experiment directory "
        f"or a single search_* directory."
    )


def _parse_metric(text: str, regex: re.Pattern[str]) -> str:
    match = regex.search(text)
    return match.group(1) if match else ""


def _iter_parameter_set_dirs(search_dir: Path) -> Iterable[Path]:
    for path in sorted(search_dir.iterdir()):
        if path.is_dir() and path.name.startswith("set_") and (path / "parameter_set.json").is_file():
            yield path


def _run_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        if "_" in stem:
            return int(stem.rsplit("_", 1)[1]), stem
        return int(stem), stem
    except ValueError:
        return (sys.maxsize, stem)


def _rows_for_search_dir(search_dir: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    for set_dir in _iter_parameter_set_dirs(search_dir):
        parameter_set = _load_json(set_dir / "parameter_set.json")
        values = dict(parameter_set.get("values", {}))
        top_k = str(values.get("aggregator.top_k", ""))
        max_check = str(values.get("index.max_check", ""))
        for run_log in sorted(set_dir.glob("run_*.log"), key=_run_sort_key):
            text = run_log.read_text(encoding="utf-8", errors="replace")
            rows.append(
                [
                    f"{set_dir.name}/{run_log.stem}",
                    top_k,
                    max_check,
                    _parse_metric(text, RECALL_RE),
                    _parse_metric(text, TOTAL_TIME_RE),
                    _parse_metric(text, QPS_RE),
                ]
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS)
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export per-run SPTAG search sweep logs to CSV.")
    parser.add_argument("path", help="Milestone experiment directory or single search_* directory")
    parser.add_argument("--output-dir", required=True, help="Directory where CSV files will be written")
    args = parser.parse_args(argv)

    root = _normalize_path(args.path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    output_dir = _normalize_path(args.output_dir)
    written: List[Path] = []
    for search_dir in _search_dirs(root):
        rows = _rows_for_search_dir(search_dir)
        output_path = output_dir / f"{search_dir.name}.csv"
        _write_csv(output_path, rows)
        written.append(output_path)
        print(f"Wrote {output_path} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
