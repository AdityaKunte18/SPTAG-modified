#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/akunte2/work/horizann/SPTAG-modified}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

JOB_ID="${JOB_ID:-sptag_sift1b_1b_10workers_local}"
SESSION_NAME="${SESSION_NAME:-sptag-local-1b-10workers}"
RUNNER_SESSION_NAME="${RUNNER_SESSION_NAME:-orchestrator-${SESSION_NAME}}"
LOG_DIR="${LOG_DIR:-$HOME/logs/sptag-sift1b-1b-10workers-local}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/index_build_output_sift1b_1b_10workers_local}"

MASTER_PORT="${MASTER_PORT:-48079}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-48080}"
SEARCH_WORKER_BASE_PORT="${SEARCH_WORKER_BASE_PORT:-49100}"
SEARCH_MASTER_PORT="${SEARCH_MASTER_PORT:-49200}"

"$PYTHON_BIN" "$REPO_ROOT/tools/sptag_local_single_host_orchestrator.py" stop \
  --repo "$REPO_ROOT" \
  --python "$PYTHON_BIN" \
  --job-id "$JOB_ID" \
  --session-name "$SESSION_NAME" \
  --log-dir "$LOG_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --master-port "$MASTER_PORT" \
  --worker-base-port "$WORKER_BASE_PORT" \
  --search-worker-base-port "$SEARCH_WORKER_BASE_PORT" \
  --search-master-port "$SEARCH_MASTER_PORT"

tmux kill-session -t "$RUNNER_SESSION_NAME" 2>/dev/null || true

rm -rf "$LOG_DIR"
rm -rf "$OUTPUT_DIR"
rm -f "$REPO_ROOT/Release"/generated_worker_*.ini
rm -f "$REPO_ROOT/Release/Aggregator.ini"
rm -f "$REPO_ROOT/ClientServerImplementation/workers.json"

echo "removed logs: $LOG_DIR"
echo "removed output: $OUTPUT_DIR"
echo "removed generated ini files and workers.json under: $REPO_ROOT"
