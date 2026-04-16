#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/akunte2/work/horizann/SPTAG-modified}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/srv/local/data/anns_data/gist}"

BASE_PATH="${BASE_PATH:-$DATA_ROOT/gist_base.fvecs}"
QUERY_PATH="${QUERY_PATH:-$DATA_ROOT/gist_query.fvecs}"
SEARCH_GT_PATH="${SEARCH_GT_PATH:-$DATA_ROOT/gist_groundtruth.ivecs}"

JOB_ID="${JOB_ID:-sptag_gist_1m_1worker_local}"
SESSION_NAME="${SESSION_NAME:-sptag-local-gist-1m-1worker}"
RUNNER_SESSION_NAME="${RUNNER_SESSION_NAME:-orchestrator-${SESSION_NAME}}"
LOG_DIR="${LOG_DIR:-$HOME/logs/sptag-gist-1m-1worker-local}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/index_build_output_gist_1m_1worker_local}"
ORCHESTRATOR_COMMAND="${ORCHESTRATOR_COMMAND:-run-all}"
RESUME_FROM_POINT="${RESUME_FROM_POINT:-0}"
MAX_POINTS="${MAX_POINTS:-1000000}"
BUILD_PHASES="${BUILD_PHASES:-1000000}"
SEARCH_CHECKPOINTS="${SEARCH_CHECKPOINTS:-1000000}"
STOP_SERVICES_AFTER_RUN="${STOP_SERVICES_AFTER_RUN:-0}"
SEARCH_MAX_CHECK_VALUES="${SEARCH_MAX_CHECK_VALUES:-1024,4096,8192,16384}"

MASTER_PORT="${MASTER_PORT:-58079}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-58080}"
SEARCH_WORKER_BASE_PORT="${SEARCH_WORKER_BASE_PORT:-59100}"
SEARCH_MASTER_PORT="${SEARCH_MASTER_PORT:-59200}"

RUNNER_LOG="${RUNNER_LOG:-$LOG_DIR/orchestrator_run_all.log}"
RUNNER_BOOTSTRAP_LOG="${RUNNER_BOOTSTRAP_LOG:-$LOG_DIR/orchestrator_runner_bootstrap.log}"
RUNNER_BOOTSTRAP_SCRIPT="${RUNNER_BOOTSTRAP_SCRIPT:-$LOG_DIR/orchestrator_runner_bootstrap.sh}"
RUN_SCRIPT="${REPO_ROOT}/tools/run_gist_1m_1worker_local.sh"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "run helper not found: $RUN_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

if tmux has-session -t "$RUNNER_SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $RUNNER_SESSION_NAME" >&2
  echo "Use: tmux attach -t $RUNNER_SESSION_NAME" >&2
  exit 1
fi

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euxo pipefail\n'
  printf 'REPO_ROOT=%q\n' "$REPO_ROOT"
  printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
  printf 'DATA_ROOT=%q\n' "$DATA_ROOT"
  printf 'BASE_PATH=%q\n' "$BASE_PATH"
  printf 'QUERY_PATH=%q\n' "$QUERY_PATH"
  printf 'SEARCH_GT_PATH=%q\n' "$SEARCH_GT_PATH"
  printf 'JOB_ID=%q\n' "$JOB_ID"
  printf 'SESSION_NAME=%q\n' "$SESSION_NAME"
  printf 'LOG_DIR=%q\n' "$LOG_DIR"
  printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
  printf 'ORCHESTRATOR_COMMAND=%q\n' "$ORCHESTRATOR_COMMAND"
  printf 'RESUME_FROM_POINT=%q\n' "$RESUME_FROM_POINT"
  printf 'MAX_POINTS=%q\n' "$MAX_POINTS"
  printf 'BUILD_PHASES=%q\n' "$BUILD_PHASES"
  printf 'SEARCH_CHECKPOINTS=%q\n' "$SEARCH_CHECKPOINTS"
  printf 'STOP_SERVICES_AFTER_RUN=%q\n' "$STOP_SERVICES_AFTER_RUN"
  printf 'SEARCH_MAX_CHECK_VALUES=%q\n' "$SEARCH_MAX_CHECK_VALUES"
  printf 'MASTER_PORT=%q\n' "$MASTER_PORT"
  printf 'WORKER_BASE_PORT=%q\n' "$WORKER_BASE_PORT"
  printf 'SEARCH_WORKER_BASE_PORT=%q\n' "$SEARCH_WORKER_BASE_PORT"
  printf 'SEARCH_MASTER_PORT=%q\n' "$SEARCH_MASTER_PORT"
  printf 'RUNNER_LOG=%q\n' "$RUNNER_LOG"
  printf 'RUNNER_BOOTSTRAP_LOG=%q\n' "$RUNNER_BOOTSTRAP_LOG"
  printf 'RUN_SCRIPT=%q\n' "$RUN_SCRIPT"
  printf 'export REPO_ROOT PYTHON_BIN DATA_ROOT BASE_PATH QUERY_PATH SEARCH_GT_PATH JOB_ID SESSION_NAME LOG_DIR OUTPUT_DIR ORCHESTRATOR_COMMAND RESUME_FROM_POINT MAX_POINTS BUILD_PHASES SEARCH_CHECKPOINTS STOP_SERVICES_AFTER_RUN SEARCH_MAX_CHECK_VALUES MASTER_PORT WORKER_BASE_PORT SEARCH_WORKER_BASE_PORT SEARCH_MASTER_PORT RUNNER_LOG RUNNER_BOOTSTRAP_LOG RUN_SCRIPT\n'
  printf 'exec >>"$RUNNER_BOOTSTRAP_LOG" 2>&1\n'
  printf 'echo "[bootstrap start $(date -Is)]"\n'
  printf 'echo "whoami=$(whoami)"\n'
  printf 'echo "pwd_before=$(pwd)"\n'
  printf 'echo "shell=$SHELL"\n'
  printf 'echo "tmux_version=$(tmux -V || true)"\n'
  printf 'env | sort\n'
  printf 'cd "$REPO_ROOT"\n'
  printf 'echo "pwd_after=$(pwd)"\n'
  printf 'set +e\n'
  printf 'bash -x "$RUN_SCRIPT" >>"$RUNNER_LOG" 2>&1\n'
  printf 'rc=$?\n'
  printf 'set -e\n'
  printf 'echo "[runner exited code=$rc]"\n'
  printf 'echo "runner log: $RUNNER_LOG"\n'
  printf 'if [[ -f "$RUNNER_LOG" ]]; then tail -n 100 "$RUNNER_LOG"; fi\n'
  printf 'exec bash\n'
} >"$RUNNER_BOOTSTRAP_SCRIPT"

chmod +x "$RUNNER_BOOTSTRAP_SCRIPT"

printf -v TMUX_CMD 'bash %q' "$RUNNER_BOOTSTRAP_SCRIPT"
tmux new-session -d -s "$RUNNER_SESSION_NAME" "$TMUX_CMD"

sleep 1

if ! tmux has-session -t "$RUNNER_SESSION_NAME" 2>/dev/null; then
  echo "runner tmux session exited immediately: $RUNNER_SESSION_NAME" >&2
  if [[ -f "$RUNNER_BOOTSTRAP_LOG" ]]; then
    echo "last runner bootstrap log lines:" >&2
    tail -n 100 "$RUNNER_BOOTSTRAP_LOG" >&2 || true
  fi
  if [[ -f "$RUNNER_LOG" ]]; then
    echo "last runner log lines:" >&2
    tail -n 50 "$RUNNER_LOG" >&2 || true
  fi
  exit 1
fi

echo "started tmux session: $RUNNER_SESSION_NAME"
echo "runner log: $RUNNER_LOG"
echo "runner bootstrap log: $RUNNER_BOOTSTRAP_LOG"
echo "attach: tmux attach -t $RUNNER_SESSION_NAME"
echo "master status:"
echo "  $PYTHON_BIN $REPO_ROOT/tools/sptag_local_single_host_orchestrator.py status --job-id $JOB_ID --master-port $MASTER_PORT"
