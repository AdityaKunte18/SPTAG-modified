#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/akunte2/work/horizann/SPTAG-modified}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/srv/local/data/anns_data/sift1b}"

JOB_ID="${JOB_ID:-sptag_sift1b_1b_10workers_local}"
SESSION_NAME="${SESSION_NAME:-sptag-local-1b-10workers}"
LOG_DIR="${LOG_DIR:-$HOME/logs/sptag-sift1b-1b-10workers-local}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/index_build_output_sift1b_1b_10workers_local}"
ORCHESTRATOR_COMMAND="${ORCHESTRATOR_COMMAND:-run-all}"
RESUME_FROM_POINT="${RESUME_FROM_POINT:-0}"
MAX_POINTS="${MAX_POINTS:-1000000000}"
BUILD_PHASES="${BUILD_PHASES:-1000000,10000000,100000000,200000000,500000000,1000000000}"
SEARCH_CHECKPOINTS="${SEARCH_CHECKPOINTS:-1000000,10000000,100000000,200000000,500000000,1000000000}"
STOP_SERVICES_AFTER_RUN="${STOP_SERVICES_AFTER_RUN:-0}"

# Use a dedicated port block so this run does not collide with older local sessions.
MASTER_PORT="${MASTER_PORT:-48079}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-48080}"
SEARCH_WORKER_BASE_PORT="${SEARCH_WORKER_BASE_PORT:-49100}"
SEARCH_MASTER_PORT="${SEARCH_MASTER_PORT:-49200}"

SEARCH_GT_PATH_BY_CHECKPOINT="${SEARCH_GT_PATH_BY_CHECKPOINT:-1000000=$DATA_ROOT/gnd/idx_1M.ivecs,10000000=$DATA_ROOT/gnd/idx_10M.ivecs,100000000=$DATA_ROOT/gnd/idx_100M.ivecs,200000000=$DATA_ROOT/gnd/idx_200M.ivecs,500000000=$DATA_ROOT/gnd/idx_500M.ivecs,1000000000=$DATA_ROOT/gnd/idx_1000M.ivecs}"

exec "$PYTHON_BIN" "$REPO_ROOT/tools/sptag_local_single_host_orchestrator.py" "$ORCHESTRATOR_COMMAND" \
  --repo "$REPO_ROOT" \
  --data-root "$DATA_ROOT" \
  --python "$PYTHON_BIN" \
  --num-workers 10 \
  --max-points "$MAX_POINTS" \
  --build-phases "$BUILD_PHASES" \
  --search-checkpoints "$SEARCH_CHECKPOINTS" \
  --run-search \
  --batch-size 5000 \
  --threads 64 \
  --value-type UInt8 \
  --search-value-type UInt8 \
  --search-agg-topk-values all \
  --search-max-check-values 1024,4096,8192,16384 \
  --search-gt-path-by-checkpoint "$SEARCH_GT_PATH_BY_CHECKPOINT" \
  --finalize-after-run \
  --job-id "$JOB_ID" \
  --session-name "$SESSION_NAME" \
  --log-dir "$LOG_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --resume-from-point "$RESUME_FROM_POINT" \
  --master-port "$MASTER_PORT" \
  --worker-base-port "$WORKER_BASE_PORT" \
  --search-worker-base-port "$SEARCH_WORKER_BASE_PORT" \
  --search-master-port "$SEARCH_MASTER_PORT" \
  --client-phase-timeout 604800 \
  $([[ "$STOP_SERVICES_AFTER_RUN" == "1" ]] && printf '%s' '--stop-services-after-run')
