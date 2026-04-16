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
LOG_DIR="${LOG_DIR:-$HOME/logs/sptag-gist-1m-1worker-local}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/index_build_output_gist_1m_1worker_local}"
ORCHESTRATOR_COMMAND="${ORCHESTRATOR_COMMAND:-run-all}"
RESUME_FROM_POINT="${RESUME_FROM_POINT:-0}"
MAX_POINTS="${MAX_POINTS:-1000000}"
BUILD_PHASES="${BUILD_PHASES:-1000000}"
SEARCH_CHECKPOINTS="${SEARCH_CHECKPOINTS:-1000000}"
STOP_SERVICES_AFTER_RUN="${STOP_SERVICES_AFTER_RUN:-0}"
SEARCH_MAX_CHECK_VALUES="${SEARCH_MAX_CHECK_VALUES:-1024,4096,8192,16384}"

# Use a dedicated port block so this run does not collide with the SIFT jobs.
MASTER_PORT="${MASTER_PORT:-58079}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-58080}"
SEARCH_WORKER_BASE_PORT="${SEARCH_WORKER_BASE_PORT:-59100}"
SEARCH_MASTER_PORT="${SEARCH_MASTER_PORT:-59200}"

exec "$PYTHON_BIN" "$REPO_ROOT/tools/sptag_local_single_host_orchestrator.py" "$ORCHESTRATOR_COMMAND" \
  --repo "$REPO_ROOT" \
  --data-root "$DATA_ROOT" \
  --base-path "$BASE_PATH" \
  --query-path "$QUERY_PATH" \
  --python "$PYTHON_BIN" \
  --num-workers 1 \
  --max-points "$MAX_POINTS" \
  --build-phases "$BUILD_PHASES" \
  --search-checkpoints "$SEARCH_CHECKPOINTS" \
  --run-search \
  --batch-size 5000 \
  --threads 64 \
  --value-type Float \
  --search-gt-path "$SEARCH_GT_PATH" \
  --search-value-type Float \
  --search-agg-topk-values all \
  --search-max-check-values "$SEARCH_MAX_CHECK_VALUES" \
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
