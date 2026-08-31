#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <deepseek_v4_flash|qwen3_6_35b_a3b> <request-config.json>" >&2
  exit 2
fi

MODEL_CASE=$1
REQUEST_CONFIG=$2
RUN_ROOT=${PROFILE_RUN_ROOT:-/vllm-workspace/graph_operator_profile_runs}
RUN_DIR="$RUN_ROOT/$MODEL_CASE"
PROFILE_DIR="$RUN_DIR/profile"
TOOL_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_URL=${PROFILE_BASE_URL:-http://localhost:8000}
active=0
stop_profile() {
  if [[ "$active" -eq 1 ]]; then
    curl -fsS -XPOST "$BASE_URL/stop_profile"
    active=0
  fi
}
trap stop_profile EXIT

curl -fsS "$BASE_URL/health"
python3 "$TOOL_DIR/run_concurrent_requests.py" \
  --config "$REQUEST_CONFIG" \
  --base-url "$BASE_URL" \
  --prompt-output "$RUN_DIR/prompt_token_ids.json" \
  --responses "$RUN_DIR/warmup_responses.json" \
  --metrics "$RUN_DIR/warmup_metrics.json"

curl -fsS -XPOST "$BASE_URL/start_profile"
active=1
python3 "$TOOL_DIR/run_concurrent_requests.py" \
  --config "$REQUEST_CONFIG" \
  --base-url "$BASE_URL" \
  --prompt-input "$RUN_DIR/prompt_token_ids.json" \
  --responses "$RUN_DIR/profiled_responses.json" \
  --metrics "$RUN_DIR/profiled_metrics.json"
stop_profile

python3 "$TOOL_DIR/extract_operator_shapes.py" \
  --runtime "$PROFILE_DIR" \
  --rank 0 \
  --output-dir "$RUN_DIR/results"
