#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <deepseek_v4_flash|qwen3_6_35b_a3b> <request.json>" >&2
  exit 2
fi

MODEL_CASE=$1
REQUEST_JSON=$2
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
curl -fsS "$BASE_URL/v1/completions" -HContent-Type:application/json \
  --data-binary "@$REQUEST_JSON" > "$RUN_DIR/warmup_response.json"
curl -fsS -XPOST "$BASE_URL/start_profile"
active=1
curl -fsS "$BASE_URL/v1/completions" -HContent-Type:application/json \
  --data-binary "@$REQUEST_JSON" > "$RUN_DIR/profiled_response.json"
stop_profile
python3 "$TOOL_DIR/extract_operator_shapes.py" \
  --capture "$PROFILE_DIR/capture_traces" \
  --runtime "$PROFILE_DIR" \
  --rank 0 \
  --output-dir "$RUN_DIR/results"
