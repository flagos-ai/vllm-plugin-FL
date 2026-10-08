#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <served-model-name> <run-dir>" >&2
  exit 2
fi

MODEL=$1
RUN_DIR=$2
RUN_DIR=$(cd "$RUN_DIR" && pwd)
PROFILE_DIR="$RUN_DIR/profile"
RESULT_DIR="$RUN_DIR/results"
TOOL_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_URL=${PROFILE_BASE_URL:-http://localhost:8000}
HEALTH_TIMEOUT=${PROFILE_HEALTH_TIMEOUT_SECONDS:-3600}
FLAGGEMS_OPLIST=${FLAGGEMS_ENABLE_OPLIST_PATH:-$RUN_DIR/flaggems_enable_oplist.txt}

if [[ ! -d "$PROFILE_DIR" ]]; then
  echo "profile directory does not exist: $PROFILE_DIR" >&2
  exit 1
fi
exec 9>"$RUN_DIR/.profile.lock"
if ! flock -n 9; then
  echo "another profile process is already using: $RUN_DIR" >&2
  exit 1
fi
if find "$PROFILE_DIR" -type f -name '*.pt.trace.json*' -print -quit |
  grep -q .; then
  echo "profile directory already contains a trace: $PROFILE_DIR" >&2
  exit 1
fi

profiling=0
stop_profile() {
  local best_effort=${1:-0}
  if [[ "$profiling" -eq 1 ]]; then
    if curl -fsS -XPOST "$BASE_URL/stop_profile"; then
      profiling=0
    elif [[ "$best_effort" -eq 1 ]]; then
      echo "warning: failed to stop profiler during cleanup" >&2
    else
      return 1
    fi
  fi
}
trap 'stop_profile 1' EXIT

deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS --max-time 10 "$BASE_URL/health" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    echo "server health check timed out after $HEALTH_TIMEOUT seconds" >&2
    exit 1
  fi
  sleep 5
done

python3 "$TOOL_DIR/run_concurrent_requests.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --prompt-output "$RUN_DIR/prompt_token_ids.json" \
  --metrics "$RUN_DIR/warmup_metrics.json"

curl -fsS -XPOST "$BASE_URL/start_profile"
profiling=1
python3 "$TOOL_DIR/run_concurrent_requests.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --prompt-input "$RUN_DIR/prompt_token_ids.json" \
  --metrics "$RUN_DIR/profiled_metrics.json"
stop_profile 0

python3 "$TOOL_DIR/extract_operator_shapes.py" \
  --runtime "$PROFILE_DIR" \
  --rank 0 \
  --output-dir "$RESULT_DIR"

if [[ -f "$FLAGGEMS_OPLIST" ]]; then
  mv "$FLAGGEMS_OPLIST" "$RESULT_DIR/flaggems_enable_oplist.txt"
fi
