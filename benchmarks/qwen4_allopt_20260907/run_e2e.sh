#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$script_dir/all_on_env.sh"

artifact_root=${ARTIFACT_ROOT:-/artifact}
runtime_dir="$artifact_root/runtime"
mkdir -p "$runtime_dir"
server_log=${SERVER_LOG:-$artifact_root/server.log}

wait_http_200() {
  local path=$1
  local output=$2
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    local code
    code=$(curl -sS -o "$output" -w '%{http_code}' "http://127.0.0.1:${PORT}${path}" || true)
    # A 200 with an empty body is valid for /health.
    if [[ "$code" == 200 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "HTTP 200 timeout for $path" >&2
  return 1
}

wait_http_200 /health "$runtime_dir/health_before_benchmark.json"

# Poll only the worker-emitted post-warmup records. The gate requires all TP8
# ranks and positive hits/misses; an import/preflight or rank-0-only line does
# not pass. No NSYS or extra workload is started here.
cache_gate="$runtime_dir/plan_cache_gate.json"
deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  set +e
  python3 "$script_dir/plan_cache_gate.py" \
    --log "$server_log" \
    --output "$cache_gate" \
    --expected-ranks "$TP_SIZE"
  gate_rc=$?
  set -e
  if (( gate_rc == 0 )); then
    break
  fi
  sleep 5
done
if (( gate_rc != 0 )); then
  echo "plan-cache gate did not pass before timeout" >&2
  exit 4
fi

python3 "$script_dir/correctness_gate.py" \
  --port "$PORT" \
  --model "$SERVED_MODEL_NAME" \
  --output "$runtime_dir/api_smoke_gate.json"

exec "$script_dir/run_benchmark.sh"
