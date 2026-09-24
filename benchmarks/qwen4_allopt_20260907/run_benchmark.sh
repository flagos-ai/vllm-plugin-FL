#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$script_dir/all_on_env.sh"

artifact_root=${ARTIFACT_ROOT:-/artifact}
benchmark_dir="$artifact_root/benchmark"
mkdir -p "$benchmark_dir"

wait_http_200() {
  local path=$1
  local output=$2
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    local code
    code=$(curl -sS -o "$output" -w '%{http_code}' "http://127.0.0.1:${PORT}${path}" || true)
    # /health is allowed to have an empty body; HTTP status is the gate.
    if [[ "$code" == 200 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "HTTP 200 timeout for $path" >&2
  return 1
}

wait_http_200 /health "$benchmark_dir/health.json"
wait_http_200 /v1/models "$benchmark_dir/models.json"

run_one() {
  local label=$1
  local prompts=$2
  local result_file="$benchmark_dir/${label}.json"
  local log_file="$benchmark_dir/${label}.log"
  local command_file="$benchmark_dir/${label}.command.txt"
  local status_file="$benchmark_dir/${label}.status.json"
  local -a command=(
    vllm bench serve
    --backend vllm
    --model "$SERVED_MODEL_NAME"
    --tokenizer "$MODEL_CONTAINER"
    --endpoint /v1/completions
    --host 127.0.0.1
    --port "$PORT"
    --dataset-name random
    --ignore-eos
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --random-range-ratio 0.0
    --request-rate inf
    --temperature 0
    --seed "$SEED"
    --max-concurrency "$CONCURRENCY"
    --num-prompts "$prompts"
    --save-result
    --save-detailed
    --result-dir "$benchmark_dir"
    --result-filename "${label}.json"
  )
  printf '%q ' "${command[@]}" >"$command_file"
  printf '\n' >>"$command_file"
  set +e
  "${command[@]}" >"$log_file" 2>&1
  local rc=$?
  set -e
  python3 - "$status_file" "$rc" "$result_file" <<'PY'
import json
import os
import sys

out, rc, result = sys.argv[1], int(sys.argv[2]), sys.argv[3]
payload = {
    "returncode": rc,
    "result_exists": os.path.isfile(result),
    "result_path": result,
    "status": "pass" if rc == 0 and os.path.isfile(result) else "fail",
}
with open(out, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
  if (( rc != 0 )); then
    echo "$label failed; preserving $log_file and any partial result" >&2
  else
    echo "$label complete"
  fi
}

# Warmup deliberately uses the same 1024/1024 shape as formal requests. It is
# not included in formal aggregates but is retained for auditability.
run_one warmup "$WARMUP_PROMPTS"
for ((round = 1; round <= FORMAL_ROUNDS; round++)); do
  run_one "formal${round}" "$FORMAL_PROMPTS"
done

python3 "$script_dir/benchmark_summary.py" \
  --benchmark-dir "$benchmark_dir" \
  --formal-rounds "$FORMAL_ROUNDS" \
  --formal-prompts "$FORMAL_PROMPTS" \
  --warmup-prompts "$WARMUP_PROMPTS" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --output "$benchmark_dir/summary.json"
