#!/usr/bin/env bash
set -euo pipefail

export GITHUB_ENV=/qualification/github-env
: > "$GITHUB_ENV"
bash .github/scripts/metax/check.sh
bash .github/scripts/metax/setup.sh
while IFS= read -r line; do
  export "$line"
done < "$GITHUB_ENV"

unset VLLM_USE_V2_MODEL_RUNNER VLLM_FL_FLAGOS_WHITELIST

export FLAGCX_PATH=/opt/FlagCX
export PYTHONPATH="$FLAGCX_PATH${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$FLAGCX_PATH/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FLAGCX_USE_HOST_COMM=0 FLAGCX_DEBUG=INFO FLAGCX_DEBUG_SUBSYS=INIT,GRAPH
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
sha256sum "$FLAGCX_PATH/build/lib/libflagcx.so" "$FLAGCX_PATH/plugin/interservice/flagcx_wrapper.py"
python - <<'RECEIPT'
import importlib.metadata as md
import json
import flag_gems, flaggems_vllm, triton
print(json.dumps({
    "flaggems": [getattr(flag_gems, "__version__", None), flag_gems.__file__],
    "flaggems_vllm": [md.version("flaggems-vllm"), flaggems_vllm.__file__],
    "triton": [triton.__version__, triton.__file__],
}))
RECEIPT
export TENSOR_PARALLEL_SIZE=4 MAX_MODEL_LEN=32768 SERVICE_TIMEOUT=2400 REQUEST_TIMEOUT=900
cd tools/adaptation-gate-cases
run_dir=/qualification/adaptation-results
mkdir -p "$run_dir"

run_case() (
  model=$1
  mode=$2
  port=$3
  export MACA_VISIBLE_DEVICES=$4
  label="$model-$mode"
  export MODEL_PATH="/data/models/Qwen/$model" SERVED_MODEL_NAME="$model" PORT="$port"
  export SERVER_PID_FILE="$run_dir/$label.pid" RESULTS_DIR="$run_dir/$label"
  mkdir -p "$RESULTS_DIR"
  printf '%s START %s\n' "$(date -Is)" "$label"

  common_args=(
    --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --max-model-len "$MAX_MODEL_LEN"
    --allowed-local-media-path "$PWD/images" --trust-remote-code
  )
  if [[ "$mode" == graph ]]; then
    common_args+=(--cudagraph-metrics)
  else
    common_args+=(--enforce-eager)
  fi
  setsid env -u VLLM_USE_V2_MODEL_RUNNER -u VLLM_FL_FLAGOS_WHITELIST bash -c '
    printf "%s\n" "$$" > "$SERVER_PID_FILE"
    printf "%q " vllm serve "$MODEL_PATH" "$@" > "$RESULTS_DIR/command.txt"
    printf "\n" >> "$RESULTS_DIR/command.txt"
    test -z "${VLLM_USE_V2_MODEL_RUNNER+x}" || exit 87
    test -z "${VLLM_FL_FLAGOS_WHITELIST+x}" || exit 87
    exec vllm serve "$MODEL_PATH" "$@"
  ' sh "${common_args[@]}" > "$run_dir/$label.serve.log" 2>&1 &
  server=$!
  stop_server() {
    kill -TERM -- "-$server" 2>/dev/null || true
    for ((i=0; i<45; i++)); do
      kill -0 -- "-$server" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
  }
  trap stop_server EXIT

  for ((i=0; i<30; i++)); do
    if test -s "$SERVER_PID_FILE"; then break; fi
    kill -0 "$server" 2>/dev/null || break
    sleep 1
  done
  test -s "$SERVER_PID_FILE" || exit 87
  python - "$server" "$RESULTS_DIR/environment.json" <<'PY' || exit 87
import json
import sys
from pathlib import Path

pid, output = sys.argv[1:]
env = dict(
    item.split(b"=", 1) for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    if b"=" in item
)
keys = (
    "GEMS_VENDOR", "VLLM_PLUGINS", "MACA_VISIBLE_DEVICES", "FLAGGEMS_CACHE_DIR",
    "VLLM_USE_V2_MODEL_RUNNER", "VLLM_FL_FLAGOS_WHITELIST", "FLAGCX_PATH",
)
receipt = {key: env[key.encode()].decode() if key.encode() in env else None for key in keys}
assert receipt["VLLM_USE_V2_MODEL_RUNNER"] is None, receipt
assert receipt["VLLM_FL_FLAGOS_WHITELIST"] is None, receipt
Path(output).write_text(json.dumps({"pid": int(pid), "initial_environment": receipt}, indent=2))
PY

  status=0
  bash ./run_test.sh > "$run_dir/$label.test.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$run_dir/$label.exit"
  printf '%s END %s status=%s\n' "$(date -Is)" "$label" "$status"
  exit "$status"
)

run_model() {
  result=0
  run_case "$1" eager "$2" "$4" || result=1
  run_case "$1" graph "$3" "$4" || result=1
  return "$result"
}

run_model Qwen3.6-27B 18971 18972 0,1,2,3 &
dense=$!
run_model Qwen3.6-35B-A3B 18973 18974 4,5,6,7 &
moe=$!
result=0
wait "$dense" || result=1
wait "$moe" || result=1
exit "$result"
