#!/usr/bin/env bash
set -euo pipefail
root=/validation
gate=/gate
export HSA_TOOLS_DISABLE_REGISTER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS="-p no:cacheprovider"
unset VLLM_USE_V2_MODEL_RUNNER VLLM_FL_FLAGOS_WHITELIST
export VLLM_CACHE_ROOT="$root/vllm-cache"
export TRITON_CACHE_DIR="$root/triton-cache"
export FLAGGEMS_CACHE_DIR="$root/flaggems-cache"
export TENSOR_PARALLEL_SIZE=8 MAX_MODEL_LEN=32768 SERVICE_TIMEOUT=1800 REQUEST_TIMEOUT=300
python3 - <<'PYTHON'
from importlib.metadata import version
import flag_gems
import torch
import triton
import vllm_fl
from vllm_fl.platform import PlatformFL

assert "/usr/local/lib/python3.12/dist-packages/vllm_fl/" in vllm_fl.__file__, vllm_fl.__file__
assert "/usr/local/lib/python3.12/dist-packages/flag_gems/" in flag_gems.__file__, flag_gems.__file__
assert "/opt/flagtree/" in triton.__file__, triton.__file__
assert version("vllm") == "0.32.0+empty"
assert version("flag-gems").endswith("+g9a51bed13")
assert PlatformFL.dist_backend == "flagcx"
assert torch.cuda.device_count() == 8
print("installed stack", version("vllm"), version("flag-gems"), version("flagcx"), flush=True)
for device in range(8):
    assert torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx1100"
    assert torch.cuda.mem_get_info(device)[0] >= 43 * 1024**3
PYTHON
cd "$gate"
run_group() (
    local model=$1 mode=$2 port=$3
    local label="$model-$mode"
    export MODEL_PATH="/models/$model" SERVED_MODEL_NAME="$model" PORT="$port"
    export SERVER_PID_FILE="$root/$label.pid" RESULTS_DIR="$root/results/$label"
    mkdir -p "$RESULTS_DIR"
    printf '%s START %s\n' "$(date -Is)" "$label"
    if [[ "$mode" = graph ]]; then
        local graph_options=()
        if [[ "$model" == "Qwen3.6-35B-A3B" ]]; then
            graph_options=(
                --kv-cache-memory-bytes 21474836480
                --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}'
            )
        fi
        setsid bash -c '
            printf "%s\n" "$$" > "$SERVER_PID_FILE"
            exec vllm serve "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
                --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
                --max-model-len "$MAX_MODEL_LEN" --allowed-local-media-path "$PWD/images" \
                --trust-remote-code --cudagraph-metrics "$@"
        ' _ "${graph_options[@]}" > "$root/$label.serve.log" 2>&1 &
    else
        setsid bash ./run_serve_eager.sh > "$root/$label.serve.log" 2>&1 &
    fi
    local server=$!
    trap 'kill -TERM -- "-$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
    local status=0
    bash ./run_test.sh > "$root/$label.test.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$root/$label.exit"
    printf '%s END %s status=%s\n' "$(date -Is)" "$label" "$status"
)
run_group Qwen3.6-35B-A3B graph 18843
run_group Qwen3.6-27B graph 18844
run_group Qwen3.6-27B eager 18845
run_group Qwen3.6-35B-A3B eager 18846
for exit_file in "$root"/*.exit; do
    test "$(cat "$exit_file")" = 0 || exit 1
done
