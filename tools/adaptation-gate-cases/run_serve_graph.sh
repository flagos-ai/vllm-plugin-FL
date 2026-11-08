#!/usr/bin/env bash
# Example: MODEL_PATH=/models/Qwen3.6-27B PORT=8001 ./run_serve_graph.sh

set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: MODEL_PATH=/path/to/model PORT=8001 $0" >&2
    exit 2
fi

model_path=${MODEL_PATH:-}
port=${PORT:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "$model_path" || -z "$port" ]]; then
    echo "MODEL_PATH and PORT are required." >&2
    echo "Usage: MODEL_PATH=/path/to/model PORT=8001 $0" >&2
    exit 2
fi
if [[ ! -d "$model_path" ]]; then
    echo "Model directory does not exist: $model_path" >&2
    exit 2
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "Port must be an integer from 1 to 65535: $port" >&2
    exit 2
fi
served_model_name=${SERVED_MODEL_NAME:-$model_path}
if [[ -z ${VLLM_PLUGINS+x} ]]; then
    export VLLM_PLUGINS=fl
fi
printf '%s\n' "$$" >"${SERVER_PID_FILE:-$script_dir/.server.pid}"

exec vllm serve "$model_path" \
    --served-model-name "$served_model_name" \
    --host 127.0.0.1 \
    --port "$((10#$port))" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --allowed-local-media-path "$script_dir/images" \
    --trust-remote-code
