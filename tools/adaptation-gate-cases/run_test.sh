#!/usr/bin/env bash
# Example: MODEL_PATH=/models/Qwen3.6-27B PORT=8000 ./run_test.sh

set -uo pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: MODEL_PATH=/path/to/model PORT=8000 $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model_path=${MODEL_PATH:-}
served_model_name=${SERVED_MODEL_NAME:-}
port=${PORT:-}
service_timeout=${SERVICE_TIMEOUT:-1800}
server_pid_file=${SERVER_PID_FILE:-"$script_dir/.server.pid"}

if [[ -z "$served_model_name" && -z "$model_path" ]] || [[ -z "$port" ]]; then
    echo "SERVED_MODEL_NAME or MODEL_PATH, and PORT are required." >&2
    echo "Usage: MODEL_PATH=/path/to/model PORT=8000 $0" >&2
    exit 2
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "Port must be an integer from 1 to 65535: $port" >&2
    exit 2
fi
port=$((10#$port))
base_url=${BASE_URL:-"http://127.0.0.1:$port/v1"}

deadline=$((SECONDS + service_timeout))
until curl --silent --fail "$base_url/models" >/dev/null 2>&1; do
    if [[ -r "$server_pid_file" ]]; then
        read -r server_pid <"$server_pid_file"
        if [[ "$server_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Service process $server_pid exited before becoming ready." >&2
            exit 1
        fi
    fi
    if (( SECONDS >= deadline )); then
        echo "Service was not ready within ${service_timeout}s: $base_url" >&2
        exit 1
    fi
    sleep 5
done

export MODEL_PATH="$model_path"
export PORT="$port"
export BASE_URL="$base_url"
export RESULTS_DIR=${RESULTS_DIR:-"$script_dir/results"}

cd "$script_dir" || exit 1
status=0
pytest -sv test_text.py || status=1
pytest -sv test_image.py || status=1
pytest -sv test_mix_text_image.py || status=1
exit "$status"
