#!/usr/bin/env bash
# Example: MODEL_PATH=/models/Qwen3.6-27B PORT=8000 ./run_serve.sh

set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: MODEL_PATH=/path/to/model PORT=8000 $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/run_serve_graph.sh"
