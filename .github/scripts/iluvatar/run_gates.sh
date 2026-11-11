#!/usr/bin/env bash
# Run the committed inventory without filtering cases or replacing fixtures.
set -euo pipefail
: "${ILUVATAR_MODEL_27B:?Set the mounted Qwen3.6-27B directory}"
: "${ILUVATAR_MODEL_35B:?Set the mounted Qwen3.6-35B-A3B directory}"
: "${FLAGCX_PATH:?Set the validated FlagCX build root; in CI, configure ILUVATAR_FLAGCX_PATH to a path visible inside the container}"
for path in "$ILUVATAR_MODEL_27B" "$ILUVATAR_MODEL_35B"; do
    test -f "$path/config.json" || { echo "Missing model: $path" >&2; exit 2; }
done
for file in build/lib/libflagcx.so plugin/interservice/flagcx_wrapper.py; do
    test -f "$FLAGCX_PATH/$file" || { echo "Missing FlagCX file: $FLAGCX_PATH/$file" >&2; exit 2; }
done
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$root"
export PYTHONPATH="$root"
export VLLM_PLUGINS=fl USE_FLAGGEMS=1 GEMS_VENDOR=iluvatar
python .github/scripts/iluvatar/verify_stack.py
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
export GRAPH_COMPILATION_CONFIG=${GRAPH_COMPILATION_CONFIG:-'{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'}
export PORT=${PORT:-8132}
out=${ILUVATAR_GATE_RESULTS:-ci-results/iluvatar-gates}
mkdir -p "$out"
out=$(cd "$out" && pwd)
server=''
cleanup() {
    if [[ -n "$server" ]]; then
        kill -TERM -- "-$server" 2>/dev/null || true
        for ((i=0; i<60; i++)); do
            if ! kill -0 -- "-$server" 2>/dev/null; then
                wait "$server" 2>/dev/null || true
                server=''
                return
            fi
            sleep 1
        done
        kill -KILL -- "-$server" 2>/dev/null || true
        wait "$server" 2>/dev/null || true
        server=''
    fi
}
trap cleanup EXIT
status=0
for model in 27b 35b; do
    if [[ "$model" == 27b ]]; then
        export MODEL_PATH="$ILUVATAR_MODEL_27B"
    else
        export MODEL_PATH="$ILUVATAR_MODEL_35B"
    fi
    for mode in eager graph; do
        group="$out/$model-$mode"
        mkdir -p "$group"
        serve_mode=graph
        if [[ "$mode" == eager ]]; then
            serve_mode=eager
        fi
        printf '%s\n' "model=$MODEL_PATH" "mode=$mode" \
            "compilation=${GRAPH_COMPILATION_CONFIG:-upstream-default}" \
            "tp=$TENSOR_PARALLEL_SIZE" "max_model_len=$MAX_MODEL_LEN" \
            >"$group/config.txt"
        git rev-parse HEAD >"$group/commit.txt"
        export SERVER_PID_FILE="$group/server.pid" RESULTS_DIR="$group/results"
        setsid bash "tools/adaptation-gate-cases/run_serve_$serve_mode.sh" >"$group/server.log" 2>&1 &
        server=$!
        if bash tools/adaptation-gate-cases/run_test.sh >"$group/tests.log" 2>&1; then
            echo "$model $mode PASS" | tee -a "$out/status.txt"
        else
            echo "$model $mode FAIL" | tee -a "$out/status.txt"
            status=1
        fi
        cleanup
    done
done
exit "$status"
