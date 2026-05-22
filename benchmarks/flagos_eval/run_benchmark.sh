#!/bin/bash
set -e

# E4.8 FINAL：分场景极限并发 + 大 batch tokens
export VLLM_PLUGINS=ascend
export VLLM_ASCEND_ENABLE_FLAGGEMS=1
export ENABLE_FLAGGEMS=1
export FLAGGEMS_DEVICE=ascend
export HCCL_OP_EXPANSION_MODE=AIV

MODEL_PATH=${1:?"Please provide model path, e.g.: ./run_benchmark.sh /workspace/Qwen3-4B/"}
OUTPUT_DIR=bench_results
mkdir -p "${OUTPUT_DIR}"

echo "=== E4.8 FINAL benchmark ==="

declare -A THROUGHPUT_SCENARIOS=(
    ["chat_1k"]="1024 1024 300"
    ["chat_4k"]="4096 1024 300"
    ["chat_6k"]="6144 1024 300"
)
declare -A LATENCY_SCENARIOS=(
    ["batch_8"]="4096 1024 8 10"
)

BASE_OPTS="--tensor-parallel-size 2 --gpu-memory-utilization 0.95"

echo "==================== THROUGHPUT TESTS ===================="
for scenario in "${!THROUGHPUT_SCENARIOS[@]}"; do
    read input_len output_len num_prompts <<< "${THROUGHPUT_SCENARIOS[$scenario]}"
    output_file="${OUTPUT_DIR}/throughput_${scenario}.json"
    case "$scenario" in
        chat_1k)
            max_model_len=3072
            max_seqs=512
            max_batched_tokens=16384
            ;;
        chat_4k)
            max_model_len=5632
            max_seqs=512
            max_batched_tokens=12288
            ;;
        chat_6k)
            max_model_len=8192
            max_seqs=256
            max_batched_tokens=8192
            ;;
    esac

    echo "--- Throughput: ${scenario} (seqs=${max_seqs}, batch=${max_batched_tokens}, max_len=${max_model_len}) ---"

    vllm bench throughput \
        --model "${MODEL_PATH}" \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --num-prompts "${num_prompts}" \
        --trust-remote-code --dtype auto --enforce-eager \
        --max-model-len "${max_model_len}" \
        ${BASE_OPTS} \
        --max-num-seqs "${max_seqs}" \
        --max-num-batched-tokens "${max_batched_tokens}" \
        --output-json "${output_file}"
    echo "Saved: ${output_file}"
done

echo ""
echo "==================== LATENCY TESTS ===================="
for scenario in "${!LATENCY_SCENARIOS[@]}"; do
    read input_len output_len batch_size num_iters <<< "${LATENCY_SCENARIOS[$scenario]}"
    output_file="${OUTPUT_DIR}/latency_${scenario}.json"
    max_model_len=6144
    max_seqs=16

    echo "--- Latency: ${scenario} ---"
    vllm bench latency \
        --model "${MODEL_PATH}" \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --batch-size "${batch_size}" --num-iters "${num_iters}" \
        --trust-remote-code --dtype auto --enforce-eager \
        --max-model-len "${max_model_len}" \
        ${BASE_OPTS} --max-num-seqs "${max_seqs}" \
        --output-json "${output_file}"
    echo "Saved: ${output_file}"
done

echo ""
echo "=== BENCHMARK COMPLETED ==="
ls -la "${OUTPUT_DIR}"/*.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/collect_benchmark_results.py" ] && python3 "${SCRIPT_DIR}/collect_benchmark_results.py" "${OUTPUT_DIR}"