#!/bin/bash
set -e

# Arguments
MODEL_PATH=${1:?"Please provide model path, e.g.: ./run_benchmark.sh /workspace/Qwen3-4B/"}

# Output directory
OUTPUT_DIR=bench_results
mkdir -p "${OUTPUT_DIR}"

echo "=== Starting benchmark for model: ${MODEL_PATH} ==="
echo "Results will be saved to: ${OUTPUT_DIR}/"
echo ""

# ============================================================================
# Configuration based on mode
# ============================================================================
declare -A THROUGHPUT_SCENARIOS=(
    # input_len output_len num_prompts
    ["chat_1k"]="1024 1024 300"
    ["chat_4k"]="4096 1024 300"
    ["chat_6k"]="6144 1024 300"
)
declare -A LATENCY_SCENARIOS=(
    # input_len output_len batch_size num_iters
    ["batch_8"]="4096 1024 8 10"
)

# ============================================================================
# Throughput Tests
# ============================================================================
echo "==================== THROUGHPUT TESTS ===================="

for scenario in "${!THROUGHPUT_SCENARIOS[@]}"; do
    read input_len output_len num_prompts <<< "${THROUGHPUT_SCENARIOS[$scenario]}"
    output_file="${OUTPUT_DIR}/throughput_${scenario}.json"

    echo ""
    echo "--- Throughput: ${scenario} (input=${input_len}, output=${output_len}, prompts=${num_prompts}) ---"

    vllm bench throughput \
        --model "${MODEL_PATH}" \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --num-prompts "${num_prompts}" \
        --trust-remote-code \
        --dtype auto \
		--enforce-eager \
        --output-json "${output_file}"

    echo "Saved: ${output_file}"
done

# ============================================================================
# Latency Tests
# ============================================================================
echo ""
echo "==================== LATENCY TESTS ===================="

for scenario in "${!LATENCY_SCENARIOS[@]}"; do
    read input_len output_len batch_size num_iters <<< "${LATENCY_SCENARIOS[$scenario]}"
    output_file="${OUTPUT_DIR}/latency_${scenario}.json"

    echo ""
    echo "--- Latency: ${scenario} (input=${input_len}, output=${output_len}, batch=${batch_size}, iters=${num_iters}) ---"

    vllm bench latency \
        --model "${MODEL_PATH}" \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --batch-size "${batch_size}" \
        --num-iters "${num_iters}" \
        --trust-remote-code \
        --dtype auto \
		--enforce-eager \
        --output-json "${output_file}"

    echo "Saved: ${output_file}"
done

echo ""
echo "==================== BENCHMARK COMPLETED ===================="
echo "All results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Files generated:"
ls -la "${OUTPUT_DIR}"/*.json

# Collect and summarize results
echo ""
echo "=== Collecting benchmark results... ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/collect_benchmark_results.py" "${OUTPUT_DIR}"
