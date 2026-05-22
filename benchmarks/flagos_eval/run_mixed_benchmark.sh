#!/bin/bash
set -e

# ============================================================================
# Mixed-Length Benchmark Script for Vendor Performance Acceptance
# ============================================================================
# Runs throughput tests with mixed input lengths in shuffled order.
# Uses a seed to ensure reproducible shuffle results.
#
# Usage:
#   ./run_mixed_benchmark.sh <model_path> [seed]
#
# Arguments:
#   model_path: Path to the model directory
#   seed: Random seed for shuffle (default: 42)
#
# Example:
#   ./run_mixed_benchmark.sh /workspace/Qwen3-4B/
#   ./run_mixed_benchmark.sh /workspace/Qwen3-4B/ 123
# ============================================================================

# Arguments
MODEL_PATH=${1:?"Please provide model path, e.g.: ./run_mixed_benchmark.sh /workspace/Qwen3-4B/"}
SEED=${2:-42}

# Output directory
OUTPUT_DIR="mixed_bench_results"
mkdir -p "${OUTPUT_DIR}"

echo "================================================================================"
echo "MIXED-LENGTH BENCHMARK FOR VENDOR ACCEPTANCE"
echo "================================================================================"
echo "Model: ${MODEL_PATH}"
echo "Seed: ${SEED}"
echo "Output directory: ${OUTPUT_DIR}/"
echo "================================================================================"
echo ""

# ============================================================================
# Test Configuration
# Format: "name:input_len:output_len:num_prompts"
# ============================================================================
declare -a TEST_CONFIGS=(
    "1k_1k:1024:1024:500"
    "2k_1k:2048:1024:300"
    "4k_1k:4096:1024:300"
    "6k_1k:6144:1024:300"
    "12k_1k:12288:1024:200"
    "32k_1k:32768:1024:10"
)

# ============================================================================
# Shuffle with seed for reproducibility
# ============================================================================
echo "Shuffling test order (seed=${SEED})..."
readarray -t TEST_CONFIGS < <(printf '%s\n' "${TEST_CONFIGS[@]}" | shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"${SEED}" -nosalt </dev/zero 2>/dev/null))

# ============================================================================
# Display test plan
# ============================================================================
echo "Test Plan:"
echo "----------"
TOTAL_SAMPLES=0
for config in "${TEST_CONFIGS[@]}"; do
    IFS=':' read -r name input_len output_len num_prompts <<< "$config"
    echo "  ${name}: ${input_len}/${output_len} tokens, ${num_prompts} samples"
    TOTAL_SAMPLES=$((TOTAL_SAMPLES + num_prompts))
done
echo "  Total: ${TOTAL_SAMPLES} samples"
echo ""

# ============================================================================
# Run throughput tests
# ============================================================================
echo "================================================================================"
echo "RUNNING THROUGHPUT TESTS"
echo "================================================================================"

TEST_COUNT=0
TOTAL_TESTS=${#TEST_CONFIGS[@]}

for config in "${TEST_CONFIGS[@]}"; do
    IFS=':' read -r name input_len output_len num_prompts <<< "$config"
    TEST_COUNT=$((TEST_COUNT + 1))
    output_file="${OUTPUT_DIR}/throughput_${name}.json"

    echo ""
    echo "--- Test ${TEST_COUNT}/${TOTAL_TESTS}: ${name} ---"
    echo "Input: ${input_len} tokens, Output: ${output_len} tokens, Samples: ${num_prompts}"
    echo ""

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
# Collect and analyze results
# ============================================================================
echo ""
echo "================================================================================"
echo "COLLECTING RESULTS"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_SCRIPT="${SCRIPT_DIR}/collect_mixed_benchmark_results.py"

if [ -f "${COLLECTOR_SCRIPT}" ]; then
    python3 "${COLLECTOR_SCRIPT}" "${OUTPUT_DIR}" "${SEED}"
else
    echo "Warning: Collector script not found at ${COLLECTOR_SCRIPT}"
    echo ""
    echo "Files generated:"
    ls -lh "${OUTPUT_DIR}"/*.json

    echo ""
    echo "Quick summary:"
    for config in "${TEST_CONFIGS[@]}"; do
        IFS=':' read -r name input_len output_len num_prompts <<< "$config"
        output_file="${OUTPUT_DIR}/throughput_${name}.json"

        if [ -f "${output_file}" ]; then
            tokens_per_sec=$(python3 -c "import json; data=json.load(open('${output_file}')); print(f\"{data.get('tokens_per_second', data.get('output_throughput', 0)):.2f}\")")
            echo "  ${name}: ${tokens_per_sec} tokens/s"
        fi
    done
fi

echo ""
echo "================================================================================"
echo "BENCHMARK COMPLETED"
echo "================================================================================"
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
