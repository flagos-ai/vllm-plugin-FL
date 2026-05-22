#!/bin/bash
set -e

# ==========================================================
# ==========================================================
export VLLM_PLUGINS=ascend                     # 避免 FL 平台插件冲突
export VLLM_ASCEND_ENABLE_FLAGGEMS=1           # 启用 FlagGems
export ENABLE_FLAGGEMS=1
export FLAGGEMS_DEVICE=ascend
export HCCL_OP_EXPANSION_MODE=AIV              # 通信融合优化

# Arguments
MODEL_PATH=${1:?"Please provide model path, e.g.: ./run_benchmark.sh /workspace/Qwen3-4B/"}

# Output directory
OUTPUT_DIR=bench_results
mkdir -p "${OUTPUT_DIR}"

echo "=== Starting optimized benchmark for model: ${MODEL_PATH} ==="
echo "Results will be saved to: ${OUTPUT_DIR}/"
echo ""

# ===========================================================================
# ============================================================================
declare -A THROUGHPUT_SCENARIOS=(
    ["chat_1k"]="1024 1024 300"
    ["chat_4k"]="4096 1024 300"
    ["chat_6k"]="6144 1024 300"
)
declare -A LATENCY_SCENARIOS=(
    ["batch_8"]="4096 1024 8 10"
)

# ============================================================================
# 通用优化参数（已验证稳定）
# ============================================================================
COMMON_OPTS="--tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-num-seqs 256"

# ============================================================================
# Throughput Tests
# ============================================================================
echo "==================== THROUGHPUT TESTS ===================="

for scenario in "${!THROUGHPUT_SCENARIOS[@]}"; do
    read input_len output_len num_prompts <<< "${THROUGHPUT_SCENARIOS[$scenario]}"
    output_file="${OUTPUT_DIR}/throughput_${scenario}.json"

    # 动态计算 max-model-len：确保能容纳当前场景的序列长度
    max_model_len=$(( input_len + output_len + 1024 ))

    echo ""
    echo "--- Throughput: ${scenario} (input=${input_len}, output=${output_len}, prompts=${num_prompts}, max_len=${max_model_len}) ---"

    vllm bench throughput \
        --model "${MODEL_PATH}" \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --num-prompts "${num_prompts}" \
        --trust-remote-code \
        --dtype auto \
        --enforce-eager \
        --max-model-len "${max_model_len}" \
        ${COMMON_OPTS} \
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

    max_model_len=$(( input_len + output_len + 1024 ))

    echo ""
    echo "--- Latency: ${scenario} (input=${input_len}, output=${output_len}, batch=${batch_size}, iters=${num_iters}, max_len=${max_model_len}) ---"

    vllm bench latency \
        --model "${MODEL_PATH}" \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --batch-size "${batch_size}" \
        --num-iters "${num_iters}" \
        --trust-remote-code \
        --dtype auto \
        --enforce-eager \
        --max-model-len "${max_model_len}" \
        ${COMMON_OPTS} \
        --output-json "${output_file}"

    echo "Saved: ${output_file}"
done

echo ""
echo "==================== BENCHMARK COMPLETED ===================="
echo "All results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Files generated:"
ls -la "${OUTPUT_DIR}"/*.json

# Collect and summarize results (if the python script exists)
echo ""
echo "=== Collecting benchmark results... ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/collect_benchmark_results.py" ]; then
    python3 "${SCRIPT_DIR}/collect_benchmark_results.py" "${OUTPUT_DIR}"
else
    echo "collect_benchmark_results.py not found, skipping summary."
fi