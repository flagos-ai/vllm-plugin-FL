#!/bin/bash
set -e

# Arguments
MODEL_PATH=${1:?"Please provide model path, e.g.: ./run_eval.sh /workspace/Qwen3-4B/ hf_xxx"}
HF_TOKEN=${2:?"Please provide HF_TOKEN, e.g.: ./run_eval.sh /workspace/Qwen3-4B/ hf_xxx"}

# Environment variables
export HF_ENDPOINT=https://hf-mirror.com            # China mirror (can be removed for overseas)
export HF_ALLOW_CODE_EVAL=1                         # Required for humaneval/mbpp
export HF_DATASETS_TRUST_REMOTE_CODE=1              # Required for mgsm etc.
export HF_TOKEN=${HF_TOKEN}

echo "=== Starting evaluation for model: ${MODEL_PATH} ==="


declare -A TASKS_FEWSHOT=(
    # --- General Tasks ---
    ["bbh"]=3                     # 06:22

    # --- Math & STEM Tasks ---
    ["gsm8k"]=4                   # 01:27

    # --- Coding Tasks ---
    ["humaneval"]=0               # 00:31
    ["mbpp"]=3                    # 00:13
)

# Special tasks (non-standard fewshot format)
declare -A SPECIAL_TASKS=(
    ["mgsm"]="mgsm_direct_zh,mgsm_native_cot_zh"                                            # Chinese only
)

# Run standard evaluation tasks
for task in "${!TASKS_FEWSHOT[@]}"; do
    fewshot=${TASKS_FEWSHOT[$task]}
    echo ""
    echo "=== Evaluating task: ${task} (num_fewshot=${fewshot}) ==="

    if ! lm_eval --model vllm \
        --model_args "trust_remote_code=True,pretrained=${MODEL_PATH},enforce_eager=True" \
        --tasks "${task}" \
        --batch_size auto \
        --trust_remote_code \
        --output_path output \
        --num_fewshot "${fewshot}" \
        --confirm_run_unsafe_code; then
        echo "Task ${task} evaluation failed!"
        exit 1
    fi
    echo "=== Task ${task} completed ==="
done

# Run special tasks (need to specify subtasks)
for task in "${!SPECIAL_TASKS[@]}"; do
    subtasks=${SPECIAL_TASKS[$task]}
    echo ""
    echo "=== Evaluating task: ${task} (${subtasks}) ==="

    if ! lm_eval --model vllm \
        --model_args "trust_remote_code=True,pretrained=${MODEL_PATH},enforce_eager=True" \
        --tasks "${subtasks}" \
        --batch_size auto \
        --trust_remote_code \
        --output_path output \
        --confirm_run_unsafe_code; then
        echo "Task ${task} evaluation failed!"
        exit 1
    fi
    echo "=== Task ${task} completed ==="
done

echo ""
echo "=== All evaluations completed! ==="
echo "Results saved to output/ directory"
echo ""

# Collect and summarize results
echo "=== Collecting results... ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/collect_eval_results.py" output
