#!/bin/bash
# Step 9: Benchmark verification
# Usage: bash scripts/benchmark.sh <MODEL_DISPLAY_NAME>
set -euo pipefail

MODEL="${1:?Usage: benchmark.sh <MODEL_DISPLAY_NAME>}"

vllm bench throughput \
    --model "/models/${MODEL}" \
    --dataset-name random \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 2 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --load-format dummy \
    --max-num-seqs 10 \
    --trust-remote-code
