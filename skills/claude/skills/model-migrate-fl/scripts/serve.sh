#!/bin/bash
# Step 10.1: Start vLLM server for verification
# Usage: bash scripts/serve.sh <MODEL_DISPLAY_NAME> [PORT]
set -euo pipefail

MODEL="${1:?Usage: serve.sh <MODEL_DISPLAY_NAME> [PORT]}"
PORT="${2:-8121}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve "/models/${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 10 \
    --load-format fastsafetensors \
    --trust-remote-code
