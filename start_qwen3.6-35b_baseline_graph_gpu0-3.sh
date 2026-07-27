#!/bin/bash
# Baseline test: PTO + Triton (current implementation), graph mode (35B-A3B, gpu0-3)

export VLLM_FL_USE_ACLNN_CHUNK_GDN=0

echo "=========================================="
echo "  Running Baseline (PTO + Triton), graph mode (35B-A3B, gpu0-3)"
echo "=========================================="
echo "VLLM_FL_USE_ACLNN_CHUNK_GDN=${VLLM_FL_USE_ACLNN_CHUNK_GDN}"
echo ""

/workspace/scripts/run_vllm_fl_profile_unified.sh \
    --model-path /models/Qwen3.6-35B-A3B \
    --model-name qwen3.6 \
    --model-tag qwen3.6-35b-a3b \
    --mode graph \
    --cudagraph-mode FULL \
    --cases "1024,1024,256" \
    --concurrency 64 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --tp 4 \
    --gmem 0.9 \
    --devices 0,1,2,3 \
    --port 8122 \
    --no-bench-profile \
    --skip-analyse \
    --package none \
    --run-label baseline_pto_c64_i1024_o1024_np256_gpu03_graph_noprof
