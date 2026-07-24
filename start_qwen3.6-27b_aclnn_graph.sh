#!/bin/bash
# New test: aclnn chunk_gated_delta_rule, graph mode

export VLLM_FL_USE_ACLNN_CHUNK_GDN=1
# 3 features: all ON
export VLLM_FL_DISABLE_CONV1D_PREPACK=0
export VLLM_FL_ENABLE_MM_AR_RMSNORM=1
export VLLM_FL_DISABLE_NPU_SLOT_MAPPING=0

echo "=========================================="
echo "  Running with aclnn chunk_gated_delta_rule, graph mode"
echo "=========================================="
echo "VLLM_FL_USE_ACLNN_CHUNK_GDN=${VLLM_FL_USE_ACLNN_CHUNK_GDN}"
echo "VLLM_FL_DISABLE_CONV1D_PREPACK=${VLLM_FL_DISABLE_CONV1D_PREPACK}"
echo "VLLM_FL_ENABLE_MM_AR_RMSNORM=${VLLM_FL_ENABLE_MM_AR_RMSNORM}"
echo "VLLM_FL_DISABLE_NPU_SLOT_MAPPING=${VLLM_FL_DISABLE_NPU_SLOT_MAPPING}"
echo ""

/workspace/scripts/run_vllm_fl_profile_unified.sh \
    --model-path /models/Qwen3.6-27B \
    --model-name qwen3.6 \
    --model-tag qwen3.6-27b \
    --mode graph \
    --cudagraph-mode FULL \
    --cases "1024,1024,256;4096,1024,256;16384,1024,256;65536,1024,256" \
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
    --run-label aclnn_chunk_3feat_on_c64_4case_i1k_4k_16k_64k_graph_noprof
