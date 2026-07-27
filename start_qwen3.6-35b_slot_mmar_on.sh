#!/bin/bash
# slot_mapping ON + mm_ar_rmsnorm ON (35B-A3B, gpu4-7)
# mm fusion threshold: M > 512 (VLLM_FL_MM_AR_RMSNORM_MIN_TOKENS).
# Rationale: decode steps have M <= concurrency (64) and must NOT fuse
# (the extra add_out all-gather cancels the win at small M); prefill chunks
# (up to 2048) fuse only above 512, same as the tuned npugraph_ex pass.

export VLLM_FL_USE_ACLNN_CHUNK_GDN=1
# features: slot_mapping ON, conv1d_prepack OFF, mm_ar_rmsnorm ON
export VLLM_FL_DISABLE_CONV1D_PREPACK=1
export VLLM_FL_ENABLE_MM_AR_RMSNORM=1
export VLLM_FL_DISABLE_NPU_SLOT_MAPPING=0
export VLLM_FL_MM_AR_RMSNORM_MIN_TOKENS=512

echo "=========================================="
echo "  slot_mapping ON + mm_ar_rmsnorm ON (35B-A3B)"
echo "=========================================="
echo "VLLM_FL_DISABLE_CONV1D_PREPACK=${VLLM_FL_DISABLE_CONV1D_PREPACK}"
echo "VLLM_FL_ENABLE_MM_AR_RMSNORM=${VLLM_FL_ENABLE_MM_AR_RMSNORM}"
echo "VLLM_FL_DISABLE_NPU_SLOT_MAPPING=${VLLM_FL_DISABLE_NPU_SLOT_MAPPING}"
echo "VLLM_FL_MM_AR_RMSNORM_MIN_TOKENS=${VLLM_FL_MM_AR_RMSNORM_MIN_TOKENS}"
echo ""

/workspace/scripts/run_vllm_fl_profile_unified.sh \
    --model-path /models/Qwen3.6-35B-A3B \
    --model-name qwen3.6 \
    --model-tag qwen3.6-35b-a3b \
    --mode graph \
    --cudagraph-mode FULL \
    --cases "1024,1024,256;4096,1024,256;16384,1024,256;65536,1024,256" \
    --concurrency 64 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --tp 4 \
    --gmem 0.9 \
    --devices 4,5,6,7 \
    --port 8123 \
    --no-bench-profile \
    --skip-analyse \
    --package none \
    --run-label slot_mmar_on_c64_4case_i1k_4k_16k_64k_graph_noprof
