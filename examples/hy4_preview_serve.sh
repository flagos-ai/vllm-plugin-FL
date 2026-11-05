#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# NVIDIA H100, two nodes, TP16/EP16, vLLM 0.28 and pinned Hy4 dependencies.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 CHECKPOINT MASTER_ADDR NODE_RANK [additional vllm arguments]"
    echo "Run on both nodes with the same checkpoint and master address; ranks are 0 and 1."
    exit 0
fi
if [[ $# -lt 3 || ! "$3" =~ ^[01]$ ]]; then
    echo "Expected CHECKPOINT MASTER_ADDR and NODE_RANK (0 or 1)." >&2
    exit 2
fi
model_path="$1"
master_addr="$2"
node_rank="$3"
shift 3
export VLLM_PLUGINS="${VLLM_PLUGINS:-fl}"
export USE_FLAGGEMS="${USE_FLAGGEMS:-1}"
export HY4_INDEXER_DTYPE="${HY4_INDEXER_DTYPE:-int8}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"
export FLAGGEMS_TOPK_K2048_TLE_V2="${FLAGGEMS_TOPK_K2048_TLE_V2:-1}"
export FLAG_GEMS_GROUPED_TOPK_SCORE_FUSION="${FLAG_GEMS_GROUPED_TOPK_SCORE_FUSION:-1}"
export FLAG_GEMS_GROUPED_TOPK_SINGLE_GROUP_FUSION="${FLAG_GEMS_GROUPED_TOPK_SINGLE_GROUP_FUSION:-1}"
export FLAG_GEMS_W8A8_QUANT_FUSION="${FLAG_GEMS_W8A8_QUANT_FUSION:-quant_silu}"
python -c "from vllm.v1.worker.model_runtime import register_model_runtime"
exec vllm serve "$model_path" \
    --tensor-parallel-size 16 --enable-expert-parallel --disable-custom-all-reduce \
    --nnodes 2 --node-rank "$node_rank" --master-addr "$master_addr" \
    --master-port 29500 --distributed-executor-backend mp \
    --all2all-backend allgather_reducescatter \
    --dtype bfloat16 --block-size 64 --max-model-len 204800 \
    --max-num-seqs 64 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --attention-backend FLASHMLA_SPARSE \
    --kv-cache-dtype fp8_ds_mla \
    --compilation-config '{"compile_mm_encoder":false,"cudagraph_mm_encoder":false,"cudagraph_capture_sizes":[4,8,16,32,64,128,256,512,1024,2048],"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_num_of_warmups":1,"max_cudagraph_capture_size":2048,"mode":"NONE"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_tensor_parallel_size":16,"draft_sample_method":"greedy","rejection_sample_method":"standard"}' \
    --limit-mm-per-prompt '{"image":16,"video":0}' \
    --reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice "$@"
