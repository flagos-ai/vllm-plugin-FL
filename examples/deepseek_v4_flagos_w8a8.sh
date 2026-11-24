#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# See deepseek_v4_flagos_w8a8.md for dependencies and a sample request.
set -euo pipefail

: "${DEEPSEEK_FLAGOS_W8A8_MODEL:?Set this to the FlagOS-Compressor W8A8 checkpoint path}"

export VLLM_PLUGINS=fl USE_FLAGGEMS=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_FL_FLAGOS_WHITELIST=sigmoid
export VLLM_ENGINE_READY_TIMEOUT_S=3000 OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export HCCL_BUFFSIZE=200 HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

exec vllm serve "$DEEPSEEK_FLAGOS_W8A8_MODEL" \
  --served-model-name deepseek-flagos-w8a8 --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel \
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
