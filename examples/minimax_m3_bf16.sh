#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${MINIMAX_M3_BF16_MODEL:?Set the BF16 checkpoint path}"
: "${NODE_IP:?Set the local Ascend node address}"
: "${COMM_IFACE:?Set the communication interface}"
: "${HEAD_ADDR:?Set the head node address}"

export VLLM_PLUGINS=fl USE_FLAGGEMS="${USE_FLAGGEMS:-1}" VLLM_USE_BREAKABLE_CUDAGRAPH=0
if [[ "$USE_FLAGGEMS" == 1 ]]; then
  unset VLLM_FL_FLAGOS_BLACKLIST
  export VLLM_FL_FLAGOS_WHITELIST=rsqrt
fi
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export VLLM_ENGINE_READY_TIMEOUT_S=3000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_HOST_IP="$NODE_IP" HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE"
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

node_args=()
if [[ "${DP_RANK:-0}" == 1 ]]; then
  node_args+=(--headless --data-parallel-start-rank 1)
fi

exec vllm serve "$MINIMAX_M3_BF16_MODEL" \
  --served-model-name minimax-m3-bf16 --trust-remote-code \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --tensor-parallel-size 16 --data-parallel-size 2 --enable-expert-parallel \
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser minimax_m3 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --data-parallel-size-local 1 --data-parallel-address "$HEAD_ADDR" \
  --data-parallel-rpc-port "${DP_RPC_PORT:-18501}" \
  --data-parallel-backend mp --distributed-executor-backend mp \
  "${node_args[@]}" "$@"
