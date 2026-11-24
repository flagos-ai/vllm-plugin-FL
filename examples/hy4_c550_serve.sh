#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    cat <<'EOF'
Run on both MetaX C550 hosts with eight accelerators each.
Set MODEL_DIR to the local HY4-Preview W8A8 checkpoint, NODE_RANK to 0 or 1,
NODE_IP to this host's address, and MASTER_ADDR to rank 0's address.
Collectives use RDMA on SOCKET_IFNAME (default inbond1). Set NET_TRANSPORT=tcp
to fall back to TCP, or SOCKET_IFNAME=auto to leave the interface variables
unset. Optional overrides: PORT (8000), MASTER_PORT (29574),
MAX_MODEL_LEN (4096), GPU_MEMORY_UTILIZATION (0.90).
EOF
    exit 0
fi

: "${MODEL_DIR:?Set MODEL_DIR to the HY4-Preview W8A8 checkpoint directory}"
: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
: "${NODE_IP:?Set NODE_IP to the local host address}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rank 0 address}"

case "$NODE_RANK" in
    0) set -- ;;
    1) set -- --headless ;;
    *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;;
esac
if [[ ! -r "$MODEL_DIR/config.json" ]]; then
    echo "Missing readable config.json in MODEL_DIR" >&2
    exit 2
fi

export GEMS_VENDOR=metax VLLM_PLUGINS=fl VLLM_HOST_IP="$NODE_IP"

SOCKET_IFNAME="${SOCKET_IFNAME:-inbond1}"
if [[ "$SOCKET_IFNAME" != auto ]]; then
    export MCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
    export NCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
    export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"
fi

# RDMA is the validated transport for a 16-way group across two C550 nodes.
# Forcing TCP raised ncclRemoteError on an ALLREDUCE during decode.
case "${NET_TRANSPORT:-rdma}" in
    rdma) export NCCL_IB_DISABLE=0 MCCL_IB_DISABLE=0 ;;
    tcp) export NCCL_IB_DISABLE=1 MCCL_IB_DISABLE=1 ;;
    *) echo "NET_TRANSPORT must be rdma or tcp" >&2; exit 2 ;;
esac
# The baseline uses vLLM's Triton INT8 linear for the compressed-tensors checkpoint.
# The FlagGems scaled_mm in the baseline MetaX runtime rejects these INT8 inputs.
export VLLM_FL_FLAGOS_BLACKLIST_APPEND="${VLLM_FL_FLAGOS_BLACKLIST_APPEND:+${VLLM_FL_FLAGOS_BLACKLIST_APPEND},}w8a8_dynamic_per_token_linear"

exec vllm serve "$MODEL_DIR" --served-model-name hy4-preview \
    --host 0.0.0.0 --port "${PORT:-8000}" \
    --tensor-parallel-size 16 --pipeline-parallel-size 1 \
    --distributed-executor-backend mp --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29574}" \
    --enable-expert-parallel --enable-ep-weight-filter \
    --all2all-backend allgather_reducescatter --load-format hy4_safetensors \
    --dtype bfloat16 --kv-cache-dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --max-num-batched-tokens 2048 --max-num-seqs 64 --enable-chunked-prefill \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" --enforce-eager \
    --no-enable-prefix-caching --disable-custom-all-reduce --seed 1 \
    "$@"
