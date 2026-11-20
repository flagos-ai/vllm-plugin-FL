#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    cat <<'EOF'
Run on both MetaX C550 hosts with eight accelerators each.
Set MODEL_DIR to the local HY4-Preview W8A8 checkpoint, NODE_RANK to 0 or 1,
NODE_IP to this host's address, and MASTER_ADDR to rank 0's address.
Set MCCL_SOCKET_IFNAME, NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME as needed
for the cluster. Optional overrides: PORT (8000), MASTER_PORT (29574),
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
export NCCL_IB_DISABLE=1 MCCL_IB_DISABLE=1
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
