#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the original checkpoint directory}"
: "${MASTER_ADDR:?}" "${NODE_RANK:?}" "${NODE_IP:?}" "${DIST_IFACE:?}"

case "$NODE_RANK" in
  0) frontend=(--api-server-count 1 --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-18317}") ;;
  1) frontend=(--headless) ;;
  *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;;
esac

if [[ -n "${SOURCE_ROOT:-}" ]]; then
  export PYTHONPATH="$SOURCE_ROOT/vllm-plugin-FL:$SOURCE_ROOT/FlagGems-vllm/src:$SOURCE_ROOT/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}"
fi
export GEMS_VENDOR=metax VLLM_PLUGINS=fl VLLM_USE_V2_MODEL_RUNNER=1 USE_FLAGTUNE=0 VLLM_FL_STRICT=1
export VLLM_HOST_IP="$NODE_IP" OMP_NUM_THREADS=4
export MCCL_SOCKET_IFNAME="$DIST_IFACE" NCCL_SOCKET_IFNAME="$DIST_IFACE" GLOO_SOCKET_IFNAME="$DIST_IFACE"
export MCCL_IB_DISABLE=1 NCCL_IB_DISABLE=1 MCCL_GRAPH_MIXING_SUPPORT=0 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
unset MCCL_P2P_DISABLE NCCL_P2P_DISABLE MCCL_BUFFSIZE NCCL_BUFFSIZE NCCL_GRAPH_MIXING_SUPPORT
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${VLLM_PYTHON:-python3}" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --config "$example_dir/metax_dp8.yaml" --node-rank "$NODE_RANK" \
  --data-parallel-address "$MASTER_ADDR" --data-parallel-rpc-port "${DP_PORT:-29931}" \
  --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29933}" "${frontend[@]}" "$@"
