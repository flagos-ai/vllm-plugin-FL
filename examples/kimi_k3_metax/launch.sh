#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    printf '%s\n' "Usage: $0 --model PATH --node-rank 0..3 --master-addr IP --cache DIR --net-interface IFACE --ib-hca HCAS [--master-port 29864] [--host 127.0.0.1] [--port 8000] [--gid-index 3] [--dry-run]"
}
die() { printf '%s\n' "$*" >&2; exit 2; }
model="" node_rank="" master_addr="" cache="" net_interface="" ib_hca=""
master_port=29864 host=127.0.0.1 port=8000 gid_index=3 dry_run=0
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=1; shift; continue ;;
    esac
    (($# >= 2)) || die "Missing value for $1"
    case "$1" in
        --model) model=$2 ;;
        --node-rank) node_rank=$2 ;;
        --master-addr) master_addr=$2 ;;
        --master-port) master_port=$2 ;;
        --host) host=$2 ;;
        --port) port=$2 ;;
        --cache) cache=$2 ;;
        --net-interface) net_interface=$2 ;;
        --ib-hca) ib_hca=$2 ;;
        --gid-index) gid_index=$2 ;;
        *) die "Unknown option: $1" ;;
    esac
    shift 2
done
[[ -n $model && -n $master_addr && -n $cache && -n $net_interface && -n $ib_hca ]] || { usage >&2; exit 2; }
[[ $node_rank == [0-3] ]] || die "node-rank must be 0, 1, 2 or 3"
[[ -z ${FLAGCX_PATH:-} ]] || die "This configuration uses MCCL; unset FLAGCX_PATH."

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cache=$(realpath -m -- "$cache")
python_bin=$(command -v python3)
environment=(
    "VLLM_PLUGINS=fl" "USE_FLAGGEMS=1" "USE_FLAGTUNE=0"
    "VLLM_FL_PREFER_ENABLED=1" "VLLM_FL_PREFER=flagos" "VLLM_FL_STRICT=1"
    "VLLM_FL_CONFIG=$script_dir/dispatch.yaml" "OMP_NUM_THREADS=2"
    "K3_CACHE_ROOT=$cache" "VLLM_CACHE_ROOT=$cache/bootstrap/vllm"
    "HF_HOME=$cache/model_metadata" "TMPDIR=$cache/bootstrap/tmp"
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM=1" "VLLM_FLOAT32_MATMUL_PRECISION=highest"
    "VLLM_PP_LAYER_PARTITION=24,24,24,21"
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200" "VLLM_ENGINE_READY_TIMEOUT_S=10800"
    "GLOO_SOCKET_IFNAME=$net_interface"
)
for prefix in NCCL MCCL; do
    environment+=(
        "${prefix}_SOCKET_IFNAME=$net_interface" "${prefix}_IB_DISABLE=0"
        "${prefix}_NET=IB" "${prefix}_IB_HCA==$ib_hca"
        "${prefix}_IB_GID_INDEX=$gid_index" "${prefix}_MIN_NCHANNELS=16"
        "${prefix}_MAX_NCHANNELS=16" "${prefix}_MAX_P2P_NCHANNELS=16"
        "${prefix}_BUFFSIZE=1048576"
    )
done
api_servers=1
if [[ $node_rank != 0 ]]; then api_servers=0; fi
command=(
    "$python_bin" -I -m vllm.entrypoints.cli.main serve "$model"
    --config "$script_dir/serve.yaml"
    --node-rank "$node_rank" --api-server-count "$api_servers"
    --master-addr "$master_addr" --master-port "$master_port"
    --host "$host" --port "$port"
)
if [[ $node_rank != 0 ]]; then command+=(--headless); fi
if ((dry_run)); then
    printf 'export %q\n' "${environment[@]}"
    printf 'exec '; printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi
export "${environment[@]}"
mkdir -p -- "$K3_CACHE_ROOT" "$VLLM_CACHE_ROOT" "$HF_HOME" "$TMPDIR"
# Dependency/source checks run in the K3 plugin admission before model allocation.
exec "${command[@]}"
