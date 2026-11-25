#!/bin/sh
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Direct native vLLM entry. MTP=off selects the validated throughput profile.
set -eu

: "${MODEL_PATH:?Set MODEL_PATH to the model directory}"
: "${QWEN38_CACHE_ROOT:?Set QWEN38_CACHE_ROOT to a service-specific cache directory}"
SCRIPT_DIR=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH="" cd -- "$SCRIPT_DIR/../.." && pwd)
MTP=${MTP:-off}
DRAFT=
case "$MTP" in
    off) CONFIG="$SCRIPT_DIR/metax.yaml" ;;
    k1) CONFIG="$SCRIPT_DIR/metax_mtp.yaml"; DRAFT=1; BUCKETS='[2,4,8,16,32,64,128]'; MAX_BUCKET=128 ;;
    k2) CONFIG="$SCRIPT_DIR/metax_mtp.yaml"; DRAFT=2; BUCKETS='[3,6,12,24,48,96,192]'; MAX_BUCKET=192 ;;
    k3) CONFIG="$SCRIPT_DIR/metax_mtp.yaml"; DRAFT=3; BUCKETS='[4,8,16,32,64,128,256]'; MAX_BUCKET=256 ;;
    *) printf '%s\n' 'MTP must be off, k1, k2 or k3' >&2; exit 2 ;;
esac

mkdir -p -- "$QWEN38_CACHE_ROOT/$MTP"
CACHE_DIR=$(CDPATH="" cd -- "$QWEN38_CACHE_ROOT/$MTP" && pwd)
export USE_FLAGGEMS=1 VLLM_FL_USE_FLAGGEMS_VLLM=1 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_FL_PREFER_ENABLED=1 VLLM_FL_PREFER=flagos VLLM_FL_OOT_ENABLED=1
export VLLM_FL_CONFIG="$REPO_ROOT/vllm_fl/dispatch/config/qwen3_8_flash_next_metax.yaml"
export VLLM_TUNED_CONFIG_FOLDER="$SCRIPT_DIR/metax_tuning/moe-configs"
export FLAGGEMS_METAX_TLE=0
export TRITON_CACHE_DIR="$CACHE_DIR/triton" GEMS_CACHE_PATH="$CACHE_DIR/gems"
export FLAGGEMS_DB_URL="${FLAGGEMS_DB_URL:-sqlite:///$CACHE_DIR/gems.db}"
export FLAGGEMS_ENABLE_OPLIST_PATH="$CACHE_DIR/enabled-ops.txt"
unset FLAGCX_PATH FLAGGEMS_FORCE_TLE

# vLLM 0.24 skips false YAML booleans; pass explicit negative CLI options.
set -- --no-enable-prefix-caching "$@"
if [ "$MTP" != off ]; then
    set -- --no-async-scheduling "$@"
fi
if [ -n "$DRAFT" ]; then
    set -- --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT,\"parallel_drafting\":false,\"use_local_argmax_reduction\":false,\"disable_padded_drafter_batch\":true}" \
        --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$BUCKETS,\"max_cudagraph_capture_size\":$MAX_BUCKET}" "$@"
fi
exec "${VLLM_BIN:-vllm}" serve "$MODEL_PATH" --config "$CONFIG" "$@"
