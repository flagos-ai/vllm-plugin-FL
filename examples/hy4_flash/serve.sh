#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${1:?Usage: bash examples/hy4_flash/serve.sh MODEL_PATH_OR_ID}"

export VLLM_PLUGINS=fl
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_FL_CONFIG="$example_dir/dispatch.yaml"
export VLLM_FL_FLAGOS_WHITELIST="grouped_topk,hy4_flash_fwht_quant_fp8,hy4_flash_gated_pool,hy4_flash_rotate_indexer,hy4_flash_select_groups,hy4_flash_expand_groups,hy4_flash_write_pool,hy4_flash_prepare_groups,hy4_flash_write_tails"
export FLASH_MODE="${FLASH_MODE:-load}"

exec vllm serve "$model" \
  --served-model-name Hy4-Flash-FP8 \
  --host 127.0.0.1 --port 8000 \
  --tokenizer-mode hy_v4 --dtype bfloat16 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --distributed-executor-backend mp \
  --max-model-len 204800 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype auto \
  --no-enable-prefix-caching --limit-mm-per-prompt '{"image":16,"video":0}' \
  --reasoning-parser hy_v4 --enable-auto-tool-choice --tool-call-parser hy_v4 \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64]}' \
  --seed 1 --no-enable-log-requests
