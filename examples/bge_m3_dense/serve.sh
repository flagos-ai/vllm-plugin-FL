#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
# MetaX C550 FP16 dense profile; all model environment settings are process-local.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [MODEL_PATH] [vllm serve options...]"
  echo "Default model: /models/bge-m3; host: 127.0.0.1; port: 8000"
  exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
model_path=/models/bge-m3
if [[ $# -gt 0 && "$1" != --* ]]; then
  model_path="$1"
  shift
fi

# Slow FGV flash_attn_varlen_func: exclude its default attention backend and
# select MetaX flash_attn.flash_attn_interface.flash_attn_varlen_func via YAML.
# Preserve existing blacklist entries; use blacklist mode for this process.
exec env -u VLLM_FL_FLAGOS_WHITELIST \
  USE_FLAGGEMS=1 GEMS_VENDOR=metax \
  VLLM_FL_PREFER_ENABLED=1 VLLM_FL_PREFER=flagos \
  VLLM_FL_USE_FLAGGEMS_VLLM=1 VLLM_FL_USE_FLAGGEMS_ATTN=1 \
  VLLM_PLUGINS=fl FLAGTREE_AABS=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  VLLM_FL_CONFIG="$repo_root/vllm_fl/dispatch/config/bge_m3_metax_fp16_dense.yaml" \
  VLLM_FL_FLAGOS_BLACKLIST_APPEND="${VLLM_FL_FLAGOS_BLACKLIST_APPEND:+${VLLM_FL_FLAGOS_BLACKLIST_APPEND},}attention_backend" \
  vllm serve "$model_path" --config "$script_dir/serve.yaml" \
    --host 127.0.0.1 --port 8000 "$@"
