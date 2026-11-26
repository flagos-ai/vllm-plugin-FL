#!/bin/bash
# Copyright 2026 FlagOS Contributors
# Install the checkout into the T-Head CI Runtime.
set -euo pipefail

export TORCH_DEVICE_BACKEND_AUTOLOAD=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_VENDOR="" VLLM_TARGET_DEVICE=empty
export VLLM_PLUGINS=fl GEMS_VENDOR=thead USE_FLAGGEMS=1
export FL_CONCURRENT_MAX_TOKENS="${FL_CONCURRENT_MAX_TOKENS:-256}"
unset FL_BACKEND PYTHONHOME

cache_dir="$(mktemp -d /tmp/thead-ci.XXXXXXXX)"
export XDG_CACHE_HOME="$cache_dir" FLAGGEMS_CACHE_DIR="$cache_dir/flaggems"
export TRITON_CACHE_DIR="$cache_dir/triton" HF_HOME="$cache_dir/huggingface"
mkdir -p "$FLAGGEMS_CACHE_DIR" "$TRITON_CACHE_DIR"

git config --global --add safe.directory "$PWD"
python .github/scripts/install_flaggems_vllm.py --source-archive
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check

python - <<'PYCODE'
from pathlib import Path

import torch
import triton
from vllm.platforms import current_platform

assert Path(triton.__file__).resolve().is_relative_to("/opt/flagtree"), triton.__file__
assert torch.cuda.is_available() and torch.cuda.device_count() == 4
assert current_platform.vendor_name == "thead", current_platform
assert current_platform.dist_backend == "flagcx", current_platform
PYCODE

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in TORCH_DEVICE_BACKEND_AUTOLOAD \
    VLLM_WORKER_MULTIPROC_METHOD VLLM_VENDOR VLLM_TARGET_DEVICE VLLM_PLUGINS \
    GEMS_VENDOR USE_FLAGGEMS FL_CONCURRENT_MAX_TOKENS XDG_CACHE_HOME \
    FLAGGEMS_CACHE_DIR TRITON_CACHE_DIR HF_HOME; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
