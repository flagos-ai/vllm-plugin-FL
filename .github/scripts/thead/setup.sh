#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for T-Head PPU CI environment.
set -euo pipefail

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    GEMS_VENDOR \
    VLLM_PLUGINS; do
    echo "${name}=${!name}" >> "${GITHUB_ENV}"
  done
fi

# vLLM, FlagGems, and test dependencies are provided by the CI image.
# Only install the checked-out plugin source for this workflow run.
pip install --no-build-isolation --no-deps -e .

# Install FlagTree (Triton backend for T-Head PPU).
# See https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ppu
FLAGTREE_VERSION="${FLAGTREE_VERSION:-0.6.1+ppu3.6}"
FLAGOS_INDEX_URL="${FLAGOS_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
pip install --no-cache-dir \
    --index-url "${FLAGOS_INDEX_URL}" \
    "flagtree===${FLAGTREE_VERSION}"

python - <<'PY'
import flag_gems
import flagtree
import torch
import vllm
import vllm_fl

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"FlagTree import ok: {getattr(flagtree, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY
