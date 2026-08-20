#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Setup script for Moore Threads MUSA CI environment.
set -euo pipefail

# Install FlagGems for test purpose
FLAGGEMS_VERSION="v5.3.4"
FLAGGEMS_DIR="$(cd .. && pwd -P)/FlagGems"
rm -rf "${FLAGGEMS_DIR}"
git clone --branch "${FLAGGEMS_VERSION}" --depth 1 https://github.com/flagos-ai/FlagGems.git "${FLAGGEMS_DIR}"
python -m pip install --no-build-isolation -e "${FLAGGEMS_DIR}"

# Install FlagGems-vllm for test purpose
FLAGGEMS_VLLM_VERSION="v0.1.1-rc0"
FLAGGEMS_VLLM_DIR="$(cd .. && pwd -P)/FlagGems-vllm"
rm -rf "${FLAGGEMS_VLLM_DIR}"
git clone --branch "${FLAGGEMS_VLLM_VERSION}" --depth 1 https://github.com/flagos-ai/FlagGems-vllm.git "${FLAGGEMS_VLLM_DIR}"
python -m pip install --no-build-isolation --no-deps -e "${FLAGGEMS_VLLM_DIR}"

git config --global --add safe.directory "$(pwd)"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${MTHREADS_VISIBLE_DEVICES:?MTHREADS_VISIBLE_DEVICES is not set}"

# Install vLLM-Plugin-FL
python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import flag_gems
import flaggems_vllm
import torch
import torch_musa
import vllm
import vllm_fl
from vllm.platforms import current_platform

assert torch.musa.is_available(), "MUSA accelerator is unavailable"
assert torch.musa.device_count() > 0, "No MUSA devices detected"
assert current_platform.device_type == "musa", current_platform.device_type

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"FlagGems-vllm grouped_topk: {callable(flaggems_vllm.grouped_topk)}")
print(f"Torch import ok: {torch.__version__}")
print(f"MUSA available: {torch.musa.is_available()}")
print(f"MUSA devices: {torch.musa.device_count()}")
print(f"Platform: {current_platform}")
PY
