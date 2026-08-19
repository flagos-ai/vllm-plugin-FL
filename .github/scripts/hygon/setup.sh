#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Hygon DCU CI environment.
set -euo pipefail

# Install FlagGems for test purpose
FLAGGEMS_VERSION="v5.4.0.dev0"
FLAGGEMS_DIR="$(cd .. && pwd -P)/FlagGems"
rm -rf "${FLAGGEMS_DIR}"
git clone --branch "${FLAGGEMS_VERSION}" --depth 1 https://github.com/flagos-ai/FlagGems.git "${FLAGGEMS_DIR}"
python -m pip install --no-build-isolation -e "${FLAGGEMS_DIR}"

# Install FlagGems-vllm for test purpose
FLAGGEMS_VLLM_VERSION="v0.1.1-rc0"
FLAGGEMS_VLLM_DIR="$(cd .. && pwd -P)/FlagGems-vllm"
rm -rf "${FLAGGEMS_VLLM_DIR}"
git clone --branch "${FLAGGEMS_VLLM_VERSION}" --depth 1 https://github.com/flagos-ai/FlagGems-vllm.git "${FLAGGEMS_VLLM_DIR}"
python -m pip install --no-build-isolation -e "${FLAGGEMS_VLLM_DIR}"

git config --global --add safe.directory "$(pwd)"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${DTK_HOME:?DTK_HOME is not set}"
: "${ROCM_PATH:?ROCM_PATH is not set}"
: "${HIP_PATH:?HIP_PATH is not set}"
: "${HSA_PATH:?HSA_PATH is not set}"
: "${HIP_CLANG_PATH:?HIP_CLANG_PATH is not set}"
: "${DEVICE_LIB_PATH:?DEVICE_LIB_PATH is not set}"
: "${LD_LIBRARY_PATH:?LD_LIBRARY_PATH is not set}"

unset VLLM_FL_IMAGE_PLUGIN_ROOT
unset HYGON_USE_IMAGE_PLUGIN

echo "DTK_HOME=${DTK_HOME}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
test -e "${HIP_PATH}/lib/libgalaxyhip.so.5"
test -e "${DTK_HOME}/llvm/lib/libomp.so"

# Install vLLM-Plugin-FL
python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import flag_gems
import flaggems_vllm
import torch
import vllm
import vllm_fl

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"FlagGems-vllm grouped_topk: {callable(flaggems_vllm.grouped_topk)}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY
