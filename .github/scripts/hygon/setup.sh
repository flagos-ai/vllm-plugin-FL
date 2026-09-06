#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Hygon DCU CI environment.
set -euo pipefail

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

python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import flag_gems
import torch
import vllm
import vllm_fl

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.cuda.is_available()}")
print(f"Accelerator count: {torch.cuda.device_count()}")
PY
