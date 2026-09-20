#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Setup script for the Iluvatar CoreX CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

# CoreX exposes a CUDA-compatible runtime, but its nvcc is not NVIDIA CUDA
# and cannot be parsed by CMake's CUDA compiler detection.  Install the
# Python package without requesting the optional native extension build.
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install "pytest-xdist>=3.0.0"

python - <<'PY'
import flag_gems
import torch
import vllm
import vllm_fl
from vllm.platforms import current_platform

assert torch.cuda.is_available(), "Iluvatar accelerator is unavailable"
assert torch.cuda.device_count() >= 4, torch.cuda.device_count()
assert current_platform.device_type == "cuda", current_platform.device_type
assert current_platform.vendor_name == "iluvatar", current_platform.vendor_name

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Iluvatar devices: {torch.cuda.device_count()}")
print(f"Platform: {current_platform}")
PY
