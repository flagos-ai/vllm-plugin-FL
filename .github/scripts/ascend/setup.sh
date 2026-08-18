#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -euo pipefail

# Install FlagGems-vllm for test purpose
FLAGGEMS_VLLM_DIR="$(cd .. && pwd -P)/FlagGems-vllm"
git clone https://github.com/flagos-ai/FlagGems-vllm.git "${FLAGGEMS_VLLM_DIR}"
git -C "${FLAGGEMS_VLLM_DIR}" checkout main
python -m pip install --no-build-isolation -e "${FLAGGEMS_VLLM_DIR}"

git config --global --add safe.directory "$(pwd)"

pip install --upgrade pip "setuptools>=77.0.3"

# Install vLLM-Plugin-FL
pip install \
    --no-build-isolation \
    --no-deps \
    -e .

python - <<'PY'
import flag_gems
import flaggems_vllm
import numpy
import torch
import torch_npu  # noqa: F401
import vllm
import vllm_fl

expected = "1.26.4"
if numpy.__version__ != expected:
    raise RuntimeError(
        f"Unexpected NumPy version: {numpy.__version__}; expected {expected}"
    )
print(f"NumPy version: {numpy.__version__}")
print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"FlagGems-vllm grouped_topk: {callable(flaggems_vllm.grouped_topk)}")
print(f"Torch import ok: {torch.__version__}")
print(f"Accelerator available: {torch.npu.is_available()}")
print(f"Accelerator count: {torch.npu.device_count()}")
PY
