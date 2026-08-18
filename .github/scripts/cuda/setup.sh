#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

# Install FlagGems-vllm for test purpose
FLAGGEMS_VLLM_DIR="$(cd .. && pwd -P)/FlagGems-vllm"
git clone https://github.com/flagos-ai/FlagGems-vllm.git "${FLAGGEMS_VLLM_DIR}"
git -C "${FLAGGEMS_VLLM_DIR}" checkout main
uv pip install --system --no-build-isolation -e "${FLAGGEMS_VLLM_DIR}"

git config --global --add safe.directory "$(pwd)"

uv pip install --system --upgrade pip

# Install vLLM-Plugin-FL
uv pip install --system --no-build-isolation -e ".[test]"

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
