#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for MetaX C550 CI environment.
set -euo pipefail

# Install FlagGems-vllm for test purpose
FLAGGEMS_VLLM_DIR="$(cd .. && pwd -P)/FlagGems-vllm"
rm -rf "${FLAGGEMS_VLLM_DIR}"
git clone https://github.com/flagos-ai/FlagGems-vllm.git "${FLAGGEMS_VLLM_DIR}"
git -C "${FLAGGEMS_VLLM_DIR}" checkout main
python -m pip install --no-build-isolation -e "${FLAGGEMS_VLLM_DIR}"

export PATH="/opt/conda/bin:${PATH}"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${MACA_VISIBLE_DEVICES:?MACA_VISIBLE_DEVICES is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH \
    GEMS_VENDOR \
    VLLM_PLUGINS \
    MACA_VISIBLE_DEVICES; do
    echo "${name}=${!name}" >> "${GITHUB_ENV}"
  done
fi

# Install vLLM-Plugin-FL
# vLLM and test dependencies are provided by the CI image.
# Only install the checked-out plugin source for this workflow run.
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
