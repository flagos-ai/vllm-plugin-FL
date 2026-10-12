#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Iluvatar CoreX CI environment.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH:-}"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH \
    VLLM_PLUGINS \
    USE_FLAGGEMS \
    GEMS_VENDOR; do
    if [[ -n "${!name:-}" ]]; then
      echo "${name}=${!name}" >> "${GITHUB_ENV}"
    fi
  done
fi

# vLLM / FlagGems / torch come from the CI image.
# Install only the checked-out plugin source for this workflow run.
python -m pip install --no-build-isolation --no-deps -e .

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
assert vllm.__version__.startswith("0.28"), vllm.__version__

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Iluvatar devices: {torch.cuda.device_count()}")
print(f"Platform: {current_platform}")
PY
