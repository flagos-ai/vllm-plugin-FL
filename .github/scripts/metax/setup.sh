#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for MetaX C550 CI environment.
set -euo pipefail

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

# vLLM, FlagGems, and test dependencies are provided by the CI image.
# Only install the checked-out plugin source for this workflow run.
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
