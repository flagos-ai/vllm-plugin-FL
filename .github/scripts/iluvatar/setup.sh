#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Iluvatar CoreX CI environment.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH:-}"
# A baked-in site-packages PYTHONPATH can shadow the freshly installed vLLM.
unset PYTHONPATH

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  # Each workflow step starts a new shell; do not reintroduce the image path.
  echo 'PYTHONPATH=' >> "${GITHUB_ENV}"
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

bash .github/scripts/iluvatar/install_flagtree.sh
bash .github/scripts/iluvatar/install_vllm.sh
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install "pytest-xdist>=3.0.0"

# Match the validated Iluvatar development stack without changing other vendors.
python -m pip install --no-build-isolation --no-deps \
  'flag-gems @ git+https://github.com/flagos-ai/FlagGems.git@2bcfe2656dad0b654077fcfe7d66e6a93542afab' \
  'flaggems-vllm @ git+https://github.com/flagos-ai/FlagGems-vllm.git@b434eafb1715849d1019ddcd480c5fc39461665c'

python - <<'PY'
from importlib import metadata

import flag_gems
import torch
import vllm
import vllm_fl
from vllm.platforms import current_platform

assert metadata.version("flag-gems") == "5.5.0.dev0"
assert metadata.version("flaggems-vllm") == "0.3.0.dev1"
assert torch.cuda.is_available(), "Iluvatar accelerator is unavailable"
assert torch.cuda.device_count() >= 4, torch.cuda.device_count()
assert current_platform.device_type == "cuda", current_platform.device_type
assert current_platform.vendor_name == "iluvatar", current_platform.vendor_name
assert vllm.__version__.split('+')[0] == '0.32.0', vllm.__version__

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"Iluvatar devices: {torch.cuda.device_count()}")
print(f"Platform: {current_platform}")
PY
