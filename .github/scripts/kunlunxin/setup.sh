#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Setup script for Kunlunxin P800 CI environment.
set -euo pipefail

if [[ -d /root/miniconda/envs/python310_torch29_cuda/bin ]]; then
  export PATH="/root/miniconda/envs/python310_torch29_cuda/bin:${PATH}"
fi

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${FLAGCX_PATH:?FLAGCX_PATH is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"
: "${USE_RESHAPE_AND_CACHE_FLASH:?USE_RESHAPE_AND_CACHE_FLASH is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -d "${FLAGCX_PATH}/build/lib" ]]; then
  export LD_LIBRARY_PATH="${FLAGCX_PATH}/build/lib:/opt/kunlun/lib:${LD_LIBRARY_PATH:-}"
else
  echo "::warning::${FLAGCX_PATH}/build/lib not found; FlagCX may be unavailable."
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH \
    LD_LIBRARY_PATH \
    GEMS_VENDOR \
    VLLM_PLUGINS \
    FLAGCX_PATH \
    CUDA_VISIBLE_DEVICES \
    USE_RESHAPE_AND_CACHE_FLASH \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS; do
    echo "${name}=${!name-}" >> "${GITHUB_ENV}"
  done
fi

# vLLM, FlagGems, FlagCX, and vendor runtime dependencies are provided by the
# Kunlunxin image. Install only the checked-out plugin source for this run.
python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import os
import sys

flagcx_path = os.environ.get("FLAGCX_PATH", "")
if flagcx_path and flagcx_path not in sys.path:
    sys.path.insert(0, flagcx_path)

import flag_gems
import torch
import vllm
import vllm_fl
from vllm.platforms import current_platform

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"FlagGems vendor: {getattr(flag_gems, 'vendor_name', 'auto-detected')}")
print(f"Torch import ok: {torch.__version__}")
print(f"CUDA-compatible accelerator available: {torch.cuda.is_available()}")
print(f"CUDA-compatible accelerator count: {torch.cuda.device_count()}")
print(f"Platform: {current_platform}")

try:
    import flagcx  # noqa: F401

    print("FlagCX import ok")
except Exception as exc:
    print(f"::warning::FlagCX import failed: {exc}")
PY
