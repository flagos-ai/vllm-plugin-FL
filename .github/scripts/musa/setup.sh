#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Setup script for Moore Threads MUSA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${MTHREADS_VISIBLE_DEVICES:?MTHREADS_VISIBLE_DEVICES is not set}"

python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import flag_gems
import torch
import torch_musa
import vllm
import vllm_fl
from vllm.platforms import current_platform

assert torch.musa.is_available(), "MUSA accelerator is unavailable"
assert torch.musa.device_count() > 0, "No MUSA devices detected"
assert current_platform.device_type == "musa", current_platform.device_type

print(f"vLLM import ok: {vllm.__version__}")
print(f"vLLM-FL import ok: {vllm_fl.__file__}")
print(f"FlagGems import ok: {getattr(flag_gems, '__version__', 'unknown')}")
print(f"Torch import ok: {torch.__version__}")
print(f"MUSA available: {torch.musa.is_available()}")
print(f"MUSA devices: {torch.musa.device_count()}")
print(f"Platform: {current_platform}")
PY
