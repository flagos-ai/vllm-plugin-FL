#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Setup script for Moore Threads MUSA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${MTHREADS_VISIBLE_DEVICES:?MTHREADS_VISIBLE_DEVICES is not set}"

python -m pip install --no-build-isolation --no-deps -e .
python -m pip install "pytest-xdist>=3.0.0"

python .github/scripts/install_flaggems_vllm.py

python - <<'PY'
from importlib.metadata import version as dist_version

import flag_gems
import flagtree  # noqa: F401 - provides the Triton build
import torch
import torch_musa
import triton
import vllm
import vllm_fl
from vllm.platforms import current_platform

# The MUSA CI image is pinned to the vLLM 0.32 adaptation stack; fail loudly
# instead of silently testing a different dependency set.
assert vllm.__version__.startswith("0.32."), vllm.__version__
assert flag_gems.__version__.startswith("5.4."), flag_gems.__version__
# ``flagtree.__version__`` mirrors Triton (3.6.0); the distribution carries the
# 0.7.x FlagTree build identifier.
assert dist_version("flagtree").startswith("0.7."), dist_version("flagtree")
assert triton.__version__.startswith("3.6."), triton.__version__
assert torch.__version__.startswith("2.9."), torch.__version__

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
