#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

uv pip install \
    --system \
    --break-system-packages \
    --no-build-isolation \
    --no-deps \
    -e .

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import flag_gems
import triton
import vllm

assert vllm.__version__.startswith("0.28.0"), vllm.__version__
assert version("flagtree") == "0.6.2a1", version("flagtree")
assert triton.__version__.startswith("3.6.0"), triton.__version__
try:
    standalone_triton = version("triton")
except PackageNotFoundError:
    standalone_triton = None
assert standalone_triton is None, standalone_triton

print(f"vLLM: {vllm.__version__}")
print(f"FlagTree: {version('flagtree')}")
print(f"Triton: {triton.__version__} (provided by FlagTree)")
print(f"FlagGems: {flag_gems.__version__}")
PY
