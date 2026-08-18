#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -euo pipefail

FLAGGEMS_VLLM_DIR="$(cd .. && pwd -P)/FlagGems-vllm"
git clone https://github.com/flagos-ai/FlagGems-vllm.git "${FLAGGEMS_VLLM_DIR}"
git -C "${FLAGGEMS_VLLM_DIR}" checkout main

git config --global --add safe.directory "$(pwd)"

pip install --upgrade pip "setuptools>=77.0.3"
pip install \
    --no-build-isolation \
    --no-deps \
    -e .

python - <<'PY'
import numpy

expected = "1.26.4"
if numpy.__version__ != expected:
    raise RuntimeError(
        f"Unexpected NumPy version: {numpy.__version__}; expected {expected}"
    )
print(f"NumPy version: {numpy.__version__}")
PY
