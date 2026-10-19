#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

uv pip install --system --break-system-packages --upgrade pip

# Existing CI images may contain an older FlagGems release.
if ! python3 .github/scripts/cuda/check_flaggems.py; then
    uv pip install --system --break-system-packages --no-build-isolation --no-deps \
        "flag-gems @ git+https://github.com/flagos-ai/FlagGems.git@v5.4.0"
fi

uv pip install --system --break-system-packages --no-build-isolation -e ".[test]"

python3 .github/scripts/install_flaggems_vllm.py

python3 .github/scripts/cuda/check_flaggems.py
