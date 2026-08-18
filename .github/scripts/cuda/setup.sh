#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

FLAGGEMS_VLLM_DIR="$(cd .. && pwd -P)/FlagGems-vllm"
git clone https://github.com/flagos-ai/FlagGems-vllm.git "${FLAGGEMS_VLLM_DIR}"
git -C "${FLAGGEMS_VLLM_DIR}" checkout main

git config --global --add safe.directory "$(pwd)"

uv pip install --system --upgrade pip
uv pip install --system --no-build-isolation -e ".[test]"
