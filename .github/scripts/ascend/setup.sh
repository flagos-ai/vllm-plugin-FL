#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Setup script for Ascend CI environment (installs only the checked-out plugin).
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

# Ensures FlagTree (Triton) stays importable even when the container env
# is stripped (mirror of the image's PYTHONPATH; harmless no-op otherwise).
export PYTHONPATH="/opt/flagtree${PYTHONPATH:+:${PYTHONPATH}}"

# Install only the checked-out vllm-plugin-FL; vllm and flaggems-vllm
# are pre-installed and constrained in the base image.
uv pip install --python /flagos/bin/python \
    --break-system-packages \
    --no-build-isolation \
    --no-deps \
    -e .
