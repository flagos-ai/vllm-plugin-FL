#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install this checkout without changing the image's native Ascend vLLM/FL runtime.
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
