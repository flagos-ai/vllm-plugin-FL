#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install the checkout into the Iluvatar CI Runtime.
set -euo pipefail

export PATH="/flagos/bin:$PATH"
# Expose the Runtime Triton without putting a baked-in checkout on sys.path.
export PYTHONPATH=/opt/flagtree
: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

git config --global --add safe.directory "$PWD"
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check

if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'PATH=%s\nPYTHONPATH=%s\n' "$PATH" "$PYTHONPATH" >> "$GITHUB_ENV"
fi
