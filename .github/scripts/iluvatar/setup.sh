#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install the checked-out plugin into the published Iluvatar CI image.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH:-}"
# Do not let a baked-in checkout shadow the source under test.
unset PYTHONPATH

: "${GEMS_VENDOR:?GEMS_VENDOR is not set}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

git config --global --add safe.directory "$(pwd)"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo 'PYTHONPATH=' >> "${GITHUB_ENV}"
  for name in PATH VLLM_PLUGINS USE_FLAGGEMS GEMS_VENDOR; do
    if [[ -n "${!name:-}" ]]; then
      echo "${name}=${!name}" >> "${GITHUB_ENV}"
    fi
  done
fi

python -m pip install --no-build-isolation --no-deps -e .
