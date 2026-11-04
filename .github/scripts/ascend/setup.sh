#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Use the dependency image built by docker/ascend/Dockerfile.
set -euo pipefail

export VLLM_USE_V2_MODEL_RUNNER=1

git config --global --add safe.directory "$(pwd)"
python docker/ascend/runtime_constraints.py verify /opt/ascend-stack \
    --vllm-version 0.32.0 --check-imports

python -m pip install --no-build-isolation --no-deps -e .
# Resolve test tools with the captured NPU stack overrides as well.
uv --config-file /opt/ascend-stack/uv.toml pip install \
    --python "$(command -v python)" 'pytest-xdist>=3.0.0'
python .github/scripts/install_flaggems_vllm.py
python docker/ascend/runtime_constraints.py verify /opt/ascend-stack \
    --vllm-version 0.32.0 --check-imports
