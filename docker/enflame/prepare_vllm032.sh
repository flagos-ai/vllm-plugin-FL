#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Shared by the Enflame Docker build and CI setup.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VENV=${1:?Usage: prepare_vllm032.sh VENV_PATH}
BUILD_ROOT=$(mktemp -d)
SOURCE="$BUILD_ROOT/vllm"

# Refresh the same installer used by the NVIDIA setup. Keep the vendor SDK
# packages intact by installing the target runtime into a virtualenv.
python -m pip install --no-deps --index-url https://pypi.org/simple uv==0.13.6
python -m uv venv --system-site-packages --python "$(command -v python)" "$VENV"
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm.git "$SOURCE"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 5df63a7203cdb9328d824a27b682e77c7b9928be
(
    cd "$SOURCE"
    VLLM_TARGET_DEVICE=empty python -m uv pip install --python "$VENV/bin/python" \
        --no-build-isolation --no-deps .
)
python -m uv pip install --python "$VENV/bin/python" --no-deps --require-hashes \
    --index-url https://pypi.org/simple \
    -r "$SCRIPT_DIR/requirements-vllm032.txt"
"$VENV/bin/python" "$SCRIPT_DIR/verify_compiler.py"
"$VENV/bin/python" "$SCRIPT_DIR/verify_index_fill.py"
