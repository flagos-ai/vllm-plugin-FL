#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

uv pip install --system --break-system-packages --upgrade pip

# Existing CI images may contain an older FlagGems release.
if ! python3 .github/scripts/nvidia/check_flaggems.py; then
    uv pip install --system --break-system-packages --no-build-isolation --no-deps \
        "flag-gems @ git+https://github.com/flagos-ai/FlagGems.git@v5.4.0"
fi

uv pip install --system --break-system-packages --no-build-isolation -e ".[test]"

python3 .github/scripts/install_flaggems_vllm.py

uv pip install --system --break-system-packages "wheel==0.46.3"
python3 .github/scripts/nvidia/install_flashinfer_ci_fix.py

python3 .github/scripts/nvidia/check_flaggems.py

# TileLang's cudart stub must not hide the real CUDA runtime from FlashInfer.
python3 - <<'PY'
import tilelang  # noqa: F401
import torch
import flashinfer.comm  # noqa: F401

print(f"FlashInfer CUDA IPC import passed on {torch.cuda.get_device_name(0)}")
PY
