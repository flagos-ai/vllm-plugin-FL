#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

pip install --upgrade pip "setuptools>=77.0.3"
# Install the published empty-device wheel without replacing the image's
# Ascend torch/torch-npu/FlagTree stack. The previous 0.20 image lacks APIs
# used by the main-branch tests and imports TileLang kernels unconditionally.
pip install --no-deps \
    "https://resource.flagos.net/repository/flagos-pypi-ascend/packages/vllm/0.24.0+flagos/vllm-0.24.0+flagos-py3-none-any.whl#sha256=8a6404fb861a4ee47e39cf395e22413c129d70e4f5895efb2dd9c9e9c77502c0"
pip install \
    --no-build-isolation \
    --no-deps \
    -e .
pip install "pytest-xdist>=3.0.0"

python .github/scripts/install_flaggems_vllm.py

python - <<'PY'
from importlib.metadata import version

import numpy
# Load torch-npu before Triton; its backend imports torch during initialization.
import torch
import torch_npu  # noqa: F401
import triton

expected_vllm = "0.24.0+flagos"
if version("vllm") != expected_vllm:
    raise RuntimeError(f"Expected vLLM {expected_vllm}, found {version('vllm')}")
if not callable(getattr(triton, "jit", None)):
    raise RuntimeError("Ascend CI requires FlagTree's working triton.jit")
for package in ("vllm", "torch", "torch-npu", "flagtree", "flag-gems"):
    print(f"{package}: {version(package)}")

# npu-smi and the mapped /dev nodes are not enough: the image ships an empty
# ASCEND_RT_VISIBLE_DEVICES, which hides every device from the runtime. Fail
# here (with the count) instead of letting device-dependent tests no-op later.
device_count = torch.npu.device_count()
print(f"Ascend devices: {device_count}")
if device_count < 2:
    raise RuntimeError(
        "Ascend CI requires at least 2 visible NPU devices, found "
        f"{device_count}; check ASCEND_RT_VISIBLE_DEVICES"
    )
smoke = (torch.ones(2, 2) * 2).npu().sum()
torch.npu.synchronize()
if smoke.cpu().item() != 8.0:
    raise RuntimeError("Ascend NPU smoke computation returned an unexpected result")

expected = "1.26.4"
if numpy.__version__ != expected:
    raise RuntimeError(
        f"Unexpected NumPy version: {numpy.__version__}; expected {expected}"
    )
print(f"NumPy version: {numpy.__version__}")
PY
