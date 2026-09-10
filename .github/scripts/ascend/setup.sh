#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

pip install --upgrade pip "setuptools>=77.0.3,<80"
pip install \
    --no-build-isolation \
    --no-deps \
    -e .

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import numpy

for package, expected_version in {
    "vllm": "0.28.0+empty",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
    "flagtree": "0.6.2a1+ascend3.5",
    "cann-shmem": "1.6.0",
}.items():
    actual = version(package)
    if actual != expected_version:
        raise RuntimeError(
            f"Unexpected {package} version: {actual}; expected {expected_version}. "
            "Rebuild the Ascend 0.28.0 image before running CI."
        )
    print(f"{package}: {actual}")

# FlagTree owns the triton module. Another distribution can overwrite it.
for package in ("triton", "triton-ascend"):
    try:
        installed = version(package)
    except PackageNotFoundError:
        continue
    raise RuntimeError(f"Remove standalone {package}=={installed}; use FlagTree")

expected = "1.26.4"
if numpy.__version__ != expected:
    raise RuntimeError(
        f"Unexpected NumPy version: {numpy.__version__}; expected {expected}"
    )
print(f"NumPy version: {numpy.__version__}")
PY
