#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

git config --global --add safe.directory "$(pwd)"

VLLM_DIST_VERSION="$(python -c 'from importlib.metadata import version; print(version("vllm"))' 2>/dev/null || true)"

if [[ "${VLLM_DIST_VERSION}" == "0.32.0+flagos" ]]; then
    # FlagOS Runtime 2.3.0 (CANN 9.2.1) + vLLM 0.32.0+flagos stack
    python "$HERE/../install_flaggems_vllm.py"

    # Preserve Runtime PYTHONPATH=/opt/flagtree compiler selection (do not pass -I).
    python -B "$HERE/stack.py" --runtime-only

    if [[ ! -e .git && -z "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL:-}" ]]; then
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL="$(python -c 'from importlib.metadata import PackageNotFoundError, version
try:
    print(version("vllm-plugin-fl"))
except PackageNotFoundError:
    print("0.5.0.dev0")')"
        export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL
    fi

    env -u VLLM_VENDOR uv pip install \
        --python "$(python -c 'import sys; print(sys.executable)')" \
        --break-system-packages \
        --no-build-isolation \
        --no-deps \
        -e .

    python - <<'PY'
from importlib.metadata import version

for package in ("pytest", "pytest-xdist", "pytest-cov", "pytest-json-report", "pytest-timeout"):
    print(f"CI tool: {package}=={version(package)}")
PY

    python -B "$HERE/stack.py"
    exit 0
fi

pip install \
    --no-build-isolation \
    --no-deps \
    -e .
pip install "pytest-xdist>=3.0.0"

python .github/scripts/install_flaggems_vllm.py

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import numpy

for package, expected_version in {
    "vllm": "0.28.0+empty",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
    "flagtree": "0.6.2a1+ascend3.5",
    "cann-shmem": "1.6.0",
    "pytest": "8.3.2",
    "coverage": "7.16.1",
    "pytest-cov": "7.1.0",
    "pytest-json-report": "1.5.0",
    "pytest-metadata": "3.1.1",
    "pytest-timeout": "2.4.0",
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
