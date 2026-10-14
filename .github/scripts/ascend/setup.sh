#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -euo pipefail

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../lib/network.sh"
fl_net_hardening

git config --global --add safe.directory "$(pwd)"

fl_net_retry pip install --upgrade pip "setuptools>=77.0.3"
fl_net_retry pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import numpy

expected = "1.26.4"
if numpy.__version__ != expected:
    raise RuntimeError(
        f"Unexpected NumPy version: {numpy.__version__}; expected {expected}"
    )
print(f"NumPy version: {numpy.__version__}")
PY
