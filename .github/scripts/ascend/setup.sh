#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -euo pipefail

# Initialize CANN Toolkit and ATB environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
source /usr/local/Ascend/nnal/atb/set_env.sh

# Install vLLM plugins and dependencies
pip install --upgrade pip "setuptools>=77.0.3"
pip install --no-build-isolation -e ".[test]"
