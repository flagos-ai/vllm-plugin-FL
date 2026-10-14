#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../lib/network.sh"
fl_net_hardening

git config --global --add safe.directory "$(pwd)"

fl_net_retry uv pip install --system --break-system-packages --upgrade pip
fl_net_retry uv pip install --system --break-system-packages --no-build-isolation -e ".[test]"
