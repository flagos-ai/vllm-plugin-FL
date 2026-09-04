#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

git config --global --add safe.directory "$(pwd)"

# FlagOps benchmark report upload uses jq to build JSON payloads.
apt-get update && apt-get install -y --no-install-recommends jq

uv pip install --system --upgrade pip
uv pip install --system --no-build-isolation -e ".[test]"
