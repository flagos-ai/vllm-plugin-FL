#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for CUDA CI environment.
set -euo pipefail

uv pip install --system --upgrade pip
uv pip install --system --no-build-isolation -e ".[test]"
