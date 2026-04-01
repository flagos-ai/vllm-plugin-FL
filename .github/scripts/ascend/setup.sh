#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -euo pipefail

pip install --upgrade pip "setuptools>=77.0.3"
pip install --no-build-isolation -e ".[test]"
