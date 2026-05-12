#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for MUSA CI environment.
set -euo pipefail

pip install --no-build-isolation -e ".[test]"
