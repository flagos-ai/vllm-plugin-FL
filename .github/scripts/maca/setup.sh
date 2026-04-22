#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for MACA CI environment.
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh && conda activate base
pip install --no-build-isolation -e ".[test]"
