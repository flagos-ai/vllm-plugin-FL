#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check NVIDIA GPU availability.
set -euo pipefail
echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking NVIDIA GPU availability ==="
nvidia-smi
