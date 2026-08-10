#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check NVIDIA GPU availability.
set -euo pipefail
echo "=== Checking NVIDIA GPU availability ==="
nvidia-smi
