#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Hygon GPU availability.
set -euo pipefail
echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Hygon GPU availability ==="
hy-smi
echo "=== Checking Hygon GPU VRAM usage ==="
hy-smi --showmeminfo vram
