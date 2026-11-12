#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check T-Head PPU availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking T-Head PPU availability ==="

# nvidia-smi is a symlink to ppu-smi on T-Head PPU SDK
nvidia-smi
