#!/bin/bash
# Copyright 2026 FlagOS Contributors
# Check T-Head PPU availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking T-Head PPU availability ==="

# nvidia-smi is a symlink to ppu-smi on T-Head PPU SDK
nvidia-smi
