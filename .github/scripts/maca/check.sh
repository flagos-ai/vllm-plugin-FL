#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check MetaX GPU availability.
set -euo pipefail
echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking MetaX GPU availability ==="
mx-smi
