#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check MThreads MUSA GPU availability.
set -euo pipefail
echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking MThreads MUSA GPU availability ==="
mthreads-gmi
