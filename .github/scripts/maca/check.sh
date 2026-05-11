#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check MetaX GPU availability.
set -euo pipefail
echo "=== Checking MetaX GPU availability ==="
mx-smi
