#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check MetaX C550 availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking MetaX C550 availability ==="

MX_SMI_BIN=""
if command -v mx-smi >/dev/null 2>&1; then
  MX_SMI_BIN="$(command -v mx-smi)"
fi

if [[ -n "${MX_SMI_BIN}" ]]; then
  echo "Using mx-smi: ${MX_SMI_BIN}"
  "${MX_SMI_BIN}" || true
else
  echo "::warning::mx-smi not found in PATH; skipping SMI output."
fi

test -d /dev/dri
test -e /dev/mxcd
