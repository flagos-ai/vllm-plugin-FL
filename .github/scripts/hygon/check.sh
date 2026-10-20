#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Hygon DCU availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Hygon DCU availability ==="

HY_SMI_BIN=""
if command -v hy-smi >/dev/null 2>&1; then
  HY_SMI_BIN="$(command -v hy-smi)"
elif [[ -x /opt/hyhal/bin/hy-smi ]]; then
  HY_SMI_BIN="/opt/hyhal/bin/hy-smi"
fi

if [[ -n "${HY_SMI_BIN}" ]]; then
  echo "Using hy-smi: ${HY_SMI_BIN}"
  "${HY_SMI_BIN}"
  "${HY_SMI_BIN}" --showmeminfo vram || true
else
  echo "::warning::hy-smi not found in PATH or /opt/hyhal/bin; skipping SMI output."
fi

test -e /dev/kfd
test -e /dev/mkfd
test -d /dev/dri
test -d /opt/hyhal
