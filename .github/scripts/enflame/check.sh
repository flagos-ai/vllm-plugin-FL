#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Enflame S60 availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Enflame S60 availability ==="

if ! command -v efsmi >/dev/null 2>&1; then
  echo "::error::efsmi is not available in the CI container."
  exit 1
fi

efsmi

python - <<'PY'
import torch
import torch_gcu  # noqa: F401

if not hasattr(torch, "gcu") or not torch.gcu.is_available():
    raise RuntimeError("Enflame GCU is not available")

count = torch.gcu.device_count()
print(f"Enflame GCU count: {count}")
if count < 2:
    raise RuntimeError(f"At least 2 GCUs are required, found {count}")
PY
