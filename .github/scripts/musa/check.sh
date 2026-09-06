#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check Moore Threads MUSA availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Moore Threads MUSA availability ==="

if command -v mthreads-gmi >/dev/null 2>&1; then
  mthreads-gmi
else
  echo "::warning::mthreads-gmi not found; checking through torch_musa."
fi

python - <<'PY'
import torch
import torch_musa

assert torch.musa.is_available(), "MUSA accelerator is unavailable"
count = torch.musa.device_count()
assert count > 0, "No MUSA devices detected"

tensor = torch.ones((32, 32), device="musa:0")
torch.musa.synchronize()

print(f"MUSA devices: {count}")
print(f"Tensor smoke: {tensor.device} {tuple(tensor.shape)}")
PY
