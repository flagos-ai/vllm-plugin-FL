#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check Iluvatar CoreX availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Iluvatar BI-V150 availability ==="

if command -v ixsmi >/dev/null 2>&1; then
  ixsmi
else
  echo "::warning::ixsmi not found; checking through torch."
fi

python - <<'PY'
import torch

assert torch.cuda.is_available(), "Iluvatar accelerator is unavailable"
count = torch.cuda.device_count()
assert count >= 4, f"At least 4 Iluvatar devices are required, found {count}"

tensor = torch.ones((32, 32), device="cuda:0")
torch.cuda.synchronize()

print(f"Iluvatar devices: {count}")
print(f"Device 0: {torch.cuda.get_device_name(0)}")
print(f"Tensor smoke: {tensor.device} {tuple(tensor.shape)}")
PY
