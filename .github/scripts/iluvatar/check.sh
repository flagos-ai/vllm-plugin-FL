#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Iluvatar CoreX BI-V150 availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Iluvatar CoreX availability ==="

if command -v ixsmi >/dev/null 2>&1; then
  ixsmi || true
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::ixsmi not found; falling back to nvidia-smi (CoreX compat)."
  nvidia-smi || true
else
  echo "::warning::Neither ixsmi nor nvidia-smi found; probing via torch."
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
