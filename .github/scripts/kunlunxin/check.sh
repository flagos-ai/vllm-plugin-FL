#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check Kunlunxin P800 availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Kunlunxin P800 availability ==="

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

if command -v xpu-smi >/dev/null 2>&1; then
  xpu-smi || true
else
  echo "::warning::xpu-smi not found; checking through torch.cuda."
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Kunlunxin CUDA-compatible accelerator is unavailable")

count = torch.cuda.device_count()
print(f"Kunlunxin visible devices: {count}")
if count < 4:
    raise RuntimeError(f"At least 4 visible Kunlunxin devices are required, found {count}")

tensor = torch.ones((32, 32), device="cuda:0")
torch.cuda.synchronize()
print(f"Tensor smoke: {tensor.device} {tuple(tensor.shape)}")
PY
