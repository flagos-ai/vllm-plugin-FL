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
import shutil
import subprocess

import torch

assert torch.cuda.is_available(), "Iluvatar accelerator is unavailable"
count = torch.cuda.device_count()
assert count >= 4, f"At least 4 Iluvatar devices are required, found {count}"


def report_device_memory():
    """Best-effort memory report when the smoke tensor fails; never masks it.

    A 4KB allocation failing here means the card is exhausted by processes
    outside this job. The subprocess timeout survives driver-level hangs;
    per-device torch queries double as an exhaustion map (a device whose
    context cannot even be created is the one that is full).
    """
    ixsmi = shutil.which("ixsmi")
    if not ixsmi:
        import os
        candidate = "/usr/local/corex/bin/ixsmi"
        if os.access(candidate, os.X_OK):
            ixsmi = candidate
    if ixsmi:
        try:
            out = subprocess.run(
                [ixsmi], timeout=15, capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                print(out.stdout)
                return
        except Exception:
            pass
    for i in range(count):
        try:
            free, total = torch.cuda.mem_get_info(i)
            print(f"device {i}: {free / 1024**3:.2f}/{total / 1024**3:.2f} GiB free/total")
        except Exception as exc:
            print(f"device {i}: memory probe failed: {exc}")


try:
    tensor = torch.ones((32, 32), device="cuda:0")
    torch.cuda.synchronize()
except Exception:
    report_device_memory()
    raise

print(f"Iluvatar devices: {count}")
print(f"Device 0: {torch.cuda.get_device_name(0)}")
print(f"Tensor smoke: {tensor.device} {tuple(tensor.shape)}")
PY
