#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Iluvatar CoreX BI-V150 availability.
set -euo pipefail

# Bound the entire process group, including diagnostics and driver calls. A
# Python/thread timer cannot interrupt an import or a hung native CUDA call.
probe_timeout=${ILUVATAR_DEVICE_CHECK_TIMEOUT_SECONDS:-120}
if [[ ! "$probe_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::ILUVATAR_DEVICE_CHECK_TIMEOUT_SECONDS must be a positive integer."
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "::error::GNU timeout is required for the Iluvatar device check."
  exit 127
fi

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Iluvatar CoreX availability (timeout ${probe_timeout}s) ==="
if timeout --kill-after=10s "${probe_timeout}s" bash -s <<'PROBE'
set -euo pipefail
echo "[device-check] management-tool: start"
if command -v ixsmi >/dev/null 2>&1; then
  ixsmi || true
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::ixsmi not found; falling back to nvidia-smi (CoreX compat)."
  nvidia-smi || true
else
  echo "::warning::Neither ixsmi nor nvidia-smi found; probing via torch."
fi
echo "[device-check] management-tool: complete"

python -u - <<'PY'
import shutil
import subprocess

print("[device-check] torch-import: start", flush=True)
import torch
print("[device-check] torch-import: complete", flush=True)

print("[device-check] cuda-availability: start", flush=True)
assert torch.cuda.is_available(), "Iluvatar accelerator is unavailable"
print("[device-check] cuda-availability: complete", flush=True)
print("[device-check] device-count: start", flush=True)
count = torch.cuda.device_count()
assert count >= 4, f"At least 4 Iluvatar devices are required, found {count}"
print(f"[device-check] device-count: complete ({count})", flush=True)


def report_device_memory():
    """Best-effort failure diagnostics, also bounded by the outer watchdog."""
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
                print(out.stdout, flush=True)
                return
        except Exception:
            pass
    for i in range(count):
        try:
            free, total = torch.cuda.mem_get_info(i)
            print(f"device {i}: {free / 1024**3:.2f}/{total / 1024**3:.2f} GiB free/total", flush=True)
        except Exception as exc:
            print(f"device {i}: memory probe failed: {exc}", flush=True)


try:
    print("[device-check] tensor-allocation: start", flush=True)
    tensor = torch.ones((32, 32), device="cuda:0")
    print("[device-check] tensor-allocation: complete", flush=True)
    print("[device-check] cuda-synchronize: start", flush=True)
    torch.cuda.synchronize()
    print("[device-check] cuda-synchronize: complete", flush=True)
except Exception:
    report_device_memory()
    raise

print(f"Iluvatar devices: {count}", flush=True)
print("[device-check] device-name: start", flush=True)
print(f"Device 0: {torch.cuda.get_device_name(0)}", flush=True)
print("[device-check] device-name: complete", flush=True)
print(f"Tensor smoke: {tensor.device} {tuple(tensor.shape)}", flush=True)
PY
PROBE
then
  echo "[device-check] complete"
else
  status=$?
  if [[ "$status" == 124 || "$status" == 137 ]]; then
    echo "::error::Iluvatar device check timed out after ${probe_timeout}s (kill grace 10s); inspect the last stage above."
  else
    echo "::error::Iluvatar device check failed with exit code ${status}."
  fi
  exit "$status"
fi
