#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Iluvatar CoreX BI-V150 availability.
set -euo pipefail

echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Checking Iluvatar CoreX availability ==="

# Bound external probes independently of the three-hour model-test job.
# SIGKILL escalation bounds ordinary stuck processes; an uninterruptible
# kernel/driver wait can still require host recovery.
command -v timeout >/dev/null 2>&1 || {
  echo "::error::GNU timeout is required for the Iluvatar device preflight."
  exit 1
}
run_management_probe() {
  local status=0
  echo "[preflight] Starting $1"
  timeout --kill-after=5s "${ILUVATAR_SMI_TIMEOUT_SECONDS:-30}s" "$1" || status=$?
  if [[ $status == 124 || $status == 137 ]]; then
    echo "::error::Iluvatar management probe $1 timed out (exit $status); check runner/driver health."
    exit "$status"
  elif [[ $status != 0 ]]; then
    echo "::warning::$1 exited with $status; continuing with the required torch smoke test."
  fi
  echo "[preflight] Finished $1"
}
if command -v ixsmi >/dev/null 2>&1; then
  run_management_probe ixsmi
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "::warning::ixsmi not found; falling back to nvidia-smi (CoreX compat)."
  run_management_probe nvidia-smi
else
  echo "::warning::Neither ixsmi nor nvidia-smi found; probing via torch."
fi

echo "[preflight] Starting bounded torch smoke test"
status=0
timeout --kill-after=5s "${ILUVATAR_TORCH_TIMEOUT_SECONDS:-120}s" python -u - <<'PY' || status=$?
import shutil
import subprocess

print("[preflight] Importing torch", flush=True)
import torch

print("[preflight] Checking torch.cuda.is_available()", flush=True)
assert torch.cuda.is_available(), "Iluvatar accelerator is unavailable"
print("[preflight] Querying device count", flush=True)
count = torch.cuda.device_count()
print(f"[preflight] Visible devices: {count}", flush=True)
assert count >= 4, f"At least 4 Iluvatar devices are required, found {count}"


def report_device_memory():
    """Best-effort memory report when the smoke tensor fails; never masks it.

    An allocation failure may indicate memory pressure or driver trouble.
    Bound the management subprocess; the outer timeout also covers these
    per-device torch queries, which can themselves block in a faulty driver.
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
    print("[preflight] Allocating smoke tensor on cuda:0", flush=True)
    tensor = torch.ones((32, 32), device="cuda:0")
    print("[preflight] Synchronizing cuda:0", flush=True)
    torch.cuda.synchronize()
except Exception:
    report_device_memory()
    raise

print(f"Iluvatar devices: {count}")
print("[preflight] Querying cuda:0 device name", flush=True)
print(f"Device 0: {torch.cuda.get_device_name(0)}")
print(f"Tensor smoke: {tensor.device} {tuple(tensor.shape)}")
PY
if [[ $status == 124 || $status == 137 ]]; then
  echo "::error::Iluvatar torch smoke test timed out (exit $status); last preflight marker identifies the blocked operation."
elif [[ $status != 0 ]]; then
  echo "::error::Iluvatar torch smoke test failed (exit $status)."
fi
exit "$status"
