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

benchmark_model=/data/models/Qwen/Qwen3-0.6B
for model_file in config.json tokenizer.json; do
  model_path="${benchmark_model}/${model_file}"
  if [[ -r "${model_path}" ]]; then
    echo "Benchmark model file is readable: ${model_path}"
  else
    echo "::warning::Benchmark model file is missing or unreadable: ${model_path}"
  fi
done

# The torch probe occasionally wedges while the CoreX user space talks to an
# older kernel driver (the process holds a device context and never returns).
# A hung probe would otherwise burn the job's full timeout, so bound every
# attempt and retry in a fresh process.
PROBE_SRC=$(cat <<'PY'
import shutil
import subprocess

print("[Iluvatar probe] Importing torch")
import torch

print(f"[Iluvatar probe] Checking torch {torch.__version__} CUDA availability")
assert torch.cuda.is_available(), "Iluvatar accelerator is unavailable"
print("[Iluvatar probe] Counting devices")
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
    print("[Iluvatar probe] Allocating a tensor on cuda:0")
    tensor = torch.ones((32, 32), device="cuda:0")
    print("[Iluvatar probe] Synchronizing cuda:0")
    torch.cuda.synchronize()
except Exception:
    report_device_memory()
    raise

print(f"Iluvatar devices: {count}")
print(f"Device 0: {torch.cuda.get_device_name(0)}")
print(f"Tensor smoke: {tensor.device} {tuple(tensor.shape)}")
PY
)

probe_ok=0
for attempt in 1 2 3; do
  echo "=== Device probe attempt ${attempt}/3 (timeout 180s) ==="
  rc=0
  timeout 180 python -u -c "${PROBE_SRC}" || rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    probe_ok=1
    break
  fi
  if [[ "${rc}" -eq 124 ]]; then
    echo "::warning::Device probe attempt ${attempt} hung for 180s and was killed; retrying in a fresh process."
  else
    echo "::warning::Device probe attempt ${attempt} failed with exit code ${rc}."
  fi
  sleep 5
done

if [[ "${probe_ok}" -ne 1 ]]; then
  echo "::error::Iluvatar device probe failed after 3 attempts."
  exit 1
fi
