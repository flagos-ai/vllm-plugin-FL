#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Huawei Ascend NPU availability.
set -euo pipefail

echo "=== Checking Ascend NPU availability ==="

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "::error::npu-smi is required but was not found in PATH."
  exit 1
fi

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  echo "::error::ASCEND_RT_VISIBLE_DEVICES is not set."
  exit 1
fi

required_devices=(
  "/dev/davinci_manager"
  "/dev/devmm_svm"
  "/dev/hisi_hdc"
)

IFS=',' read -r -a visible_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
for device_id in "${visible_devices[@]}"; do
  device_id="${device_id//[[:space:]]/}"
  if [[ -z "${device_id}" ]]; then
    continue
  fi
  required_devices+=("/dev/davinci${device_id}")
done

for device_path in "${required_devices[@]}"; do
  if [[ ! -e "${device_path}" ]]; then
    echo "::error::Missing Ascend device path: ${device_path}"
    exit 1
  fi
done

npu-smi info
echo "Ascend device check passed."
