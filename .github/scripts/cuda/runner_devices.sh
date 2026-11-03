#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Match the GPU partitions named by the two DSW runner workspaces.
# Source this file before Python/CUDA initialization; sourcing alone has no effect.

configure_cuda_runner_devices() {
    if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
        return 0
    fi
    local default_devices
    case "${RUNNER_NAME:-}" in
        dsw-h20-03) default_devices="0,1,2,3" ;;
        dsw-h20-47) default_devices="4,5,6,7" ;;
        *) return 0 ;;
    esac

    _CUDA_RUNNER_ASSIGNED_DEVICES=""
    # An explicitly empty value is intentional too. Change the device order
    # only when generating a numeric mask; existing masks keep their semantics.
    if [[ "${CUDA_VISIBLE_DEVICES+x}" != "x" ]]; then
        CUDA_VISIBLE_DEVICES="$default_devices"
        CUDA_DEVICE_ORDER="PCI_BUS_ID"
        _CUDA_RUNNER_ASSIGNED_DEVICES="$default_devices"
    fi
    export CUDA_VISIBLE_DEVICES
    if [[ "${CUDA_DEVICE_ORDER+x}" == "x" ]]; then
        export CUDA_DEVICE_ORDER
    fi
    # Each Actions step starts another shell. Export covers this setup process;
    # GITHUB_ENV carries the same selection into the later pytest/server steps.
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" >> "$GITHUB_ENV"
        if [[ "${CUDA_DEVICE_ORDER+x}" == "x" ]]; then
            printf 'CUDA_DEVICE_ORDER=%s\n' "$CUDA_DEVICE_ORDER" >> "$GITHUB_ENV"
        fi
    fi
    printf '[cuda setup] runner=%s CUDA_VISIBLE_DEVICES=%q CUDA_DEVICE_ORDER=%q\n' \
        "$RUNNER_NAME" "$CUDA_VISIBLE_DEVICES" "${CUDA_DEVICE_ORDER-<default>}"
}

report_cuda_runner_devices() {
    if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
        return 0
    fi
    case "${RUNNER_NAME:-}" in
        dsw-h20-03|dsw-h20-47) ;;
        *) return 0 ;;
    esac

    # nvidia-smi reports physical NVML indices; Torch enumerates the masked
    # logical devices. Their UUIDs verify the actual binding, including order.
    python - "${_CUDA_RUNNER_ASSIGNED_DEVICES:-}" <<'PY'
import csv
import io
import json
import os
import subprocess
import sys
from uuid import UUID

import torch

inventory = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
physical = {}
for row in csv.reader(io.StringIO(inventory)):
    index, gpu_uuid, pci_bus = (value.strip() for value in row)
    physical[gpu_uuid.lower().removeprefix("gpu-")] = {
        "nvml_index": int(index),
        "pci_bus_id": pci_bus,
    }

devices = []
for logical_index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(logical_index)
    gpu_uuid = properties.uuid
    if hasattr(gpu_uuid, "bytes"):
        normalized = str(UUID(bytes=bytes(gpu_uuid.bytes)))
    else:
        normalized = str(gpu_uuid).lower().removeprefix("gpu-")
    devices.append(
        {
            "logical_index": logical_index,
            "torch_uuid": f"GPU-{normalized}",
            **physical.get(normalized, {}),
        }
    )
print(
    "[cuda setup] GPU binding "
    + json.dumps(
        {
            "runner": os.environ["RUNNER_NAME"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "logical_device_count": len(devices),
            "devices": devices,
        },
        sort_keys=True,
    ),
    flush=True,
)
if sys.argv[1]:
    expected = [int(index) for index in sys.argv[1].split(",")]
    actual = [device.get("nvml_index") for device in devices]
    if actual != expected:
        raise RuntimeError(
            f"H20 runner GPU binding mismatch: expected NVML {expected}, got {actual}"
        )
PY
}
