# Copyright (c) 2025 BAAI. All rights reserved.

import json
import os
from typing import Optional, Tuple

import flag_gems
try:
    # FlagGems<=5.0.2: DeviceDetector lives in device.
    from flag_gems.runtime.backend.device import DeviceDetector
except (ImportError, ModuleNotFoundError, FileNotFoundError):
    # FlagGems>5.0.2: DeviceDetector lives in device_finder.
    from flag_gems.runtime.backend.device_finder import DeviceDetector
from flag_gems.runtime import backend

_OP_CONFIG: Optional[dict[str, str]] = None

# Mapping used by dispatch registration to resolve the current runtime platform
# into a backend directory under dispatch/backends/vendor.
#
# Field definitions and sources:
# - top-level key (vendor_name): normalized hardware vendor identifier.
#   Source: runtime platform detection (current_platform.vendor_name) and
#   fallback device probing (DeviceInfo.vendor_name).
# - device_type: compute class reported by runtime, such as "cuda" or "npu".
#   Source: runtime platform detection (current_platform.device_type) and
#   fallback device probing (DeviceInfo.device_type).
# - device_name: runtime device family/product alias used by vLLM platform.
#   Source: runtime platform detection (current_platform.device_name).
#
# Values are normalized to lowercase and matched against available backend
# subdirectories (for example, cuda/ascend/metax/iluvatar/mthreads).
VENDOR_DEVICE_MAP: dict[str, dict[str, str]] = {
    # Registered backend: vendor/cuda
    "nvidia": {"device_type": "cuda", "device_name": "nvidia"},
    # Registered backend: vendor/ascend
    "ascend": {"device_type": "npu", "device_name": "npu"},
    # Registered backend: vendor/iluvatar
    "iluvatar": {"device_type": "cuda", "device_name": "cuda"},
    # Registered backend: vendor/metax
    "metax": {"device_type": "cuda", "device_name": "metax"},
    # Registered backend: vendor/musa
    "mthreads": {"device_type": "musa", "device_name": "musa"},
    # Registered backend: vendor/sunrise
    "sunrise": {"device_type": "ptpu", "device_name": "ptpu"},
    # Registered backend: vendor/hygon
    "hygon": {"device_type": "cuda", "device_name": "cuda"},
    # Registered backend: vendor/thead (Alibaba T-Head PPU)
    "thead": {"device_type": "ppu", "device_name": "ppu"},
    # Registered backend: vendor/gcu (Enflame GCU / torch_gcu)
    "enflame": {"device_type": "gcu", "device_name": "gcu"},
    # Registered backend: vendor/txda
    "tsingmicro": {"device_type": "txda", "device_name": "txda"},
    # Registered backend: vendor/kunlunxin
    "kunlunxin": {"device_type": "cuda", "device_name": "kunlunxin"},
}


def _get_vendor_device_field(vendor_name: str, field: str) -> str:
    """Get a required field from VENDOR_DEVICE_MAP for the specified vendor."""
    if not isinstance(vendor_name, str) or not vendor_name.strip():
        raise ValueError("vendor_name must be a non-empty string.")

    normalized_vendor = vendor_name
    device_info = VENDOR_DEVICE_MAP.get(normalized_vendor)
    if not isinstance(device_info, dict):
        raise ValueError(
            f"Vendor '{normalized_vendor}' not found in VENDOR_DEVICE_MAP."
        )

    value = device_info.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Field '{field}' for vendor '{normalized_vendor}' is missing "
            "or empty in VENDOR_DEVICE_MAP."
        )
    return value


def get_device_type(vendor_name: str) -> str:
    """Return the configured device_type for the given vendor."""
    return _get_vendor_device_field(vendor_name, "device_type")


def get_device_name(vendor_name: str) -> str:
    """Return the configured device_name for the given vendor."""
    return _get_vendor_device_field(vendor_name, "device_name")


def use_flaggems(default: bool = True) -> bool:
    if os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() not in ("true", "1"):
        return False
    return flag_gems.is_available(default=default)


def get_op_config() -> dict[str, str]:
    global _OP_CONFIG
    if _OP_CONFIG is not None:
        return _OP_CONFIG

    # Read platform dispatch configuration from `devices.yaml`
    devices_yaml = os.path.join(os.path.dirname(__file__), "devices.yaml")
    if not os.path.exists(devices_yaml):
        raise FileNotFoundError(
            f"Dispatch configuration not found at {devices_yaml}"
        )

    with open(devices_yaml) as f:
        import yaml

        devices = yaml.safe_load(f)

    detector = DeviceDetector()
    vendor = str(detector.vendor).lower().split(".")[-1]

    device_type = get_device_type(vendor)
    device_name = get_device_name(vendor)

    # Build dispatch op config
    _OP_CONFIG = {
        "device_type": device_type,
        "vendor": vendor,
        "device_name": device_name,
    }
    return _OP_CONFIG


def get_support_layers() -> Tuple[dict[str, str], dict[str, list[dict[str, str]]]]:
    """
    Retrieve layer/op support information from the YAML config.

    Returns:
        supported_layers: { key: layer_name, value: op_name }
        supported_layers_ops: { key: layer_name, value: list of { key: op_name, value: [flags] } }
    """
    devices_yaml = os.path.join(os.path.dirname(__file__), "devices.yaml")
    if not os.path.exists(devices_yaml):
        raise FileNotFoundError(
            f"Dispatch configuration not found at {devices_yaml}"
        )

    with open(devices_yaml) as f:
        import yaml

        devices = yaml.safe_load(f)

    detector = DeviceDetector()
    vendor = str(detector.vendor).lower().split(".")[-1]

    device_type = get_device_type(vendor)

    # Structure:
    # devices:
    #   - device_type: cuda
    #     vendors:
    #       - vendor_name: nvidia
    #         ops: { <layer_name>: <op_name>, ... }
    #         supported_layers_ops: { <layer_name>: [ { <op_name>: [<flag>, ...] }, ... ], ... }
    #       ...
    #   ...

    for device in devices.get("devices", []):
        if device.get("device_type") != device_type:
            continue
        for v in device.get("vendors", []):
            if v.get("vendor_name") != vendor:
                continue
            supported_layers = v.get("ops", {})
            supported_layers_ops = v.get("supported_layers_ops", {})
            return supported_layers, supported_layers_ops

    raise ValueError(
        f"No device configuration found for device_type={device_type}, vendor={vendor}"
    )
