# Copyright (c) 2025 BAAI. All rights reserved.

import json
import os
from typing import Optional, Tuple

import flag_gems
from flag_gems.runtime.backend.device import DeviceDetector
from flag_gems.runtime import backend

_OP_CONFIG: Optional[dict[str, str]] = None


def use_flaggems(default: bool = True) -> bool:
    if os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() not in ("true", "1"):
        return False
    prefer_backend = os.environ.get("VLLM_FL_PREFER", "").strip()
    if prefer_backend and prefer_backend.lower() != "flagos":
        return False
    value = os.environ.get("USE_FLAGGEMS", None)
    if value is None:
        return default
    return value.lower() in ("true", "1")


def get_flag_gems_whitelist_blacklist() -> Tuple[
    Optional[list[str]], Optional[list[str]]
]:
    """
    Get FlagGems operator whitelist and blacklist.

    Priority (highest to lowest):
    1. VLLM_FL_FLAGOS_WHITELIST env var: Only these ops use FlagGems
    2. VLLM_FL_FLAGOS_BLACKLIST env var: These ops don't use FlagGems
    3. Platform config flagos_blacklist: Default blacklist from config file

    Note: VLLM_FL_FLAGOS_WHITELIST and VLLM_FL_FLAGOS_BLACKLIST cannot be set
    simultaneously. If whitelist is set, it completely overrides any blacklist.

    Returns:
        Tuple[Optional[list[str]], Optional[list[str]]]:
            A tuple of (whitelist, blacklist). Each is None if not set.

    Raises:
        ValueError: If both VLLM_FL_FLAGOS_WHITELIST and VLLM_FL_FLAGOS_BLACKLIST
                    are set simultaneously.
    """
    whitelist_str = os.environ.get("VLLM_FL_FLAGOS_WHITELIST", "")
    blacklist_str = os.environ.get("VLLM_FL_FLAGOS_BLACKLIST", "")

    # Check if both env vars are set (conflict)
    if whitelist_str and blacklist_str:
        raise ValueError(
            "Cannot set both VLLM_FL_FLAGOS_WHITELIST and VLLM_FL_FLAGOS_BLACKLIST "
            "simultaneously. Please set only one of them."
        )

    whitelist = None
    blacklist = None

    # Priority 1: Whitelist from env var (completely overrides blacklist)
    if whitelist_str:
        whitelist = [op.strip() for op in whitelist_str.split(",") if op.strip()]
        return whitelist, None  # Whitelist overrides any blacklist

    # Priority 2: Blacklist from env var
    if blacklist_str:
        blacklist = [op.strip() for op in blacklist_str.split(",") if op.strip()]
        return None, blacklist

    # Priority 3: Blacklist from platform config
    try:
        from vllm_fl.dispatch.config import get_flagos_blacklist
        config_blacklist = get_flagos_blacklist()
        if config_blacklist:
            blacklist = config_blacklist
    except Exception:
        pass

    return whitelist, blacklist


def use_flaggems_op(op_name: str, default: bool = True) -> bool:
    """
    Check if FlagGems should be used for a specific operator.

    Priority (highest to lowest):
    1. VLLM_FL_FLAGOS_WHITELIST env var: Only these ops use FlagGems
    2. VLLM_FL_FLAGOS_BLACKLIST env var: These ops don't use FlagGems
    3. Platform config flagos_blacklist: Default blacklist from config file
    4. Default: Use FlagGems for all ops

    Note: Whitelist and blacklist (env vars) cannot be set simultaneously.
    If whitelist is set, it completely overrides the config file blacklist.
    """
    if not use_flaggems(default=default):
        return False

    # Get whitelist/blacklist with proper priority
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()

    # If whitelist is set, only allow ops in whitelist
    if whitelist is not None:
        return op_name in whitelist

    # If blacklist is set (from env or config), deny ops in blacklist
    if blacklist is not None:
        return op_name not in blacklist

    # Default: allow all ops
    return True


def _load_op_config_from_env() -> None:
    global _OP_CONFIG
    config_path = os.environ.get("VLLM_FL_OP_CONFIG", None)
    if config_path is None or not config_path.strip():
        _OP_CONFIG = None
        return
    if not os.path.isfile(config_path):
        raise ValueError(f"VLLM_FL_OP_CONFIG file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid VLLM_FL_OP_CONFIG JSON file.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("VLLM_FL_OP_CONFIG must be a JSON object.")
    normalized: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("VLLM_FL_OP_CONFIG must map strings to strings.")
        normalized[key] = value
    _OP_CONFIG = normalized


def get_op_config() -> Optional[dict[str, str]]:
    return _OP_CONFIG


_load_op_config_from_env()


class DeviceInfo:
    def __init__(self):
        self.device = DeviceDetector()
        self.supported_device = ["nvidia", "ascend"]
        backend.set_torch_backend_device_fn(self.device.vendor_name)

    @property
    def dispatch_key(self):
        return self.device.dispatch_key

    @property
    def vendor_name(self):
        return self.device.vendor_name

    @property
    def device_type(self):
        return self.device.name

    @property
    def torch_device_fn(self):
        # torch_device_fn is like 'torch.cuda' object
        return backend.gen_torch_device_object()

    @property
    def torch_backend_device(self):
        # torch_backend_device is like 'torch.backend.cuda' object
        return backend.get_torch_backend_device_fn()

    def get_supported_device(self):
        if self.vendor_name in self.supported_device:
            raise NotImplementedError(f"{self.vendor_name} is not support now!")
        return True


def get_flaggems_all_ops() -> list[str]:
    """
    Get all FlagGems operator names from flag_gems._FULL_CONFIG.
    """
    try:
        pass
    except Exception:
        return []
    ops = flag_gems.all_registered_ops()

    return ops


# OOT operator names as registered in custom_ops.py (op_name lowercase)
OOT_OP_NAMES = [
    "silu_and_mul",
    "rms_norm",
    "rotary_embedding",
    "fused_moe",
    "unquantized_fused_moe_method",
]


def get_oot_whitelist() -> Optional[list[str]]:
    """
    Get OOT operator whitelist from VLLM_FL_OOT_WHITELIST environment variable.

    If set, only the specified OOT operators will be registered.
    Comma-separated list of OOT operator names (e.g., "silu_and_mul,rms_norm").

    Returns:
        List of OOT operator names to register, or None if not set (register all).
    """
    whitelist_str = os.environ.get("VLLM_FL_OOT_WHITELIST", "")
    if not whitelist_str:
        return None
    return [op.strip() for op in whitelist_str.split(",") if op.strip()]


def get_oot_blacklist() -> Optional[list[str]]:
    """
    Get OOT operator blacklist from environment variable or platform config.

    Priority (highest to lowest):
    1. VLLM_FL_OOT_WHITELIST env var: If set, blacklist is ignored
    2. VLLM_FL_OOT_BLACKLIST env var: These ops won't be registered
    3. Platform config oot_blacklist: Default blacklist from config file

    Returns:
        List of OOT operator names to NOT register, or None if not set.
    """
    # If whitelist is set, blacklist is ignored
    whitelist_str = os.environ.get("VLLM_FL_OOT_WHITELIST", "")
    if whitelist_str:
        return None

    # Priority 2: Blacklist from env var
    blacklist_str = os.environ.get("VLLM_FL_OOT_BLACKLIST", "")
    if blacklist_str:
        return [op.strip() for op in blacklist_str.split(",") if op.strip()]

    # Priority 3: Blacklist from platform config
    try:
        from vllm_fl.dispatch.config import get_oot_blacklist as config_get_oot_blacklist
        config_blacklist = config_get_oot_blacklist()
        if config_blacklist:
            return config_blacklist
    except Exception:
        pass

    return None


def is_oot_enabled() -> bool:
    """
    Check if OOT registration is enabled.

    Controlled by VLLM_FL_OOT_ENABLED environment variable.
    Default is True (enabled).

    Returns:
        True if OOT registration is enabled, False otherwise.
    """
    if os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() not in ("true", "1"):
        return False
    enabled_str = os.environ.get("VLLM_FL_OOT_ENABLED", "1")
    return enabled_str.lower() in ("1", "true")


if __name__ == "__main__":
    device = DeviceInfo()
