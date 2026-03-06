# Copyright (c) 2026 BAAI. All rights reserved.

"""
Hardware-specific operator configuration loader utilities.

This module provides automatic loading of operator configurations based on
the detected hardware platform.

Configuration Priority (highest to lowest):
1. VLLM_FL_CONFIG: User-specified config file path (complete override)
2. Environment variables: Override specific items from platform config
   - VLLM_FL_PREFER: Backend preference (flagos, vendor, reference)
   - VLLM_FL_STRICT: Strict mode (1 or 0)
   - VLLM_FL_PER_OP: Per-operator backend order
   - VLLM_FL_FLAGOS_BLACKLIST: FlagOS operator blacklist
   - VLLM_FL_OOT_BLACKLIST: OOT operator blacklist
3. Platform-specific config file: Default values (auto-detected)
4. Built-in default values

Supported platforms:
- ascend: Huawei Ascend NPU
- cuda: NVIDIA GPU
- (more platforms can be added)

Configuration files are stored in this directory as YAML files:
- ascend.yaml
- cuda.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

# Directory containing config files (config/)
_CONFIG_DIR = Path(__file__).parent


def get_platform_name() -> str:
    """
    Detect the current hardware platform using platform vendor_name.

    This function uses current_platform.vendor_name to accurately distinguish
    between different hardware vendors (NVIDIA, METAX, Ascend, etc.).

    Returns:
        Platform name string based on vendor_name: 'nvidia', 'metax', 'ascend', etc.
    """
    try:
        from vllm.platforms import current_platform

        if hasattr(current_platform, 'vendor_name'):
            return current_platform.vendor_name
    except Exception:
        pass

    return 'unknown'


def get_config_path(platform: Optional[str] = None) -> Optional[Path]:
    """
    Get the configuration file path for the specified or detected platform.

    Args:
        platform: Platform name. If None, auto-detect.

    Returns:
        Path to the config file, or None if not found.
    """
    if platform is None:
        platform = get_platform_name()

    # Try platform-specific config
    config_file = _CONFIG_DIR / f"{platform}.yaml"
    if config_file.exists():
        return config_file

    return None


def load_platform_config(platform: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Load the configuration for the specified or detected platform.

    Args:
        platform: Platform name. If None, auto-detect.

    Returns:
        Configuration dictionary, or None if no config found.
    """
    config_path = get_config_path(platform)
    if config_path is None:
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if isinstance(config, dict) else None
    except Exception:
        return None


def get_per_op_order(config: Optional[dict] = None) -> Optional[dict[str, list[str]]]:
    """
    Extract per-op backend order from config.

    Args:
        config: Configuration dict. If None, load from platform config.

    Returns:
        Dict mapping op names to backend order lists.
    """
    if config is None:
        config = load_platform_config()
    if config is None:
        return None

    per_op = config.get('per_op', {})
    if not isinstance(per_op, dict):
        return None

    result = {}
    for op_name, backends in per_op.items():
        if isinstance(backends, list):
            result[op_name] = [str(b) for b in backends]
        elif isinstance(backends, str):
            result[op_name] = [backends]

    return result if result else None


def get_flagos_blacklist(config: Optional[dict] = None) -> Optional[list[str]]:
    """
    Extract FlagOS operator blacklist from config.

    Args:
        config: Configuration dict. If None, load from platform config.

    Returns:
        List of blacklisted FlagOS operator names.
    """
    if config is None:
        config = load_platform_config()
    if config is None:
        return None

    blacklist = config.get('flagos_blacklist', [])
    if isinstance(blacklist, list):
        return [str(op) for op in blacklist]
    return None


def get_oot_blacklist(config: Optional[dict] = None) -> Optional[list[str]]:
    """
    Extract OOT operator blacklist from config.

    Args:
        config: Configuration dict. If None, load from platform config.

    Returns:
        List of blacklisted OOT operator names.
    """
    if config is None:
        config = load_platform_config()
    if config is None:
        return None

    blacklist = config.get('oot_blacklist', [])
    if isinstance(blacklist, list):
        return [str(op) for op in blacklist]
    return None


def get_effective_config() -> dict[str, Any]:
    """
    Get the effective configuration, considering environment variable overrides.

    Priority:
    1. VLLM_FL_CONFIG environment variable (user-specified config file)
    2. Platform-specific config file (auto-detected)
    3. Default config file
    4. Empty config (no restrictions)

    Returns:
        Effective configuration dictionary.
    """
    # Check for user-specified config file
    user_config_path = os.environ.get('VLLM_FL_CONFIG', '').strip()
    if user_config_path and os.path.isfile(user_config_path):
        try:
            with open(user_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict):
                return config
        except Exception:
            pass

    # Load platform config
    platform_config = load_platform_config()
    if platform_config:
        return platform_config

    # Return empty config
    return {}
