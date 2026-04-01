# Copyright (c) 2026 BAAI. All rights reserved.

"""
Built-in operator implementations registration.

This module registers DEFAULT (FlagGems) and REFERENCE (PyTorch) implementations
for all supported operators by calling register_builtins from each backend.
"""

from __future__ import annotations

import importlib
import os

from .config import get_vendor_device_map
from .registry import OpRegistry
from .logger_manager import get_logger

logger = get_logger()

# Directory containing vendor backends
_VENDOR_BACKENDS_DIR = os.path.join(os.path.dirname(__file__), "backends", "vendor")


def _find_vendor_backend_dir(
    vendor_name: str,
    available_vendor_dirs: set[str],
) -> str | None:
    """Return the backend directory for *vendor_name*, or None if not found.

    Resolves *vendor_name* against get_vendor_device_map() and picks the first
    candidate directory that exists in *available_vendor_dirs*. The "maca" alias
    is treated as "metax" for MetaX runtime compatibility.
    """
    # Keep compatibility with MetaX runtime naming.
    if vendor_name == "maca":
        vendor_name = "metax"
    vendor_map = get_vendor_device_map()
    if vendor_name not in vendor_map:
        return None
    value = vendor_map[vendor_name]
    device_type = value.get("device_type")
    device_name = value.get("device_name")
    return next(
        (c for c in (vendor_name, device_name, device_type) if c in available_vendor_dirs),
        None,
    )


def _get_current_vendor_backend_dirs(available_vendor_dirs: set[str]) -> set[str]:
    """Detect current platform vendor name and return its backend directory."""
    try:
        from vllm.platforms import current_platform

        vendor_name = getattr(current_platform, "vendor_name", None)
        if not isinstance(vendor_name, str) or not vendor_name:
            return None
        return _find_vendor_backend_dir(vendor_name, available_vendor_dirs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to detect current vendor backend from current_platform."
        ) from exc


def _register_vendor_backends(registry: OpRegistry) -> None:
    """
    Auto-discover and register all vendor backends.

    Scans the vendor directory for subdirectories containing register_ops.py
    and calls their register_builtins function.

    Args:
        registry: Registry to register into
    """
    if not os.path.isdir(_VENDOR_BACKENDS_DIR):
        logger.debug(f"Vendor backends directory not found: {_VENDOR_BACKENDS_DIR}")
        return

    available_vendor_dirs = {
        vendor_name
        for vendor_name in os.listdir(_VENDOR_BACKENDS_DIR)
        if os.path.isdir(os.path.join(_VENDOR_BACKENDS_DIR, vendor_name))
        and not vendor_name.startswith("_")
    }

    current_vendor_dir = _get_current_vendor_backend_dirs(available_vendor_dirs)
    if not current_vendor_dir:
        logger.warning(
            "Unable to detect current vendor backend; skipping vendor backend registration"
        )
        return

    logger.info(
        "Registering vendor backends for current platform: %s",
        current_vendor_dir,
    )

    for vendor_name in available_vendor_dirs:
        vendor_path = os.path.join(_VENDOR_BACKENDS_DIR, vendor_name)

        if vendor_name != current_vendor_dir:
            logger.debug(
                "Skipping vendor backend '%s' for current platform",
                vendor_name,
            )
            continue

        # Skip if no register_ops.py exists
        register_ops_path = os.path.join(vendor_path, "register_ops.py")
        if not os.path.isfile(register_ops_path):
            continue

        # Try to import and register
        module_name = f".backends.vendor.{vendor_name}.register_ops"
        try:
            mod = importlib.import_module(module_name, package="vllm_fl.dispatch")
            if hasattr(mod, "register_builtins"):
                mod.register_builtins(registry)
                logger.debug(f"Registered {vendor_name} operators")
            else:
                logger.debug(f"No register_builtins function in {module_name}")
        except Exception as e:
            logger.debug(f"{vendor_name} operators not available: {e}")


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all built-in operator implementations.

    This function registers:
    - DEFAULT implementations (FlagGems)
    - REFERENCE implementations (PyTorch)
    - VENDOR implementations (auto-discovered)
    - External plugins (via entry points and environment variable)

    Args:
        registry: Registry to register into
    """
    # Register FlagGems (DEFAULT) implementations
    try:
        from .backends.flaggems.register_ops import register_builtins as register_flaggems

        register_flaggems(registry)
        logger.debug("Registered FlagGems operators")
    except Exception as e:
        logger.warning(f"Failed to register FlagGems operators: {e}")

    # Register PyTorch (REFERENCE) implementations
    try:
        from .backends.reference.register_ops import register_builtins as register_reference

        register_reference(registry)
        logger.debug("Registered Reference operators")
    except Exception as e:
        logger.warning(f"Failed to register Reference operators: {e}")

    # Auto-discover and register VENDOR implementations
    _register_vendor_backends(registry)

    # Discover and register external plugins
    try:
        from .discovery import discover_plugins
        plugin_count = discover_plugins(registry)
        if plugin_count > 0:
            logger.debug(f"Registered {plugin_count} external plugins")
    except Exception as e:
        logger.debug(f"Plugin discovery failed: {e}")
