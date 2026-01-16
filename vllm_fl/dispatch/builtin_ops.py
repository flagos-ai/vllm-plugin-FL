# Copyright (c) 2026 BAAI. All rights reserved.

"""
Built-in operator implementations registration.

This module registers DEFAULT (FlagGems) and REFERENCE (PyTorch) implementations
for all supported operators by calling register_builtins from each backend.
"""

from __future__ import annotations

import importlib
import os

from .registry import OpRegistry
from .logger_manager import get_logger

logger = get_logger()

# Directory containing vendor backends
_VENDOR_BACKENDS_DIR = os.path.join(os.path.dirname(__file__), "backends", "vendor")


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

    for vendor_name in os.listdir(_VENDOR_BACKENDS_DIR):
        vendor_path = os.path.join(_VENDOR_BACKENDS_DIR, vendor_name)

        # Skip non-directories and special files
        if not os.path.isdir(vendor_path) or vendor_name.startswith("_"):
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
