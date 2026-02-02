# Copyright (c) 2026 BAAI. All rights reserved.

"""
Plugin discovery mechanism for vllm-plugin-FL dispatch.

This module provides functionality to discover and load external plugins
that can register additional operator implementations.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, List, Optional, Tuple

from .logger_manager import get_logger

# Entry point group name for plugin discovery
PLUGIN_GROUP = "vllm_fl.plugin"

# Environment variable for specifying plugin modules
PLUGIN_MODULES_ENV = "VLLM_FL_PLUGIN_MODULES"

logger = get_logger()

# Track discovered plugins: (name, source, success)
_discovered_plugins: List[Tuple[str, str, bool]] = []


def _get_entry_points():
    """
    Get entry points for the plugin group.

    Returns:
        List of entry points for the vllm_fl.plugin group
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        try:
            from importlib_metadata import entry_points
        except ImportError:
            logger.debug(
                "importlib.metadata not available, skipping entry points discovery"
            )
            return []

    try:
        eps = entry_points()

        # Python 3.10+ style
        if hasattr(eps, "select"):
            return list(eps.select(group=PLUGIN_GROUP))

        # Python 3.9 style (dict-like)
        if isinstance(eps, dict):
            return eps.get(PLUGIN_GROUP, [])

        # Fallback for older versions
        if hasattr(eps, "get"):
            return eps.get(PLUGIN_GROUP, [])

        return []

    except Exception as e:
        logger.warning(f"Error accessing entry points: {e}")
        return []


def _call_register_function(
    obj: Any,
    registry: Any,
    source_name: str,
) -> bool:
    """
    Call the register function on a plugin object.

    Args:
        obj: Plugin object (module or callable)
        registry: OpRegistry instance to register into
        source_name: Name of the plugin source for logging

    Returns:
        True if registration was successful, False otherwise
    """
    # If obj is directly callable (not a class), call it
    if callable(obj) and not isinstance(obj, type):
        try:
            obj(registry)
            logger.info(f"Registered plugin from {source_name} (direct callable)")
            return True
        except Exception as e:
            logger.error(f"Error calling plugin {source_name}: {e}")
            return False

    # Look for register function
    register_fn = getattr(obj, "vllm_fl_register", None) or getattr(
        obj, "register", None
    )

    if callable(register_fn):
        try:
            register_fn(registry)
            logger.info(f"Registered plugin from {source_name}")
            return True
        except Exception as e:
            logger.error(f"Error calling register function in {source_name}: {e}")
            return False

    logger.debug(f"No register function found in {source_name}")
    return False


def discover_from_entry_points(registry: Any) -> int:
    """
    Discover and load plugins from entry points.

    Args:
        registry: OpRegistry instance to register into

    Returns:
        Number of successfully loaded plugins
    """
    loaded = 0
    entry_points_list = _get_entry_points()

    if not entry_points_list:
        logger.debug(f"No entry points found for group: {PLUGIN_GROUP}")
        return 0

    logger.debug(f"Found {len(entry_points_list)} entry points")

    for ep in entry_points_list:
        ep_name = getattr(ep, "name", str(ep))
        try:
            logger.debug(f"Loading entry point: {ep_name}")
            obj = ep.load()

            if _call_register_function(obj, registry, f"entry_point:{ep_name}"):
                _discovered_plugins.append((ep_name, "entry_point", True))
                loaded += 1
            else:
                _discovered_plugins.append((ep_name, "entry_point", False))

        except Exception as e:
            logger.error(f"Failed to load entry point {ep_name}: {e}")
            _discovered_plugins.append((ep_name, "entry_point", False))

    return loaded


def discover_from_env_modules(registry: Any) -> int:
    """
    Discover and load plugins from environment variable.

    The VLLM_FL_PLUGIN_MODULES environment variable should contain
    a comma-separated list of module names to import.

    Args:
        registry: OpRegistry instance to register into

    Returns:
        Number of successfully loaded plugins
    """
    modules_str = os.environ.get(PLUGIN_MODULES_ENV, "").strip()

    if not modules_str:
        return 0

    loaded = 0
    module_names = [m.strip() for m in modules_str.split(",") if m.strip()]

    logger.debug(f"Loading plugins from env var: {module_names}")

    for mod_name in module_names:
        try:
            logger.debug(f"Importing module: {mod_name}")
            mod = importlib.import_module(mod_name)

            if _call_register_function(mod, registry, f"env_module:{mod_name}"):
                _discovered_plugins.append((mod_name, "env_module", True))
                loaded += 1
            else:
                _discovered_plugins.append((mod_name, "env_module", False))

        except ImportError as e:
            logger.error(f"Failed to import plugin module {mod_name}: {e}")
            _discovered_plugins.append((mod_name, "env_module", False))
        except Exception as e:
            logger.error(f"Error loading plugin module {mod_name}: {e}")
            _discovered_plugins.append((mod_name, "env_module", False))

    return loaded


def discover_plugins(registry: Any) -> int:
    """
    Main plugin discovery function.

    Discovers and registers plugins from:
    1. Entry points (group: 'vllm_fl.plugin')
    2. Environment variable modules (VLLM_FL_PLUGIN_MODULES)

    Args:
        registry: OpRegistry instance to register plugins to

    Returns:
        Number of successfully loaded plugins
    """
    if registry is None:
        logger.warning("Registry is None, skipping plugin discovery")
        return 0

    logger.debug("Starting plugin discovery...")

    total = 0

    # Discover from entry points
    total += discover_from_entry_points(registry)

    # Discover from environment variable
    total += discover_from_env_modules(registry)

    logger.debug(f"Plugin discovery complete. Loaded {total} plugins.")

    return total


def get_discovered_plugins() -> List[Tuple[str, str, bool]]:
    """
    Get list of discovered plugins.

    Returns:
        List of tuples (name, source, success)
    """
    return _discovered_plugins.copy()


def clear_discovered_plugins() -> None:
    """Clear the discovered plugins list (for testing)."""
    _discovered_plugins.clear()
