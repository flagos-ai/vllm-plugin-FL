# Copyright (c) 2026 BAAI. All rights reserved.

"""
Dispatch mechanism for vllm-plugin-FL.

This module provides a flexible operator dispatch system that allows
selecting between different backend implementations (FlagGems, PyTorch, etc.)
based on availability and policy configuration.

Usage:
    from vllm_fl.dispatch import get_default_manager, call_op

    # Call an operator through the dispatch system
    result = call_op("silu_and_mul", x)

    # Or use the manager directly
    manager = get_default_manager()
    fn = manager.resolve("rmsnorm")
    result = fn(x, residual, weight, epsilon)

Environment Variables:
    VLLM_FL_PREFER: Preferred backend ("flaggems", "vendor", "reference")
    VLLM_FL_STRICT: Enable strict mode ("1" or "0")
    VLLM_FL_DENY_VENDORS: Comma-separated list of denied vendors
    VLLM_FL_ALLOW_VENDORS: Comma-separated list of allowed vendors
    VLLM_FL_PER_OP: Per-operator order (format: op1=a|b|c;op2=x|y)
    VLLM_FL_PLUGIN_MODULES: Comma-separated list of plugin modules to load
    VLLM_FL_LOG_LEVEL: Log level for dispatch module (DEBUG, INFO, WARNING, ERROR)
    VLLM_FL_DISPATCH_DEBUG: Enable debug printing ("1" or "0", default: "0")
        When enabled, prints:
        - Detailed list of registered operators and implementations at initialization
        - Selected backend for each operator call
"""

from .types import OpImpl, BackendImplKind, BackendPriority, match_token
from .registry import OpRegistry, OpRegistrySnapshot
from .policy import (
    SelectionPolicy,
    PolicyManager,
    get_policy,
    set_global_policy,
    reset_global_policy,
    policy_context,
    with_strict_mode,
    with_preference,
    with_allowed_vendors,
    with_denied_vendors,
    PREFER_DEFAULT,
    PREFER_VENDOR,
    PREFER_REFERENCE,
)
from .manager import OpManager, get_default_manager, reset_default_manager
from .ops import VLLMFLBackendBase
from .discovery import (
    discover_plugins,
    get_discovered_plugins,
    clear_discovered_plugins,
    PLUGIN_GROUP,
    PLUGIN_MODULES_ENV,
)
from .logger_manager import get_logger, set_log_level


def call_op(op_name: str, *args, **kwargs):
    """
    Convenience function to call an operator through the default manager.

    Args:
        op_name: Name of the operator
        *args, **kwargs: Arguments passed to the operator

    Returns:
        Result from the operator implementation
    """
    return get_default_manager().call(op_name, *args, **kwargs)


def resolve_op(op_name: str):
    """
    Convenience function to resolve an operator through the default manager.

    Args:
        op_name: Name of the operator

    Returns:
        Callable implementation function
    """
    return get_default_manager().resolve(op_name)


__all__ = [
    # Types
    "OpImpl",
    "BackendImplKind",
    "BackendPriority",
    "match_token",
    # Registry
    "OpRegistry",
    "OpRegistrySnapshot",
    # Policy
    "SelectionPolicy",
    "PolicyManager",
    "get_policy",
    "set_global_policy",
    "reset_global_policy",
    "policy_context",
    "with_strict_mode",
    "with_preference",
    "with_allowed_vendors",
    "with_denied_vendors",
    "PREFER_DEFAULT",
    "PREFER_VENDOR",
    "PREFER_REFERENCE",
    # Manager
    "OpManager",
    "get_default_manager",
    "reset_default_manager",
    # Backend base
    "VLLMFLBackendBase",
    # Plugin discovery
    "discover_plugins",
    "get_discovered_plugins",
    "clear_discovered_plugins",
    "PLUGIN_GROUP",
    "PLUGIN_MODULES_ENV",
    # Logging
    "get_logger",
    "set_log_level",
    # Convenience functions
    "call_op",
    "resolve_op",
]
