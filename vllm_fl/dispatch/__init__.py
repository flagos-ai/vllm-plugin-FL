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
    fn = manager.resolve("rms_norm")
    result = fn(x, residual, weight, epsilon)

Environment Variables:
    VLLM_FL_CONFIG: Path to YAML configuration file (highest priority, overrides env vars)
    VLLM_FL_PREFER: Preferred backend ("flagos", "vendor", "reference")
    VLLM_FL_STRICT: Strict mode: "1" = fail immediately on error (no fallback), "0" = try fallback (default)
    VLLM_FL_DENY_VENDORS: Comma-separated list of denied vendors
    VLLM_FL_ALLOW_VENDORS: Comma-separated list of allowed vendors
    VLLM_FL_PER_OP: Per-operator order (format: op1=a|b|c;op2=x|y)
    VLLM_FL_PLUGIN_MODULES: Comma-separated list of plugin modules to load
    VLLM_FL_LOG_LEVEL: Log level for dispatch module (DEBUG, INFO, WARNING, ERROR)
    VLLM_FL_DISPATCH_DEBUG: Enable debug printing ("1" or "0", default: "0")
        When enabled, prints:
        - Detailed list of registered operators and implementations at initialization
        - Selected backend for each operator call

Configuration File (YAML):
    When VLLM_FL_CONFIG is set, the dispatch system loads configuration from the
    specified YAML file. Example:

        # vllm_fl_dispatch.yaml

        # Preferred backend type: flagos, vendor, or reference
        prefer: vendor

        # Strict mode:
        #   true  = fail immediately on error, no fallback
        #   false = try next backend on failure (default)
        strict: true

        # Vendor whitelist (optional)
        allow_vendors:
          - cuda

        # Vendor blacklist (optional)
        deny_vendors:
          - ascend

        # Per-operator backend selection order (optional)
        # Only the backends listed will be tried, in the specified order.
        # If you only list 2 options, only those 2 will be attempted.
        #
        # Supported tokens:
        #   - flagos        : FlagOS default implementation
        #   - reference     : PyTorch reference implementation
        #   - vendor        : Any available vendor backend (auto-detect)
        #   - vendor:cuda   : Only CUDA vendor backend
        #   - vendor:ascend : Only Ascend vendor backend
        op_backends:
          rms_norm:
            - vendor        # Try any available vendor first
            - flagos        # Then try flagos
            # reference not listed, so it won't be used

          silu_and_mul:
            - vendor:cuda   # Only try CUDA, not other vendors
            - flagos
            - reference
"""

import os

from .types import OpImpl, BackendImplKind, BackendPriority, match_token
from .registry import OpRegistry, OpRegistrySnapshot
from .policy import (
    SelectionPolicy,
    PolicyManager,
    get_policy,
    get_policy_epoch,
    set_global_policy,
    reset_global_policy,
    policy_context,
    policy_from_config,
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
from .io_dumper import (
    enable_io_dump,
    disable_io_dump,
    io_dump_step,
    is_dump_enabled,
)
from .io_common import list_model_layers, register_tensor_stat, tensor_stats


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


# Fast-path opt-out: set VLLM_FL_OP_FAST_PATH=0 to disable per-op fn caching
# in hot OOT layers and route every call back through OpManager.call.
_OP_FAST_PATH_ENABLED = os.environ.get("VLLM_FL_OP_FAST_PATH", "1") == "1"


class CachedOp:
    """Resolve an op once at the call site and refresh on policy changes.

    OpManager.call preserves fallback and IO-dump hooks, but it also pays the
    manager/fallback path on every invocation. Hot layer paths can use CachedOp
    to call the resolved implementation directly after the first lookup.

    The cache is invalidated by both OpManager.policy_epoch and
    PolicyManager.policy_epoch. The latter matters for policy_context() and
    set_global_policy(), which can change the effective backend without
    touching the OpManager instance.

    Cache refresh is best-effort under concurrent calls. If another thread
    changes policy at the same time, a call may observe the previous impl once
    before the next epoch check refreshes it.
    """

    __slots__ = (
        "_op_name",
        "_impl",
        "_use_manager_call",
        "_manager_id",
        "_manager_epoch",
        "_policy_epoch",
    )

    def __init__(self, op_name: str) -> None:
        self._op_name = op_name
        self._impl = None
        self._use_manager_call = False
        self._manager_id = -1
        self._manager_epoch = -1
        self._policy_epoch = -1

    def __call__(self, *args, **kwargs):
        mgr = get_default_manager()

        if not _OP_FAST_PATH_ENABLED:
            return mgr.call(self._op_name, *args, **kwargs)

        if is_dump_enabled():
            return mgr.call(self._op_name, *args, **kwargs)

        manager_epoch = mgr.policy_epoch
        manager_id = id(mgr)
        policy_epoch = get_policy_epoch()
        if (
            self._manager_id != manager_id
            or self._manager_epoch != manager_epoch
            or self._policy_epoch != policy_epoch
        ):
            self._impl = None
            self._use_manager_call = False

        if self._use_manager_call:
            return mgr.call(self._op_name, *args, **kwargs)

        impl = self._impl
        if (
            impl is None
            or self._manager_id != manager_id
            or self._manager_epoch != manager_epoch
            or self._policy_epoch != policy_epoch
        ):
            impl = mgr._resolve_impl(self._op_name)
            mgr._record_first_use(self._op_name, impl)
            self._impl = impl
            # resolve() can initialize the manager and bump its epoch.
            self._manager_id = manager_id
            self._manager_epoch = mgr.policy_epoch
            self._policy_epoch = get_policy_epoch()

        try:
            return impl.fn(*args, **kwargs)
        except Exception:
            self._impl = None
            if get_policy().strict:
                raise
            mgr._mark_failed_impl(self._op_name, impl.impl_id)
            self._use_manager_call = True
            return mgr.call(self._op_name, *args, **kwargs)


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
    "get_policy_epoch",
    "set_global_policy",
    "reset_global_policy",
    "policy_context",
    "policy_from_config",
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
    # IO Dump
    "enable_io_dump",
    "disable_io_dump",
    "io_dump_step",
    "list_model_layers",
    "register_tensor_stat",
    "tensor_stats",
    # Convenience functions
    "call_op",
    "resolve_op",
    "CachedOp",
]
