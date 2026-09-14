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

import hashlib
import json
import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Iterable, Optional

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


@dataclass(frozen=True)
class FrozenOpSelection:
    """One logical operator selected for a frozen execution phase."""

    op_name: str
    impl_id: str
    kind: str
    vendor: Optional[str]
    callable_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "impl_id": self.impl_id,
            "kind": self.kind,
            "vendor": self.vendor,
            "callable": self.callable_name,
        }


@dataclass(frozen=True)
class FrozenDispatchManifest:
    """Stable dispatch choices used by one compiled execution phase."""

    policy_fingerprint: str
    selections: tuple[FrozenOpSelection, ...]
    unresolved: tuple[tuple[str, str], ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_fingerprint": self.policy_fingerprint,
            "selections": [selection.to_dict() for selection in self.selections],
            "unresolved": [
                {"op_name": op_name, "error": error}
                for op_name, error in self.unresolved
            ],
            "fingerprint": self.fingerprint,
        }


_CACHED_OPS: "weakref.WeakSet[CachedOp]" = weakref.WeakSet()
_FREEZE_LOCK = threading.RLock()
_DISPATCH_FROZEN = False
_FROZEN_MANIFEST: Optional[FrozenDispatchManifest] = None


def _callable_name(fn: Any) -> str:
    module = getattr(fn, "__module__", type(fn).__module__)
    qualname = getattr(fn, "__qualname__", type(fn).__qualname__)
    return f"{module}.{qualname}"


def _is_torch_compiling() -> bool:
    """Query Dynamo lazily so importing dispatch does not require torch."""

    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except (AttributeError, ImportError):
        return False


def is_dispatch_frozen() -> bool:
    """Return whether runtime policy selection has been frozen."""

    return _DISPATCH_FROZEN


def get_frozen_dispatch_manifest() -> Optional[FrozenDispatchManifest]:
    """Return the manifest for the current frozen execution phase."""

    return _FROZEN_MANIFEST


def _make_frozen_manifest(
    policy_fingerprint: str,
    resolved: dict[str, OpImpl],
    unresolved: dict[str, str],
) -> FrozenDispatchManifest:
    selections = tuple(
        FrozenOpSelection(
            op_name=op_name,
            impl_id=impl.impl_id,
            kind=impl.kind.value,
            vendor=impl.vendor,
            callable_name=_callable_name(impl.fn),
        )
        for op_name, impl in sorted(resolved.items())
    )
    unresolved_items = tuple(sorted(unresolved.items()))
    payload = {
        "policy_fingerprint": policy_fingerprint,
        "selections": [selection.to_dict() for selection in selections],
        "unresolved": list(unresolved_items),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FrozenDispatchManifest(
        policy_fingerprint=policy_fingerprint,
        selections=selections,
        unresolved=unresolved_items,
        fingerprint=fingerprint,
    )


def freeze_dispatch(
    op_names: Optional[Iterable[str]] = None,
    *,
    strict: bool = True,
) -> FrozenDispatchManifest:
    """Resolve ``CachedOp`` instances before Dynamo starts tracing.

    Frozen calls contain no policy lookup, manager access, lock, IO dump, or
    exception-driven backend fallback.  ``strict=False`` is useful for a model
    process that imported optional operator modules it will never execute; an
    unresolved operator still fails immediately if it is called later.
    """

    global _DISPATCH_FROZEN, _FROZEN_MANIFEST

    requested = set(op_names) if op_names is not None else None
    with _FREEZE_LOCK:
        mgr = get_default_manager()
        mgr.ensure_initialized()

        instances = [
            cached_op
            for cached_op in list(_CACHED_OPS)
            if requested is None or cached_op.op_name in requested
        ]
        names = sorted({cached_op.op_name for cached_op in instances})

        resolved: dict[str, OpImpl] = {}
        unresolved: dict[str, str] = {}
        for op_name in names:
            try:
                impl = mgr._resolve_impl(op_name)
                mgr._record_first_use(op_name, impl)
                resolved[op_name] = impl
            except Exception as exc:
                unresolved[op_name] = f"{type(exc).__name__}: {exc}"

        if strict and unresolved:
            details = "; ".join(
                f"{op_name}: {error}" for op_name, error in unresolved.items()
            )
            raise RuntimeError(f"Unable to freeze vLLM-FL dispatch: {details}")

        for cached_op in instances:
            cached_op._frozen_impl = resolved.get(cached_op.op_name)
            cached_op._freeze_error = unresolved.get(cached_op.op_name)

        manifest = _make_frozen_manifest(
            get_policy().fingerprint(), resolved, unresolved
        )
        _FROZEN_MANIFEST = manifest
        _DISPATCH_FROZEN = True
        return manifest


def thaw_dispatch() -> None:
    """Leave the frozen phase.

    Existing compiled graphs must be discarded before this is used in a model
    process.  The function primarily exists for post-fork reset and tests.
    """

    global _DISPATCH_FROZEN, _FROZEN_MANIFEST

    with _FREEZE_LOCK:
        _DISPATCH_FROZEN = False
        _FROZEN_MANIFEST = None
        for cached_op in list(_CACHED_OPS):
            cached_op._clear_all_caches()


def _reset_frozen_dispatch_after_fork() -> None:
    """Reset without acquiring a lock that may be owned by a vanished thread."""

    global _FREEZE_LOCK, _DISPATCH_FROZEN, _FROZEN_MANIFEST

    _FREEZE_LOCK = threading.RLock()
    _DISPATCH_FROZEN = False
    _FROZEN_MANIFEST = None
    for cached_op in list(_CACHED_OPS):
        cached_op._clear_all_caches()


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
        "__weakref__",
        "_op_name",
        "_impl",
        "_frozen_impl",
        "_freeze_error",
        "_use_manager_call",
        "_manager_id",
        "_manager_epoch",
        "_policy_epoch",
    )

    def __init__(self, op_name: str) -> None:
        self._op_name = op_name
        self._impl = None
        self._frozen_impl = None
        self._freeze_error = None
        self._use_manager_call = False
        self._manager_id = -1
        self._manager_epoch = -1
        self._policy_epoch = -1
        _CACHED_OPS.add(self)

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def frozen_impl_id(self) -> Optional[str]:
        impl = self._frozen_impl
        return None if impl is None else impl.impl_id

    def _clear_all_caches(self) -> None:
        self._impl = None
        self._frozen_impl = None
        self._freeze_error = None
        self._use_manager_call = False
        self._manager_id = -1
        self._manager_epoch = -1
        self._policy_epoch = -1

    def __call__(self, *args, **kwargs):
        # This is the only path used by a frozen compiled runner.  Keep it
        # before every manager/policy/debug check so Dynamo sees a stable
        # callable and no dispatch control-plane state.
        frozen_impl = self._frozen_impl
        if frozen_impl is not None:
            return frozen_impl.fn(*args, **kwargs)

        if _DISPATCH_FROZEN:
            detail = f" ({self._freeze_error})" if self._freeze_error else ""
            raise RuntimeError(
                f"CachedOp '{self._op_name}' was not bound before the dispatch "
                f"phase was frozen{detail}. Rebuild the compiled runner after "
                "registering the operator."
            )

        if _is_torch_compiling():
            raise RuntimeError(
                f"CachedOp '{self._op_name}' entered torch.compile before "
                "vllm_fl.dispatch.freeze_dispatch() was called"
            )

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
    "FrozenOpSelection",
    "FrozenDispatchManifest",
    "freeze_dispatch",
    "thaw_dispatch",
    "is_dispatch_frozen",
    "get_frozen_dispatch_manifest",
]


# A manager and its registrations are process-local.  Never inherit frozen
# callables across a fork; each worker binds again after loading its model.
try:
    os.register_at_fork(after_in_child=_reset_frozen_dispatch_after_fork)
except AttributeError:
    pass
