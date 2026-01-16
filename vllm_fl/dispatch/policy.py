# Copyright (c) 2026 BAAI. All rights reserved.

"""
Selection policy management for operator dispatch.
"""

from __future__ import annotations

import contextvars
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# Valid preference values for VLLM_FL_PREFER
PREFER_DEFAULT = "flaggems"
PREFER_VENDOR = "vendor"
PREFER_REFERENCE = "reference"

VALID_PREFER_VALUES = frozenset({PREFER_DEFAULT, PREFER_VENDOR, PREFER_REFERENCE})


@dataclass(frozen=True)
class SelectionPolicy:
    """
    Policy for selecting operator implementations.

    Attributes:
        prefer: Which implementation kind to prefer. One of:
            - "flaggems": Prefer DEFAULT (FlagGems) implementations
            - "vendor": Prefer VENDOR (CUDA) implementations
            - "reference": Prefer REFERENCE (PyTorch) implementations
        strict: If True, raise error when primary implementation fails
        per_op_order: Per-operator custom selection order
        deny_vendors: Set of vendor names to deny
        allow_vendors: Set of vendor names to allow (whitelist)
    """
    prefer: str = PREFER_DEFAULT
    strict: bool = False
    per_op_order: Tuple[Tuple[str, Tuple[str, ...]], ...] = field(default_factory=tuple)
    deny_vendors: FrozenSet[str] = field(default_factory=frozenset)
    allow_vendors: Optional[FrozenSet[str]] = None

    def __post_init__(self):
        if self.prefer not in VALID_PREFER_VALUES:
            raise ValueError(
                f"Invalid prefer value: '{self.prefer}'. "
                f"Must be one of: {', '.join(sorted(VALID_PREFER_VALUES))}"
            )

    @classmethod
    def from_dict(
        cls,
        prefer: str = PREFER_DEFAULT,
        strict: bool = False,
        per_op_order: Optional[Dict[str, List[str]]] = None,
        deny_vendors: Optional[Set[str]] = None,
        allow_vendors: Optional[Set[str]] = None,
    ) -> "SelectionPolicy":
        """Create a SelectionPolicy from dictionary-like arguments."""
        per_op_tuple = tuple()
        if per_op_order:
            per_op_tuple = tuple(
                (k, tuple(v)) for k, v in sorted(per_op_order.items())
            )

        return cls(
            prefer=prefer.lower(),
            strict=strict,
            per_op_order=per_op_tuple,
            deny_vendors=frozenset(deny_vendors) if deny_vendors else frozenset(),
            allow_vendors=frozenset(allow_vendors) if allow_vendors else None,
        )

    @property
    def per_op_order_dict(self) -> Dict[str, List[str]]:
        """Get per_op_order as a mutable dict for easier access."""
        return {k: list(v) for k, v in self.per_op_order}

    def get_per_op_order(self, op_name: str) -> Optional[List[str]]:
        """Get order for a specific operator."""
        for name, order in self.per_op_order:
            if name == op_name:
                return list(order)
        return None

    def get_default_order(self) -> List[str]:
        """Get the default selection order based on preference setting."""
        if self.prefer == PREFER_REFERENCE:
            return ["reference", "flaggems", "vendor"]
        elif self.prefer == PREFER_VENDOR:
            return ["vendor", "flaggems", "reference"]
        else:  # PREFER_DEFAULT
            return ["flaggems", "vendor", "reference"]

    def is_vendor_allowed(self, vendor_name: str) -> bool:
        """Check if a vendor is allowed by this policy."""
        if vendor_name in self.deny_vendors:
            return False
        if self.allow_vendors is not None and vendor_name not in self.allow_vendors:
            return False
        return True

    def fingerprint(self) -> str:
        """Generate a unique fingerprint for this policy (used for caching)."""
        parts = [
            f"prefer={self.prefer}",
            f"st={int(self.strict)}",
        ]

        if self.allow_vendors:
            parts.append(f"allow={','.join(sorted(self.allow_vendors))}")

        if self.deny_vendors:
            parts.append(f"deny={','.join(sorted(self.deny_vendors))}")

        if self.per_op_order:
            per_op_str = ";".join(
                f"{k}={'|'.join(v)}" for k, v in self.per_op_order
            )
            parts.append(f"per={per_op_str}")

        return ";".join(parts)

    def __hash__(self) -> int:
        return hash((
            self.prefer,
            self.strict,
            self.per_op_order,
            self.deny_vendors,
            self.allow_vendors,
        ))


class PolicyManager:
    """
    Singleton manager for selection policies.

    Supports:
    - Global policy (from environment or set programmatically)
    - Context-local policy (using context managers)
    - Policy epoch tracking for cache invalidation
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        if hasattr(self, '_policy_epoch'):
            return

        self._policy_epoch = 0
        self._policy_epoch_lock = threading.Lock()
        self._global_policy = None
        self._global_policy_lock = threading.Lock()

        self._policy_var = contextvars.ContextVar(
            "vllm_fl_selection_policy",
            default=None,
        )

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__()
        return cls._instance

    def get_policy_epoch(self) -> int:
        """Get the current policy epoch."""
        return self._policy_epoch

    def bump_policy_epoch(self) -> int:
        """Bump the policy epoch and return the new value."""
        with self._policy_epoch_lock:
            self._policy_epoch += 1
            return self._policy_epoch

    def get_policy(self) -> SelectionPolicy:
        """Get the current effective policy (context or global)."""
        ctx_policy = self._policy_var.get()
        if ctx_policy is not None:
            return ctx_policy

        if self._global_policy is None:
            with self._global_policy_lock:
                if self._global_policy is None:
                    self._global_policy = self._policy_from_env()
        return self._global_policy

    def set_global_policy(self, policy: SelectionPolicy) -> SelectionPolicy:
        """Set the global policy and return the old policy."""
        with self._global_policy_lock:
            old_policy = self._global_policy
            self._global_policy = policy
            self.bump_policy_epoch()
            return old_policy if old_policy else self._policy_from_env()

    def reset_global_policy(self) -> None:
        """Reset the global policy to environment defaults."""
        with self._global_policy_lock:
            self._global_policy = None
            self.bump_policy_epoch()

    def create_policy_context(self, policy: SelectionPolicy):
        """Create a context manager for temporary policy override."""
        return _PolicyContext(self, policy)

    def _get_policy_var(self):
        return self._policy_var

    @staticmethod
    def _parse_csv_set(value: str) -> Set[str]:
        """Parse a comma-separated string into a set."""
        if not value:
            return set()
        return {x.strip() for x in value.split(",") if x.strip()}

    @staticmethod
    def _parse_per_op(value: str) -> Dict[str, List[str]]:
        """Parse per-op order string (format: op1=a|b|c;op2=x|y)."""
        if not value:
            return {}

        result: Dict[str, List[str]] = {}
        parts = [p.strip() for p in value.split(";") if p.strip()]

        for part in parts:
            if "=" not in part:
                continue
            op_name, order_str = part.split("=", 1)
            op_name = op_name.strip()
            order = [x.strip() for x in order_str.split("|") if x.strip()]
            if op_name and order:
                result[op_name] = order

        return result

    def _policy_from_env(self) -> SelectionPolicy:
        """
        Create a SelectionPolicy from environment variables.

        Environment variables:
        - VLLM_FL_PREFER: Preference (flaggems, vendor, reference)
        - VLLM_FL_STRICT: Enable strict mode (1 or 0)
        - VLLM_FL_DENY_VENDORS: Comma-separated list of denied vendors
        - VLLM_FL_ALLOW_VENDORS: Comma-separated list of allowed vendors
        - VLLM_FL_PER_OP: Per-op order (format: op1=a|b|c;op2=x|y)
        """
        prefer_str = os.environ.get("VLLM_FL_PREFER", "").strip().lower()
        if prefer_str and prefer_str in VALID_PREFER_VALUES:
            pass
        else:
            prefer_str = PREFER_DEFAULT

        strict = os.environ.get("VLLM_FL_STRICT", "0").strip() == "1"

        deny_str = os.environ.get("VLLM_FL_DENY_VENDORS", "").strip()
        deny_vendors = self._parse_csv_set(deny_str) if deny_str else None

        allow_str = os.environ.get("VLLM_FL_ALLOW_VENDORS", "").strip()
        allow_vendors = self._parse_csv_set(allow_str) if allow_str else None

        per_op_str = os.environ.get("VLLM_FL_PER_OP", "").strip()
        per_op_order = self._parse_per_op(per_op_str) if per_op_str else None

        return SelectionPolicy.from_dict(
            prefer=prefer_str,
            strict=strict,
            per_op_order=per_op_order,
            deny_vendors=deny_vendors,
            allow_vendors=allow_vendors,
        )


class _PolicyContext:
    """Context manager for temporary policy override."""

    def __init__(self, manager: PolicyManager, policy: SelectionPolicy):
        self._manager = manager
        self._policy = policy
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "_PolicyContext":
        policy_var = self._manager._get_policy_var()
        self._token = policy_var.set(self._policy)
        self._manager.bump_policy_epoch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            policy_var = self._manager._get_policy_var()
            policy_var.reset(self._token)
            self._manager.bump_policy_epoch()


# Convenience functions for easier access
def get_policy_epoch() -> int:
    """Get the current policy epoch."""
    return PolicyManager.get_instance().get_policy_epoch()


def bump_policy_epoch() -> int:
    """Bump the policy epoch and return the new value."""
    return PolicyManager.get_instance().bump_policy_epoch()


def get_policy() -> SelectionPolicy:
    """Get the current effective policy (context or global)."""
    return PolicyManager.get_instance().get_policy()


def set_global_policy(policy: SelectionPolicy) -> SelectionPolicy:
    """Set the global policy and return the old policy."""
    return PolicyManager.get_instance().set_global_policy(policy)


def reset_global_policy() -> None:
    """Reset the global policy to environment defaults."""
    PolicyManager.get_instance().reset_global_policy()


def policy_from_env() -> SelectionPolicy:
    """Create a SelectionPolicy from environment variables."""
    return PolicyManager.get_instance()._policy_from_env()


def policy_context(policy: SelectionPolicy) -> _PolicyContext:
    """
    Create a context manager to temporarily override the policy.

    Example:
        >>> with policy_context(my_policy):
        ...     # Use my_policy in this context
        ...     result = manager.resolve("op_name")
    """
    return _PolicyContext(PolicyManager.get_instance(), policy)


# Convenience context managers
def with_strict_mode() -> _PolicyContext:
    """Context manager to enable strict mode."""
    current = get_policy()
    strict_policy = SelectionPolicy.from_dict(
        prefer=current.prefer,
        strict=True,
        per_op_order={k: list(v) for k, v in current.per_op_order},
        deny_vendors=set(current.deny_vendors),
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    )
    return policy_context(strict_policy)


def with_preference(prefer: str) -> _PolicyContext:
    """
    Context manager to set implementation preference.

    Args:
        prefer: One of "flaggems", "vendor", or "reference"

    Example:
        >>> with with_preference("vendor"):
        ...     # Prefer vendor implementations in this context
        ...     result = manager.resolve("op_name")
    """
    current = get_policy()
    policy = SelectionPolicy.from_dict(
        prefer=prefer,
        strict=current.strict,
        per_op_order={k: list(v) for k, v in current.per_op_order},
        deny_vendors=set(current.deny_vendors),
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    )
    return policy_context(policy)


def with_allowed_vendors(*vendors: str) -> _PolicyContext:
    """Context manager to set allowed vendors whitelist."""
    current = get_policy()
    policy = SelectionPolicy.from_dict(
        prefer=current.prefer,
        strict=current.strict,
        per_op_order={k: list(v) for k, v in current.per_op_order},
        deny_vendors=set(current.deny_vendors),
        allow_vendors=set(vendors),
    )
    return policy_context(policy)


def with_denied_vendors(*vendors: str) -> _PolicyContext:
    """Context manager to add denied vendors to blacklist."""
    current = get_policy()
    denied = set(current.deny_vendors)
    denied.update(vendors)
    policy = SelectionPolicy.from_dict(
        prefer=current.prefer,
        strict=current.strict,
        per_op_order={k: list(v) for k, v in current.per_op_order},
        deny_vendors=denied,
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    )
    return policy_context(policy)
