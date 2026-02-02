# Copyright (c) 2026 BAAI. All rights reserved.

"""
Core type definitions for the dispatch mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Set


class BackendImplKind(str, Enum):
    """
    Kind of backend implementation.

    - DEFAULT: Default implementation (FlagGems)
    - REFERENCE: Reference implementation (PyTorch native)
    - VENDOR: Vendor-specific implementation (CUDA, etc.)
    """
    DEFAULT = "flaggems"
    REFERENCE = "reference"
    VENDOR = "vendor"

    def __str__(self) -> str:
        return self.value


class BackendPriority:
    """
    Standard priority values for different backend types.

    Higher priority implementations are selected first when available.
    """
    DEFAULT = 150     # Default implementations (FlagGems)
    VENDOR = 100      # Vendor-specific implementations
    REFERENCE = 50    # Reference implementations (PyTorch, lowest)


@dataclass(frozen=True)
class OpImpl:
    """
    Operator implementation descriptor.

    Attributes:
        op_name: Name of the operator (e.g., "silu_and_mul", "rmsnorm")
        impl_id: Unique identifier for this implementation (e.g., "default.flaggems")
        kind: Type of implementation (DEFAULT, REFERENCE, VENDOR)
        fn: The actual implementation function
        vendor: Vendor name (required if kind is VENDOR)
        priority: Priority for selection (higher = preferred)
        supported_dtypes: Set of supported data types (optional)
        min_arch: Minimum architecture requirement (optional)
    """
    op_name: str
    impl_id: str
    kind: BackendImplKind
    fn: Callable[..., Any]
    vendor: Optional[str] = None
    priority: int = 0
    supported_dtypes: Optional[Set[str]] = None
    min_arch: Optional[str] = None

    def __post_init__(self):
        if self.kind == BackendImplKind.VENDOR and not self.vendor:
            raise ValueError(
                f"OpImpl with kind=VENDOR must specify vendor name: {self.impl_id}"
            )

    def is_available(self) -> bool:
        """
        Check if this implementation is available.

        Looks for a _is_available attribute on the function.
        """
        avail_fn = getattr(self.fn, "_is_available", None)
        if callable(avail_fn):
            try:
                return bool(avail_fn())
            except Exception:
                return False
        return True


# Token patterns for matching implementations
TOKEN_PATTERNS = {
    "flaggems": lambda impl: impl.kind == BackendImplKind.DEFAULT,
    "reference": lambda impl: impl.kind == BackendImplKind.REFERENCE,
    "vendor": lambda impl: impl.kind == BackendImplKind.VENDOR,
}


def match_token(impl: OpImpl, token: str) -> bool:
    """
    Check if an implementation matches a selection token.

    Supported token formats:
    - "flaggems": Match DEFAULT implementations
    - "reference": Match REFERENCE implementations
    - "vendor": Match any VENDOR implementation
    - "vendor:CUDA": Match VENDOR with specific vendor name
    - "impl:default.flaggems": Match specific impl_id

    Args:
        impl: Implementation to check
        token: Selection token

    Returns:
        True if implementation matches the token
    """
    if token in TOKEN_PATTERNS:
        return TOKEN_PATTERNS[token](impl)

    if token.startswith("vendor:"):
        vendor_name = token.split(":", 1)[1]
        return impl.kind == BackendImplKind.VENDOR and impl.vendor == vendor_name

    if token.startswith("impl:"):
        impl_id = token.split(":", 1)[1]
        return impl.impl_id == impl_id

    return False
