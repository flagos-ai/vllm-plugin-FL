# Copyright (c) 2026 BAAI. All rights reserved.
"""Reuse the shared reference implementation; no vendor-specific math."""

from vllm_fl.dispatch.backends.reference.impl.activation import (
    silu_and_mul_torch as silu_and_mul_iluvatar,
)

__all__ = ["silu_and_mul_iluvatar"]
