# Copyright (c) 2026 BAAI. All rights reserved.
"""Reuse the shared reference implementation; no vendor-specific math."""

from vllm_fl.dispatch.backends.reference.impl.normalization import (
    rms_norm_torch as rms_norm_iluvatar,
)

__all__ = ["rms_norm_iluvatar"]
