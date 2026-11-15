# Copyright (c) 2026 BAAI. All rights reserved.
"""Reuse the shared reference implementation; no vendor-specific math."""

from vllm_fl.dispatch.backends.reference.impl.rotary import (
    rotary_embedding_torch as rotary_embedding_iluvatar,
)
