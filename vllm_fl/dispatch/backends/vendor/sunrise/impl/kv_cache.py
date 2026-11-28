# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Zero-copy Sunrise kernel views over vLLM 0.32 logical KV storage."""

import torch


def split_kv_cache(cache: torch.Tensor, head_size: int):
    """Return K and V as [blocks, tokens, heads, dim] without copying storage."""
    if cache.ndim == 5:
        # Legacy vendor cache: [2, blocks, tokens, heads, dim].
        return cache.unbind(0)
    if cache.ndim != 4 or cache.shape[-1] != 2 * head_size:
        raise ValueError(
            "Sunrise requires dense unquantized KV with equal K/V head sizes; "
            f"got {tuple(cache.shape)} for head_size={head_size}."
        )
    # The native 0.32 allocator owns [blocks, heads, tokens, K+V] storage.
    # Preserve its layer packing, block stride and scheduler-visible block IDs.
    return (
        cache[..., :head_size].transpose(1, 2),
        cache[..., head_size:].transpose(1, 2),
    )
