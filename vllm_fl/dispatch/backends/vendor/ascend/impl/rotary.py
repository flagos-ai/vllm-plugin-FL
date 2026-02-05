# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend rotary embedding operator implementations.
Based on vllm-ascend official implementation.
"""

from __future__ import annotations

import torch


def rotary_embedding_ascend(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using Ascend NPU.

    Args:
        query: Query tensor [num_tokens, num_heads, rotary_dim]
        key: Key tensor [num_tokens, num_kv_heads, rotary_dim]
        cos: Cosine cache [max_seq_len, rotary_dim // 2]
        sin: Sine cache [max_seq_len, rotary_dim // 2]
        position_ids: Position indices [num_tokens]
        rotary_interleaved: Whether to use interleaved rotary (False = neox style)
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    import torch_npu

    # query/key shape: [num_tokens, num_heads, rotary_dim]
    num_tokens = query.shape[0]
    rotary_dim = query.shape[-1]

    # Reconstruct cos_sin_cache from separate cos and sin
    # cos/sin: [max_seq_len, rotary_dim // 2]
    # cos_sin_cache: [max_seq_len, rotary_dim] where first half is cos, second half is sin
    cos_sin_cache = torch.cat([cos, sin], dim=-1)

    # Save original shapes
    query_shape = query.shape
    key_shape = key.shape

    # Flatten query/key for _npu_rotary_embedding: [num_tokens, num_heads * rotary_dim]
    query_flat = query.contiguous().view(num_tokens, -1)
    key_flat = key.contiguous().view(num_tokens, -1)

    # is_neox_style is the opposite of rotary_interleaved
    is_neox_style = not rotary_interleaved

    # Apply rotary embedding using NPU kernel (in-place operation)
    torch_npu._npu_rotary_embedding(
        position_ids,
        query_flat,
        key_flat,
        rotary_dim,  # head_size = rotary_dim
        cos_sin_cache,
        is_neox_style,
    )

    # Restore original shapes
    q_embed = query_flat.view(query_shape)
    k_embed = key_flat.view(key_shape)

    return q_embed, k_embed
