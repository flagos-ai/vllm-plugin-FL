# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend rotary embedding operator implementations.
Based on vllm-ascend official implementation.
"""

from __future__ import annotations

import torch


def initialize_rotary_embedding_ascend() -> None:
    """Resolve torch_npu's lazy ATB registration before model tracing.

    Merely reading _npu_rotary_embedding returns a lazy wrapper. Its first
    call patches torch_npu/ATB module globals, which must happen outside
    forward. The matching torch_npu registration entrypoint is cached.
    """
    from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions

    _register_atb_extensions()


def rotary_embedding_ascend(
    obj,
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
        obj: The calling obj (for interface consistency)
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
    if query.dtype == torch.float32:
        cos_selected = cos[position_ids].unsqueeze(1)
        sin_selected = sin[position_ids].unsqueeze(1)
        if rotary_interleaved:
            cos_selected = cos_selected.repeat_interleave(2, dim=-1)
            sin_selected = sin_selected.repeat_interleave(2, dim=-1)

            def rotate(x):
                return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        else:
            cos_selected = torch.cat((cos_selected, cos_selected), dim=-1)
            sin_selected = torch.cat((sin_selected, sin_selected), dim=-1)

            def rotate(x):
                first, second = x.chunk(2, dim=-1)
                return torch.cat((-second, first), dim=-1)
        return tuple(x * cos_selected + rotate(x) * sin_selected for x in (query, key))

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
