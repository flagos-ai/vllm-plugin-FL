# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend rotary embedding operator implementations.
Based on vllm-ascend official implementation.
"""

from __future__ import annotations

import torch


def _can_use_ascend_rotary_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> bool:
    supported_dtypes = (torch.float16, torch.bfloat16)
    return (
        query.dim() == 3
        and key.dim() == 3
        and cos.dim() == 2
        and sin.dim() == 2
        and position_ids.dim() == 1
        and query.shape[0] == key.shape[0] == position_ids.numel()
        and query.shape[0] > 0
        and query.shape[1] > 0
        and key.shape[1] > 0
        and query.shape[-1] == key.shape[-1]
        and query.shape[-1] > 0
        and query.shape[-1] % 2 == 0
        and cos.shape == sin.shape
        and cos.shape[-1] * 2 == query.shape[-1]
        and position_ids.is_contiguous()
        and query.device.type == "npu"
        and query.dtype in supported_dtypes
        and key.device == query.device
        and key.dtype == query.dtype
        and cos.device == query.device
        and cos.dtype == query.dtype
        and sin.device == query.device
        and sin.dtype == query.dtype
        and position_ids.device == query.device
        and position_ids.dtype == torch.int64
    )


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
    # ATB setup errors are asynchronous and cannot be recovered by the normal
    # dispatch fallback. Validate its complete device/dtype contract before
    # importing torch_npu or launching the kernel.
    position_ids = position_ids.contiguous()
    if not _can_use_ascend_rotary_kernel(query, key, cos, sin, position_ids):
        from vllm_fl.dispatch.backends.reference.impl.rotary import (
            rotary_embedding_torch,
        )

        return rotary_embedding_torch(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved,
            inplace,
        )

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
    query_work = query if inplace else query.clone()
    key_work = key if inplace else key.clone()
    query_flat = query_work.contiguous().view(num_tokens, -1)
    key_flat = key_work.contiguous().view(num_tokens, -1)

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

    if inplace:
        if query_work is not query or not query.is_contiguous():
            query.copy_(q_embed)
        if key_work is not key or not key.is_contiguous():
            key.copy_(k_embed)
        return query, key

    return q_embed, k_embed
