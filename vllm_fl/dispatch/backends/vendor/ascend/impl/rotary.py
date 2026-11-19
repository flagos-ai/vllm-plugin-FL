# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend rotary embedding operator implementations.
Based on vllm-ascend official implementation.
"""

from __future__ import annotations

import torch


def _validate_torch_npu_rotary(query, key, cos, sin, position_ids):
    if (
        query.device.type not in ("npu", "privateuseone")
        or query.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or query.ndim != 3
        or query.shape[-1] not in (64, 128)
    ):
        raise ValueError(
            "Ascend rotary multiplication requires NPU [tokens, heads, 64 or 128] "
            "FP16, BF16 or FP32 tensors"
        )
    if key is not None and (
        key.ndim != 3
        or key.shape[0] != query.shape[0]
        or key.shape[-1] != query.shape[-1]
        or key.device != query.device
        or key.dtype != query.dtype
    ):
        raise ValueError(
            "Rotary query and key require the same tokens, width, device and dtype"
        )
    if (
        cos.ndim != 2
        or cos.shape != sin.shape
        or cos.shape[-1] * 2 != query.shape[-1]
        or cos.device != query.device
        or sin.device != query.device
    ):
        raise ValueError(
            "Rotary coefficient caches must match the query width and device"
        )
    if (
        position_ids.ndim != 1
        or position_ids.shape[0] != query.shape[0]
        or position_ids.device != query.device
        or position_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "RoPE requires one integer position per query token on its device"
        )


def rotary_embedding_torch_npu(
    obj,
    query: torch.Tensor,
    key: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Adapt FL's Q/K RoPE contract to the public Ascend rotary multiply."""
    _validate_torch_npu_rotary(query, key, cos, sin, position_ids)
    if query.shape[0] == 0:
        return query, key

    import torch_npu

    positions = position_ids.long()
    mode = "interleave" if rotary_interleaved else "half"

    def expand_coefficients(coefficients):
        selected = coefficients.index_select(0, positions).to(dtype=query.dtype)
        if rotary_interleaved:
            selected = selected.repeat_interleave(2, dim=-1)
        else:
            selected = torch.cat((selected, selected), dim=-1)
        return selected[:, None, None, :].contiguous()

    cos_full, sin_full = expand_coefficients(cos), expand_coefficients(sin)

    def rotate(tensor):
        return torch_npu.npu_rotary_mul(
            tensor.unsqueeze(1).contiguous(), cos_full, sin_full, rotary_mode=mode
        ).squeeze(1)

    query_out = rotate(query)
    key_out = None if key is None else rotate(key)
    if inplace:
        query.copy_(query_out)
        if key is not None:
            key.copy_(key_out)
        return query, key
    return query_out, key_out


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
