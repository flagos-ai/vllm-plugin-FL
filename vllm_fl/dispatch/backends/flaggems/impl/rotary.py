# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems rotary embedding operator implementations.
"""

from __future__ import annotations

import torch


def rotary_embedding_flaggems(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        query: Query tensor
        key: Key tensor
        cos: Cosine cache
        sin: Sine cache
        position_ids: Position indices
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    return gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=inplace,
    )


def apply_rotary_emb_flaggems(
    obj, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Adapt single-tensor ApplyRotaryEmb to the existing Q/K FlagGems op."""
    rotary_dim = cos.shape[-1] * 2
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    if rotary_dim > x.shape[-1]:
        raise ValueError("rotary_dim must not exceed head_dim")
    if x.ndim not in (3, 4):
        raise ValueError("ApplyRotaryEmb expects a 3-D or 4-D input")
    if rotary_dim == 0:
        return x

    rotary_input = x[..., :rotary_dim]
    if x.ndim == 3:
        rotary_input = rotary_input.unsqueeze(0)
    batch_size, seq_len = rotary_input.shape[:2]

    if cos.ndim == 2:
        if cos.shape[0] == 1:
            cos = cos.expand(seq_len, -1)
            sin = sin.expand(seq_len, -1)
        elif cos.shape[0] < seq_len:
            raise ValueError("cos/sin cache is shorter than the input sequence")
        cos = cos[:seq_len]
        sin = sin[:seq_len]
        position_ids = None
    elif cos.ndim == 3:
        cos = cos.expand(batch_size, seq_len, -1)
        sin = sin.expand(batch_size, seq_len, -1)
        cos = cos.reshape(batch_size * seq_len, -1)
        sin = sin.reshape(batch_size * seq_len, -1)
        position_ids = torch.arange(
            batch_size * seq_len, device=x.device, dtype=torch.int32
        ).reshape(batch_size, seq_len)
    else:
        raise ValueError("cos/sin must be 2-D or 3-D")

    if obj.enable_fp32_compute:
        rotary_input = rotary_input.float()
    rotary_input = rotary_input.contiguous()
    cos = cos.to(dtype=rotary_input.dtype).contiguous()
    sin = sin.to(dtype=rotary_input.dtype).contiguous()

    # FlagGems requires both Q and K. One head of Q supplies the unused K
    # result; the out-of-place call keeps the input and unrotated tail intact.
    rotated, _ = rotary_embedding_flaggems(
        obj,
        rotary_input,
        rotary_input[:, :, :1, :],
        cos,
        sin,
        position_ids,
        rotary_interleaved=not obj.is_neox_style,
        inplace=False,
    )
    if x.ndim == 3:
        rotated = rotated.squeeze(0)
    if obj.enable_fp32_compute:
        rotated = rotated.to(x.dtype)
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
