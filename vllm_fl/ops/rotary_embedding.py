# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional

import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm_fl.dispatch import CachedOp

_rotary_embedding = CachedOp("rotary_embedding")


def _apply_rotary_emb_native(
    self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply partial RoPE with PyTorch ops on the current device."""
    rotary_dim = cos.shape[-1] * 2
    rotated = self.forward_native(x[..., :rotary_dim], cos, sin)
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def apply_rotary_emb_platform(
    self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Use the platform implementation when FlagGems RoPE is disabled."""
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        return self.forward_cuda(x, cos, sin)
    # vLLM has no single-tensor ApplyRotaryEmb kernel for other FL vendors.
    # Its native path needs the rotary slice for partial RoPE.
    return _apply_rotary_emb_native(self, x, cos, sin)


def apply_rotary_emb_flaggems(
    self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Adapt vLLM's single-tensor RoPE interface to the FlagGems Q/K op.

    FlagGems rotates the entire last dimension of both Q and K. Pass the
    rotary slice as Q and one head of it as K, use the out-of-place Q result,
    and retain the unrotated tail. The K result is required by its API.
    """
    rotary_dim = cos.shape[-1] * 2
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    if rotary_dim > x.shape[-1]:
        raise ValueError("rotary_dim must not exceed head_dim")
    if x.ndim not in (3, 4):
        raise ValueError("ApplyRotaryEmb expects a 3-D or 4-D input")
    if rotary_dim == 0:
        return x

    from flag_gems.modules.rotary_embedding import gems_rope_forward

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

    if self.enable_fp32_compute:
        rotary_input = rotary_input.float()
    rotary_input = rotary_input.contiguous()
    cos = cos.to(dtype=rotary_input.dtype).contiguous()
    sin = sin.to(dtype=rotary_input.dtype).contiguous()

    rotated, _ = gems_rope_forward(
        rotary_input,
        rotary_input[:, :, :1, :],
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=not self.is_neox_style,
        inplace=False,
    )
    if x.ndim == 3:
        rotated = rotated.squeeze(0)
    if self.enable_fp32_compute:
        rotated = rotated.to(x.dtype)
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


class RotaryEmbeddingFL(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base,
            is_neox_style, dtype
        )

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        positions = positions.flatten()
        num_tokens = positions.shape[0]

        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

        q_embed, k_embed = _rotary_embedding(
            self,
            query_rot,
            key_rot,
            cos,
            sin,
            positions,
            not self.is_neox_style,  # rotary_interleaved
            True,  # inplace
        )

        if self.rotary_dim < self.head_size:
            query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
        else:
            query = q_embed.reshape(query_shape)
            key = k_embed.reshape(key_shape)

        return query, key


__all__ = ["RotaryEmbeddingFL"]
