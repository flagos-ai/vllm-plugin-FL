# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm_fl.dispatch import CachedOp

_rotary_embedding = CachedOp("rotary_embedding")


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
        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None
        self._cached_src: Optional[torch.Tensor] = None
        _rotary_embedding.pre_resolve()

    def _get_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_cos is not None
            and self._cached_device == device
            and self._cached_src is self.cos_sin_cache
        ):
            return self._cached_cos, self._cached_sin
        if self.cos_sin_cache.device != device:
            self.cos_sin_cache = self.cos_sin_cache.to(device)
        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
        self._cached_cos = cos.contiguous()
        self._cached_sin = sin.contiguous()
        self._cached_device = device
        self._cached_src = self.cos_sin_cache
        return self._cached_cos, self._cached_sin

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        cos, sin = self._get_cos_sin(positions.device)
        positions_flat = positions.view(-1)
        num_tokens = positions_flat.shape[0]

        query_shape = query.shape
        key_shape = key.shape

        if self.rotary_dim == self.head_size:
            q_view = query.view(num_tokens, -1, self.head_size)
            k_view = key.view(num_tokens, -1, self.head_size)
            q_embed, k_embed = _rotary_embedding(
                self,
                q_view,
                k_view,
                cos,
                sin,
                positions_flat,
                not self.is_neox_style,
                True,
            )
            return q_embed.view(query_shape), k_embed.view(key_shape)

        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        key_pass = key[..., self.rotary_dim :]

        q_embed, k_embed = _rotary_embedding(
            self,
            query_rot,
            key_rot,
            cos,
            sin,
            positions_flat,
            not self.is_neox_style,
            True,
        )
        query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
        key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
        return query, key


__all__ = ["RotaryEmbeddingFL"]
