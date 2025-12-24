# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional, Union
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from flag_gems.modules.rotary_embedding import gems_rope_forward

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
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)
        
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
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

        q_embed, k_embed = gems_rope_forward(
            query_rot,
            key_rot,
            cos,
            sin,
            position_ids=positions,
            rotary_interleaved=not self.is_neox_style,
            inplace=True,  # set inplace to True for vLLM compatibility
        )

        if self.rotary_dim < self.head_size:
            query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
        else:
            query = q_embed.reshape(query_shape)
            key = k_embed.reshape(key_shape)

        return query, key
    
__all__ = ["RotaryEmbeddingFL"]
