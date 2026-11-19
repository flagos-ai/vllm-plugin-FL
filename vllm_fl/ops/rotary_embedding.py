# Copyright (c) 2025 BAAI. All rights reserved.

import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

from vllm_fl.dispatch import CachedOp, get_default_manager

_rotary_embedding = CachedOp("rotary_embedding")
_apply_rotary_emb = CachedOp("apply_rotary_emb")


def apply_rotary_emb_dispatch(
    self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Route vLLM's single-tensor RoPE through the FL backend policy."""
    return _apply_rotary_emb(self, x, cos, sin)


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
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        self._rotary_inplace = True
        if current_platform.device_type == "npu":
            manager = get_default_manager()
            candidates = manager.resolve_candidates("rotary_embedding")
            self._rotary_inplace = candidates[0].impl_id not in (
                "vendor.ascend.torch_npu",
                "reference.torch",
            )
            if any(impl.impl_id == "vendor.ascend" for impl in candidates):
                from vllm_fl.dispatch.backends.vendor.ascend.impl.rotary import (
                    initialize_rotary_embedding_ascend,
                )

                initialize_rotary_embedding_ascend()

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if key is None and self._rotary_inplace:
            # Legacy Q/K providers require both tensors.
            return super().forward_native(positions, query, key)
        cos_sin_cache = self.cos_sin_cache.to(positions.device)
        # Follow upstream's compile-safe buffer handling, retaining FL's
        # device-only conversion and the existing cache dtype.
        if current_platform.device_type != "npu" or not torch.compiler.is_compiling():
            self.cos_sin_cache = cos_sin_cache
        positions = positions.flatten()
        num_tokens = positions.shape[0]

        query_heads = query.reshape(num_tokens, -1, self.head_size)
        key_heads = None if key is None else key.reshape(num_tokens, -1, self.head_size)

        query_rot = query_heads[..., : self.rotary_dim]
        key_rot = None if key_heads is None else key_heads[..., : self.rotary_dim]

        cos, sin = cos_sin_cache.chunk(2, dim=-1)

        q_embed, k_embed = _rotary_embedding(
            self,
            query_rot,
            key_rot,
            cos,
            sin,
            positions,
            not self.is_neox_style,  # rotary_interleaved
            self._rotary_inplace,
        )

        if self.rotary_dim < self.head_size:
            q_embed = torch.cat((q_embed, query_heads[..., self.rotary_dim :]), dim=-1)
            if key_heads is not None:
                k_embed = torch.cat(
                    (k_embed, key_heads[..., self.rotary_dim :]), dim=-1
                )

        return (
            q_embed.reshape(query.shape),
            None if key is None else k_embed.reshape(key.shape),
        )

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # vLLM selects HIP before OOT on ROCm platforms.
        if key is None:
            # The FL Q/K provider requires a key; retain native HIP/AITER here.
            return super().forward_hip(positions, query, key)
        return self.forward_oot(positions, query, key)


__all__ = ["RotaryEmbeddingFL"]
