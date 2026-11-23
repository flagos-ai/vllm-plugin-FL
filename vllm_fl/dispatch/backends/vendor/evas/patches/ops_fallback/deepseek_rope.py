# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU Deepseek scaling-rope patch.

Upstream ``forward_native`` expands the per-dimension cos/sin cache with
``torch.repeat_interleave(2, dim=-1)`` (GPT-J / non-neox), which writes out of
bounds on LPU when fed the real gathered ``cos_sin_cache`` and corrupts the
adjacent ``seq_lens`` / ``cache_seqlens`` buffers; the bug is
allocation-layout-sensitive, so only the real-gather input path exhibits it.
The patch expands with ``torch.stack(...).reshape(...)`` instead — identical
arithmetic — verified e2e token-correct on deepseek-v3.1.

"""

import torch

_PATCHED = False


def _forward_native_no_repeat_interleave(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """PyTorch-native forward for DeepseekScalingRotaryEmbedding.

    Identical to upstream ``forward_native`` except the GPT-J (non-neox)
    cos/sin expansion uses ``torch.stack + reshape`` instead of the
    LPU-unsafe ``torch.repeat_interleave(2, dim=-1)``.
    """
    from vllm.model_executor.layers.rotary_embedding.common import (
        rotate_gptj,
        rotate_neox,
    )

    assert key is not None
    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    query_rot = query[..., : self.rotary_dim]
    key_rot = key[..., : self.rotary_dim]
    if self.rotary_dim < self.head_size:
        query_pass = query[..., self.rotary_dim:]
        key_pass = key[..., self.rotary_dim:]

    cos_sin = cos_sin_cache[
        torch.add(positions, offsets) if offsets is not None else positions
    ]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if self.is_neox_style:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    else:
        cos = torch.stack((cos, cos), dim=-1).reshape(cos.shape[0], -1).unsqueeze(-2)
        sin = torch.stack((sin, sin), dim=-1).reshape(sin.shape[0], -1).unsqueeze(-2)

    rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin

    if self.rotary_dim < self.head_size:
        query = torch.cat((query_rot, query_pass), dim=-1)
        key = torch.cat((key_rot, key_pass), dim=-1)
    else:
        query = query_rot
        key = key_rot
    return query, key


def apply_lpu_deepseek_rope_patch() -> None:
    """Replace DeepseekScalingRotaryEmbedding.forward_native on LPU."""
    global _PATCHED
    if _PATCHED:
        return
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekScalingRotaryEmbedding,
    )

    DeepseekScalingRotaryEmbedding.forward_native = (
        _forward_native_no_repeat_interleave
    )
    _PATCHED = True
