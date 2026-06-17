# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def rotary_embedding_maca(
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
    Apply rotary position embedding using mcoplib's _C implementation.

    Adapts from dispatch interface (cos, sin separate) to _C op interface
    (cos_sin_cache combined, head_size, is_neox).
    """
    from vllm._custom_ops import rotary_embedding

    # Reconstruct cos_sin_cache: shape [max_pos, rotary_dim]
    cos_sin_cache = torch.cat([cos, sin], dim=-1)
    head_size = obj.head_size
    is_neox = not rotary_interleaved

    rotary_embedding(
        position_ids,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )
    return query, key
