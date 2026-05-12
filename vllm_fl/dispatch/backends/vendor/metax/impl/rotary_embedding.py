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
    Apply rotary position embedding using vLLM's CUDA implementation.
    """

    from vllm._custom_ops import rotary_embedding

    rotary_embedding(
        position_ids,
        query,
        key,
        cos,
        sin,
        rotary_interleaved,
    )
    return query, key
