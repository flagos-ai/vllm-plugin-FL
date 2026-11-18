# Copyright (c) 2026 BAAI. All rights reserved.

from functools import lru_cache

import torch

from vllm_fl.dispatch import get_default_manager
from vllm_fl.dispatch.binding import OperatorBinding


@lru_cache(maxsize=1)
def _scaled_dot_product_attention(manager) -> OperatorBinding:
    return OperatorBinding(manager, "scaled_dot_product_attention")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Noncausal SDPA for BNSD tensors with an optional additive bias.

    The bias is added to the scaled QK scores before softmax. This op uses
    the default scale of the inverse square root of the head dimension.
    """
    binding = _scaled_dot_product_attention(get_default_manager())
    return binding(query, key, value, attn_bias)
