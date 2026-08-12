# Copyright (c) 2026 BAAI. All rights reserved.

import torch


def fused_indexer_q_rope_quant_maca(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
):
    from vllm.v1.attention.ops.deepseek_v4_ops import fused_indexer_q_rope_quant

    return fused_indexer_q_rope_quant(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        use_fp4,
    )
