# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX rotary embedding operator implementations.
"""


def topk_softmax_maca(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    from vllm._custom_ops import topk_softmax

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )
    return topk_weights, topk_indices
