# Copyright (c) 2026 BAAI. All rights reserved.
"""EMPTY-vLLM MoE routing adapters for Iluvatar."""

import importlib

# Routing math lives in the shared reference backend.


def topk_softmax(topk_weights, topk_indices, token_expert_indices,
                 gating_output, renormalize=False, e_score_correction_bias=None):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        _get_padding_mask,
    )

    from vllm_fl.dispatch import call_op

    if e_score_correction_bias is None:
        weights, indices = call_op(
            "topk_softmax", topk_weights, topk_indices,
            token_expert_indices, gating_output, renormalize,
        )
        # Some backends return a new normalized tensor rather than mutating it.
        topk_weights.copy_(weights)
        topk_indices.copy_(indices)
    else:
        from vllm_fl.dispatch.backends.reference.impl.fused_moe import (
            topk_softmax as reference_topk_softmax,
        )

        reference_topk_softmax(
            topk_weights, topk_indices, token_expert_indices,
            gating_output, renormalize, e_score_correction_bias,
        )
    padding = _get_padding_mask(topk_indices.shape[0])
    if padding is not None:
        topk_indices.masked_fill_(padding[:, None], -1)
    return topk_weights, topk_indices


def patch_moe_router():
    """Replace only Iluvatar's missing CUDA extension entry points."""
    for name in ("fused_topk_router", "fused_topk_bias_router"):
        module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router." + name)
        module.vllm_topk_softmax = topk_softmax
