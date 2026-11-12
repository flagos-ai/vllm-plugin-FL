# SPDX-License-Identifier: Apache-2.0
"""Portable square-root-softplus routing, including token-to-expert hashes."""

import torch


def topk_softplus_sqrt(
    hidden_states,
    gating_output,
    topk,
    renormalize,
    e_score_correction_bias=None,
    indices_type=None,
    input_tokens=None,
    hash_indices_table=None,
):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        _topk_softplus_sqrt_torch,
    )

    if hidden_states.shape[0] != gating_output.shape[0]:
        raise ValueError("Routing token counts must match")
    if (input_tokens is None) != (hash_indices_table is None):
        raise ValueError("Hash routing requires both input tokens and a hash table")
    if hash_indices_table is not None:
        if input_tokens.shape != (gating_output.shape[0],):
            raise ValueError("Hash routing requires one token ID per row")
        if hash_indices_table.ndim != 2 or hash_indices_table.shape[1] != topk:
            raise ValueError("Hash table must have [vocabulary, topk] shape")
        # Runner padding uses -1, which must select the dummy token, not the
        # final vocabulary entry through PyTorch's negative-index semantics.
        input_tokens = torch.where(input_tokens == -1, 0, input_tokens)
    shape = (gating_output.shape[0], topk)
    weights = torch.empty(shape, dtype=torch.float32, device=gating_output.device)
    indices = torch.empty(
        shape,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=gating_output.device,
    )
    scratch = torch.empty(shape, dtype=torch.int32, device=gating_output.device)
    # Reuse upstream's pure torch math: selection-only bias, unbiased weights,
    # optional normalization. The router owns the single output scaling step.
    return _topk_softplus_sqrt_torch(
        weights,
        indices,
        scratch,
        gating_output,
        renormalize,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
    )
