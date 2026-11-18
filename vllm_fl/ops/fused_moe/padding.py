# SPDX-License-Identifier: Apache-2.0
"""Preserve vLLM's row-padding contract at the FlagGems expert boundary."""
from __future__ import annotations

import torch


def prepare_padded_moe_inputs(hidden, weights, ids, padding_mask):
    """Replace only metadata-marked padding; preserve active expert routes.

    Native padded routers can emit -1 IDs with nonzero normalized weights.
    FlagGems alignment requires valid indices. Use a finite dummy row/expert
    and zero its weight, then explicitly mask the final output as well.
    No tensor values are read by Python, including during CUDA graph capture.
    """
    if padding_mask is None:
        return hidden, weights, ids
    if (
        padding_mask.dtype != torch.bool
        or padding_mask.ndim != 1
        or padding_mask.shape[0] != hidden.shape[0]
        or padding_mask.device != hidden.device
    ):
        raise ValueError("MoE padding mask must be a device-local bool row vector")
    if weights.shape != ids.shape or ids.shape[0] != hidden.shape[0]:
        raise ValueError("MoE routing rows must match hidden-state rows")
    mask = padding_mask.unsqueeze(-1)
    return (
        hidden.masked_fill(mask, 0),
        weights.masked_fill(mask, 0),
        ids.masked_fill(mask, 0),
    )


def zero_padded_moe_rows(output, padding_mask):
    """Zero padding explicitly, including dummy-expert NaN/Inf results."""
    if padding_mask is not None:
        output.masked_fill_(padding_mask.unsqueeze(-1), 0)
    return output


def install_no_tle_moe_alignment():
    """Select the pinned dependency's existing capture-safe CUDA aligner."""
    import importlib

    moe = importlib.import_module("flaggems_vllm.ops.fused_moe")
    align = importlib.import_module("flaggems_vllm.ops.moe_align_block_size")
    fn = getattr(align, "moe_align_block_size_no_tle", None)
    if fn is None or getattr(moe, "moe_align_block_size", None) is fn:
        return False
    moe.moe_align_block_size = fn
    return True
