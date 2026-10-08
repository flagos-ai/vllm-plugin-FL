# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise MoE helper fallbacks (topk_softmax, moe_sum, moe_align_block_size)."""

from __future__ import annotations

from typing import Optional

import torch
from vllm.utils.math_utils import round_up


def topk_softmax_sunrise(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-native ``topk_softmax`` for PTPU.

    Matches the contract of the FlagGems and CUDA implementations:

    * ``gating_output``  ``[num_tokens, num_experts]`` — raw router logits.
    * ``topk_weights``   ``[num_tokens, top_k]``      — *output*, in-place,
      typically ``float32``.
    * ``topk_indices``   ``[num_tokens, top_k]``      — *output*, in-place,
      typically ``int32`` or ``int64``.
    * ``token_expert_indices`` ``[num_tokens, top_k]`` — allocated by the
      caller; vLLM's ``fused_topk`` helper immediately discards it (see
      ``vllm_fl/ops/fused_moe/router.py::fused_topk``), so we deliberately
      do **not** write to it. This matches the existing
      ``_native_topk_with_bias`` patch in ``sunrise/patch.py``.
    * ``renormalize``    — when ``True``, the top-k weights are rescaled
      to sum to 1 *after* selection (standard MoE router renormalisation,
      matching vLLM ``FusedTopKRouter``).

    Numerical notes:

    * Softmax runs in the gating dtype (PTPU softmax accumulates in fp32).
    * Uses unsorted ``torch.topk`` (``sorted=False``) because vLLM's
      downstream MoE kernels don't require sorted top-k.

    Returns ``(topk_weights, topk_indices)`` (the same handles passed in)
    so callers can use either positional or chained access.
    """
    if gating_output.dim() != 2:
        raise ValueError(
            "topk_softmax_sunrise: gating_output must be 2-D "
            f"[num_tokens, num_experts]; got shape {tuple(gating_output.shape)}"
        )

    top_k = topk_weights.shape[-1]

    scores = gating_output.softmax(dim=-1)

    chosen_indices = torch.topk(scores, k=top_k, dim=-1, sorted=False).indices
    chosen_weights = scores.gather(dim=-1, index=chosen_indices)
    if renormalize:
        chosen_weights = chosen_weights / chosen_weights.sum(dim=-1, keepdim=True)

    topk_weights.copy_(chosen_weights.to(topk_weights.dtype))
    topk_indices.copy_(chosen_indices.to(topk_indices.dtype))
    return topk_weights, topk_indices


def moe_sum_sunrise(inp: torch.Tensor, out: torch.Tensor) -> None:
    """Reduce ``inp`` across its top-k axis into ``out``.

    Contract matches vLLM's ``moe_sum`` op:

    * ``inp`` shape ``[..., top_k, hidden]`` — per-token, per-expert
      outputs already weighted by the router.
    * ``out`` shape ``[..., hidden]``         — *output*, in-place.

    Precision: for ``bf16`` / ``fp16`` inputs we **must** match FlagGems'
    Triton kernel, which accumulates in ``float32`` and only casts on store
    (see FlagGems ``moe_sum``). A plain bf16 ``torch.sum`` on PTPU can
    accumulate in the output dtype and lose precision across many layers.
    """
    if inp.dim() < 2 or out.dim() < 1:
        raise ValueError(
            "moe_sum_sunrise: expected inp ndim>=2 and out ndim>=1, "
            f"got inp.shape={tuple(inp.shape)}, out.shape={tuple(out.shape)}"
        )
    if inp.shape[-1] != out.shape[-1]:
        raise ValueError(
            "moe_sum_sunrise: hidden dim mismatch: "
            f"inp.shape[-1]={inp.shape[-1]} vs out.shape[-1]={out.shape[-1]}"
        )

    if inp.dtype in (torch.bfloat16, torch.float16):
        acc = torch.sum(inp, dim=-2, dtype=torch.float32)
        out.copy_(acc.to(out.dtype))
    else:
        torch.sum(inp, dim=-2, out=out)


def moe_align_block_size_sunrise(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference ``moe_align_block_size`` for PTPU (opt-in fallback).

    Default dispatch keeps FlagGems Triton first. Enable this path via
    ``VLLM_FL_PER_OP`` / yaml reorder only when the FlagGems kernel hangs or
    misbehaves for a given expert/token shape.
    """
    del ignore_invalid_experts  # reference path counts all experts in topk_ids

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )

    # Match CUDA/vLLM buffer sizing (ceil) for both sorted_ids and expert_ids.
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sorted_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        expert_token_counts[expert_id] = (sorted_expert_ids == expert_id).sum()

    expert_padded_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            sorted_ids[
                current_pos : current_pos + num_expert_tokens
            ] = expert_tokens

            expert_blocks_needed = int(
                expert_padded_counts[expert_id].item() // block_size
            )
            expert_id_new = (
                expert_map[expert_id] if expert_map is not None else expert_id
            )
            expert_ids[
                current_block : current_block + expert_blocks_needed
            ] = expert_id_new

            current_pos += int(expert_padded_counts[expert_id].item())
            current_block += expert_blocks_needed

    total_padded_tokens = int(expert_padded_counts.sum().item())
    num_tokens_post_pad.fill_(total_padded_tokens)

    return sorted_ids, expert_ids, num_tokens_post_pad


__all__ = [
    "topk_softmax_sunrise",
    "moe_sum_sunrise",
    "moe_align_block_size_sunrise",
]
