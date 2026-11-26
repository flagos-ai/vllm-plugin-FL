# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU workarounds for the vendor-neutral ``reference`` MoE backend.

The reference backend is the last-resort pure-PyTorch MoE implementation.  The
upstream version keeps every routing/index tensor in int64 and reduces the
whole token axis at once, which is fine on CUDA but breaks on Enflame GCU:

* gcu300 has no 64-bit element-wise aten/Triton kernels
  (``64-bit data type not supported on GCU300``) -> the ``expert_map`` and
  bucket arithmetic must stay int32 (``num_tokens*topk`` is far below 2**31);
* gcu300's FlagGems sum kernel runs out of registers when the reduction axis
  exceeds ~4096 elements -> accumulate per-expert counts in 4096-column chunks;
* the slot map is built token-major x topk; a broadcast int64 add aborts with
  ``This binary op pattern not support``;
* on the sampler hot path a ``gather`` of the selected scores costs an extra
  device sync, so take the weights straight from ``torch.topk``.

Everything is installed by monkey-patching the reference backend **on GCU only**
from ``apply_gcu_patches()``; other vendors keep the pristine upstream code.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)
_PATCH_STATE = {"applied": False}


def moe_align_block_size_gcu(
    self,
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch fallback for fused MoE block alignment (capture-safe).

    ``topk_ids`` holds global expert ids. With ``ignore_invalid_experts`` the
    map is applied while bucketing, so ids mapped to ``-1`` -- or otherwise
    out of range -- belong to another rank and are dropped instead of being
    routed into a neighbouring expert's bucket. Without the flag, buckets are
    built from global ids and the returned ``expert_ids`` are remapped
    through ``expert_map`` afterwards, matching upstream.
    """
    device = topk_ids.device
    # Enflame S60/gcu300 has no 64-bit Triton/aten element-wise support
    # ("64-bit data type not supported on GCU300"), so every index tensor
    # here stays int32: num_tokens * topk and the padded offsets are far
    # below 2**31 for any served batch.
    flat = topk_ids.reshape(-1).to(torch.int32)
    # Upstream's top-k kernel writes -1 for padded rows, so screen the raw
    # ids before they index anything: a negative or out-of-range id must not
    # reach ``expert_map`` (where -1 selects its last entry) nor the bucket
    # ranking below (where it would overwrite the last expert's slots).
    # With a map the raw ids are global, so the map length is the bound.
    raw_count = expert_map.numel() if expert_map is not None else num_experts
    raw_valid = (flat >= 0) & (flat < raw_count)
    # Mirror upstream: with ``ignore_invalid_experts`` the map is applied
    # while bucketing, so ids owned by another rank are dropped; without it
    # buckets are built from global ids and the returned expert ids are
    # remapped at the end instead.
    remap_output = expert_map is not None and not ignore_invalid_experts
    if expert_map is not None and ignore_invalid_experts:
        mapped = expert_map.to(device=device, dtype=torch.int32)[
            torch.where(raw_valid, flat, torch.zeros_like(flat))
        ]
        valid = raw_valid & (mapped >= 0) & (mapped < num_experts)
        ids = mapped
    else:
        valid = raw_valid
        ids = flat
    local = torch.where(valid, ids, torch.zeros_like(ids))
    num_valid = flat.numel()
    same_expert = (
        local.unsqueeze(0)
        == torch.arange(num_experts, device=device).unsqueeze(1)
    ) & valid.unsqueeze(0)
    # gcu300's FlagGems sum kernel runs out of registers when the
    # reduction axis is longer than ~4096 elements, so accumulate the
    # per-expert counts in column chunks (identical result).
    same_expert_i32 = same_expert.to(torch.int32)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for chunk_start in range(0, num_valid, 4096):
        counts += same_expert_i32[:, chunk_start : chunk_start + 4096].sum(
            dim=1, dtype=torch.int32
        )
    padded = (counts + block_size - 1) // block_size * block_size
    offsets = torch.cumsum(padded, 0) - padded
    ends = offsets + padded
    total = padded.sum()
    if pad_sorted_ids:
        total = (total + block_size - 1) // block_size * block_size
    # Static upper bound keeps every buffer size independent of the routing
    # data, which is required while capturing CUDA graphs. The extra slot is
    # the sink for tokens dropped by ``expert_map``.
    max_total = num_valid + num_experts * (block_size - 1) + 1
    max_total = (max_total + block_size - 1) // block_size * block_size
    sorted_ids = torch.full(
        (max_total,), num_valid, dtype=torch.int32, device=device
    )
    slot = torch.arange(num_valid, device=device, dtype=torch.int32)
    ranks = (torch.cumsum(same_expert_i32, dim=1) - 1)[local, slot]
    dest = offsets[local] + ranks
    dest = torch.where(valid, dest, torch.full_like(dest, max_total - 1))
    sorted_ids[dest] = torch.where(
        valid, slot, torch.full_like(slot, num_valid)
    ).to(torch.int32)
    num_blocks = max_total // block_size
    block_start = torch.arange(num_blocks, device=device, dtype=torch.int32) * block_size
    owner = (block_start.unsqueeze(0) >= offsets.unsqueeze(1)) & (
        block_start.unsqueeze(0) < ends.unsqueeze(1)
    )
    expert_of_block = torch.where(
        owner.any(dim=0),
        owner.to(torch.int32).argmax(dim=0).to(torch.int32),
        torch.full((num_blocks,), -1, dtype=torch.int32, device=device),
    )
    expert_ids = torch.where(
        block_start < total,
        expert_of_block,
        torch.full((num_blocks,), -1, dtype=torch.int32, device=device),
    )
    num_tokens_post_padded = total.reshape(1).to(torch.int32)
    if remap_output:
        safe = expert_ids.clamp(min=0)
        mapped = expert_map.to(device=device, dtype=torch.int32)[safe]
        expert_ids = torch.where(expert_ids >= 0, mapped, expert_ids)
    return sorted_ids, expert_ids, num_tokens_post_padded


def topk_softmax_gcu(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
):
    """Write routing outputs in place; correction bias affects selection only.

    Padding belongs to the caller's execution context, not this reference op.
    Source indices use the slot-major layout expected by the MoE kernels.
    """
    scores = torch.softmax(gating_output.float(), dim=-1)
    selection_scores = scores
    if e_score_correction_bias is not None:
        selection_scores = scores + e_score_correction_bias.float()
    topk_result = torch.topk(selection_scores, topk_indices.shape[1], dim=-1)
    indices = topk_result.indices
    if e_score_correction_bias is None:
        # Without a routing-correction bias the selection scores *are* the
        # weights, so take them straight from topk. The gather it replaces runs
        # once per MoE layer per token and, on gcu300, forced a device sync in
        # FlagGems' gather wrapper -- a measurable chunk of decode latency.
        weights = topk_result.values
    else:
        weights = scores.gather(1, indices)
    if renormalize:
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
    topk_weights.copy_(weights)
    topk_indices.copy_(indices)
    # The slot map is token-major x topk. Enflame S60/gcu300 topsaten has no
    # int64 element-wise kernels: a broadcast int64 add ([N,1] + [K]) aborts the
    # process with "This binary op pattern not support" (TOPSATEN_STATUS_NOT_SUPPORT)
    # as soon as TORCH_GCU_ENABLE_INT64_AND_UINT64=1 makes int64 storage real.
    # num_tokens * topk is far below 2**31, so build the map in int32 and widen
    # only when the destination tensor is int64.
    num_tokens = gating_output.shape[0]
    num_topk = topk_indices.shape[1]
    rows = torch.arange(num_tokens, device=gating_output.device, dtype=torch.int32)
    slots = torch.arange(num_topk, device=gating_output.device, dtype=torch.int32)
    slot_map = rows[:, None] + slots * num_tokens
    if token_expert_indices.dtype != torch.int32:
        slot_map = slot_map.to(token_expert_indices.dtype)
    token_expert_indices.copy_(slot_map)
    return topk_weights, topk_indices


def apply_reference_moe_gcu_patch() -> None:
    """Install the GCU variants of the reference MoE helpers."""
    if _PATCH_STATE["applied"]:
        return
    # Imported lazily: these modules pull in vLLM, which must already be usable
    # when the GCU backend initialises.
    from vllm_fl.dispatch.backends.reference import reference as _reference
    from vllm_fl.dispatch.backends.reference.impl import fused_moe as _ref_impl

    _reference.ReferenceBackend.moe_align_block_size = moe_align_block_size_gcu
    _ref_impl.topk_softmax = topk_softmax_gcu
    logger.info(
        "Patched reference MoE backend for GCU (int32 routing, chunked counts, "
        "topk weights without gather)"
    )
    _PATCH_STATE["applied"] = True
