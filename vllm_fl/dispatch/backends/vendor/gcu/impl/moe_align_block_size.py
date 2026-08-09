# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix for ``moe_align_block_size``.

The upstream ``moe_align_block_size`` delegates to ``ops.moe_align_block_size``
which is a CUDA-only custom op.  On GCU this op is not available and causes a
RuntimeError.

This patch replaces the community function with a version that dispatches
through ``vllm_fl``'s ``CachedOp`` mechanism, which selects a GCU-compatible
implementation (currently the reference PyTorch implementation).
"""

from __future__ import annotations

import logging

import torch

from vllm_fl.dispatch import CachedOp

logger = logging.getLogger(__name__)

_patched = False

# Resolved once, refreshed when the dispatch policy changes.
_gcu_moe_align = CachedOp("moe_align_block_size")



"""
Pure PyTorch (torch native) implementation of ``moe_align_block_size``.

Reference: vLLM community CUDA kernel at
``vllm/csrc/moe/moe_align_sum_kernels.cu`` and the Python wrapper at
``vllm/model_executor/layers/fused_moe/moe_align_block_size.py``.

This module provides a fully vectorised, GPU-friendly implementation that
does not depend on any custom C++/CUDA operator.  It is suitable for:
- hardware backends that lack a compiled ``moe_align_block_size`` op,
- debugging / testing the behaviour of the CUDA kernel,
- serving as a readable reference for the algorithm.
"""

def _round_up(x: int, multiple: int) -> int:
    """Round *x* up to the next multiple of *multiple*."""
    return (x + multiple - 1) // multiple * multiple


def moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch implementation of ``moe_align_block_size``.

    Aligns token distribution across experts so that every expert handles a
    number of tokens divisible by *block_size*, which is required for
    efficient block matrix multiplication.

    The algorithm mirrors the vLLM CUDA kernel:

    1. Flatten ``topk_ids`` (shape ``[total_tokens, top_k]``) into a 1-D
       sequence of expert assignments.
    2. Count how many tokens are routed to each expert.
    3. Pad every expert's count to a multiple of *block_size*.
    4. Compute prefix-sums (cumsum) over the padded counts to obtain
       write-offsets.
    5. Fill ``expert_ids`` (one entry per block) with the owning expert
       index; unused blocks are set to -1.
    6. Sort token indices by expert, writing them into
       ``sorted_token_ids``; unused slots receive *numel* as sentinel.
    7. Optionally remap ``expert_ids`` via *expert_map* (for expert
       parallelism).

    Parameters
    ----------
    topk_ids : torch.Tensor
        Shape ``[total_tokens, top_k]``, dtype integer.  The top-k expert
        indices for each token.
    block_size : int
        Block size used in block matrix multiplication (typically 32, 64,
        or 128).
    num_experts : int
        Total number of experts (global count, not per-shard).
    expert_map : torch.Tensor or None
        Shape ``[num_experts]``, dtype integer.  Maps a global expert index
        to a local index on the current expert-parallel shard.  Experts not
        present on this shard are mapped to -1.
    pad_sorted_ids : bool
        When True, the length of the returned ``sorted_token_ids`` tensor is
        rounded up to a multiple of *block_size*.
    ignore_invalid_experts : bool
        When True and *expert_map* is given, tokens assigned to invalid
        experts (mapped to -1) are excluded from counting and sorting.
        When False, all tokens participate in counting; invalid experts are
        only marked **after** the alignment via ``expert_map[expert_ids]``.

    Returns
    -------
    sorted_token_ids : torch.Tensor
        1-D int32 tensor.  Token indices grouped by expert, padded with
        *numel* sentinel values.
    expert_ids : torch.Tensor
        1-D int32 tensor.  Expert index for each block; -1 for padding.
    num_tokens_post_padded : torch.Tensor
        Scalar (shape ``[1]``) int32 tensor.  Number of valid entries in
        *sorted_token_ids*.
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    cur_device = topk_ids.device
    topk_ids = topk_ids.to('cpu')
    if expert_map is not None:
        expert_map = expert_map.to('cpu')
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_experts < 0:
        raise ValueError("num_experts must be non-negative")

    device = topk_ids.device
    num_routes = topk_ids.numel()  # total_tokens * top_k
    sentinel = num_routes  # used as "invalid index" marker

    # ------------------------------------------------------------------
    # Compute output tensor sizes (matches the vLLM Python wrapper)
    # ------------------------------------------------------------------
    max_num_tokens_padded = num_routes + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = _round_up(max_num_tokens_padded, block_size)
    if num_routes < num_experts:
        max_num_tokens_padded = min(
            num_routes * block_size, max_num_tokens_padded
        )

    max_num_m_blocks = _round_up(max_num_tokens_padded, block_size) // block_size

    # ------------------------------------------------------------------
    # Allocate output tensors (initialised with sentinel / -1)
    # ------------------------------------------------------------------
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        sentinel,
        dtype=torch.int32,
        device=device,
    )
    expert_ids = torch.full(
        (max_num_m_blocks,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    # ------------------------------------------------------------------
    # Step 1 – Flatten & determine expert-for-token
    # ------------------------------------------------------------------
    flat_ids = topk_ids.view(-1).to(torch.int64)

    # Decide whether expert_map filtering happens *inside* the alignment
    # (ignore_invalid_experts=True) or *after* (default).
    use_expert_map_internally = expert_map is not None and ignore_invalid_experts

    if use_expert_map_internally:
        # --- Map + filter in one pass --------------------------------
        mapped_ids = expert_map[flat_ids.clamp(min=0, max=num_experts - 1)]
        valid_mask = (flat_ids >= 0) & (flat_ids < num_experts) & (mapped_ids != -1)
        expert_for_token = mapped_ids[valid_mask].to(torch.int64)
    else:
        # --- Only drop out-of-range expert ids ------------------------
        valid_mask = (flat_ids >= 0) & (flat_ids < num_experts)
        expert_for_token = flat_ids[valid_mask]

    # Original flattened indices of every *valid* token
    token_indices = valid_mask.nonzero(as_tuple=False).flatten()

    # ------------------------------------------------------------------
    # Step 2 – Count tokens per expert
    # ------------------------------------------------------------------
    token_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if expert_for_token.numel() > 0:
        token_counts.scatter_add_(
            0,
            expert_for_token.to(torch.int64),
            torch.ones(expert_for_token.numel(), dtype=torch.int32, device=device),
        )

    # ------------------------------------------------------------------
    # Step 3 – Pad counts & compute prefix-sum offsets
    # ------------------------------------------------------------------
    padded_counts = (
        (token_counts + block_size - 1) // block_size
    ) * block_size
    cumsum = torch.cumsum(padded_counts, dim=0)
    cumsum_shifted = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), cumsum]
    )
    total_tokens_padded = cumsum[-1].item()

    # ------------------------------------------------------------------
    # Step 4 – Fill expert_ids (one id per block)
    # ------------------------------------------------------------------
    for e in range(num_experts):
        if padded_counts[e] == 0:
            continue
        start_block = cumsum_shifted[e].item() // block_size
        end_block = cumsum_shifted[e + 1].item() // block_size
        expert_ids[start_block:end_block] = e

    # ------------------------------------------------------------------
    # Step 5 – Fill sorted_token_ids (sort tokens by expert)
    # ------------------------------------------------------------------
    if expert_for_token.numel() > 0:
        # Stable sort: tokens of the same expert keep original order.
        sorted_order = expert_for_token.argsort(stable=True)
        sorted_experts = expert_for_token[sorted_order]
        sorted_tok_indices = token_indices[sorted_order]

        # Compute each token's rank *within* its expert group.
        # `boundaries[i] == 1` iff position *i* starts a new expert.
        boundaries = torch.cat(
            [
                torch.ones(1, dtype=torch.int32, device=device),
                (sorted_experts[1:] != sorted_experts[:-1]).to(torch.int32),
            ]
        )
        within_rank = torch.cumsum(boundaries, dim=0) - 1

        # Absolute write position = expert_offset + within_expert_rank
        write_positions = (
            cumsum_shifted[sorted_experts.to(torch.int64)] + within_rank
        )
        sorted_token_ids[write_positions] = sorted_tok_indices.to(torch.int32)

    # ------------------------------------------------------------------
    # Step 6 – Post-process expert_map (when not applied internally)
    # ------------------------------------------------------------------
    if expert_map is not None and not ignore_invalid_experts:
        valid_exp_mask = expert_ids >= 0
        expert_ids[valid_exp_mask] = expert_map[
            expert_ids[valid_exp_mask].to(torch.int64)
        ]

    num_tokens_post_pad = torch.tensor(
        [total_tokens_padded], dtype=torch.int32, device=device
    )

    sorted_token_ids = sorted_token_ids.to(cur_device)
    expert_ids = expert_ids.to(cur_device)
    num_tokens_post_pad = num_tokens_post_pad.to(cur_device)
    return sorted_token_ids, expert_ids, num_tokens_post_pad



def moe_align_block_size_gcu(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GCU-compatible ``moe_align_block_size`` via the FlagOS dispatch system."""
    return _gcu_moe_align(
        topk_ids,
        block_size,
        num_experts,
        expert_map,
        pad_sorted_ids=pad_sorted_ids,
        ignore_invalid_experts=ignore_invalid_experts,
    )


def apply_moe_align_block_size_gcu_patch() -> None:
    """Patch ``moe_align_block_size`` for GCU devices."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    # Modules that hold a reference to moe_align_block_size (either as
    # the original definition site or via ``from ... import``).
    _IMPORTERS: list[str] = [
        "vllm.model_executor.layers.fused_moe.moe_align_block_size",
        "vllm.model_executor.layers.fused_moe.fused_moe",
    ]

    try:
        for module_name in _IMPORTERS:
            try:
                mod = __import__(module_name, fromlist=["moe_align_block_size"])
            except ImportError:
                continue
            if hasattr(mod, "moe_align_block_size"):
                # mod.moe_align_block_size = moe_align_block_size_gcu
                mod.moe_align_block_size = moe_align_block_size_torch

        _patched = True
        logger.info(
            "Patched moe_align_block_size for GCU (using dispatch system)"
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch moe_align_block_size for GCU: %s",
            exc,
        )
