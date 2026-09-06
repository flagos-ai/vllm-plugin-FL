# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU fallbacks for vLLM penalty paths that use Triton atomics.

Two penalty implementations exist in vLLM 0.20:

1. **V1 gpu_model_runner** (used on PTPU): ``vllm.v1.sample.sampler`` ->
   ``model_executor.layers.utils.apply_penalties`` ->
   ``get_token_bin_counts_and_mask`` which calls ``scatter_add_``. FlagGems
   routes ``scatter_add_`` to Triton ``atomic_add`` / ``cmpxchg``, which PTPU
   does not support.

2. **V2 gpu/model_runner** (not used yet): ``vllm.v1.worker.gpu.sample.penalties``
   Triton ``_bincount_kernel`` with ``atomic_or`` / ``atomic_add``.

Both are patched here with PyTorch references that avoid device atomics.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def _get_token_bin_counts_and_mask_pytorch(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for ``get_token_bin_counts_and_mask`` (no atomics)."""
    device = tokens.device
    bin_counts = torch.zeros(
        (num_seqs, vocab_size), dtype=torch.long, device=device
    )
    mask = torch.zeros((num_seqs, vocab_size), dtype=torch.bool, device=device)

    for seq_idx in range(num_seqs):
        # Use Python-side iteration to avoid FlagGems boolean reductions
        # (e.g. ``valid.any()``) and scatter_add_ atomics on PTPU.
        counts: dict[int, int] = {}
        for tok in tokens[seq_idx].tolist():
            if tok < vocab_size:
                counts[tok] = counts.get(tok, 0) + 1
        for tok, cnt in counts.items():
            bin_counts[seq_idx, tok] = cnt
            mask[seq_idx, tok] = True

    return bin_counts, mask


def _bincount_pytorch(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    """PyTorch reference for V2 ``penalties.bincount`` (no device atomics)."""
    del max_prefill_len  # only used for Triton grid sizing

    prompt_bin_mask[expanded_idx_mapping] = 0
    output_bin_counts[expanded_idx_mapping] = 0

    if expanded_idx_mapping.numel() == 0:
        return

    for req_idx in expanded_idx_mapping.unique().tolist():
        p_len = int(prompt_len[req_idx].item())
        pf_len = int(prefill_len[req_idx].item())
        if p_len > 0:
            prompt_tokens = all_token_ids[req_idx, :p_len].tolist()
            for tok in prompt_tokens:
                word_idx = tok // 32
                prompt_bin_mask[req_idx, word_idx] |= 1 << (tok % 32)

        if pf_len > p_len:
            counts: dict[int, int] = {}
            for tok in all_token_ids[req_idx, p_len:pf_len].tolist():
                counts[tok] = counts.get(tok, 0) + 1
            for tok, cnt in counts.items():
                output_bin_counts[req_idx, tok] = cnt


def patch_ptpu_penalties_bincount() -> None:
    if os.environ.get("VLLM_FL_SUNRISE_PTPU_PENALTIES_BINCOUNT", "1") == "0":
        logger.info(
            "PTPU penalties patch disabled by "
            "VLLM_FL_SUNRISE_PTPU_PENALTIES_BINCOUNT=0"
        )
        return

    _patch_v1_penalty_bin_counts()
    _patch_v2_penalty_bincount()


def _patch_v1_penalty_bin_counts() -> None:
    try:
        from vllm.model_executor.layers import utils as layers_utils
    except Exception as exc:
        logger.warning(
            "Failed to import model_executor.layers.utils "
            "(skipping V1 penalties patch): %s",
            exc,
        )
        return

    if getattr(layers_utils, "_sunrise_ptpu_penalties_patched", False):
        return

    layers_utils.get_token_bin_counts_and_mask = _get_token_bin_counts_and_mask_pytorch
    layers_utils._sunrise_ptpu_penalties_patched = True
    logger.info(
        "Patched model_executor.layers.utils.get_token_bin_counts_and_mask for "
        "PTPU (PyTorch fallback; avoids FlagGems scatter_add_ cmpxchg in "
        "repetition_penalty path)"
    )


def _patch_v2_penalty_bincount() -> None:
    try:
        from vllm.v1.worker.gpu.sample import penalties as penalties_mod
    except Exception as exc:
        logger.warning(
            "Failed to import vLLM V2 penalties module "
            "(skipping bincount patch): %s",
            exc,
        )
        return

    if getattr(penalties_mod, "_sunrise_ptpu_bincount_patched", False):
        return

    penalties_mod.bincount = _bincount_pytorch
    penalties_mod._sunrise_ptpu_bincount_patched = True
    logger.info(
        "Patched v1.worker.gpu.sample.penalties.bincount for PTPU "
        "(PyTorch fallback for future V2 model runner)"
    )
