# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU-safe ``get_masked_input_and_mask`` for TP>1 ``VocabParallelEmbedding``.

The upstream mask helper's ``<`` / ``>=`` each launch a native LPU kernel;
under TP>1 every worker launches them concurrently and, with
``ENABLE_DOUBLE_DIE=1``, the driver segfaults on the lookup
(``hpe_runtime.c:2392``) -- the same class of crash as the MoE ``valid_mask``
lt (``fused/fused_moe.py``). This wrapper runs the **unmodified upstream
function** on a host copy of the token row and moves the two results back.
Inert at TP1: upstream skips the mask path there.

"""

from __future__ import annotations

import torch

from vllm.model_executor.layers import vocab_parallel_embedding as _vpe_mod

_ORIG_GET_MASKED = None
_PATCHED = False


def _lpu_get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run upstream ``get_masked_input_and_mask`` host-side, return device tensors.

    ``input_`` is the token-id row (``(seq,)`` int64) — tiny, the host copy is
    free.  The returned values are identical to upstream: same boolean logic,
    only the execution device of the four pointwise comparisons changes (see
    module docstring).
    """
    masked_host, mask_host = _ORIG_GET_MASKED(
        input_.cpu(),
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    return masked_host.to(device=input_.device), mask_host.to(device=input_.device)


def apply_lpu_vocab_embedding_patch() -> None:
    """Replace ``get_masked_input_and_mask`` with the host-side wrapper on LPU."""
    global _PATCHED, _ORIG_GET_MASKED
    if _PATCHED:
        return
    _ORIG_GET_MASKED = _vpe_mod.get_masked_input_and_mask
    _vpe_mod.get_masked_input_and_mask = _lpu_get_masked_input_and_mask
    _PATCHED = True
