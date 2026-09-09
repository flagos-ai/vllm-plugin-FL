# Copyright (c) 2026 BAAI. All rights reserved.

"""Monkey-patch ``vllm._custom_ops.apply_repetition_penalties`` for txda.

The upstream dispatcher (``vllm/_custom_ops.py``) routes based on
``logits.is_cuda``::

    if logits.is_cuda and logits.is_contiguous():
        apply_repetition_penalties_cuda(...)   # torch.ops._C.apply_repetition_penalties_
    else:
        apply_repetition_penalties_torch(...)  # pure-torch fallback

On txda the logits tensor lives on ``PrivateUse1`` but ``logits.is_cuda``
returns ``True`` (torch_txda aliases CUDA), so the CUDA branch is taken and
``torch.ops._C.apply_repetition_penalties_`` is invoked.  That custom op has
no CPU kernel; torch_txda's generic backend fallback copies the tensor to CPU
and re-dispatches on the ``CPU`` key, which then fails with::

    NotImplementedError: Could not run '_C::apply_repetition_penalties_'
    with arguments from the 'CPU' backend.

This patch replaces the dispatcher with one that always uses the
device-agnostic pure-torch implementation (``apply_repetition_penalties_torch``),
which performs the exact same in-place math and works on txda.  The call site
(``vllm/model_executor/layers/utils.py:apply_penalties``) imports the symbol
from ``vllm._custom_ops`` inside the function body, so rebinding the module
attribute is enough.  It is applied only when the txda backend loads, so other
vendors keep the upstream behaviour.
"""

import torch

import vllm._custom_ops as _custom_ops

# Prefer vLLM's own pure-torch fallback so we stay in sync with upstream math.
# Fall back to FL's equivalent impl if a future vLLM version renames/removes it.
_apply_repetition_penalties_torch = getattr(
    _custom_ops, "apply_repetition_penalties_torch", None
)
if _apply_repetition_penalties_torch is None:
    from vllm_fl.ops._C_ops_registry import (
        _apply_repetition_penalties_impl as _apply_repetition_penalties_torch,
    )


def _apply_repetition_penalties_txda(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Device-agnostic replacement for ``apply_repetition_penalties``.

    Always routes to the pure-torch path so txda never dispatches the
    ``_C::apply_repetition_penalties_`` custom op (which lacks a CPU kernel).
    """
    _apply_repetition_penalties_torch(
        logits, prompt_mask, output_mask, repetition_penalties
    )


# Apply the patch
_custom_ops.apply_repetition_penalties = _apply_repetition_penalties_txda
