# Copyright (c) 2026 BAAI. All rights reserved.
"""Fused Triton rewrite of ``get_masked_input_and_mask`` for GCU.

Why the upstream / eager version is slow here
---------------------------------------------
Upstream keeps the pointwise decomposition because ``@torch.compile`` fuses it
into one kernel.  On GCU the engine runs eager (``TORCHGCU_INDUCTOR_ENABLE=0``
/ ``--enforce-eager``), so the decomposition costs **~13 separate aten
launches** (3 comparisons, 2 ``&``, ``|``, ``~``, 3 scalar ``*``, ``+``, ``-``,
``*``) for a tensor of only ``num_tokens`` elements.  With ~0.3-1 ms CPU launch
overhead per device op on GCU, that is several ms of pure launch latency on the
critical path of *every* forward step (this layer runs once per step).

What this patch does
--------------------
One Triton launch writes both outputs (``masked_input`` + ``input_mask``).

GCU300 caveats handled:
* ``int64`` data path: a kernel mixing an ``i64`` load with two stores does not
  pass register allocation ("ran out of registers during register allocation").
  Token ids always fit in ``int32``, so the ``int64`` buffer is read through a
  free ``view(torch.int32)`` (little-endian low word, stride ``WORD=2``).  The
  ``int64`` *output* is kept so that ``masked_input.long()`` in
  ``VocabParallelEmbedding.forward`` stays a no-op instead of a cast kernel.
  Exactness of the low-word read: two's complement makes the low 32 bits equal
  the value for every ``|id| < 2**31`` (``-1`` included), and the subtraction
  only runs on lanes already known to lie in ``[org_start, org_end)``, so it
  cannot overflow either.  A divergence would need ``|token_id| >= 2**32`` with
  the low word landing inside this shard -- and the eager path truncates the
  same way (int64 arithmetic is silently int32 on GCU), so this kernel is not
  narrower than what it replaces.
* mask quirk: ``tl.arange``-derived masks used by ``tl.load``/``tl.store`` may
  only use ``<`` / ``~(a < b)``; the range tests are written as
  ``(v < end) & ~(v < start)``.
* scalar specialization: Triton constant-folds int args that are ``0``/``1``;
  with ``org_start == 1`` this kernel hit the same register allocator failure,
  so every scalar bound is passed through ``do_not_specialize``.
* requires ``ENABLE_I64_CHECK=0``: that is what turns the triton_gcu
  ``enable_i64`` path on (``enable_i64 = not get_bool_env("ENABLE_I64_CHECK",
  True)``).  With the default (unset/1) the 64-bit type verifier rejects the
  int64 store outright -> ``Pipeline run failed: PassManager execution failed``.

Semantics stay bit-identical with upstream: out-of-shard tokens get
``masked_input == 0`` / ``input_mask == True``, in-shard tokens get
``token_id - offset``.
"""

from __future__ import annotations

import importlib
import logging

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

GCU_NUM_GRID = 48
GCU_MAX_WARPS = 4
_BLOCK = 256

_PATCH_TARGET = "vllm.model_executor.layers.vocab_parallel_embedding"
_PATCH_NAME = "get_masked_input_and_mask"

_patched = False


# ---------------------------------------------------------------------------
# reference path (CPU tensors only -- device tensors always take Triton)
# ---------------------------------------------------------------------------
def get_masked_input_and_mask_eager(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


# ---------------------------------------------------------------------------
# GCU Triton kernel
# ---------------------------------------------------------------------------
@triton.jit(
    do_not_specialize=[
        "n",
        "num_spc",
        "org_start",
        "org_end",
        "add_start",
        "add_end",
        "added_offset",
    ]
)
def _get_masked_input_and_mask_kernel_gcu(
    x_ptr,  # token ids: int32 view of the int64 buffer (WORD=2) or int32 (WORD=1)
    out_ptr,  # [n] masked input ids (int64)
    mask_ptr,  # [n] bool, True == out of this shard
    n,
    num_spc,  # runtime arg (not constexpr): the grid size follows num_tokens and
    # would otherwise trigger a fresh JIT compilation for every batch size
    org_start,
    org_end,
    add_start,
    add_end,
    added_offset,
    BLOCK: tl.constexpr,
    WORD: tl.constexpr,
    HAS_ADDED: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    for base in range(pid * BLOCK, n, num_spc * BLOCK):
        offs = base + lane
        m = offs < n  # single '<' comparison -> safe on GCU300
        v = tl.load(x_ptr + offs * WORD, mask=m, other=0)

        org_ok = (v < org_end) & ~(v < org_start)
        if HAS_ADDED:
            add_ok = (v < add_end) & ~(v < add_start)
            valid = org_ok | add_ok
            offset = tl.where(org_ok, org_start, added_offset)
        else:
            valid = org_ok
            offset = org_start

        tl.store(
            out_ptr + offs,
            tl.where(valid, v - offset, 0).to(out_ptr.dtype.element_ty),
            mask=m,
        )
        tl.store(mask_ptr + offs, (valid.to(tl.int32) == 0), mask=m)


def get_masked_input_and_mask_gcu(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-launch replacement for ``get_masked_input_and_mask``."""
    if input_.device.type not in ("gcu", "cuda"):
        return get_masked_input_and_mask_eager(
            input_,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
        )
    if not input_.is_contiguous():
        # ``.contiguous()`` is a no-op on the hot path (input_ids always is);
        # for the rare strided view it costs **one** copy kernel, which still
        # beats the ~13-launch eager decomposition.
        input_ = input_.contiguous()

    shape = input_.shape
    n = input_.numel()
    masked_input = torch.empty(shape, dtype=torch.int64, device=input_.device)
    input_mask = torch.empty(shape, dtype=torch.bool, device=input_.device)
    if n == 0:
        return masked_input, input_mask

    # int64 -> int32 view: no copy, no device kernel (little-endian low word).
    if input_.element_size() == 8:
        x = input_.view(torch.int32)
        word = 2
    else:
        x = input_
        word = 1

    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    grid = (min(triton.cdiv(n, _BLOCK), GCU_NUM_GRID),)
    _get_masked_input_and_mask_kernel_gcu[grid](
        x,
        masked_input,
        input_mask,
        n,
        grid[0],
        org_vocab_start_index,
        org_vocab_end_index,
        added_vocab_start_index,
        added_vocab_end_index,
        added_offset,
        BLOCK=_BLOCK,
        WORD=word,
        HAS_ADDED=added_vocab_start_index != added_vocab_end_index,
        num_warps=GCU_MAX_WARPS,
    )
    return masked_input, input_mask

_patched = False


def apply_patch_get_masked_input_and_mask() -> None:
    """Route ``get_masked_input_and_mask`` to the fused GCU Triton kernel."""
    global _patched
    if _patched:
        return
    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return
    try:
        mod = importlib.import_module(_PATCH_TARGET)
        if not hasattr(mod, _PATCH_NAME):
            # circular import: module still partially initialised -> retry later
            logger.debug("get_masked_input_and_mask patch deferred (partial import)")
            return
        setattr(mod, _PATCH_NAME, get_masked_input_and_mask_gcu)
        _patched = True
        logger.info(
            "Patched get_masked_input_and_mask -> fused GCU Triton kernel "
            "(1 launch instead of ~13, BLOCK=%d, grid<=%d)",
            _BLOCK,
            GCU_NUM_GRID,
        )
    except Exception as exc:
        logger.debug("get_masked_input_and_mask patch deferred (will retry): %s", exc)
