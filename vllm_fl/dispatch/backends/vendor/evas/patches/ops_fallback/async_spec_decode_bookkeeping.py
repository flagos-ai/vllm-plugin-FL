# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU patch: async spec-decode bookkeeping without int64 device arithmetic.

Upstream opens with ``prev_positions.clamp(min=0)`` on an int64 buffer, which
LPU aten rejects (``ERR01002``).  The stand-in narrows that buffer to int32 for
the clamp and keeps the gather index int64; the upstream quirks are load-bearing.
Both model runners hold their own reference, so the rebind sweeps every holder.

"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False

# The true upstream callable, captured on first install so that a re-entry after
# a strict-mode raise still compares against it instead of against our own
# stand-in (which would silently stop reporting holders that refused the rebind).
_ORIGINAL = None

# Raise, instead of logging, when a holder of the upstream callable could not be
# rebound (strict mode is the CI guard; the run itself only logs).
_STRICT_REBIND_ENV = "VLLM_FL_STRICT_UPSTREAM_REBIND"

_TARGET_NAME = "update_num_computed_tokens_for_batch_change"

# Module the callable is *defined* in.  It is also rebound (the fallback for a
# caller that binds it late), and it is the definition, not a consumer copy.
_DEFINITION_MODULES = ("vllm.v1.spec_decode.utils",)


def _gather_index_and_mask(prev_positions):
    """Upstream ``prev_positions.clamp(min=0)`` plus the ``>= 0`` mask.

    Split out because the clamp is a *bounds guard*, not a value fix: a
    non-participating row keeps its ``-1`` in ``nonneg`` but gathers row 0, so
    the index never leaves the buffers.  The guard cannot be pinned from the
    results -- on CPU a ``-1`` index merely wraps and the ``where`` discards
    whatever it read -- so this seam is what lets a test assert the index
    itself.

    LPU aten rejects clamp on int64 and ``prev_positions`` is the only int64
    argument, so it is normalised first (dsv4_compressor does the same for its
    slot mapping).  Positions and the ``-1`` sentinel are exact in int32, and
    the index returns to int64 because that is the dtype advanced indexing
    takes on this backend.
    """
    pos = prev_positions.to(torch.int32)
    nonneg = pos >= 0
    return nonneg, pos.clamp(min=0).to(prev_positions.dtype)


def _lpu_update_num_computed_tokens_for_batch_change(
    num_computed_tokens,
    num_accepted_tokens,
    prev_positions,
    valid_sampled_token_count,
    prev_num_draft_tokens,
    cpu_num_computed_tokens,
) -> None:
    """LPU stand-in for ``update_num_computed_tokens_for_batch_change``.

    In-place, returns ``None``: both writes go through ``copy_`` into the
    arguments the caller passed.  Rebinding a parameter (or returning a tensor)
    would silently lose the write -- the caller keeps no reference to the result.
    """
    n = prev_positions.shape[0]
    if n == 0:
        # Upstream is a no-op for an empty batch as well (every gather and write
        # is empty); returning early only skips the launches.
        return

    nonneg, gather_indices = _gather_index_and_mask(prev_positions)

    # Whole buffers, not [:n]: a current row can map to a previous index >= n
    # when the batch shrank.  Advanced indexing materialises, so these are
    # previous-step snapshots taken before either write below.
    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = nonneg & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    num_computed_tokens[:n].copy_(
        torch.where(participating, corrected, cpu_num_computed_tokens)
    )
    num_accepted_tokens.copy_(
        torch.where(participating, valid_counts, num_accepted_tokens)
    )


def _rebind_all_holders(original, replacement) -> None:
    """Rebind ``original`` to ``replacement`` in every namespace holding it.

    The two model runners import this callable with ``from ... import``, which
    copies it into their own namespace at import time; assigning in the defining
    module would never reach that copy.  So the sweep *is* the install
    mechanism here, and a rebound consumer copy is expected rather than a leak
    (the sibling kernel patches report the opposite way round, because there the
    definition module is the only legitimate holder).

    A namespace that refuses the ``setattr`` is a real defect: it keeps calling
    the upstream body, so the int64 clamp comes back as "sometimes it still
    raises".  Those are logged as an error and, under
    ``VLLM_FL_STRICT_UPSTREAM_REBIND=1``, raised.
    """
    import sys as _sys

    rebound: list[str] = []
    blocked: list[str] = []
    for mod_name, mod in list(_sys.modules.items()):
        if mod is None or mod_name.startswith(("torch", "triton")):
            continue
        if getattr(mod, _TARGET_NAME, None) is not original:
            continue
        try:
            setattr(mod, _TARGET_NAME, replacement)
        except Exception:  # noqa: BLE001 - some modules are not writable
            blocked.append(mod_name)
            continue
        rebound.append(mod_name)

    if rebound:
        logger.info(
            "async spec-decode bookkeeping: rebound in %s",
            [m for m in rebound if m not in _DEFINITION_MODULES] or rebound,
        )
    if blocked:
        logger.error(
            "async spec-decode bookkeeping: module namespaces holding the "
            "upstream callable that could NOT be rebound (they keep running the "
            "int64 clamp): %s", blocked
        )
        print(f"[async-spec-decode-bookkeeping] UNRESOLVED holders: {blocked}")
        if os.environ.get(_STRICT_REBIND_ENV) == "1":
            raise RuntimeError(
                f"upstream async spec-decode bookkeeping callable could not be "
                f"rebound in {blocked}"
            )


def apply_lpu_async_spec_decode_bookkeeping_patch() -> None:
    """Rebind the async spec-decode bookkeeping to an int64-free body on LPU.

    Applied unconditionally on LPU: the rebind is inert unless async
    speculative decoding runs, which no framework code here can be keyed off
    (``VLLM_FL_DSV4_MTP`` is read by the e2e script only).
    """
    global _PATCHED, _ORIGINAL

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return
    if _PATCHED:
        return

    import vllm.v1.spec_decode.utils as _utils

    if _ORIGINAL is None:
        _ORIGINAL = getattr(_utils, _TARGET_NAME)
    # The sweep rebinds the defining module too, so there is one install path
    # rather than two.  A copy imported after this point reads the rebound
    # module and is correct by construction.
    replacement = _lpu_update_num_computed_tokens_for_batch_change

    # The sentinel is set after the sweep: it means the install completed, and a
    # strict-mode raise must not leave a patch that reports itself as installed.
    _rebind_all_holders(_ORIGINAL, replacement)
    _PATCHED = True
