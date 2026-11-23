# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU patch: drafter-proposer stand-ins for the padded spec-decode prep.

``vllm/v1/spec_decode/utils.py`` runs two Triton kernels per padded-drafter
step; each builds a runtime-scalar ``if`` with tensor branches that
``evas-triton-opt`` aborts on, so the first draft step dies.  Both are rebound
in the ``llm_base_proposer`` namespace to host-driven stand-ins; wrappers keep
upstream bodies and the three sibling kernels stay upstream as unreachable at
one draft token.  They trade the padded path's no-blocking-CPU invariant
(``llm_base_proposer.py:1027``).

"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False

# Raise, instead of logging, when an upstream alias is found un-rebound.
_STRICT_REBIND_ENV = "VLLM_FL_STRICT_UPSTREAM_REBIND"

# Names read from the proposer module's own namespace.  Both are launched as
# ``kernel[(grid,)](...)``.
_KERNEL_NAMES = (
    "eagle_prepare_next_token_padded_kernel",
    "eagle_prepare_inputs_padded_kernel",
)

# Module the kernels are *defined* in.  A copy there is not a leak -- it is the
# source -- so it is rebound silently and kept out of the leak report.
_DEFINITION_MODULES = ("vllm.v1.spec_decode.utils",)


class _TorchKernelLauncher:
    """Give a plain function the ``kernel[(grid,)](args...)`` call form.

    The upstream call sites launch the templates as Triton kernels
    (``eagle_prepare_next_token_padded_kernel[grid](...)``); the replacement is
    subscripted the same way.  Mirrors ``vllm/utils/cpu_triton_utils._FuncWrapper``,
    which the upstream CPU path uses for the same purpose.
    """

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _grid):
        return self._fn


def _lpu_eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,
    discard_request_mask_ptr,
    backup_next_token_ids_ptr,
    next_token_ids_ptr,
    valid_sampled_tokens_count_ptr,
    vocab_size,
    num_sampled_tokens_per_req,
    batch_size,
    stride_sampled_token_ids,
    BLOCK_SIZE_TOKENS=None,
):
    """LPU stand-in for ``eagle_prepare_next_token_padded_kernel``.

    Per request: a discarded row takes ``backup`` and reports a count of 0;
    otherwise the row's *last* valid sampled token wins, with ``backup`` again
    when the row holds none.  Keeping the kernel's quirks:

    - validity is ``(id != -1) & (id < vocab_size) & (pos < num_sampled)`` -- a
      negative id other than -1 counts as valid, there is no lower bound;
    - the last valid position is the maximum index among the valid ones, and its
      token is picked by the kernel's sum trick rather than a second load; ties
      cannot arise because the index is unique;
    - the final tile's out-of-range lanes are ``-1`` (the kernel loads with
      ``other=-1``), so they are simply invalid.

    The kernel reads row ``req`` at ``sampled_token_ids_ptr + req * stride``,
    which only matches tensor indexing while the inner stride is 1; this version
    indexes normally, so a view with a unit inner stride and a padded row stride
    behaves identically, and a non-unit inner stride is handled correctly
    instead of misread.  ``BLOCK_SIZE_TOKENS`` only sizes the Triton tile.
    ``batch_size`` must be the row count (the caller passes
    ``sampled_token_ids.shape[0]``); it is checked rather than trusted.
    """
    if discard_request_mask_ptr.dtype != torch.bool:
        raise ValueError(
            "eagle_prepare_next_token_padded: discard_request_mask must be "
            f"bool, got {discard_request_mask_ptr.dtype}"
        )
    if backup_next_token_ids_ptr.dtype != torch.int32:
        raise ValueError(
            "eagle_prepare_next_token_padded: backup_next_token_ids must be "
            f"int32, got {backup_next_token_ids_ptr.dtype}"
        )
    rows, width = sampled_token_ids_ptr.shape
    if rows != batch_size:
        raise ValueError(
            f"eagle_prepare_next_token_padded: {rows} sampled-token rows for "
            f"batch_size={batch_size}"
        )
    if num_sampled_tokens_per_req > width:
        # The kernel would read past the end of the row here; upstream always
        # passes the row width, so this means the caller changed.
        raise ValueError(
            f"eagle_prepare_next_token_padded: num_sampled_tokens_per_req="
            f"{num_sampled_tokens_per_req} exceeds the row width {width}"
        )
    if rows == 0:
        return

    tokens = sampled_token_ids_ptr.detach().cpu()
    offs = torch.arange(width, dtype=torch.int64)
    valid = (tokens != -1) & (tokens < vocab_size)
    valid &= (offs < num_sampled_tokens_per_req).unsqueeze(0)

    valid_count = valid.sum(dim=1, dtype=torch.int32)
    if width:
        last_valid_index = torch.where(valid, offs, -1).amax(dim=1)
        last_valid_token = (
            tokens * (offs == last_valid_index.unsqueeze(1))
        ).sum(dim=1, dtype=torch.int32)
    else:
        # A zero-width row has no positions at all (the kernel's tile would be
        # empty as well), so the reduction has no maximum to take and every row
        # falls back below.
        last_valid_token = torch.zeros(rows, dtype=torch.int32)

    # These two are caller-owned buffers that can be wider than this step's
    # batch (they are sized once for the largest batch); the kernel only reads
    # the first ``num_reqs`` entries.
    backup = backup_next_token_ids_ptr.detach().cpu().reshape(-1)[:rows]
    discarded = discard_request_mask_ptr.detach().cpu().reshape(-1)[:rows]
    use_backup = discarded | (valid_count == 0)

    # int32 on both ends: upstream allocates the two outputs with
    # ``torch.empty(..., dtype=torch.int32)`` / ``new_empty``.  Only the first
    # ``rows`` entries are written, as the kernel does -- a wider caller buffer
    # keeps its tail untouched.
    next_token_ids_ptr[:rows].copy_(
        torch.where(use_backup, backup, last_valid_token).to(
            next_token_ids_ptr.dtype
        ).to(next_token_ids_ptr.device)
    )
    valid_sampled_tokens_count_ptr[:rows].copy_(
        torch.where(discarded, torch.zeros_like(valid_count), valid_count).to(
            valid_sampled_tokens_count_ptr.dtype
        ).to(valid_sampled_tokens_count_ptr.device)
    )


def _lpu_eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,
    valid_sampled_tokens_count_ptr,
    query_start_loc_gpu_ptr,
    token_indices_to_sample_ptr,
    num_rejected_tokens_gpu_ptr,
    num_reqs,
):
    """LPU stand-in for ``eagle_prepare_inputs_padded_kernel``.

    Per request: ``num_draft`` is the difference to the previous cumulative
    count (the first entry is inclusive, there is no leading zero), the rejected
    count is ``num_draft + 1 - valid_count`` and is forced to 0 when nothing was
    drafted at all, and the token to sample sits that many slots before the end
    of the request's query block.

    The rejected-count rule is not an invention: upstream's *unpadded* Python
    path computes the same quantity as
    ``n + 1 - len(sampled_token_ids[i]) if n > 0 else 0``
    (``llm_base_proposer.py:1107-1110``), which is what makes the two paths
    comparable.  A ``0 <= num_rejected <= num_draft`` check is deliberately not
    added -- the kernel does not have one, and inventing a failure mode here
    would only hide a caller bug behind a host exception.
    """
    available = int(cu_num_draft_tokens_ptr.shape[0])
    if available < num_reqs:
        raise ValueError(
            f"eagle_prepare_inputs_padded: {available} cumulative draft counts "
            f"for num_reqs={num_reqs}"
        )
    if query_start_loc_gpu_ptr.shape[0] < num_reqs + 1:
        raise ValueError(
            "eagle_prepare_inputs_padded: query_start_loc has "
            f"{query_start_loc_gpu_ptr.shape[0]} entries, needs {num_reqs + 1}"
        )
    if num_reqs == 0:
        return

    # Slice to num_reqs: the buffers may be wider, and the kernel reads only
    # the first num_reqs entries of each.
    cu = cu_num_draft_tokens_ptr.detach().cpu().to(torch.int32)[:num_reqs]
    valid_count = valid_sampled_tokens_count_ptr.detach().cpu().to(
        torch.int32
    )[:num_reqs]
    query_start_loc = query_start_loc_gpu_ptr.detach().cpu().to(torch.int32)

    previous = torch.cat([torch.zeros(1, dtype=torch.int32), cu[:-1]])
    num_draft = cu - previous
    num_rejected = torch.where(
        num_draft > 0, num_draft + 1 - valid_count, torch.zeros_like(num_draft)
    )
    index_to_sample = (
        query_start_loc[1:num_reqs + 1] - 1 - num_rejected
    )

    # Only the first num_reqs entries are written, as the kernel does.
    token_indices_to_sample_ptr[:num_reqs].copy_(
        index_to_sample.to(token_indices_to_sample_ptr.dtype).to(
            token_indices_to_sample_ptr.device
        )
    )
    num_rejected_tokens_gpu_ptr[:num_reqs].copy_(
        num_rejected.to(num_rejected_tokens_gpu_ptr.dtype).to(
            num_rejected_tokens_gpu_ptr.device
        )
    )


def _sweep_leaked_aliases(originals: dict) -> None:
    """Rebind any remaining alias of the two upstream kernels, then report.

    The kernels are reached through the namespace of
    ``vllm.v1.spec_decode.llm_base_proposer``, so rebinding them *there* is
    sufficient today.  But ``from ... import X`` copies a name into the
    importing module's namespace, and one such copy anywhere would silently keep
    launching the upstream Triton kernel -- the failure mode this patch exists
    to remove would come back as "sometimes it still crashes".  The sweep runs
    at patch-install time and flags whatever it finds.

    The defining module (``utils.py``) is rebound but not reported: it is where
    the names come from, and nothing there calls them, so a copy is expected
    rather than a leak.  Identity is compared against the objects captured
    before the rebind, so re-entry cannot mistake our own stand-ins for upstream
    ones.  A module that refuses the rebind is reported as unresolved rather
    than as a leak that was handled.
    """
    import sys as _sys

    leaked: dict[str, list[str]] = {}
    unresolved: dict[str, list[str]] = {}
    for name, (original, replacement) in originals.items():
        holders = []
        blocked = []
        for mod_name, mod in list(_sys.modules.items()):
            if mod is None or mod_name.startswith(("torch", "triton")):
                continue
            if getattr(mod, name, None) is not original:
                continue
            try:
                setattr(mod, name, replacement)
            except Exception:  # noqa: BLE001 - some modules are not writable
                blocked.append(mod_name)
                continue
            if mod_name not in _DEFINITION_MODULES:
                holders.append(mod_name)
        if blocked:
            unresolved[name] = blocked
        if holders:
            leaked[name] = holders
    if leaked:
        logger.error(
            "spec-decode proposer stand-ins: upstream kernel aliases still "
            "bound in other module namespaces, rebound there too: %s",
            leaked,
        )
        print(f"[spec-decode-proposer] rebound leaked aliases: {leaked}")
    if unresolved:
        # These keep launching the upstream Triton kernel: the abort this patch
        # exists to remove would come back as "sometimes it still crashes".
        logger.error(
            "spec-decode proposer stand-ins: module namespaces holding an "
            "upstream kernel alias that could NOT be rebound: %s", unresolved
        )
        print(f"[spec-decode-proposer] UNRESOLVED aliases: {unresolved}")
    if (leaked or unresolved) and os.environ.get(_STRICT_REBIND_ENV) == "1":
        raise RuntimeError(
            f"upstream spec-decode proposer kernel aliases leaked into "
            f"{leaked} / unresolved in {unresolved}"
        )


_STANDINS = {
    "eagle_prepare_next_token_padded_kernel":
        _lpu_eagle_prepare_next_token_padded_kernel,
    "eagle_prepare_inputs_padded_kernel":
        _lpu_eagle_prepare_inputs_padded_kernel,
}


def apply_lpu_spec_decode_proposer_patch() -> None:
    """Rebind the drafter proposer's padded-prep kernels to torch.

    Applied unconditionally on LPU: the rebindings are inert unless speculative
    decoding is enabled, which no framework code here can be keyed off
    (``VLLM_FL_DSV4_MTP`` is read by the e2e script only).
    """
    global _PATCHED

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return
    if _PATCHED:
        return

    import vllm.v1.spec_decode.llm_base_proposer as _proposer

    replacements = {
        name: _TorchKernelLauncher(_STANDINS[name]) for name in _KERNEL_NAMES
    }
    originals = {
        name: (getattr(_proposer, name), replacement)
        for name, replacement in replacements.items()
    }
    for name, replacement in replacements.items():
        setattr(_proposer, name, replacement)

    # The sentinel is set after the sweep: it means the install completed, and a
    # strict-mode raise must not leave a patch that reports itself as installed.
    _sweep_leaked_aliases(originals)
    _PATCHED = True
