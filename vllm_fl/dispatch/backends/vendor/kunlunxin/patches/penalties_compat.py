# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: FlagTree's XPU Triton backend cannot lower either of the V2 sampler's
# penalty kernels in ``vllm.v1.worker.gpu.sample.penalties``.
#
# ``_bincount_kernel`` (penalties.py:217) fails in ``TritonXPULegalize``::
#
#     .../penalties.py:244:31: error: 'arith.divsi' op requires the same type
#     for all operands and results
#     RuntimeError: PassManager::run failed ... [`TritonXPULegalize`]
#
# ``_penalties_kernel`` (penalties.py:105) fails one pass earlier, in
# ``ConvertTritonToTritonXPU``::
#
#     out of resource: uni_sram PassManager::run failed
#     note: Pipeline failed while executing [`ConvertTritonToTritonXPU`]
#
# Both are reached from ``warmup_kernels`` (vllm/v1/worker/gpu/warmup.py:363 ->
# penalties.py:70 ``bincount``), so every V2 run aborts during warmup even when
# no request actually asks for penalties.
#
# Unlike the M-RoPE ``scf.if`` bug handled in ``rope_compat``, this one does not
# yield to a source rewrite.  Measured on P800 (FlagTree triton,
# /opt/flagtree/triton/backends/xpu):
#
#     * ``_bincount_kernel`` fails at BLOCK_SIZE 1024/256/64 and with the index
#       mapping in both int64 and int32 -- so neither block size nor dtype is
#       the trigger.
#     * Rewriting ``//``/``%`` as ``>> 5``/``& 31``, and folding both atomic
#       offsets into explicit int32, both still fail.
#     * Splitting the kernel in half: the ``atomic_or`` half and the
#       ``atomic_add`` half each fail on their own.
#     * The minimal reproducer is an atomic with an int32 scalar base plus a
#       tensor offset.  Holding everything else fixed:
#
#           atomic offset form                   result
#           tensor offset only                   OK
#           int64 scalar base + tensor offset    OK
#           int32 scalar base + tensor offset    FAIL
#           program_id base + tensor offset      OK
#           explicit int32 casts / folded        OK
#
#       The real kernel's base comes from ``tl.load(expanded_idx_mapping_ptr
#       + token_idx)``, and the pass mishandles it regardless of how the
#       arithmetic is spelled.
#     * ``_penalties_kernel`` fails at every BLOCK_SIZE from 8192 down to 64,
#       and reports ``Required: 0, Hardware limit: 0``, so it is not a genuine
#       SRAM budget problem either.
#
# Replace both entry points with torch implementations.  Penalties are an
# opt-in, rarely used sampling feature -- ``PenaltiesState.apply``
# (penalties.py:85) already early-returns when no request in the batch uses
# them, and ``apply_staged_writes`` (penalties.py:62) only calls ``bincount``
# for newly added penalty requests -- so the slower torch path costs nothing on
# the common path and keeps the feature correct rather than unavailable.
#
# The torch code mirrors the kernels bit for bit, including the packed int32
# prompt bitmask layout (``1 << (tok % 32)`` at word ``tok // 32``, where bit 31
# wraps to a negative int32 exactly as ``tl.int32`` does) and the draft-token
# accumulation that ``_penalties_kernel`` folds into ``output_bin_counts``.
# Rows are processed in chunks because the unpacked bitmask is vocab-sized.
# --------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_penalties_torch"

# Rows handled per chunk.  The unpacked prompt bitmask is vocab-sized, so a
# whole batch at once would allocate hundreds of MB for a path that is rarely
# taken.
_ROW_CHUNK = 16


def _bincount(
    expanded_idx_mapping,
    all_token_ids,
    prompt_len,
    prefill_len,
    prompt_bin_mask,
    output_bin_counts,
    max_prefill_len,
) -> None:
    """torch equivalent of ``penalties._bincount_kernel``.

    Rebuilds, for each request named by ``expanded_idx_mapping``, the packed
    bitmask of its prompt tokens and the per-token counts of its already
    generated output tokens.
    """
    import torch

    idx = expanded_idx_mapping.to(torch.int64)
    # Use index_fill_ instead of `tensor[idx] = 0` to avoid a sync, as upstream
    # does.
    prompt_bin_mask.index_fill_(0, idx, 0)
    output_bin_counts.index_fill_(0, idx, 0)

    num_reqs = idx.numel()
    if num_reqs == 0 or max_prefill_len <= 0:
        return

    device = prompt_bin_mask.device
    num_words = prompt_bin_mask.shape[1]
    vocab_size = output_bin_counts.shape[1]

    tokens = all_token_ids.index_select(0, idx)[:, :max_prefill_len].to(torch.int64)
    p_len = prompt_len.index_select(0, idx).to(torch.int64).unsqueeze(1)
    pf_len = prefill_len.index_select(0, idx).to(torch.int64).unsqueeze(1)
    col = torch.arange(tokens.shape[1], device=device).unsqueeze(0)

    prompt_valid = col < p_len
    output_valid = (col >= p_len) & (col < pf_len)

    shifts = torch.arange(32, device=device, dtype=torch.int64)
    weights = torch.ones(32, device=device, dtype=torch.int64) << shifts

    for lo in range(0, num_reqs, _ROW_CHUNK):
        hi = min(lo + _ROW_CHUNK, num_reqs)
        rows = idx[lo:hi]
        tok = tokens[lo:hi]
        p_ok = prompt_valid[lo:hi]
        o_ok = output_valid[lo:hi]
        n = hi - lo

        # Prompt bitmask.  Invalid slots are routed to a scratch column so they
        # cannot clobber a real token's bit.
        dense = torch.zeros(n, num_words * 32 + 1, dtype=torch.bool, device=device)
        dense.scatter_(1, torch.where(p_ok, tok, dense.shape[1] - 1), True)
        bits = dense[:, : num_words * 32].view(n, num_words, 32).to(torch.int64)
        words = (bits * weights).sum(-1)
        prompt_bin_mask.index_copy_(0, rows, words.to(prompt_bin_mask.dtype))

        # Output token counts, same scratch-column trick.
        counts = torch.zeros(n, vocab_size + 1, dtype=torch.int32, device=device)
        counts.scatter_add_(
            1,
            torch.where(o_ok, tok, vocab_size),
            torch.ones(n, tok.shape[1], dtype=torch.int32, device=device),
        )
        output_bin_counts.index_copy_(
            0, rows, counts[:, :vocab_size].to(output_bin_counts.dtype)
        )


def _apply_penalties(
    logits,
    expanded_idx_mapping,
    token_ids,
    expanded_local_pos,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    prompt_bin_mask,
    output_bin_counts,
) -> None:
    """torch equivalent of ``penalties._penalties_kernel``, in place on ``logits``."""
    import torch

    num_tokens, vocab_size = logits.shape
    if num_tokens == 0:
        return

    device = logits.device
    req = expanded_idx_mapping.to(torch.int64)
    rep = repetition_penalty.index_select(0, req).to(torch.float32)
    freq = frequency_penalty.index_select(0, req).to(torch.float32)
    pres = presence_penalty.index_select(0, req).to(torch.float32)

    # Same early-out as the kernel: rows with no active penalty are untouched.
    active = (rep != 1.0) | (freq != 0.0) | (pres != 0.0)
    sel = active.nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return

    local_pos = expanded_local_pos.index_select(0, sel).to(torch.int64)
    max_pos = int(local_pos.max().item())
    num_words = prompt_bin_mask.shape[1]
    shifts = torch.arange(32, device=device, dtype=torch.int32)

    for lo in range(0, sel.numel(), _ROW_CHUNK):
        hi = min(lo + _ROW_CHUNK, sel.numel())
        rows = sel[lo:hi]
        req_rows = req.index_select(0, rows)
        n = hi - lo

        lg = logits.index_select(0, rows).to(torch.float32)

        counts = output_bin_counts.index_select(0, req_rows).to(torch.int32)
        if max_pos > 0:
            # Fold in the draft tokens at earlier positions of the same request,
            # which the kernel accumulates into output_bin_counts on the fly.
            pos = local_pos[lo:hi]
            start = rows - pos
            padded = torch.zeros(n, vocab_size + 1, dtype=torch.int32, device=device)
            for step in range(max_pos):
                taken = pos > step
                src = (start + step + 1).clamp_(0, token_ids.shape[0] - 1)
                tok = token_ids.index_select(0, src).to(torch.int64)
                padded.scatter_add_(
                    1,
                    torch.where(taken, tok, vocab_size).unsqueeze(1),
                    torch.ones(n, 1, dtype=torch.int32, device=device),
                )
            counts = counts + padded[:, :vocab_size]
        output_bin_mask = counts > 0

        # Repetition penalty.  rep == 1.0 makes the scale a no-op, which is why
        # the kernel's per-row `use_rep_penalty` branch needs no counterpart.
        words = prompt_bin_mask.index_select(0, req_rows).to(torch.int32)
        prompt_mask = (
            ((words.unsqueeze(-1) >> shifts) & 1)
            .to(torch.bool)
            .view(n, num_words * 32)[:, :vocab_size]
        )
        scale = torch.where(
            prompt_mask | output_bin_mask, rep.index_select(0, rows).unsqueeze(1), 1.0
        )
        # Positive logits are divided by the penalty, negative ones multiplied.
        lg *= torch.where(lg > 0, 1.0 / scale, scale)

        lg -= freq.index_select(0, rows).unsqueeze(1) * counts
        lg -= pres.index_select(0, rows).unsqueeze(1) * output_bin_mask

        logits.index_copy_(0, rows, lg.to(logits.dtype))


def apply_penalties_compat() -> None:
    """Swap the V2 penalty kernels for torch code FlagTree does not have to lower."""
    try:
        import importlib

        penalties = importlib.import_module("vllm.v1.worker.gpu.sample.penalties")
    except ImportError:
        # Builds without the V2 runner have nothing to patch.
        return

    if getattr(penalties, _PATCH_FLAG, False):
        return

    if not hasattr(penalties, "bincount") or not hasattr(penalties, "apply_penalties"):
        return

    penalties.bincount = _bincount
    penalties.apply_penalties = _apply_penalties
    setattr(penalties, _PATCH_FLAG, True)
    logger.info(
        "kunlunxin penalties_compat: replaced bincount and apply_penalties with "
        "torch implementations; FlagTree's XPU backend cannot lower either "
        "penalty kernel (TritonXPULegalize / ConvertTritonToTritonXPU)"
    )
