# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: FlagTree's XPU Triton backend cannot lower an early ``return`` that sits
# after the kernel has already emitted tensor work.  The V2 model runner's
# ``vllm.v1.worker.gpu.input_batch._combine_sampled_and_draft_tokens_kernel``
# stores the logits indices, then bails out for prefill-only requests::
#
#     tl.store(logits_indices_ptr + cu_num_logits_start + block, ...)
#     ...
#     if seq_len <= prefill_len:
#         # Handling prefill tokens. No sampled or draft tokens.
#         return
#     ...                      # more tensor work follows
#
# so every V2 run aborts while preparing the first batch with::
#
#     .../vllm/v1/worker/gpu/input_batch.py:441:18: error: operation with block
#     successors must terminate its parent block
#     note: Pipeline failed while executing [`TritonXPULegalize` on
#     'builtin.module' operation]
#     triton.runtime.errors.OutOfResources: out of resource: uni_sram
#
# The ``OutOfResources`` wrapper is misleading -- it reports "Required: 0,
# Hardware limit: 0", so this is a lowering failure, not an SRAM budget problem,
# and reducing the block size does not help.
#
# Measured on P800 (FlagTree triton, /opt/flagtree/triton/backends/xpu), varying
# only how the conditional is written and holding everything else fixed:
#
#     construct                                  BLOCK=64  256  1024
#     masked store, then early return               FAIL   FAIL  FAIL
#     masked store, then if-guarded body            OK     OK    OK
#
# The block size is irrelevant here, unlike the M-RoPE case in rope_compat.
# What matters is that the ``return`` leaves a block that already has tensor
# successors.  The three other early-return kernels in the same module --
# _prepare_prefill_inputs_kernel, _prepare_pos_seq_lens_kernel and
# _post_update_kernel -- lower without complaint, because each returns before
# emitting any tensor work or returns in tail position inside its branch.
#
# Removing the early return is necessary but not sufficient, because two further
# FlagTree limitations constrain how the condition may be re-expressed.
#
# 1. A branch that *yields a tensor* cannot be lowered either.  Wrapping the
#    rest of the body in ``if seq_len > prefill_len:`` moves the draft-token
#    load inside a branch, which fails with::
#
#        error: LLVM Translation failed for operation:
#        builtin.unrealized_conversion_cast
#        RuntimeError: failed to translate module to LLVM IR
#
#    This is the same defect rope_compat works around.  Note that the launcher
#    uses BLOCK_SIZE = next_power_of_2(num_speculative_steps +
#    num_new_sampled_tokens), which is 1 without speculative decoding, so the
#    failure is not block-size dependent here.
#
# 2. A *scalar* bool AND-ed into a tensor mask is silently dropped.  This one
#    does not fail to compile -- it produces wrong results, which is worse.
#    Measured with an all-false predicate, counting elements actually written
#    where zero were expected:
#
#        mask expression                      BLOCK=1   2    4    64
#        block < n            (n = 0)            0      0    0     0   correct
#        (block < n) & active (active False)    48     48   48    48   WRONG
#        block < where(active, n, 0)             0      0    0     0   correct
#
#    So the request-level predicate must be folded into the element bound, not
#    conjoined with the mask.
#
# 3. A 1-wide access to an int64 tensor, at a row index that was itself loaded
#    from an int64 tensor, cannot be lowered.  Both int64 operands and the
#    width of 1 are required to trigger it; the index dtype is what matters,
#    not a cast applied afterwards:
#
#        idx dtype  src dtype  BLOCK=1  BLOCK=4
#        int64      int64       FAIL      OK
#        int64      int32        OK       OK
#        int32      int64        OK       OK
#        int32      int32        OK       OK
#
#    This is reached in the default configuration: req_states.idx_mapping and
#    req_states.draft_tokens are both int64 (vllm/v1/worker/gpu/states.py),
#    and BLOCK_SIZE is next_power_of_2(num_speculative_steps +
#    num_new_sampled_tokens), which is 1 whenever speculative decoding is off.
#    Casting the loaded index or the loaded value does not help; widening the
#    access to two elements does.
#
# The kernel below therefore keeps a single exit, puts only scalar work inside a
# branch, folds ``active`` into the draft-token bound with ``tl.where``, and
# widens the draft-token access to at least two elements.  The control flow is
# otherwise identical to upstream: the statements skipped for a prefill-only
# request are the same ones the early return skipped, and the widened lane is
# always masked off.  Verified against a torch reference across 60
# configurations (num_speculative_steps 0/1/3/7/15 x 3/5/6/16 requests x 3
# seeds, each mixing prefill-only, decode-with-prompt-tail,
# exactly-at-prefill-boundary and pure-decode requests), using the dtypes the
# runner actually passes (int64 idx_mapping, draft_tokens and
# last_sampled_tokens; int32 input_ids): the rewritten kernel reproduces both
# input_ids and logits_indices exactly.
# --------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_input_batch_no_early_return"


def _build_kernel():
    """Build the single-exit variant of the sampled/draft token combiner."""
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _combine_sampled_and_draft_tokens_kernel(
        input_ids_ptr,
        idx_mapping_ptr,
        last_sampled_tokens_ptr,
        query_start_loc_ptr,
        seq_lens_ptr,
        prefill_len_ptr,
        draft_tokens_ptr,
        draft_tokens_stride,
        cu_num_logits_ptr,
        logits_indices_ptr,
        BLOCK_SIZE: tl.constexpr,
        NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
    ):
        batch_idx = tl.program_id(0)
        req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

        # Get the number of logits and draft tokens.
        cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
        cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
        num_logits = cu_num_logits_end - cu_num_logits_start
        num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

        # Compute the logits indices.
        block = tl.arange(0, BLOCK_SIZE)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        logits_start = query_end - num_logits
        tl.store(
            logits_indices_ptr + cu_num_logits_start + block,
            logits_start + block,
            mask=block < num_logits,
        )

        seq_len = tl.load(seq_lens_ptr + batch_idx)
        prefill_len = tl.load(prefill_len_ptr + req_state_idx)
        # Upstream returns early when seq_len <= prefill_len (prefill tokens
        # only, so there is nothing to combine). FlagTree cannot lower that
        # return here, so express the same condition without leaving the body.
        active = seq_len > prefill_len

        # A branch that only loads and stores scalars lowers without trouble.
        if NUM_NEW_SAMPLED_TOKENS > 0:
            # Keep prompt-tail slots intact; only rewrite generated-token slots.
            if active and (seq_len - num_logits) >= prefill_len:
                # Write the last sampled token ID to input_ids.
                last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
                tl.store(input_ids_ptr + logits_start, last_token_id)

        # Write the draft tokens (if any) to input_ids. Both guards -- "this
        # request is past its prefill" and "there are draft tokens" -- are
        # folded into the element bound, because neither may be expressed as a
        # branch around tensor work nor as a scalar conjunct in the mask (see
        # the notes above the kernel).
        #
        # The draft block is widened to at least two elements: a 1-wide access
        # to an int64 tensor at an index that was itself loaded from an int64
        # tensor cannot be lowered. The mask is unchanged, so the extra lane is
        # always masked off and the widening is not observable.
        DRAFT_BLOCK: tl.constexpr = max(BLOCK_SIZE, 2)
        draft_block = tl.arange(0, DRAFT_BLOCK)
        num_draft_tokens_eff = tl.where(active, num_draft_tokens, 0)
        mask = draft_block < num_draft_tokens_eff
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + draft_block,
            mask=mask,
            other=0,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + draft_block,
            draft_tokens,
            mask=mask,
        )

    return _combine_sampled_and_draft_tokens_kernel


_PATCH_FLAG_LAUNCHERS = "_vllm_fl_kunlunxin_input_batch_launchers_torch"


def _make_prepare_pos_seq_lens_torch():
    """Pure-torch replacement for prepare_pos_seq_lens.

    FlagTree silently miscompiles _prepare_pos_seq_lens_kernel at decode time,
    producing wrong position IDs and sequence lengths from the second request
    onward in a concurrent batch.  The torch implementation is semantically
    identical and runs in a Python loop (batch sizes are always small).
    """
    import torch

    def prepare_pos_seq_lens_torch(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        pos,
        seq_lens,
    ):
        num_reqs = idx_mapping.shape[0]
        max_num_reqs = seq_lens.shape[0]
        for r in range(num_reqs):
            req_idx = idx_mapping[r].item()
            nc = num_computed_tokens[req_idx].item()
            qs = query_start_loc[r].item()
            qe = query_start_loc[r + 1].item()
            ql = qe - qs
            seq_lens[r] = nc + ql
            if ql > 0:
                pos[qs:qe] = torch.arange(
                    nc, nc + ql, device=pos.device, dtype=pos.dtype
                )
        # Pad unused seq_lens entries (matches the special last thread block).
        if num_reqs < max_num_reqs:
            seq_lens[num_reqs:] = 0

    return prepare_pos_seq_lens_torch


def _make_get_num_sampled_and_rejected_torch():
    """Pure-torch replacement for get_num_sampled_and_rejected."""

    def get_num_sampled_and_rejected_torch(
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    ):
        for r in range(idx_mapping.shape[0]):
            req_idx = idx_mapping[r].item()
            is_cp = seq_lens[r].item() < prefill_len[req_idx].item()
            nl = cu_num_logits[r + 1].item() - cu_num_logits[r].item()
            if is_cp:
                num_sampled[r] = 0
                num_rejected[r] = 0
            else:
                ns = num_sampled[r].item()
                num_rejected[r] = nl - ns

    return get_num_sampled_and_rejected_torch


def _make_post_update_torch():
    """Pure-torch replacement for post_update.

    _post_update_kernel writes each newly-sampled token into all_token_ids and
    advances num_computed_tokens — it runs on every decode step.  FlagTree
    silently miscompiles the scalar loop ``for i in range(num_sampled)`` inside
    the kernel, corrupting the entire token-id history from step 2 onward and
    causing ``!`` to be produced for all subsequent tokens.
    """

    def post_update_torch(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    ):
        for r in range(idx_mapping.shape[0]):
            req_idx = idx_mapping[r].item()
            if req_idx < 0:
                continue
            ns = num_sampled[r].item()
            tlen = total_len[req_idx].item()
            if ns > 0:
                last_sampled_tokens[req_idx] = sampled_tokens[r, ns - 1]
                total_len[req_idx] = tlen + ns
                for i in range(ns):
                    tok = sampled_tokens[r, i]
                    all_token_ids[req_idx, tlen + i] = tok
                    if output_bin_counts is not None:
                        tid = tok.item()
                        if tid < output_bin_counts.shape[1]:
                            output_bin_counts[req_idx, tid] += 1
            ql = (
                (query_start_loc[r + 1] - query_start_loc[r]).item()
                if query_start_loc is not None
                else 0
            )
            delta = ql - num_rejected[r].item()
            if delta != 0:
                num_computed_tokens[req_idx] = (
                    num_computed_tokens[req_idx] + delta
                )

    return post_update_torch


def _make_prepare_prefill_inputs_torch():
    """Pure-torch replacement for prepare_prefill_inputs.

    _prepare_prefill_inputs_kernel JIT-compiles during the first concurrent
    batch. Its early-return branch is safe from a lowering standpoint but
    FlagTree produces wrong ``input_ids`` for requests that are not at their
    prefill boundary, causing decode tokens to be fed back as prefill tokens.
    """
    import torch

    def prepare_prefill_inputs_torch(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    ):
        num_reqs = idx_mapping.shape[0]
        num_lookahead = next_prefill_tokens.shape[0]
        for r in range(num_reqs):
            req_idx = idx_mapping[r].item()
            pl = prefill_len[req_idx].item()
            nc = num_computed_tokens[req_idx].item()
            if nc >= pl:
                continue
            qs = query_start_loc[r].item()
            qe = query_start_loc[r + 1].item()
            ql = qe - qs
            if ql > 0:
                input_ids[qs:qe] = all_token_ids[req_idx, nc: nc + ql]
            if num_lookahead > 0:
                lookahead_start = nc + ql
                for li in range(num_lookahead):
                    pos = lookahead_start + li
                    tok = (
                        all_token_ids[req_idx, pos]
                        if pos < pl
                        else torch.zeros(1, dtype=next_prefill_tokens.dtype,
                                         device=next_prefill_tokens.device)[0]
                    )
                    next_prefill_tokens[li, req_idx] = tok

    return prepare_prefill_inputs_torch


def apply_input_batch_compat() -> None:
    """Swap in a single-exit token combiner that FlagTree can lower,
    and replace the input-batch launcher functions with pure-torch versions."""
    try:
        from vllm.v1.worker.gpu import input_batch
    except ImportError:
        # Builds without the V2 runner have nothing to patch.
        return

    # --- Triton kernel: _combine_sampled_and_draft_tokens_kernel ---
    current = getattr(input_batch, "_combine_sampled_and_draft_tokens_kernel", None)
    if current is not None and not getattr(current, _PATCH_FLAG, False):
        kernel = _build_kernel()
        setattr(kernel, _PATCH_FLAG, True)
        input_batch._combine_sampled_and_draft_tokens_kernel = kernel
        logger.info(
            "kunlunxin input_batch_compat: replaced "
            "_combine_sampled_and_draft_tokens_kernel with a single-exit form; "
            "FlagTree's XPU backend cannot lower an early return that follows "
            "tensor work"
        )

    # --- Python launcher functions: replace with pure-torch versions ---
    if getattr(input_batch, _PATCH_FLAG_LAUNCHERS, False):
        return

    patched = []

    for attr, factory, desc in (
        (
            "prepare_pos_seq_lens",
            _make_prepare_pos_seq_lens_torch,
            "prepare_pos_seq_lens → torch (FlagTree miscompiles "
            "_prepare_pos_seq_lens_kernel, producing wrong positions)",
        ),
        (
            "get_num_sampled_and_rejected",
            _make_get_num_sampled_and_rejected_torch,
            "get_num_sampled_and_rejected → torch (FlagTree miscompiles "
            "_get_num_sampled_and_rejected_kernel)",
        ),
        (
            "post_update",
            _make_post_update_torch,
            "post_update → torch (FlagTree miscompiles _post_update_kernel; "
            "corrupts all_token_ids from step 2 onward, producing '!' tokens)",
        ),
        (
            "prepare_prefill_inputs",
            _make_prepare_prefill_inputs_torch,
            "prepare_prefill_inputs → torch (FlagTree miscompiles "
            "_prepare_prefill_inputs_kernel)",
        ),
    ):
        if hasattr(input_batch, attr):
            fn = factory()
            setattr(fn, _PATCH_FLAG_LAUNCHERS, True)
            setattr(input_batch, attr, fn)
            patched.append(attr)
            logger.info("kunlunxin input_batch_compat: %s", desc)

    if patched:
        setattr(input_batch, _PATCH_FLAG_LAUNCHERS, True)
        logger.info(
            "kunlunxin input_batch_compat: patched %d launcher(s) with "
            "pure-torch implementations to prevent FlagTree silent "
            "miscompilation during decode",
            len(patched),
        )

    # model_runner.py imports these names directly at the top of the module
    # (lines 118-123: combine_sampled_and_draft_tokens, expand_idx_mapping,
    # post_update, post_update_num_computed_tokens, prepare_pos_seq_lens,
    # prepare_prefill_inputs).  Patching input_batch alone does not reach those
    # already-bound module-level names; we must also overwrite them in
    # model_runner so the functions called at lines 1450, 1461, 1692, 2331 use
    # the pure-torch versions.
    try:
        from vllm.v1.worker.gpu import model_runner as _mr
        mr_patched = []
        for attr in patched:
            fn = getattr(input_batch, attr, None)
            if fn is not None and hasattr(_mr, attr):
                setattr(_mr, attr, fn)
                mr_patched.append(attr)
        if mr_patched:
            logger.info(
                "kunlunxin input_batch_compat: propagated %d torch launcher(s) "
                "into model_runner module namespace: %s",
                len(mr_patched),
                ", ".join(mr_patched),
            )
    except ImportError:
        pass

    # sampler.py and rejection_sampler.py also import get_num_sampled_and_rejected
    # directly by name (sampler.py:20, rejection_sampler.py:17), so the
    # already-bound module-level name there still calls the Triton kernel after
    # input_batch is patched.  Propagate into those modules too.
    gnsr_fn = getattr(input_batch, "get_num_sampled_and_rejected", None)
    if gnsr_fn is not None:
        for mod_path in (
            "vllm.v1.worker.gpu.sample.sampler",
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
        ):
            try:
                import importlib
                _mod = importlib.import_module(mod_path)
                if hasattr(_mod, "get_num_sampled_and_rejected"):
                    setattr(_mod, "get_num_sampled_and_rejected", gnsr_fn)
                    logger.info(
                        "kunlunxin input_batch_compat: propagated "
                        "get_num_sampled_and_rejected → torch into %s",
                        mod_path,
                    )
            except ImportError:
                pass
