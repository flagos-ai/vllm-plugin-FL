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


def apply_input_batch_compat() -> None:
    """Swap in a single-exit token combiner that FlagTree can lower."""
    try:
        from vllm.v1.worker.gpu import input_batch
    except ImportError:
        # Builds without the V2 runner have nothing to patch.
        return

    current = getattr(input_batch, "_combine_sampled_and_draft_tokens_kernel", None)
    if current is None or getattr(current, _PATCH_FLAG, False):
        return

    kernel = _build_kernel()
    setattr(kernel, _PATCH_FLAG, True)
    input_batch._combine_sampled_and_draft_tokens_kernel = kernel
    logger.info(
        "kunlunxin input_batch_compat: replaced "
        "_combine_sampled_and_draft_tokens_kernel with a single-exit form; "
        "FlagTree's XPU backend cannot lower an early return that follows "
        "tensor work"
    )
