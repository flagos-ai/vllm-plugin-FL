# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Replace postprocess_mamba_fused_kernel with a FlagTree-compilable version.

FlagTree's TritonXPU backend aborts with
    "operation with block successors must terminate its parent block"
on any top-level ``return`` inside a Triton kernel that is not in tail
position.  ``postprocess_mamba_fused_kernel`` has four such returns:

  1. ``if batch_idx >= num_reqs: return``
  2. ``if req_idx < 0: return``   (HAS_IDX_MAPPING path)
  3. ``if not needs_copy: return``
  4. ``if src_block_idx == dest_block_idx and accept_token_bias == 0: return``

This module rewrites the kernel using nested ``if`` guards in place of the
early exits; the semantics are identical.  The ``_copy_mamba_state_block``
device function is imported from upstream and called unchanged — its own
internal ``return`` statements live inside a device function, not a kernel,
and do not trigger the same compiler path.

The ``is_conv_state_dim_first`` helper and the ``_TEMPORAL_TILES`` constant
are also read from the upstream module so this shim stays in sync when the
upstream model moves them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_postprocess_mamba_no_early_return"


def _build_kernel():
    """Build the replacement kernel after vllm.v1.worker.mamba_utils is live."""
    from vllm.triton_utils import tl, triton
    import vllm.v1.worker.mamba_utils as _m

    _copy_mamba_state_block = _m._copy_mamba_state_block

    @triton.jit(do_not_specialize=["num_reqs"])
    def _postprocess_mamba_fused_kernel_kl(
        num_accepted_tokens_ptr,
        mamba_state_idx_ptr,
        num_scheduled_tokens_ptr,
        num_computed_tokens_ptr,
        num_draft_tokens_ptr,
        block_table_ptrs_ptr,
        block_table_stride_req: tl.int64,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        state_dim_row_count_ptr,
        state_dim_row_stride_ptr,
        num_accepted_tokens_out_ptr,
        idx_mapping_ptr,
        num_reqs,
        block_size: tl.constexpr,
        COPY_BLOCK_SIZE: tl.constexpr,
        CONV_STATE_DIM_FIRST: tl.constexpr,
        HAS_IDX_MAPPING: tl.constexpr = False,
        PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,
        TEMPORAL_TILES: tl.constexpr = 1,
    ):
        """postprocess_mamba_fused_kernel rewritten without early returns.

        Semantically identical to the upstream kernel; the four early ``return``
        statements are replaced with nested ``if`` guards so FlagTree's
        TritonXPU backend can lower the kernel without hitting the
        "block successors must terminate parent block" assertion.
        """
        batch_idx = tl.program_id(0)
        state_idx = tl.program_id(1)
        tile_idx = tl.program_id(2)

        # Guard 1: out-of-bounds batch (replaces `if batch_idx >= num_reqs: return`)
        if batch_idx < num_reqs:

            # Guard 2: invalid idx mapping (replaces `if req_idx < 0: return`)
            if HAS_IDX_MAPPING:
                req_idx = tl.load(idx_mapping_ptr + batch_idx)
                valid_req = req_idx >= 0
            else:
                req_idx = batch_idx
                valid_req = True

            if valid_req:
                num_accepted = tl.load(num_accepted_tokens_ptr + req_idx)
                src_block_idx = tl.load(mamba_state_idx_ptr + req_idx)

                if PRECOMPUTED_NEW_COMPUTED:
                    new_num_computed = tl.load(num_computed_tokens_ptr + req_idx)
                    num_tokens_running_state = new_num_computed - num_accepted + 1
                else:
                    num_scheduled = tl.load(num_scheduled_tokens_ptr + req_idx)
                    num_computed = tl.load(num_computed_tokens_ptr + req_idx)
                    num_draft = tl.load(num_draft_tokens_ptr + req_idx)
                    num_tokens_running_state = (
                        num_computed + num_scheduled - num_draft
                    )
                    new_num_computed = num_tokens_running_state + num_accepted - 1

                aligned_new_computed = (new_num_computed // block_size) * block_size
                needs_copy = aligned_new_computed >= num_tokens_running_state

                # Guard 3: nothing to copy (replaces `if not needs_copy: return`)
                if needs_copy:
                    accept_token_bias = aligned_new_computed - num_tokens_running_state
                    dest_block_idx = aligned_new_computed // block_size - 1

                    # Conditional write for the self-copy case (unchanged logic).
                    is_self_copy = src_block_idx == dest_block_idx
                    do_count_write = (
                        is_self_copy & (state_idx == 0) & (tile_idx == 0)
                    )
                    if do_count_write:
                        tl.store(num_accepted_tokens_out_ptr + req_idx, 1)

                    # Guard 4: skip no-op self-copy
                    # (replaces `if src == dst and bias == 0: return`)
                    is_noop = is_self_copy & (accept_token_bias == 0)
                    if not is_noop:
                        _copy_mamba_state_block(
                            state_idx,
                            req_idx,
                            src_block_idx,
                            dest_block_idx,
                            accept_token_bias,
                            block_table_ptrs_ptr,
                            block_table_stride_req,
                            state_base_addrs_ptr,
                            state_block_strides_ptr,
                            state_elem_sizes_ptr,
                            state_inner_sizes_ptr,
                            state_conv_widths_ptr,
                            state_group_indices_ptr,
                            state_dim_row_count_ptr,
                            state_dim_row_stride_ptr,
                            tile_idx,
                            COPY_BLOCK_SIZE,
                            CONV_STATE_DIM_FIRST,
                            TEMPORAL_TILES,
                        )

    return _postprocess_mamba_fused_kernel_kl


def apply_postprocess_mamba_compat() -> None:
    """Replace postprocess_mamba_fused_kernel with a FlagTree-safe version.

    Idempotent.  Silently skips if the upstream module cannot be imported
    (e.g. a build that does not ship the V2 worker).
    """
    try:
        import vllm.v1.worker.mamba_utils as mod
    except ImportError:
        logger.debug(
            "vllm.v1.worker.mamba_utils not importable; "
            "postprocess_mamba_compat is a no-op"
        )
        return

    current = getattr(mod, "postprocess_mamba_fused_kernel", None)
    if current is None:
        return
    if getattr(current, _PATCH_FLAG, False):
        return

    kernel = _build_kernel()
    setattr(kernel, _PATCH_FLAG, True)
    mod.postprocess_mamba_fused_kernel = kernel
    logger.info(
        "Replaced postprocess_mamba_fused_kernel with the "
        "FlagTree-safe (no early-return) version"
    )
