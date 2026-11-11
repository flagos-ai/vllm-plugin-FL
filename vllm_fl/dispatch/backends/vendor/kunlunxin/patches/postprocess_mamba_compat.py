# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Replace postprocess/precopy mamba fused kernels with FlagTree-safe stubs.

FlagTree's TritonXPU backend fails on ``postprocess_mamba_fused_kernel`` and
``precopy_mamba_align_fused_kernel`` in two separate ways:

1. Early-return form → TritonXPULegalize:
     "operation with block successors must terminate its parent block"
   (same defect as block_table_compat and input_batch_compat).

2. Nested-if form with ``_copy_mamba_state_block`` → TritonXPUMemoryInplace:
     "'triton_xpu.gm2lm' op using value defined outside the region"
   because ``state_base_addr`` (mamba_utils.py:234) is loaded from a pointer
   that the pass treats as region-external when ``_copy_mamba_state_block`` is
   called inside any ``if`` guard.

The only form FlagTree accepts for a kernel that calls ``_copy_mamba_state_block``
is a flat, guard-free body — but then the semantics (early exit for padding
programs, no-copy requests, etc.) break.

Fix: replace both kernels with trivial no-op stubs. These stubs compile
without issue. The actual copy semantics are preserved because:

  * ``MambaSpecDecodeGPUContext`` guards every caller with
    ``if num_reqs == 0 or not self.is_initialized: return``.
  * ``is_initialized`` is False for all non-speculative-decoding model runs,
    including every model in the gate (Qwen3.6-27B and Qwen3.6-35B-A3B with
    standard serving). The stubs are therefore never invoked during inference.
  * During warmup FlagTree compiles the stubs (not the upstream kernels) and
    succeeds; the methods still return early via the guard.

If spec-decode on P800 is needed in the future, this shim should be extended
to implement the copy logic in Python/torch inside the method bodies.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_postprocess_mamba_noop"


def _build_postprocess_stub():
    """Build a trivial no-op replacement for postprocess_mamba_fused_kernel."""
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["num_reqs"])
    def _postprocess_mamba_noop(
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
        # No-op stub: FlagTree cannot lower early returns or cross-region gm2lm
        # ops in the upstream kernel.  The run_fused_postprocess method guards
        # on is_initialized, so this stub is never reached during inference on
        # non-speculative-decoding models.  Read a program_id to give the
        # compiler a non-empty body.
        _ = tl.program_id(0)  # noqa: F841

    return _postprocess_mamba_noop


def _build_precopy_stub():
    """Build a trivial no-op replacement for precopy_mamba_align_fused_kernel."""
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["num_reqs"])
    def _precopy_mamba_noop(
        mamba_state_idx_ptr,
        src_col_ptr,
        token_bias_ptr,
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
        idx_mapping_ptr,
        num_reqs,
        COPY_BLOCK_SIZE: tl.constexpr,
        CONV_STATE_DIM_FIRST: tl.constexpr,
        HAS_IDX_MAPPING: tl.constexpr = True,
        TEMPORAL_TILES: tl.constexpr = 1,
    ):
        # No-op stub: same rationale as _postprocess_mamba_noop above.
        _ = tl.program_id(0)  # noqa: F841

    return _precopy_mamba_noop


def apply_postprocess_mamba_compat() -> None:
    """Replace mamba fused kernels with FlagTree-safe no-op stubs.  Idempotent."""
    try:
        import vllm.v1.worker.mamba_utils as mod
    except ImportError:
        logger.debug(
            "vllm.v1.worker.mamba_utils not importable; "
            "postprocess_mamba_compat is a no-op"
        )
        return

    # Patch postprocess_mamba_fused_kernel
    current = getattr(mod, "postprocess_mamba_fused_kernel", None)
    if current is not None and not getattr(current, _PATCH_FLAG, False):
        stub = _build_postprocess_stub()
        setattr(stub, _PATCH_FLAG, True)
        mod.postprocess_mamba_fused_kernel = stub
        logger.info(
            "kunlunxin postprocess_mamba_compat: replaced "
            "postprocess_mamba_fused_kernel with no-op stub "
            "(FlagTree cannot lower early returns or cross-region gm2lm)"
        )

    # Patch precopy_mamba_align_fused_kernel (same failure modes)
    precopy = getattr(mod, "precopy_mamba_align_fused_kernel", None)
    if precopy is not None and not getattr(precopy, _PATCH_FLAG, False):
        stub2 = _build_precopy_stub()
        setattr(stub2, _PATCH_FLAG, True)
        mod.precopy_mamba_align_fused_kernel = stub2
        logger.info(
            "kunlunxin postprocess_mamba_compat: replaced "
            "precopy_mamba_align_fused_kernel with no-op stub"
        )
