# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: FlagTree's XPU Triton backend cannot lower a 3-D broadcast, i.e. an
# access whose operands each carry two broadcast axes.  The V2 model runner's
# ``vllm.v1.worker.mamba_utils.get_aligned_state_indices_multi_group_kernel``
# gathers every Mamba group's state indices in one launch by building a
# [group, row, slot] pointer::
#
#     tl.load(
#         block_tables[:, None, None]
#         + rows[None, :, None] * block_table_stride_req
#         + first_state_slot[None, :, None]
#         + state_slots[None, None, :],
#         mask=(valid_group[:, None, None]
#               & valid_row[None, :, None]
#               & valid_state_slot[None, None, :]),
#     )
#
# so every V2 run of a Mamba-hybrid model aborts during kernel warmup with::
#
#     .../vllm/v1/worker/mamba_utils.py:39:0: error: Failures have been
#     detected while processing an MLIR pass pipeline
#     note: Pipeline failed while executing [`TritonXPUOffsetAnalysis` on
#     'builtin.module' operation]
#     [broadcastOpCalFunc] Unsupported broadcast 2 dims
#     UNREACHABLE executed at .../Dialect/TritonXPU/Transforms/OffsetAnalysis.cpp:623
#     triton.runtime.errors.OutOfResources: out of resource: uni_sram
#
# The backend says so in as many words -- "Unsupported broadcast 2 dims" -- and
# then hits an UNREACHABLE in its offset analysis.  The OutOfResources wrapper
# is again misleading: it reports "Required: 0, Hardware limit: 0", so nothing
# is exhausted and shrinking the block sizes does not help.
#
# Measured on P800 (FlagTree triton, /opt/flagtree/triton/backends/xpu),
# holding everything else fixed and varying only the rank of the access:
#
#     access shape                              result
#     1-D  (32,)                                 OK
#     2-D  (32, 4)       one broadcast axis       OK
#     3-D  (4, 32, 4)    two broadcast axes       FAIL
#     3-D with a pointer-tensor base              FAIL
#
# The group axis is a compile-time constant (``NUM_GROUPS``), so unroll it with
# tl.static_range and keep every access 2-D: one [row, slot] gather and scatter
# per group.  Each iteration loads that group's block-table base as a scalar,
# which also drops the ``BLOCK_GROUPS`` padding and its masking.  Total work is
# unchanged -- the same NUM_GROUPS x BLOCK_ROWS x BLOCK_STATE_SLOTS elements
# move, just as NUM_GROUPS separate 2-D accesses instead of one 3-D access.
#
# Verified against a torch reference across 48 configurations (1/3/4 groups x
# 1/5/32/40 requests x 1/3 state slots x cache block sizes 128/896, covering
# both a partial and a multi-program row grid): the aligned state indices match
# exactly.
# --------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_mamba_state_indices_2d"


def _build_kernel():
    """Build the 2-D-per-group variant of the aligned-state-indices kernel."""
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["num_requests"])
    def get_aligned_state_indices_multi_group_kernel(
        block_table_ptrs_ptr,
        seq_lens_ptr,
        state_indices_ptr,
        block_table_stride_req: tl.int64,
        seq_lens_stride: tl.constexpr,
        state_indices_stride_0: tl.constexpr,
        state_indices_stride_1: tl.constexpr,
        state_indices_stride_2: tl.constexpr,
        num_requests,
        CACHE_BLOCK_SIZE: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
        NUM_STATE_SLOTS: tl.constexpr,
        BLOCK_STATE_SLOTS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid_row = rows < num_requests

        seq_lens = tl.load(
            seq_lens_ptr + rows * seq_lens_stride,
            mask=valid_row,
            other=1,
        )
        first_state_slot = tl.maximum((seq_lens - 1) // CACHE_BLOCK_SIZE, 0)

        state_slots = tl.arange(0, BLOCK_STATE_SLOTS)
        valid_state_slot = state_slots < NUM_STATE_SLOTS
        mask = valid_row[:, None] & valid_state_slot[None, :]

        # Upstream gathers all groups at once through a [group, row, slot]
        # pointer. FlagTree cannot lower that 3-D broadcast, so walk the groups
        # one at a time and keep each access 2-D. NUM_GROUPS is a constexpr, so
        # this unrolls at compile time and BLOCK_GROUPS padding is unnecessary.
        for group in tl.static_range(NUM_GROUPS):
            group_base_addr = tl.load(block_table_ptrs_ptr + group)
            block_table = group_base_addr.to(tl.pointer_type(tl.int32))
            state_indices = tl.load(
                block_table
                + rows[:, None] * block_table_stride_req
                + first_state_slot[:, None]
                + state_slots[None, :],
                mask=mask,
            )
            tl.store(
                state_indices_ptr
                + group * state_indices_stride_0
                + rows[:, None] * state_indices_stride_1
                + state_slots[None, :] * state_indices_stride_2,
                state_indices,
                mask=mask,
            )

    return get_aligned_state_indices_multi_group_kernel


def apply_mamba_utils_compat() -> None:
    """Swap in a 2-D-per-group state-index kernel that FlagTree can lower."""
    try:
        import importlib

        mamba_utils = importlib.import_module("vllm.v1.worker.mamba_utils")
    except ImportError:
        # Builds without the V2 Mamba path have nothing to patch.
        return

    current = getattr(
        mamba_utils, "get_aligned_state_indices_multi_group_kernel", None
    )
    if current is None or getattr(current, _PATCH_FLAG, False):
        return

    kernel = _build_kernel()
    setattr(kernel, _PATCH_FLAG, True)
    mamba_utils.get_aligned_state_indices_multi_group_kernel = kernel
    logger.info(
        "kunlunxin mamba_utils_compat: replaced "
        "get_aligned_state_indices_multi_group_kernel with a 2-D-per-group "
        "form; FlagTree's XPU backend cannot lower a 3-D broadcast "
        "(Unsupported broadcast 2 dims)"
    )
