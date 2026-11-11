# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: FlagTree's XPU Triton backend cannot lower an early ``return`` that
# sits after the kernel has already emitted tensor work.  The V2 model runner's
# ``vllm.v1.worker.gpu.block_table._compute_slot_mappings_kernel`` dedicates its
# last program to padding the tail of the slot mapping and then returns::
#
#     if batch_idx == tl.num_programs(1) - 1:
#         actual_num_tokens = tl.load(query_start_loc + batch_idx)
#         for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
#             offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
#             tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
#         return
#     ...                      # the slot-mapping body follows
#
# so every V2 run aborts while preparing the first batch with::
#
#     .../vllm/v1/worker/gpu/block_table.py:352:69: error: 'arith.select' op
#     failed to verify that all of {true_value, false_value, result} have same type
#     .../vllm/v1/worker/gpu/block_table.py:273:0: error: Failures have been
#     detected while processing an MLIR pass pipeline
#     triton.runtime.errors.OutOfResources: out of resource: uni_sram
#
# The reported location is misleading twice over.  The ``OutOfResources``
# wrapper reports "Required: 0, Hardware limit: 0", so this is a lowering
# failure rather than an SRAM budget problem, and the ``arith.select``
# complaint points at the ``tl.where`` on line 352 even though that expression
# is not the cause: the same kernel with the padding branch removed, and the
# same ``tl.where`` in place, compiles cleanly.  Restoring just the early
# return reproduces the failure, which is the same defect input_batch_compat
# works around in _combine_sampled_and_draft_tokens_kernel.
#
# Measured on P800 (FlagTree triton, /opt/flagtree/triton/backends/xpu), with
# the real argument dtypes (bool slot_mapping_enabled and dcp_sharded, int64
# block_table_strides, int32 block table and block_sizes, int64 positions and
# slot_mappings, int64 idx_mapping) and TRITON_BLOCK_SIZE=1024:
#
#     kernel form                                   result
#     upstream (padding branch ends in return)       FAIL
#     padding branch hoisted into if/else            OK
#     padding branch removed entirely                OK
#
# Rewrite the padding program as the ``if`` arm of an if/else whose ``else``
# arm holds the slot-mapping body.  The kernel keeps exactly one exit, no
# tensor work precedes a conditional return, and the division of labour between
# programs is unchanged: the last program still pads only, and every other
# program still computes slot mappings only.
# --------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_block_table_no_early_return"


def _build_kernel():
    """Build the single-exit variant of the slot-mapping kernel."""
    from vllm.triton_utils import tl, triton
    from vllm.v1.worker.gpu.buffer_utils import _load_ptr

    @triton.jit
    def _compute_slot_mappings_kernel(
        max_num_tokens,
        idx_mapping,  # [num_reqs]
        query_start_loc,  # [num_reqs + 1]
        pos,  # [num_tokens]
        block_table_ptrs,  # [num_kv_cache_groups]
        block_table_strides,  # [num_kv_cache_groups]
        block_sizes,  # [num_kv_cache_groups]
        slot_mapping_enabled,  # [num_kv_cache_groups]
        dcp_sharded,  # [num_kv_cache_groups]
        slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
        slot_mappings_stride,
        cp_rank,
        CP_SIZE: tl.constexpr,
        CP_INTERLEAVE: tl.constexpr,
        PAD_ID: tl.constexpr,
        TRITON_BLOCK_SIZE: tl.constexpr,
    ):
        # kv cache group id
        group_id = tl.program_id(0)
        batch_idx = tl.program_id(1)
        slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

        # Upstream handles the padding program first and returns. FlagTree
        # cannot lower that return, so the two roles become the arms of one
        # if/else instead.
        is_padding_program = batch_idx == tl.num_programs(1) - 1
        if is_padding_program:
            # Pad remaining slots to -1. This is needed for CUDA graphs.
            # Start from actual token count (not padded) to cover the gap
            # between actual tokens and padded tokens that can contain stale
            # valid slot IDs from previous chunks during chunked prefill.
            actual_num_tokens = tl.load(query_start_loc + batch_idx)
            for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
                offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                tl.store(
                    slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens
                )
        else:
            block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
            block_table_stride = tl.load(block_table_strides + group_id)
            kv_block_size = tl.load(block_sizes + group_id)
            mapping_enabled = tl.load(slot_mapping_enabled + group_id)
            if CP_SIZE != 1:
                sharded = tl.load(dcp_sharded + group_id)

            req_state_idx = tl.load(idx_mapping + batch_idx)
            # idx_mapping == -1 marks a dummy (or CUDA-graph padding) request
            # that owns no blocks: never read its block-table row and emit PAD
            # for its tokens.
            is_real_req = req_state_idx >= 0
            start_idx = tl.load(query_start_loc + batch_idx)
            end_idx = tl.load(query_start_loc + batch_idx + 1)
            for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
                offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

                if CP_SIZE == 1:
                    # Common case: Context parallelism is not used.
                    local_positions = positions
                    is_local = True
                else:
                    # Context parallelism is used.
                    virtual_block_size = kv_block_size * CP_SIZE
                    virtual_block_indices = positions // virtual_block_size
                    virtual_block_offsets = positions % virtual_block_size
                    is_local = (
                        virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
                    )
                    rounds = virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE)
                    remainder = virtual_block_offsets % CP_INTERLEAVE
                    local_offsets = rounds * CP_INTERLEAVE + remainder
                    local_positions = (
                        virtual_block_indices * kv_block_size + local_offsets
                    )
                    local_positions = tl.where(sharded, local_positions, positions)
                    is_local = ~sharded | is_local

                block_indices = tl.where(
                    mapping_enabled, local_positions // kv_block_size, 0
                )
                block_offsets = local_positions % kv_block_size
                block_numbers = tl.load(
                    block_table_ptr + req_state_idx * block_table_stride
                    + block_indices,
                    mask=is_local & is_real_req,
                    other=0,
                )
                slot_ids = block_numbers * kv_block_size + block_offsets
                if CP_SIZE != 1:
                    slot_ids = tl.where(is_local, slot_ids, PAD_ID)

                slot_ids = tl.where(mapping_enabled & is_real_req, slot_ids, PAD_ID)
                tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)

    return _compute_slot_mappings_kernel


def apply_block_table_compat() -> None:
    """Swap in a single-exit slot-mapping kernel that FlagTree can lower."""
    try:
        import importlib

        block_table = importlib.import_module("vllm.v1.worker.gpu.block_table")
    except ImportError:
        # Builds without the V2 runner have nothing to patch.
        return

    current = getattr(block_table, "_compute_slot_mappings_kernel", None)
    if current is None or getattr(current, _PATCH_FLAG, False):
        return

    kernel = _build_kernel()
    setattr(kernel, _PATCH_FLAG, True)
    block_table._compute_slot_mappings_kernel = kernel
    logger.info(
        "kunlunxin block_table_compat: replaced _compute_slot_mappings_kernel "
        "with a single-exit form; FlagTree's XPU backend cannot lower an early "
        "return that follows tensor work"
    )
