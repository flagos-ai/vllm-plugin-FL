# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from the vllm-ascend project (PR #12096, slot mapping Triton kernel
# optimization for Ascend NPU), ported to the vLLM 0.13.0 block table layout.

"""Ascend-optimized slot mapping computation for the vLLM v1 block table.

The upstream vLLM 0.13.0 ``vllm.v1.worker.block_table.BlockTable`` computes
slot mapping with NumPy on the CPU and then copies the result to the device
(``commit_slot_mapping``). This module backports the Ascend-local Triton
kernel from vllm-ascend PR #12096 so the computation runs directly on the
NPU and the H2D commit of slot mapping is skipped.

Key differences from the original PR (adaptation decisions):

1. vLLM 0.13.0 ``BlockTable.compute_slot_mapping`` takes NumPy
   ``(req_indices, positions)`` instead of on-device
   ``(num_reqs, query_start_loc, positions)``. ``compute_slot_mapping_npu``
   below rebuilds ``query_start_loc`` from ``num_scheduled_tokens`` and
   uploads it together with ``positions`` (small H2D copies) before launching
   the kernel.
2. vLLM 0.13.0 allocates ``slot_mapping`` as ``torch.int64`` (the PR targets
   an int32 buffer). The kernel therefore casts ``slot_ids`` to the pointer's
   element type at store time instead of assuming int32.
3. In vLLM 0.13.0 the block table already stores kernel-block ids (physical
   blocks are expanded by ``map_to_kernel_blocks`` when hybrid blocks are
   used), so ``KV_CACHE_BLOCK_SIZE`` is passed as
   ``block_size * blocks_per_kv_block`` and the runtime ``block_size``
   argument is the kernel block size, keeping the kernel's index math
   identical to the upstream NumPy path.
4. PCP/DCP (context parallelism) is not covered by the PR kernel launch path;
   when ``pcp_world_size * dcp_world_size > 1`` we fall back to the upstream
   NumPy implementation.
"""

import numpy as np
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

# TILE size of the token loop inside the kernel; also the granularity of the
# padding programs. Same value as the PR (and upstream vLLM).
_BLOCK_SIZE = 1024


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


# NOTE(FL): keep num_reqs out of specialization too — otherwise the first
# high-concurrency step after single-request warmup triggers a one-time
# Triton JIT recompile that shows up as a huge TTFT spike on that case.
@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens", "num_reqs"])
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    num_reqs,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32
    block_table_stride,
    block_size,
    slot_mapping_ptr,  # [max_num_tokens]
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx >= num_reqs:
        # Pad remaining slots for CUDA graph compatibility. Use one program per
        # BLOCK_SIZE tile instead of making a single program sweep the tail.
        pad_block_idx = req_idx - num_reqs
        offsets = num_tokens + pad_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(slot_mapping_ptr + offsets, PAD_ID, mask=offsets < max_num_tokens)
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1)
    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE

    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        positions = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)

        virtual_block_indices = positions // virtual_block_size
        virtual_block_offsets = positions - virtual_block_size * virtual_block_indices

        if TOTAL_CP_WORLD_SIZE == 1:
            is_local = mask
            local_block_offsets = virtual_block_offsets
        else:
            interleave_chunks = virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
            rank_in_chunk = interleave_chunks - TOTAL_CP_WORLD_SIZE * (interleave_chunks // TOTAL_CP_WORLD_SIZE)
            is_local = rank_in_chunk == TOTAL_CP_RANK
            rounds = virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            remainder_base = virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
            remainder = virtual_block_offsets - CP_KV_CACHE_INTERLEAVE_SIZE * remainder_base
            local_block_offsets = rounds * CP_KV_CACHE_INTERLEAVE_SIZE + remainder

        local_block_indices = local_block_offsets // block_size
        block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_indices

        # Non-contiguous block_table loads degrade to scalar on Ascend. Positions
        # are grouped by request, so a token tile only spans a small block window.
        valid_block_indices = tl.where(mask, block_indices, 2147483647)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        relative_block_indices = tl.where(mask & is_local, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)

        slot_offsets = local_block_offsets - block_size * local_block_indices
        slot_ids = block_numbers * block_size + slot_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        # NOTE(FL): vLLM 0.13.0 allocates slot_mapping as int64 (the PR targets
        # an int32 buffer); cast to the pointer element type so both work.
        tl.store(slot_mapping_ptr + offsets,
                 slot_ids.to(slot_mapping_ptr.dtype.element_ty),
                 mask=mask)


def _launch_kernel(block_table,
                   num_reqs: int,
                   num_tokens: int,
                   query_start_loc: torch.Tensor,
                   positions: torch.Tensor) -> None:
    """Launch the Triton kernel for a single vLLM 0.13.0 ``BlockTable``."""
    # NOTE(FL): in vLLM 0.13.0 ``block_table.block_size`` is already the
    # kernel block size and the block table stores kernel-block ids, so the
    # physical block size is reconstructed as block_size * blocks_per_kv_block.
    blocks_per_kv_block = block_table.blocks_per_kv_block
    physical_block_size = block_table.block_size * blocks_per_kv_block
    num_pad_tokens = max(block_table.max_num_batched_tokens - num_tokens, 0)
    num_pad_blocks = cdiv(num_pad_tokens, _BLOCK_SIZE)
    _compute_slot_mapping_kernel[(num_reqs + num_pad_blocks, )](
        num_tokens,
        block_table.max_num_batched_tokens,
        num_reqs,
        query_start_loc,
        positions,
        block_table.block_table.gpu,
        block_table.block_table.gpu.stride(0),
        block_table.block_size,
        block_table.slot_mapping.gpu,
        KV_CACHE_BLOCK_SIZE=physical_block_size,
        BLOCKS_PER_KV_BLOCK=blocks_per_kv_block,
        # The CP>1 math in the kernel is kept for parity with the PR, but the
        # launcher below only routes CP==1 here (CP>1 falls back to NumPy).
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=block_table.cp_kv_cache_interleave_size,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=_BLOCK_SIZE,
        BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2(
            cdiv(_BLOCK_SIZE, block_table.block_size) + 1),
    )


def compute_slot_mapping_npu(multi_group_block_table,
                             num_reqs: int,
                             num_scheduled_tokens: np.ndarray,
                             positions_np: np.ndarray) -> bool:
    """Compute slot mapping on the NPU for a ``MultiGroupBlockTable``.

    Returns True when the slot mapping was computed on the device (the caller
    must then skip ``commit_slot_mapping``), or False when the caller should
    use the upstream NumPy path (non-NPU platform, Triton unavailable, or
    context parallelism enabled).
    """
    if current_platform.device_type != "npu" or not HAS_TRITON:
        return False
    first = multi_group_block_table.block_tables[0]
    if first.pcp_world_size * first.dcp_world_size > 1:
        # The PR kernel launch path only covers CP==1; keep the upstream
        # NumPy implementation for PCP/DCP.
        return False

    num_tokens = positions_np.shape[0]
    # vLLM 0.13.0 hands us NumPy arrays; rebuild query_start_loc and upload
    # it together with positions (adaptation decision 1 in the module doc).
    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
    device = first.device
    query_start_loc = torch.from_numpy(query_start_loc_np).to(
        device, non_blocking=True)
    # positions_np is a view of the runner's pinned buffer, so this H2D copy
    # is asynchronous. The kernel consumes positions as int64, matching the
    # buffer dtype.
    positions = torch.from_numpy(positions_np).to(device, non_blocking=True)

    for block_table in multi_group_block_table.block_tables:
        _launch_kernel(block_table, num_reqs, num_tokens, query_start_loc,
                       positions)
    return True
