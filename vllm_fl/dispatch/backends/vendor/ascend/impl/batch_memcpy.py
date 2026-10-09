# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vllm-ascend's vllm_ascend/ops/triton/batch_memcpy.py.

from vllm.triton_utils import tl, triton

BATCH_MEMCPY_BLOCK_SIZE = 8192


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    """Copy each byte range described by the pointer metadata tensors."""
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    # FlagTree's pointer analysis can fail when these casts live in the loop.
    src_ptr = src_ptr.to(tl.pointer_type(tl.uint8))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.uint8))

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size
        curr_src_ptr = src_ptr + i + offsets
        curr_dst_ptr = dst_ptr + i + offsets

        # Mamba states are streaming copies; bypass L1 to avoid cache pollution.
        data = tl.load(curr_src_ptr, mask=mask, cache_modifier=".cg")
        tl.store(curr_dst_ptr, data, mask=mask, cache_modifier=".cg")


def batch_memcpy(src_ptrs, dst_ptrs, sizes) -> None:
    """Launch the Ascend-safe Mamba state-copy kernel."""
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    batch_memcpy_kernel[(batch,)](
        src_ptrs,
        dst_ptrs,
        sizes,
        BLOCK_SIZE=BATCH_MEMCPY_BLOCK_SIZE,
    )
