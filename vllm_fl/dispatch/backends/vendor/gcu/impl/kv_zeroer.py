# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM v0.28.0, vllm/v1/worker/utils.py (2cf0a6915ce5).

"""Zero KV segments without exceeding Enflame's launch-grid limits."""

import logging

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d

logger = logging.getLogger(__name__)

GCU_MAX_GRID_X = 65535
_patched = False


@triton.jit(do_not_specialize=["n_jobs"])
def _zero_kv_blocks_kernel_gcu(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    n_jobs,
    BLOCK_SIZE: tl.constexpr,
    N_SEGS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
):
    # Flatten the upstream (block, segment, chunk) grid. A bounded number of
    # programs strides through all jobs, keeping grid.y and grid.z equal to 1.
    job_index = tl.program_id(0).to(tl.int64)
    while job_index < n_jobs:
        block_index = job_index // (N_SEGS * MAX_CHUNKS)
        seg_index = (job_index // MAX_CHUNKS) % N_SEGS
        chunk_index = job_index % MAX_CHUNKS
        block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
        page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
        chunk_offset = chunk_index * BLOCK_SIZE
        if chunk_offset < page_size_el:
            block_id = tl.load(block_ids_ptr + block_index)
            seg_addr = tl.load(seg_addrs_ptr + seg_index)
            ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
            block_offset = block_id.to(tl.int64) * block_stride_el.to(tl.int64)
            cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            tl.store(
                ptr + block_offset + cols,
                tl.zeros([BLOCK_SIZE], dtype=tl.int32),
                mask=cols < page_size_el,
            )
        job_index += tl.num_programs(0).to(tl.int64)


def zero_block_ids_gcu(self, block_ids: list[int]) -> None:
    """Use vLLM's original segment metadata and asynchronous block-ID copy."""
    if not block_ids or self._meta is None:
        return
    seg_addrs, seg_block_strides, seg_page_sizes, max_chunks, blk_size, n_segs = (
        self._meta
    )
    idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
    n_jobs = len(block_ids) * n_segs * max_chunks
    num_programs = min(n_jobs, GCU_MAX_GRID_X)
    _zero_kv_blocks_kernel_gcu[(num_programs,)](
        seg_addrs,
        seg_block_strides,
        seg_page_sizes,
        idx,
        n_jobs,
        BLOCK_SIZE=blk_size,
        N_SEGS=n_segs,
        MAX_CHUNKS=max_chunks,
    )


def apply_kv_zeroer_gcu_patch() -> None:
    """Install in each Enflame worker; leave other platforms unchanged."""
    global _patched
    if _patched:
        return
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) not in ("gcu", "enflame"):
        return
    from vllm.v1.worker import utils as worker_utils

    # Keep the official kernel and metadata builder intact. Existing instances
    # and warmup resolve this class method at call time.
    worker_utils.KVBlockZeroer.zero_block_ids = zero_block_ids_gcu
    _patched = True
    logger.info("Installed Enflame KV block zeroer with a bounded 1-D grid")
