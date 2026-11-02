# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Ascend v0.24.0rc1.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bind vLLM Mamba state copy to Ascend-compatible buffers and kernel."""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker import mamba_utils


@triton.jit
def _batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    src = tl.load(src_ptrs + pid).to(tl.pointer_type(tl.uint8))
    dst = tl.load(dst_ptrs + pid).to(tl.pointer_type(tl.uint8))
    size = tl.load(sizes + pid)
    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = i + offsets < size
        data = tl.load(src + i + offsets, mask=mask, cache_modifier=".cg")
        tl.store(dst + i + offsets, data, mask=mask, cache_modifier=".cg")


def _batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch and sizes.shape[0] == batch
    _batch_memcpy_kernel[(batch,)](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=8192)


def apply_mamba_copy_patch() -> None:
    """Install once in an Ascend worker before MambaBuffers are created."""
    create = mamba_utils.MambaCopyBuffers.create
    if getattr(create.__func__, "_fl_ascend_copy_patch", False):
        return

    @classmethod
    def create_with_int64(cls, max_num_reqs, kv_cache_config, copy_funcs, make_buffer):
        return create(
            max_num_reqs, kv_cache_config, copy_funcs,
            lambda n, dtype: make_buffer(n, dtype=torch.int64 if dtype == torch.uint64 else dtype),
        )

    create_with_int64.__func__._fl_ascend_copy_patch = True
    mamba_utils.MambaCopyBuffers.create = create_with_int64
    mamba_utils.batch_memcpy_kernel = _batch_memcpy_kernel
    mamba_utils.batch_memcpy = _batch_memcpy
