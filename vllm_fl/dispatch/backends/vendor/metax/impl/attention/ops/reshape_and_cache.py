# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
# Launcher adapted from vLLM v0.32.0 (5df63a7203cd); use the upstream kernel.
"""Stride-aware cache insertion with a C550-compatible launch configuration."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    _is_supported_kv_cache_dtype,
    reshape_and_cache_kernel_flash,
)


def triton_reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    # [num_blocks, block_size, num_heads, head_size]
    key_cache: torch.Tensor,
    # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
):
    if slot_mapping.numel() == 0:
        return

    num_heads = key.shape[1]
    head_size = key.shape[2]

    use_head_major_layout = key_cache.ndim == 5
    if use_head_major_layout:
        block_size = key_cache.shape[3]
        x = key_cache.shape[4]
        head_stride = key_cache.stride(1)
        dim_stride_k = key_cache.stride(2)
        dim_stride_v = value_cache.stride(2)
    else:
        block_size = key_cache.shape[1]
        x = 1
        dim_stride_k = 0
        dim_stride_v = 0
        head_stride = key_cache.stride()[2]
    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    assert _is_supported_kv_cache_dtype(kv_cache_dtype), (
        f"Triton reshape-and-cache cannot store kv_cache_dtype={kv_cache_dtype} "
        f"on this device: an FP8 KV cache needs native fp8e4nv (SM89+). Use "
        f"--kv-cache-dtype bfloat16 (or float16 on SM75)."
    )
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else key_cache.dtype
    )

    if key_cache.dtype != kv_cache_torch_dtype and is_quantized_kv_cache(
        kv_cache_dtype
    ):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        key_cache = key_cache.view(kv_cache_torch_dtype)
        value_cache = value_cache.view(kv_cache_torch_dtype)
    FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)
    # heuristics instead of autotuning
    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    # MACA C550 uses 64-thread warps and supports at most 512 threads per
    # block. The upstream CUDA launch (16 warps, 10 stages) exceeds that
    # limit. Four warps and one stage also keep register/shared memory small.
    TILE_SIZE = min(512, TILE_SIZE)
    num_warps = 4
    num_stages = 1

    # TODO(ngl): maybe replace with static launch grid to avoid overhead if
    #   using cudagraphs
    grid = lambda meta: (
        slot_mapping.shape[0],
        triton.cdiv(n, meta["TILE_SIZE"]),
    )

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        # strides
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        dim_stride_k=dim_stride_k,
        dim_stride_v=dim_stride_v,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        x=x,
        USE_HEAD_MAJOR_LAYOUT=use_head_major_layout,
        FP8_KV_CACHE=FP8_KV_CACHE,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
