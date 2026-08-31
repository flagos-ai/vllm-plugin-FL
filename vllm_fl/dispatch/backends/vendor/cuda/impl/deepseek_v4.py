# Copyright (c) 2026 BAAI. All rights reserved.

"""CUDA vendor implementation of DeepSeek-V4-specific operators."""

from __future__ import annotations

import torch

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    _fused_inv_rope_fp8_quant_per_head,
)

from vllm_fl.ops.deepseek_v4_int8_woa import (
    fused_inv_rope_quant_int8_triton,
)


def deepseek_v4_inv_rope_quant_int8_cuda(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Triton implementation on NVIDIA CUDA devices."""
    return fused_inv_rope_quant_int8_triton(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
    )


def deepseek_v4_inv_rope_quant_fp8_cuda(
    o,
    positions,
    cos_sin_cache,
    heads_per_group,
    quant_group_size,
    chunks_per_head,
    rope_start,
    half_rope,
    tma_aligned_scales,
    fp8_max,
    tma_aligned_T,
    num_tokens,
    n_groups,
    d,
    scale_inner,
):
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d), dtype=torch.float8_e4m3fn, device=o.device
    )
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_T,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_T, 1, tma_aligned_T),
    )
    grid = (tma_aligned_T, n_groups * heads_per_group)
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        scale_stride_k=scale_buf.stride(2),
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=rope_start,
        HALF_ROPE=half_rope,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        USE_GDC=False,
        launch_pdl=False,
        num_stages=1,
        num_warps=1,
    )
    return fp8_buf, scale_buf


def deepseek_v4_int8_scaled_mm_cuda(x, weight, scale_a, scale_b, out_dtype, bias=None):
    from vllm import _custom_ops as ops

    return ops.cutlass_scaled_mm(
        x,
        weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=out_dtype,
    )


def deepseek_v4_mhc_pre_cuda(*args):
    return torch.ops.vllm.mhc_pre_tilelang(*args)


def deepseek_v4_mhc_fused_post_pre_cuda(*args):
    return torch.ops.vllm.mhc_fused_post_pre_tilelang(*args)


def deepseek_v4_mhc_post_cuda(*args):
    return torch.ops.vllm.mhc_post_tilelang(*args)


def deepseek_v4_hc_head_cuda(*args):
    return torch.ops.vllm.hc_head_fused_kernel_tilelang(*args)
