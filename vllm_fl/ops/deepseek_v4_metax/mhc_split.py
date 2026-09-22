# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

import os
import importlib
import torch
import triton
import triton.language as tl


@triton.jit
def mhc_rms(
    residual_ptr,
    rms_ptr,
    num_tokens,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    sq = 0.0
    res_base = pid_n * res_stride_n
    for k in tl.static_range(4):
        head_base = res_base + k * res_stride_i
        for h_start in range(0, hidden_size, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < hidden_size
            v = tl.load(
                residual_ptr + head_base + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            sq += tl.sum(v * v)
    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)
    tl.store(rms_ptr + pid_n, rms_inv)


@triton.jit
def mhc_coeff(
    gemm_out_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    rms_ptr,
    pre_mix_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    num_tokens,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BT: tl.constexpr,
):
    pid_n = tl.program_id(0) * BT + tl.arange(0, BT)
    valid = pid_n < num_tokens
    rms_inv = tl.load(rms_ptr + pid_n, mask=valid, other=0.0)
    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)
    go_base = pid_n * HC_MULT3
    pre_mix_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 0, mask=valid, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 0)
        )
        + hc_pre_eps
    )
    pre_mix_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 1, mask=valid, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 1)
        )
        + hc_pre_eps
    )
    pre_mix_2 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 2, mask=valid, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 2)
        )
        + hc_pre_eps
    )
    pre_mix_3 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 3, mask=valid, other=0.0)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 3)
        )
        + hc_pre_eps
    )
    tl.store(pre_mix_ptr + pid_n * 4 + 0, pre_mix_0, mask=valid)
    tl.store(pre_mix_ptr + pid_n * 4 + 1, pre_mix_1, mask=valid)
    tl.store(pre_mix_ptr + pid_n * 4 + 2, pre_mix_2, mask=valid)
    tl.store(pre_mix_ptr + pid_n * 4 + 3, pre_mix_3, mask=valid)
    post_0 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 4, mask=valid, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 4)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 0, post_0, mask=valid)
    post_1 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 5, mask=valid, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 5)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 1, post_1, mask=valid)
    post_2 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 6, mask=valid, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 6)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 2, post_2, mask=valid)
    post_3 = (
        tl.sigmoid(
            tl.load(gemm_out_ptr + go_base + 7, mask=valid, other=0.0)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 7)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 3, post_3, mask=valid)
    cb = 8
    cm_00 = tl.load(
        gemm_out_ptr + go_base + cb + 0, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 0)
    cm_01 = tl.load(
        gemm_out_ptr + go_base + cb + 1, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 1)
    cm_02 = tl.load(
        gemm_out_ptr + go_base + cb + 2, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 2)
    cm_03 = tl.load(
        gemm_out_ptr + go_base + cb + 3, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 3)
    cm_10 = tl.load(
        gemm_out_ptr + go_base + cb + 4, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 4)
    cm_11 = tl.load(
        gemm_out_ptr + go_base + cb + 5, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 5)
    cm_12 = tl.load(
        gemm_out_ptr + go_base + cb + 6, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 6)
    cm_13 = tl.load(
        gemm_out_ptr + go_base + cb + 7, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 7)
    cm_20 = tl.load(
        gemm_out_ptr + go_base + cb + 8, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 8)
    cm_21 = tl.load(
        gemm_out_ptr + go_base + cb + 9, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 9)
    cm_22 = tl.load(
        gemm_out_ptr + go_base + cb + 10, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 10)
    cm_23 = tl.load(
        gemm_out_ptr + go_base + cb + 11, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 11)
    cm_30 = tl.load(
        gemm_out_ptr + go_base + cb + 12, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 12)
    cm_31 = tl.load(
        gemm_out_ptr + go_base + cb + 13, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 13)
    cm_32 = tl.load(
        gemm_out_ptr + go_base + cb + 14, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 14)
    cm_33 = tl.load(
        gemm_out_ptr + go_base + cb + 15, mask=valid, other=0.0
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 15)
    rm = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    cm_00 = tl.exp(cm_00 - rm)
    cm_01 = tl.exp(cm_01 - rm)
    cm_02 = tl.exp(cm_02 - rm)
    cm_03 = tl.exp(cm_03 - rm)
    rs = cm_00 + cm_01 + cm_02 + cm_03
    inv_rs = 1.0 / rs
    cm_00 = cm_00 * inv_rs + hc_sinkhorn_eps
    cm_01 = cm_01 * inv_rs + hc_sinkhorn_eps
    cm_02 = cm_02 * inv_rs + hc_sinkhorn_eps
    cm_03 = cm_03 * inv_rs + hc_sinkhorn_eps
    rm = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    cm_10 = tl.exp(cm_10 - rm)
    cm_11 = tl.exp(cm_11 - rm)
    cm_12 = tl.exp(cm_12 - rm)
    cm_13 = tl.exp(cm_13 - rm)
    rs = cm_10 + cm_11 + cm_12 + cm_13
    inv_rs = 1.0 / rs
    cm_10 = cm_10 * inv_rs + hc_sinkhorn_eps
    cm_11 = cm_11 * inv_rs + hc_sinkhorn_eps
    cm_12 = cm_12 * inv_rs + hc_sinkhorn_eps
    cm_13 = cm_13 * inv_rs + hc_sinkhorn_eps
    rm = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    cm_20 = tl.exp(cm_20 - rm)
    cm_21 = tl.exp(cm_21 - rm)
    cm_22 = tl.exp(cm_22 - rm)
    cm_23 = tl.exp(cm_23 - rm)
    rs = cm_20 + cm_21 + cm_22 + cm_23
    inv_rs = 1.0 / rs
    cm_20 = cm_20 * inv_rs + hc_sinkhorn_eps
    cm_21 = cm_21 * inv_rs + hc_sinkhorn_eps
    cm_22 = cm_22 * inv_rs + hc_sinkhorn_eps
    cm_23 = cm_23 * inv_rs + hc_sinkhorn_eps
    rm = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    cm_30 = tl.exp(cm_30 - rm)
    cm_31 = tl.exp(cm_31 - rm)
    cm_32 = tl.exp(cm_32 - rm)
    cm_33 = tl.exp(cm_33 - rm)
    rs = cm_30 + cm_31 + cm_32 + cm_33
    inv_rs = 1.0 / rs
    cm_30 = cm_30 * inv_rs + hc_sinkhorn_eps
    cm_31 = cm_31 * inv_rs + hc_sinkhorn_eps
    cm_32 = cm_32 * inv_rs + hc_sinkhorn_eps
    cm_33 = cm_33 * inv_rs + hc_sinkhorn_eps
    cs0 = cm_00 + cm_10 + cm_20 + cm_30
    cs1 = cm_01 + cm_11 + cm_21 + cm_31
    cs2 = cm_02 + cm_12 + cm_22 + cm_32
    cs3 = cm_03 + cm_13 + cm_23 + cm_33
    inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps)
    inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
    inv_cs2 = 1.0 / (cs2 + hc_sinkhorn_eps)
    inv_cs3 = 1.0 / (cs3 + hc_sinkhorn_eps)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_20 *= inv_cs0
    cm_30 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1
    cm_21 *= inv_cs1
    cm_31 *= inv_cs1
    cm_02 *= inv_cs2
    cm_12 *= inv_cs2
    cm_22 *= inv_cs2
    cm_32 *= inv_cs2
    cm_03 *= inv_cs3
    cm_13 *= inv_cs3
    cm_23 *= inv_cs3
    cm_33 *= inv_cs3
    for _ in tl.static_range(sinkhorn_repeat - 1):
        rs0 = cm_00 + cm_01 + cm_02 + cm_03
        rs1 = cm_10 + cm_11 + cm_12 + cm_13
        rs2 = cm_20 + cm_21 + cm_22 + cm_23
        rs3 = cm_30 + cm_31 + cm_32 + cm_33
        inv_rs0 = 1.0 / (rs0 + hc_sinkhorn_eps)
        inv_rs1 = 1.0 / (rs1 + hc_sinkhorn_eps)
        inv_rs2 = 1.0 / (rs2 + hc_sinkhorn_eps)
        inv_rs3 = 1.0 / (rs3 + hc_sinkhorn_eps)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_02 *= inv_rs0
        cm_03 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        cm_12 *= inv_rs1
        cm_13 *= inv_rs1
        cm_20 *= inv_rs2
        cm_21 *= inv_rs2
        cm_22 *= inv_rs2
        cm_23 *= inv_rs2
        cm_30 *= inv_rs3
        cm_31 *= inv_rs3
        cm_32 *= inv_rs3
        cm_33 *= inv_rs3
        cs0 = cm_00 + cm_10 + cm_20 + cm_30
        cs1 = cm_01 + cm_11 + cm_21 + cm_31
        cs2 = cm_02 + cm_12 + cm_22 + cm_32
        cs3 = cm_03 + cm_13 + cm_23 + cm_33
        inv_cs0 = 1.0 / (cs0 + hc_sinkhorn_eps)
        inv_cs1 = 1.0 / (cs1 + hc_sinkhorn_eps)
        inv_cs2 = 1.0 / (cs2 + hc_sinkhorn_eps)
        inv_cs3 = 1.0 / (cs3 + hc_sinkhorn_eps)
        cm_00 *= inv_cs0
        cm_01 *= inv_cs1
        cm_02 *= inv_cs2
        cm_03 *= inv_cs3
        cm_10 *= inv_cs0
        cm_11 *= inv_cs1
        cm_12 *= inv_cs2
        cm_13 *= inv_cs3
        cm_20 *= inv_cs0
        cm_21 *= inv_cs1
        cm_22 *= inv_cs2
        cm_23 *= inv_cs3
        cm_30 *= inv_cs0
        cm_31 *= inv_cs1
        cm_32 *= inv_cs2
        cm_33 *= inv_cs3
    co = pid_n * 16
    tl.store(comb_mix_ptr + co + 0, cm_00, mask=valid)
    tl.store(comb_mix_ptr + co + 1, cm_01, mask=valid)
    tl.store(comb_mix_ptr + co + 2, cm_02, mask=valid)
    tl.store(comb_mix_ptr + co + 3, cm_03, mask=valid)
    tl.store(comb_mix_ptr + co + 4, cm_10, mask=valid)
    tl.store(comb_mix_ptr + co + 5, cm_11, mask=valid)
    tl.store(comb_mix_ptr + co + 6, cm_12, mask=valid)
    tl.store(comb_mix_ptr + co + 7, cm_13, mask=valid)
    tl.store(comb_mix_ptr + co + 8, cm_20, mask=valid)
    tl.store(comb_mix_ptr + co + 9, cm_21, mask=valid)
    tl.store(comb_mix_ptr + co + 10, cm_22, mask=valid)
    tl.store(comb_mix_ptr + co + 11, cm_23, mask=valid)
    tl.store(comb_mix_ptr + co + 12, cm_30, mask=valid)
    tl.store(comb_mix_ptr + co + 13, cm_31, mask=valid)
    tl.store(comb_mix_ptr + co + 14, cm_32, mask=valid)
    tl.store(comb_mix_ptr + co + 15, cm_33, mask=valid)


@triton.jit
def mhc_mix(
    residual_ptr,
    pre_mix_ptr,
    layer_input_ptr,
    num_tokens,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    res_base = pid_n * res_stride_n
    pre_mix_0 = tl.load(pre_mix_ptr + pid_n * 4 + 0)
    pre_mix_1 = tl.load(pre_mix_ptr + pid_n * 4 + 1)
    pre_mix_2 = tl.load(pre_mix_ptr + pid_n * 4 + 2)
    pre_mix_3 = tl.load(pre_mix_ptr + pid_n * 4 + 3)
    h_start = tl.program_id(1) * BLOCK_H
    if h_start < hidden_size:
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        r0 = tl.load(
            residual_ptr + res_base + 0 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        r1 = tl.load(
            residual_ptr + res_base + 1 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        acc = pre_mix_0 * r0 + pre_mix_1 * r1
        r2 = tl.load(
            residual_ptr + res_base + 2 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        r3 = tl.load(
            residual_ptr + res_base + 3 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre_mix_2 * r2 + pre_mix_3 * r3
        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            acc.to(tl.bfloat16),
            mask=h_mask,
        )


def run(
    g,
    scale,
    bias,
    residual,
    post,
    comb,
    layer,
    rms_eps,
    pre_eps,
    sinkhorn_eps,
    post_mult,
    repeat,
):
    n = residual.shape[0]
    rms = torch.empty((n,), device=residual.device, dtype=torch.float32)
    pre = torch.empty((n, 4), device=residual.device, dtype=torch.float32)
    mhc_rms[n,](
        residual,
        rms,
        n,
        *residual.stride(),
        4096,
        16384,
        rms_eps=rms_eps,
        BLOCK_H=4096,
        num_warps=4,
        num_stages=1,
    )
    mhc_coeff[triton.cdiv(n, 64),](
        g,
        scale,
        bias,
        rms,
        pre,
        post,
        comb,
        n,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=post_mult,
        sinkhorn_repeat=repeat,
        HC_MULT3=24,
        BT=64,
        num_warps=1,
        num_stages=1,
    )
    mhc_mix[n, 2](
        residual,
        pre,
        layer,
        n,
        *residual.stride(),
        layer.stride(0),
        layer.stride(1),
        4096,
        BLOCK_H=2048,
        num_warps=4,
        num_stages=1,
    )


_VERIFY_LIMIT = int(os.environ.get("VLLM_FL_METAX_MHC_PREFILL_SPLIT_VERIFY", "0"))
_VERIFY_COUNT = 0
_TAIL_STATS = {"split": 0, "fallback": 0}


def _original(
    g,
    scale,
    bias,
    residual,
    post,
    comb,
    layer,
    rms_eps,
    pre_eps,
    sinkhorn_eps,
    post_mult,
    repeat,
):
    m = importlib.import_module("vllm_fl.ops.deepseek_v4_metax.mhc")
    (n, _, h) = residual.shape
    bucket = (
        1 if n <= 512 else 2 if n <= 1024 else 3 if n <= 2048 else 4 if n <= 4096 else 5
    )
    m.mhc_pre_fused_kernel_hc_mult_4[n,](
        g,
        scale,
        bias,
        residual,
        post,
        comb,
        layer,
        n,
        bucket,
        *residual.stride(),
        layer.stride(0),
        layer.stride(1),
        h,
        4 * h,
        rms_eps=rms_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=post_mult,
        sinkhorn_repeat=repeat,
        HC_MULT3=24,
    )


@torch.library.custom_op("vllm_fl::dsv4_mhc_prefill_split_tail", mutates_args=())
def tail(
    g: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _VERIFY_COUNT
    (n, _, h) = residual.shape
    post = torch.empty((n, 4), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n, 16), device=residual.device, dtype=torch.float32)
    layer = torch.empty((n, h), device=residual.device, dtype=torch.bfloat16)
    eligible = (
        n >= 1024
        and h == 4096
        and (residual.shape[1] == 4)
        and residual.is_contiguous()
        and (residual.dtype == torch.bfloat16)
        and g.is_contiguous()
        and (g.dtype == torch.float32)
        and (scale.dtype == torch.float32)
        and (bias.dtype == torch.float32)
        and scale.is_contiguous()
        and bias.is_contiguous()
    )
    if eligible:
        _TAIL_STATS["split"] += 1
        run(
            g,
            scale,
            bias,
            residual,
            post,
            comb,
            layer,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_mult,
            repeat,
        )
    else:
        _TAIL_STATS["fallback"] += 1
        _original(
            g,
            scale,
            bias,
            residual,
            post,
            comb,
            layer,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_mult,
            repeat,
        )
    return (post, comb, layer)


@tail.register_fake
def _tail_fake(
    g, scale, bias, residual, rms_eps, pre_eps, sinkhorn_eps, post_mult, repeat
):
    (n, _, h) = residual.shape
    return (
        torch.empty((n, 4), device=residual.device, dtype=torch.float32),
        torch.empty((n, 16), device=residual.device, dtype=torch.float32),
        torch.empty((n, h), device=residual.device, dtype=torch.bfloat16),
    )
