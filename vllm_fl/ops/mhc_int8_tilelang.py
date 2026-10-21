# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""TileLang MHC kernels that also emit dynamic per-token INT8 activations."""

from __future__ import annotations

import math

import tilelang
import tilelang.language as T
import torch

from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    ENABLE_PDL,
    compute_num_split,
    mhc_fused_tilelang,
    mhc_post_tilelang,
    pass_configs,
)
from vllm.utils.deep_gemm import is_deep_gemm_supported, tf32_hc_prenorm_gemm
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op


@tilelang.jit(pass_configs=pass_configs)
def mhc_fused_shared_bf16_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 128,
    split_k: int = 1,
):
    """Fuse MHC post mapping with prenorm FMA through BF16 shared memory.

    The mixed residual is rounded to BF16 exactly once, matching the
    materialized MHC post output. The BF16 value is written to the residual
    output and consumed from shared memory by the prenorm FMA.
    """
    m = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")
    h = hidden
    h_per_split = h // split_k
    h_iters = h_per_split // n_thr
    num_warps = n_thr // 32

    comb_mix: T.Tensor((m, hc, hc), T.float32)
    residual_in: T.Tensor((m, hc, h), T.bfloat16)
    post_mix: T.Tensor((m, hc), T.float32)
    x_in: T.Tensor((m, h), T.bfloat16)
    weight_t: T.Tensor((n_out, hc, h), T.float32)
    yp_out: T.Tensor((split_k, m, n_out), T.float32)
    rp_out: T.Tensor((split_k, m), T.float32)
    residual_out: T.Tensor((m, hc, h), T.bfloat16)

    with T.Kernel(m, split_k, threads=n_thr) as (i_n, i_ks):
        tid = T.get_thread_binding()
        warp_id = tid // 32
        lane = tid % 32
        h_split_start = i_ks * h_per_split

        residual_shared = T.alloc_shared((hc, h_per_split), T.bfloat16)
        post_shared = T.alloc_shared((hc,), T.float32)
        comb_shared = T.alloc_shared((hc, hc), T.float32)
        reduce_shared = T.alloc_shared((num_warps, n_out + 1), T.float32)

        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        new_r = T.alloc_local((hc,), T.float32)
        acc = T.alloc_local((n_out,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        weight_bits = T.alloc_var(T.uint32)
        T.clear(acc)
        T.clear(sqr)

        if ENABLE_PDL:
            T.pdl_sync()

        T.copy(post_mix[i_n, 0], post_shared)
        T.copy(comb_mix[i_n, 0, 0], comb_shared)
        for j in T.unroll(hc):
            pm[j] = post_shared[j]
        for j in T.unroll(hc):
            for k in T.unroll(hc):
                cm[k, j] = comb_shared[k, j]

        for it in T.serial(h_iters):
            h_local = it * n_thr + tid
            h_idx = h_split_start + h_local
            for j in T.unroll(hc):
                new_r[j] = pm[j] * x_in[i_n, h_idx]
                for k in T.unroll(hc):
                    new_r[j] += cm[k, j] * residual_in[i_n, k, h_idx]
                residual_shared[j, h_local] = T.bfloat16(new_r[j])
        T.sync_threads()

        for it in T.serial(h_iters):
            h_local = it * n_thr + tid
            h_idx = h_split_start + h_local
            for j in T.unroll(hc):
                r = T.float32(residual_shared[j, h_local])
                residual_out[i_n, j, h_idx] = residual_shared[j, h_local]
                sqr[0] += r * r
                for n in T.unroll(n_out):
                    weight_bits = T.reinterpret(weight_t[n, j, h_idx], T.uint32)
                    weight_bits = (
                        weight_bits + 0xFFF + ((weight_bits >> 13) & 1)
                    ) & 0xFFFFE000
                    acc[n] += T.reinterpret(weight_bits, T.float32) * r

        for n in T.unroll(n_out):
            acc[n] = T.warp_reduce_sum(acc[n])
        sqr[0] = T.warp_reduce_sum(sqr[0])

        if lane == 0:
            for n in T.unroll(n_out):
                reduce_shared[warp_id, n] = acc[n]
            reduce_shared[warp_id, n_out] = sqr[0]
        T.sync_threads()

        if warp_id == 0:
            if lane < n_out:
                value = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    value += reduce_shared[w, lane]
                yp_out[i_ks, i_n, lane] = value
            if lane == n_out:
                value = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    value += reduce_shared[w, n_out]
                rp_out[i_ks, i_n] = value

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(pass_configs=pass_configs)
def mhc_pre_big_fuse_with_norm_int8_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    layer_input_q,
    layer_input_scale,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    """MHC mixing + RMSNorm + symmetric per-token INT8 quantization.

    The BF16 output is retained for compressor/indexer projections and the
    W4A16 MoE path.  The INT8 output and scale are consumed directly by W8A8
    linears, removing the following dynamic_scaled_int8_quant launch.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]
    layer_input_q: T.Tensor[[num_tokens, hidden_size], T.int8]
    layer_input_scale: T.Tensor[[num_tokens, 1], T.float32]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + hc_pre_eps
                )

            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)
                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)
                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]
                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            # Produce the numerically authoritative BF16 value first, exactly
            # matching the old kernel's rounding boundary, and accumulate its
            # global absolute maximum for dynamic symmetric INT8 quantization.
            amax_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(amax_per_pos)
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)
                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    normalized = T.bfloat16(
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )
                    ol[i1_h] = normalized
                    amax_per_pos[i1_h] = T.max(
                        amax_per_pos[i1_h], T.abs(T.float32(normalized))
                    )
                T.copy(ol, layer_input[i, i0_h * hidden_block])

            absmax = T.alloc_fragment(1, T.float32)
            T.reduce_max(amax_per_pos, absmax, dim=0)
            quant_scale = T.alloc_fragment(1, T.float32)
            quant_scale[0] = absmax[0] / 127.0
            layer_input_scale[i, 0] = quant_scale[0]

            # Scale is known only after the full hidden reduction.  The third
            # pass stays in this kernel and replaces the standalone CUDA quant
            # launch.  Reading BF16 here also preserves exact quant semantics.
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                x_shared = T.alloc_shared(hidden_block, T.bfloat16)
                x_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(layer_input[i, i0_h * hidden_block], x_shared)
                T.copy(x_shared, x_local)
                for i1_h in T.Parallel(hidden_block):
                    inv_scale = T.if_then_else(absmax[0] > 0.0, 127.0 / absmax[0], 0.0)
                    q = T.clamp(T.round(x_local[i1_h] * inv_scale), -127.0, 127.0)
                    layer_input_q[i, i0_h * hidden_block + i1_h] = T.int8(q)

        if ENABLE_PDL:
            T.pdl_trigger()


def _allocate_outputs(residual: torch.Tensor):
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    num_tokens = residual.view(-1, hc_mult, hidden_size).shape[0]
    device = residual.device
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    layer_input_q = torch.empty(
        num_tokens, hidden_size, dtype=torch.int8, device=device
    )
    layer_input_scale = torch.empty(num_tokens, 1, dtype=torch.float32, device=device)
    return (
        outer_shape,
        hc_mult,
        hidden_size,
        post_mix,
        comb_mix,
        layer_input,
        layer_input_q,
        layer_input_scale,
    )


def _run_big_fuse(
    residual_flat,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_weight,
    norm_eps,
    post_mix,
    comb_mix,
    layer_input,
    layer_input_q,
    layer_input_scale,
    gemm_out_mul,
    gemm_out_sqrsum,
    n_splits,
    hc_mult,
    hidden_size,
):
    mhc_pre_big_fuse_with_norm_int8_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        layer_input_q,
        layer_input_scale,
        norm_weight,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        n_splits,
        hc_mult,
    )


def mhc_pre_int8_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert norm_weight is not None
    (
        outer_shape,
        hc_mult,
        hidden_size,
        post_mix,
        comb_mix,
        layer_input,
        layer_input_q,
        layer_input_scale,
    ) = _allocate_outputs(residual)
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_hidden_size = hc_mult * hidden_size
    norm_weight = norm_weight.to(torch.bfloat16).contiguous()
    if is_deep_gemm_supported():
        n_splits = compute_num_split(64, hc_hidden_size, cdiv(num_tokens, 64))
    else:
        n_splits = 1
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    residual_2d = residual_flat.view(num_tokens, hc_hidden_size)
    if is_deep_gemm_supported():
        tf32_hc_prenorm_gemm(residual_2d, fn, gemm_out_mul, gemm_out_sqrsum, n_splits)
    else:
        _tilelang_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            hc_mult,
        )
    _run_big_fuse(
        residual_flat,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
        post_mix,
        comb_mix,
        layer_input,
        layer_input_q,
        layer_input_scale,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
        hc_mult,
        hidden_size,
    )
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
        layer_input_q.view(*outer_shape, hidden_size),
        layer_input_scale.view(*outer_shape, 1),
    )


def mhc_fused_post_pre_int8_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    assert norm_weight is not None
    (
        outer_shape,
        hc_mult,
        hidden_size,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        layer_input_q,
        layer_input_scale,
    ) = _allocate_outputs(residual)
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)
    hc_mult3 = hc_mult * (2 + hc_mult)
    hc_hidden_size = hc_mult * hidden_size
    norm_weight = norm_weight.to(torch.bfloat16).contiguous()
    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = num_tokens <= 16
    if use_small_fma:
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
    elif use_deep_gemm:
        n_splits = compute_num_split(64, hc_hidden_size, cdiv(num_tokens, 64))
    else:
        n_splits = 1
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    residual_cur = torch.empty_like(residual_flat)
    if use_small_fma:
        mhc_fused_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_n=tile_n,
            n_splits=n_splits,
        )
    else:
        mhc_post_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            hc_mult,
            hidden_size,
        )
        residual_cur_2d = residual_cur.view(num_tokens, hc_hidden_size)
        if use_deep_gemm:
            tf32_hc_prenorm_gemm(
                residual_cur_2d, fn, gemm_out_mul, gemm_out_sqrsum, n_splits
            )
        else:
            _tilelang_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                hidden_size,
                hc_mult,
            )
    _run_big_fuse(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        layer_input_q,
        layer_input_scale,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
        hc_mult,
        hidden_size,
    )
    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
        layer_input_q.view(*outer_shape, hidden_size),
        layer_input_scale.view(*outer_shape, 1),
    )


def _mhc_pre_int8_fake(residual, *args, **kwargs):
    del args, kwargs
    outer_shape, hc_mult, hidden_size = (
        residual.shape[:-2],
        residual.shape[-2],
        residual.shape[-1],
    )
    return (
        torch.empty(
            *outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            *outer_shape, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            *outer_shape, hidden_size, dtype=torch.bfloat16, device=residual.device
        ),
        torch.empty(
            *outer_shape, hidden_size, dtype=torch.int8, device=residual.device
        ),
        torch.empty(*outer_shape, 1, dtype=torch.float32, device=residual.device),
    )


def _mhc_fused_post_pre_int8_fake(x, residual, *args, **kwargs):
    del x, args, kwargs
    return (torch.empty_like(residual), *_mhc_pre_int8_fake(residual))


direct_register_custom_op(
    op_name="mhc_pre_int8_tilelang",
    op_func=mhc_pre_int8_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_int8_fake,
)
direct_register_custom_op(
    op_name="mhc_fused_post_pre_int8_tilelang",
    op_func=mhc_fused_post_pre_int8_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_int8_fake,
)


__all__ = [
    "mhc_pre_int8_tilelang",
    "mhc_fused_post_pre_int8_tilelang",
]
