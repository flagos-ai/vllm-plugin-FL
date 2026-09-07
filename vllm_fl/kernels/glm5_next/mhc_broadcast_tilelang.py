# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA first-layer mHC broadcast specialization for GLM5-Next.

The checkpoint stores the first mHC projection for four residual streams, but
the model input initially contains one stream.  The generic path materializes
four identical copies before running the projection.  This module performs the
algebraically equivalent projection with the stream-summed weight and expands
the residual only while writing the fused mHC output.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

if not current_platform.is_cuda():
    raise ImportError("GLM5-Next mHC broadcast TileLang requires NVIDIA CUDA")
if not has_tilelang():
    raise ImportError(
        "tilelang is required for GLM5-Next mHC broadcast; install tilelang"
    )

# Keep the reference import order.  flashinfer must bind the real libcudart
# before TileLang can load libcudart_stub on sm100 systems.
with contextlib.suppress(Exception):
    import flashinfer.comm  # noqa: F401

import tilelang
import tilelang.language as T

ENABLE_PDL = current_platform.is_arch_support_pdl()

pass_configs: dict[tilelang.PassConfigKey, Any] = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}


@tilelang.jit(pass_configs=pass_configs)
def mhc_pre_big_fuse_broadcast_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    residual_out,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 1,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    """Finish first-layer mHC pre, residual broadcast, and fused RMSNorm."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[  # type: ignore[no-redef, valid-type]
        [n_splits, num_tokens, gemm_last_dim], T.float32
    ]
    gemm_out_sqrsum: T.Tensor[  # type: ignore[no-redef, valid-type]
        [n_splits, num_tokens], T.float32
    ]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[  # type: ignore[no-redef, valid-type]
        [hc_mult3], T.float32
    ]
    residual: T.Tensor[  # type: ignore[no-redef, valid-type]
        [num_tokens, hidden_size], T.bfloat16
    ]
    residual_out: T.Tensor[  # type: ignore[no-redef, valid-type]
        [num_tokens, hc_mult, hidden_size], T.bfloat16
    ]
    post_mix: T.Tensor[  # type: ignore[no-redef, valid-type]
        [num_tokens, hc_mult], T.float32
    ]
    comb_mix: T.Tensor[  # type: ignore[no-redef, valid-type]
        [num_tokens, hc_mult * hc_mult], T.float32
    ]
    layer_input: T.Tensor[  # type: ignore[no-redef, valid-type]
        [num_tokens, hidden_size], T.bfloat16
    ]
    norm_weight: T.Tensor[  # type: ignore[no-redef, valid-type]
        [hidden_size], T.bfloat16
    ]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / hidden_size + rms_eps)
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
                        mixes_shared[j + hc_mult] * hc_scale[1]
                        + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2]
                    * hc_scale[2]
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
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared(hidden_block, T.bfloat16)
                xl = T.alloc_fragment(hidden_block, T.float32)
                T.copy(residual[i, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                for i_hc, i1_h in T.Parallel(hc_mult, hidden_block):
                    residual_out[i, i_hc, i0_h * hidden_block + i1_h] = xs[
                        i1_h
                    ]

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(
                        ol[i1_h]
                    )

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


def mhc_pre_broadcast_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    norm_weight: torch.Tensor,
    norm_eps: float,
    fn_broadcast: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run specialized first-layer mHC pre from an unexpanded ``(T, H)``."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16 and residual.dim() == 2
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32 and hc_scale.shape == (3,)
    assert hc_base.dtype == torch.float32

    num_tokens, hidden_size = residual.shape
    hc_mult = fn.shape[1] // hidden_size
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * (2 + hc_mult)
    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_base.shape == (hc_mult3,)
    assert fn_broadcast.dtype == torch.float32
    assert fn_broadcast.shape == (hc_mult3, hidden_size)
    assert norm_weight.shape == (hidden_size,)

    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))
    residual_out = torch.empty(
        num_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    tf32_hc_prenorm_gemm(
        residual,
        fn_broadcast,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    mhc_pre_big_fuse_broadcast_with_norm_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        residual_out,
        post_mix,
        comb_mix,
        layer_input,
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
    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        layer_input,
    )


__all__ = [
    "mhc_pre_big_fuse_broadcast_with_norm_tilelang",
    "mhc_pre_broadcast_tilelang",
]
