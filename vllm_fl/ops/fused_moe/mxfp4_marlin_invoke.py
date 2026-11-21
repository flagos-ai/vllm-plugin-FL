# SPDX-License-Identifier: Apache-2.0
# Adapted from FlagGems-vllm b5c2c224, fused_marlin_moe.py.
# Copyright (c) 2025 BAAI. All rights reserved.
# Real EM, launch grid and kernel arguments remain unchanged.
from flaggems_vllm.ops.fused_marlin_moe import _w4a16_mxfp4_moe_gemm_kernel, triton


def _invoke_w4a16_mxfp4_moe_gemm(
    A,
    B,
    C,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    *,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    block_size_k: int,
    group_size: int,
    compute_type,
    swap_ab: bool = False,
    fold_scale: bool = False,
):
    M_a = A.size(0)
    K = A.size(1)
    N = B.size(2)
    EM = sorted_token_ids.size(0)
    if M_a < block_m:
        EM = min(EM, M_a * top_k * block_m)
    # Coarsen only the tuning key. Real EM below still controls the grid and
    # valid-token/padding behavior; singleton specialization stays unchanged.
    em_bucket = 0 if M_a == 1 else (1 << (EM - 1).bit_length() if EM >= 8192 else EM)

    if C.ndim == 3:
        stride_cm = C.stride(1)
        stride_cn = C.stride(2)
    else:
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _w4a16_mxfp4_moe_gemm_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        A.size(0) * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        stride_cm,
        stride_cn,
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        SWAP_AB=swap_ab,
        EM_BUCKET=em_bucket,
        FOLD_SCALE=fold_scale,
    )
