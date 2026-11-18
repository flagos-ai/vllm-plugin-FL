# SPDX-License-Identifier: Apache-2.0
# Adapted from FlagGems-vllm b5c2c224, fused_marlin_moe.py.
# Copyright (c) 2025 BAAI. All rights reserved.
"""64-wide down-projection K tile for unpadded I192 TP16 experts.

This is NEW plugin kernel logic beyond PR #4483/#4943. It reuses their FP4
bit decode and concat helpers; the changed scale indexing is independently
validated against the unchanged padded-K128 kernel and dense MXFP4 reference.
"""
from vllm.triton_utils import triton, tl
from flaggems_vllm.ops.fused_marlin_moe import (
    _concat_k, _dequant_fp4_bf16, _dequant_fp4_bf16_fold, _dequant_fp4_fp16,
    _write_zeros_to_output,
)


@triton.jit
def _mxfp4_down_k64_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWAP_AB: tl.constexpr,
    EM_BUCKET: tl.constexpr,
    FOLD_SCALE: tl.constexpr = False,
):
    BLOCK_SIZE_K_PACK: tl.constexpr = BLOCK_SIZE_K // 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
            SWAP_AB,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_bk = tl.arange(0, BLOCK_SIZE_K_PACK)

    if SWAP_AB:
        a_base = a_ptr + (offs_token[None, :] // top_k * stride_am)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bk[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    scale_base = b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b_packed = tl.load(b_ptrs)

        # A 64-wide tile spans TWO group32 scales. Each scale covers four
        # of the 8 sub-vectors returned by #4943 FP4 decoding, so duplicate
        # scale arguments (s0,s0,s1,s1); never read nonexistent groups 2/3.
        g0 = k * BLOCK_SIZE_K // GROUP_SIZE_K
        sc0 = tl.load(scale_base + (g0 + 0) * stride_bsg)
        sc1 = sc0
        sc2 = tl.load(scale_base + (g0 + 1) * stride_bsg)
        sc3 = sc2
        if SWAP_AB:
            s0, s1, s2, s3 = sc0[:, None], sc1[:, None], sc2[:, None], sc3[:, None]
        else:
            s0, s1, s2, s3 = sc0[None, :], sc1[None, :], sc2[None, :], sc3[None, :]

        if FOLD_SCALE:
            bs = _dequant_fp4_bf16_fold(b_packed, s0, s1, s2, s3)
        elif compute_type == tl.float16:
            bs = _dequant_fp4_fp16(b_packed, s0, s1, s2, s3)
        else:
            bs = _dequant_fp4_bf16(b_packed, s0, s1, s2, s3)

        k_logical_base = k * BLOCK_SIZE_K
        # One mma over the whole K tile; the activation is loaded once as a
        # full tile instead of once per sub-tile.
        bs_full = _concat_k(bs, BLOCK_SIZE_N, BLOCK_SIZE_K, SWAP_AB)
        offs_ak_full = tl.arange(0, BLOCK_SIZE_K)
        if SWAP_AB:
            a_full_ptrs = a_base + (k_logical_base + offs_ak_full[:, None]) * stride_ak
            a_full = tl.load(a_full_ptrs, mask=token_mask[None, :], other=0.0)  # (K, M)
            accumulator = tl.dot(bs_full, a_full, acc=accumulator)  # (N, M)
        else:
            a_full_ptrs = a_base + (k_logical_base + offs_ak_full[None, :]) * stride_ak
            a_full = tl.load(a_full_ptrs, mask=token_mask[:, None], other=0.0)  # (M, K)
            accumulator = tl.dot(a_full, bs_full, acc=accumulator)  # (M, N)

        b_ptrs += BLOCK_SIZE_K_PACK * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * (
            moe_weight[None, :] if SWAP_AB else moe_weight[:, None]
        )

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if SWAP_AB:
        c_ptrs = c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn[:, None]
        c_mask = token_mask[None, :] & (offs_cn[:, None] < N)
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_down_k64(A, B, C, B_scale, topk_weights, sorted_token_ids,
                    expert_ids, num_tokens_post_padded, *, mul_routed_weight,
                    top_k, block_m, block_size_k, group_size, compute_type,
                    swap_ab=False, fold_scale=False):
    if block_size_k != 64 or group_size != 32 or A.shape[1] % 64:
        raise ValueError("K64 down kernel requires K multiple of 64 and group32")
    if top_k != 1:
        raise ValueError("K64 extension is limited to the down projection")
    n = B.shape[2]
    em = sorted_token_ids.shape[0]
    if A.shape[0] < block_m:
        em = min(em, A.shape[0] * top_k * block_m)
    stride_cm = C.stride(1) if C.ndim == 3 else C.stride(0)
    stride_cn = C.stride(2) if C.ndim == 3 else C.stride(1)
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, 64),)
    _mxfp4_down_k64_kernel[grid](
        A, B, C, B_scale, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, n, A.shape[1], em, A.shape[0] * top_k,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), B.stride(2),
        stride_cm, stride_cn, B_scale.stride(0), B_scale.stride(1),
        B_scale.stride(2), BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64, GROUP_SIZE_M=1, GROUP_SIZE_K=32,
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
        compute_type=compute_type, SWAP_AB=swap_ab,
        EM_BUCKET=0 if A.shape[0] == 1 else em,
        FOLD_SCALE=fold_scale, num_warps=4, num_stages=2)
