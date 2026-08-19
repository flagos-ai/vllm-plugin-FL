# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU-optimized Triton kernel for W8A8 block-scaled FP8 matmul.

Replaces ``flag_gems.w8a8_block_fp8_matmul`` on GCU with a ``tl.make_block_ptr``
based kernel that lets the GCU compiler lower A/B tile loads to DMA.

VPD profile (Qwen3.5-397B-A17B-FP8, TP=8):
  _w8a8_triton_block_scaled_mm = 21.64% of GPU time (122.66s, 19200 calls)

Launch config (auto-tuned on GCU):
  BLOCK_SIZE_M=128 BLOCK_SIZE_N=128 BLOCK_SIZE_K=128
  GROUP_SIZE_M=64 num_warps=4 num_stages=2
"""

from __future__ import annotations

import logging

import torch
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

_patched = False

# ---------------------------------------------------------------------------
# GCU hardware constants (must match fused_moe.py)
# ---------------------------------------------------------------------------
GCU_NUM_GRID = 48
GCU_MAX_WARPS = 4


# ---------------------------------------------------------------------------
# Optimized Triton kernel (make_block_ptr for DMA-friendly tile loads)
# ---------------------------------------------------------------------------

@triton.jit
def _w8a8_triton_block_scaled_mm_gcu(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # GCU Fixed-Grid strided loop
    NUM_SPC: tl.constexpr,
):
    """Triton-accelerated W8A8 block-scaled matmul with make_block_ptr.

    Uses 1-D Fixed Grid + strided loop (GCU constraint: grid <= 48)
    and ``tl.make_block_ptr`` for A/B/C tile loads/stores so the GCU
    compiler can lower them to DMA.
    """
    pid_start = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_pid = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for pid in range(pid_start, total_pid, NUM_SPC):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Matmul operand tiles via block pointers (lower to DMA on GCU).
        a_block_ptr = tl.make_block_ptr(
            base=A, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0),
        )
        # B is stored [N, K] (K contiguous), i.e. transposed: view as (K, N).
        b_block_ptr = tl.make_block_ptr(
            base=B, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(0, 1),
        )

        # Per-group scale indices (1-D, tiny — keep as masked gathers,
        # no DMA benefit).
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        As_ptrs = As + offs_am * stride_As_m
        Bs_ptrs = Bs + (offs_bn // group_n) * stride_Bs_n
        am_mask = offs_am < M
        bn_mask = offs_bn < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_block_ptr, boundary_check=(0, 1),
                        padding_option="zero")
            b = tl.load(b_block_ptr, boundary_check=(0, 1),
                        padding_option="zero")

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_s = tl.load(As_ptrs + offs_ks * stride_As_k,
                          mask=am_mask, other=0.0)
            b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k,
                          mask=bn_mask, other=0.0)

            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        if C.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif C.dtype.element_ty == tl.float16:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator.to(tl.float32)

        c_block_ptr = tl.make_block_ptr(
            base=C, shape=(M, N), strides=(stride_cm, stride_cn),
            offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0),
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------

def _w8a8_block_scaled_mm_gcu(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the GCU-optimized W8A8 block-scaled FP8 matmul kernel.

    Args:
        A: [M, K] fp8 activation tensor.
        B: [N, K] fp8 weight tensor (transposed layout, K contiguous).
        As: [M, K//group_k] activation scales.
        Bs: [N//group_n, K//group_k] weight scales.
        block_size: [group_n, group_k] quantization block shape.
        output_dtype: output dtype (bf16/fp16/fp32).

    Returns:
        C: [M, N] output tensor.
    """
    M, K = A.shape
    N, _ = B.shape
    group_n, group_k = block_size

    C = torch.empty((M, N), dtype=output_dtype, device=A.device)

    # Launch config (auto-tuned on GCU for Qwen3.5 decode shapes).
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    GROUP_SIZE_M = 64
    num_warps = GCU_MAX_WARPS
    num_stages = 2

    # GCU Fixed-Grid: 1-D, <= GCU_NUM_GRID.
    total_pid = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    grid = (min(total_pid, GCU_NUM_GRID),)

    _w8a8_triton_block_scaled_mm_gcu[grid](
        A, B, C, As, Bs,
        M, N, K,
        group_n, group_k,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),
        C.stride(0), C.stride(1),
        As.stride(0), As.stride(1),
        Bs.stride(1), Bs.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SPC=GCU_NUM_GRID,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_w8a8_block_scaled_mm_gcu_patch() -> None:
    """Patch ``flaggems_fp8_block_gemm`` to use the GCU-optimized Triton
    kernel with ``tl.make_block_ptr`` for DMA-friendly tile loads.

    On GCU, the default ``flag_gems.w8a8_block_fp8_matmul`` uses
    pointer-arithmetic tile loads which the GCU compiler cannot lower to
    DMA.  This patch replaces the op implementation with a block-pointer
    based kernel, while keeping the same custom op schema so that
    ``torch.compile`` graph tracing still works.
    """
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    try:
        # Override the custom op implementation on the GCU (PrivateUse1)
        # backend.  The op "vllm:flaggems_fp8_block_gemm" was registered in
        # vllm_fl/quantization/fp8.py on the default (CPU) backend.
        # torch.library.impl with allow_override=True replaces the impl
        # for the specified backend key.
        vllm_lib = torch.library.Library("vllm", "IMPL")
        vllm_lib.impl(
            "vllm::flaggems_fp8_block_gemm",
            _w8a8_block_scaled_mm_gcu,
            "PrivateUse1",
            allow_override=True,
        )

        _patched = True
        logger.info(
            "Patched flaggems_fp8_block_gemm for GCU "
            "(make_block_ptr, grid <= %d, NUM_SPC=%d, num_warps <= %d)",
            GCU_NUM_GRID,
            GCU_NUM_GRID,
            GCU_MAX_WARPS,
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch flaggems_fp8_block_gemm for GCU: %s",
            exc,
        )
