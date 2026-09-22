# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

"""Graph-visible MHC pre projection custom op for DSV4 W8A8 MetaX experiments.

Registers torch.ops.vllm_fl.dsv4_mhc_pre_mm(x_flat, fn_bf16) -> [M, 24] bf16.
Runtime dispatch:
  * decode small-N exact shape [M<=128, 16384] x [24,16384] bf16 -> custom split-K Triton kernel
  * otherwise -> torch.mm(x_flat, fn_bf16.t()) preserving original semantics
"""

from __future__ import annotations

import os
from typing import Any

import torch
import triton
import triton.language as tl


_MHC_PRE_SMALLN_PARTIAL_CACHE: dict[tuple[Any, int, int, int], torch.Tensor] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _smalln_enabled() -> bool:
    return os.environ.get("VLLM_FL_METAX_MHC_PRE_SMALLN", "0") == "1"


@triton.jit
def _mhc_pre_mm_smalln_partial_kernel(
    a_ptr,  # [M, K] bf16, contiguous
    b_ptr,  # [N, K] bf16, contiguous (fn_bf16)
    partial_ptr,  # [SPLITS, MAX_M, N_PAD] fp32
    M: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    SPLITS: tl.constexpr,
    MAX_M: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    k_start = (K * pid_s) // SPLITS
    k_end = (K * (pid_s + 1)) // SPLITS
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_end),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_end),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        partial_ptr + (pid_s * MAX_M + offs_m[:, None]) * N_PAD + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _mhc_pre_mm_smalln_reduce_kernel(
    partial_ptr,  # [SPLITS, MAX_M, N_PAD] fp32
    out_ptr,  # [M, N] bf16, contiguous
    M: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    SPLITS: tl.constexpr,
    MAX_M: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(0, SPLITS):
        v = tl.load(
            partial_ptr + (s * MAX_M + offs_m[:, None]) * N_PAD + offs_n[None, :],
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += v
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _get_partial(
    device: torch.device, splits: int, max_m: int, n_pad: int
) -> torch.Tensor:
    key = (device, splits, max_m, n_pad)
    buf = _MHC_PRE_SMALLN_PARTIAL_CACHE.get(key)
    if buf is None:
        buf = torch.empty((splits, max_m, n_pad), dtype=torch.float32, device=device)
        _MHC_PRE_SMALLN_PARTIAL_CACHE[key] = buf
    return buf


def _use_smalln(x_flat: torch.Tensor, fn_bf16: torch.Tensor) -> bool:
    if not _smalln_enabled():
        return False
    if x_flat.dim() != 2 or fn_bf16.dim() != 2:
        return False
    # DSV4 TP8 decode exact MHC projection shape.  Layout allowlist avoids hidden clone.
    return (
        x_flat.is_cuda
        and fn_bf16.is_cuda
        and x_flat.dtype is torch.bfloat16
        and fn_bf16.dtype is torch.bfloat16
        and x_flat.shape[0] <= _env_int("VLLM_FL_METAX_MHC_PRE_SMALLN_MAX_M", 128)
        and x_flat.shape[1] == 16384
        and fn_bf16.shape[0] == 24
        and fn_bf16.shape[1] == 16384
        and x_flat.stride(1) == 1
        and fn_bf16.stride(1) == 1
        and x_flat.is_contiguous()
        and fn_bf16.is_contiguous()
    )


def _smalln_mm(x_flat: torch.Tensor, fn_bf16: torch.Tensor) -> torch.Tensor:
    M = x_flat.shape[0]
    K = x_flat.shape[1]
    N = fn_bf16.shape[0]
    splits = _env_int("VLLM_FL_METAX_MHC_PRE_SMALLN_SPLITS", 8)
    block_m = _env_int("VLLM_FL_METAX_MHC_PRE_SMALLN_BM", 16)
    block_k = _env_int("VLLM_FL_METAX_MHC_PRE_SMALLN_BK", 64)
    warps = _env_int("VLLM_FL_METAX_MHC_PRE_SMALLN_WARPS", 4)
    stages = _env_int("VLLM_FL_METAX_MHC_PRE_SMALLN_STAGES", 1)
    max_m = _env_int("VLLM_FL_METAX_MHC_PRE_SMALLN_MAX_M", 128)
    n_pad = _env_int("VLLM_FL_METAX_MHC_PRE_SMALLN_N_PAD", 32)
    if max_m < M:
        max_m = 128
    partial = _get_partial(x_flat.device, splits, max_m, n_pad)
    out_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=x_flat.device)
    grid = (triton.cdiv(M, block_m), splits)
    _mhc_pre_mm_smalln_partial_kernel[grid](
        x_flat,
        fn_bf16,
        partial,
        M,
        K,
        x_flat.stride(0),
        x_flat.stride(1),
        fn_bf16.stride(0),
        fn_bf16.stride(1),
        SPLITS=splits,
        MAX_M=max_m,
        N=N,
        N_PAD=n_pad,
        BLOCK_M=block_m,
        BLOCK_N=n_pad,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    _mhc_pre_mm_smalln_reduce_kernel[(triton.cdiv(M, block_m),)](
        partial,
        out_bf16,
        M,
        out_bf16.stride(0),
        out_bf16.stride(1),
        SPLITS=splits,
        MAX_M=max_m,
        N=N,
        N_PAD=n_pad,
        BLOCK_M=block_m,
        BLOCK_N=n_pad,
        num_warps=warps,
        num_stages=stages,
    )
    return out_bf16


# Register once.  Importing this module before Dynamo capture is enough for
# torch.ops.vllm_fl.dsv4_mhc_pre_mm to be visible in mhc_pre.py.
try:

    @torch.library.custom_op("vllm_fl::dsv4_mhc_pre_mm", mutates_args=())
    def mhc_pre_mm(x_flat: torch.Tensor, fn_bf16: torch.Tensor) -> torch.Tensor:
        if _use_smalln(x_flat, fn_bf16):
            return _smalln_mm(x_flat, fn_bf16)
        # Semantic fallback for prefill / abnormal shapes.  This is not a no-op:
        # it is the original MHC projection math and returns bf16 just like torch.mm.
        return torch.mm(x_flat, fn_bf16.t())

    @mhc_pre_mm.register_fake
    def _mhc_pre_mm_fake(x_flat: torch.Tensor, fn_bf16: torch.Tensor) -> torch.Tensor:
        if x_flat.dim() != 2 or fn_bf16.dim() != 2:
            raise RuntimeError("mhc_pre_mm expects 2D x_flat and fn_bf16")
        return x_flat.new_empty((x_flat.shape[0], fn_bf16.shape[0]))

except Exception as exc:  # keep import failure explicit for service startup/debug
    _REGISTRATION_ERROR = exc
    raise
else:
    _REGISTRATION_ERROR = None
