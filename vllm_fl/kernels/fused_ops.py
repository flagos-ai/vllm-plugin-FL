# Copyright (c) 2026 BAAI. All rights reserved.

"""
Fused Operators for vLLM-FL.

This module provides custom fused kernels that combine multiple operations
into single GPU kernel launches to reduce memory bandwidth and kernel launch overhead.

Key Fusions:
1. RMSNorm + Linear: Fuse normalization with following linear projection
2. SiLU + Mul (already in vLLM, wrapped here)
3. Gate + Up projection for SwiGLU

Performance Benefits:
- Reduced memory traffic (intermediate results stay in registers/shared memory)
- Fewer kernel launches
- Better GPU utilization
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# =============================================================================
# Fused RMSNorm + Linear Kernel (Correct 2-pass implementation)
# =============================================================================

@triton.jit
def _rmsnorm_linear_pass1_kernel(
    # Input
    X_ptr,          # [M, K]
    # Output
    Variance_ptr,   # [M] - stores sum of squares
    # Dimensions
    M, K,
    stride_xm, stride_xk,
    BLOCK_K: tl.constexpr,
):
    """Pass 1: Compute sum of squares for each row."""
    pid_m = tl.program_id(0)

    # Accumulate sum of squares as scalar
    sum_sq = 0.0

    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        x_ptrs = X_ptr + pid_m * stride_xm + k_offs * stride_xk
        x = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    # Store variance (sum_sq / K)
    tl.store(Variance_ptr + pid_m, sum_sq / K)


@triton.jit
def _rmsnorm_linear_pass2_kernel(
    # Inputs
    X_ptr,          # [M, K]
    W_norm_ptr,     # [K]
    W_linear_ptr,   # [N, K]
    Variance_ptr,   # [M]
    # Output
    Out_ptr,        # [M, N]
    # Dimensions
    M, N, K,
    eps,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Pass 2: Normalize and multiply with linear weight."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load precomputed variance for this row
    variance = tl.load(Variance_ptr + pid_m)
    rstd = tl.rsqrt(variance + eps)

    # Output accumulator
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        # Load and normalize input
        x_ptrs = X_ptr + pid_m * stride_xm + k_offs * stride_xk
        x = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        x_norm = x * rstd

        # Apply RMSNorm weight
        w_norm = tl.load(W_norm_ptr + k_offs, mask=mask_k, other=1.0).to(tl.float32)
        x_norm = x_norm * w_norm

        # Load linear weights and accumulate
        # W_linear: [N, K], load column for this k block
        w_ptrs = W_linear_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # acc += sum(x_norm * w, axis=1)
        acc += tl.sum(x_norm[None, :] * w, axis=1)

    # Store output
    out_ptrs = Out_ptr + pid_m * stride_om + offs_n * stride_on
    tl.store(out_ptrs, acc, mask=mask_n)


def fused_rmsnorm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm + Linear operation (2-pass implementation).

    Args:
        x: Input tensor [batch, hidden] or [batch, seq, hidden]
        norm_weight: RMSNorm weight [hidden]
        linear_weight: Linear weight [out_features, hidden]
        eps: Epsilon for numerical stability

    Returns:
        Output tensor [batch, out_features] or [batch, seq, out_features]

    Note:
        Uses 2-pass algorithm:
        Pass 1: Compute variance for each row
        Pass 2: Normalize and multiply with linear weight
    """
    # Handle 3D input
    original_shape = x.shape
    if x.dim() == 3:
        batch, seq, hidden = x.shape
        x = x.view(batch * seq, hidden)
    else:
        batch, hidden = x.shape
        seq = None

    M, K = x.shape
    N = linear_weight.shape[0]

    # Ensure contiguous
    x = x.contiguous()
    norm_weight = norm_weight.contiguous()
    linear_weight = linear_weight.contiguous()

    # Allocate intermediate and output buffers
    variance = torch.empty(M, device=x.device, dtype=torch.float32)
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_K = 256
    BLOCK_N = 64

    # Pass 1: Compute variance
    _rmsnorm_linear_pass1_kernel[(M,)](
        x, variance,
        M, K,
        x.stride(0), x.stride(1),
        BLOCK_K=BLOCK_K,
    )

    # Pass 2: Normalize and linear projection
    grid = (M, triton.cdiv(N, BLOCK_N))
    _rmsnorm_linear_pass2_kernel[grid](
        x, norm_weight, linear_weight, variance,
        out,
        M, N, K,
        eps,
        x.stride(0), x.stride(1),
        linear_weight.stride(0), linear_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Restore shape if needed
    if seq is not None:
        out = out.view(batch, seq, N)

    return out


# =============================================================================
# Fused SiLU + Mul + Linear (Gate-Up Projection)
# =============================================================================

@triton.jit
def _fused_silu_mul_kernel(
    X_ptr,      # Input [M, 2*K]
    Out_ptr,    # Output [M, K]
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused SiLU activation with element-wise multiplication.

    Computes: silu(x[:, :K]) * x[:, K:]
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K

    # Load gate part (first half)
    gate_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    gate = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    # Load up part (second half)
    up_ptrs = X_ptr + offs_m[:, None] * stride_xm + (offs_k[None, :] + K) * stride_xk
    up = tl.load(up_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    # SiLU activation: x * sigmoid(x)
    gate_sigmoid = tl.sigmoid(gate.to(tl.float32))
    gate_silu = gate * gate_sigmoid

    # Element-wise multiply
    out = gate_silu * up

    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_k[None, :])


def fused_silu_mul(x: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU + Mul operation.

    Args:
        x: Input tensor [..., 2*hidden]

    Returns:
        Output tensor [..., hidden]
    """
    # Flatten leading dimensions
    original_shape = x.shape[:-1]
    K = x.shape[-1] // 2
    x = x.view(-1, 2 * K)
    M = x.shape[0]

    # Allocate output
    out = torch.empty(M, K, device=x.device, dtype=x.dtype)

    # Grid configuration
    BLOCK_M = 32
    BLOCK_K = 256

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _fused_silu_mul_kernel[grid](
        x, out,
        M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    # Restore shape
    return out.view(*original_shape, K)


# =============================================================================
# Fused Add + RMSNorm (Residual + Norm) - Using vLLM's optimized kernel
# =============================================================================

# Try to import vLLM's optimized fused ops
_vllm_fused_available = False
try:
    from vllm._custom_ops import fused_add_rms_norm as _vllm_fused_add_rms_norm
    from vllm._custom_ops import rms_norm as _vllm_rms_norm
    _vllm_fused_available = True
    logger.info("vLLM fused ops available - using optimized kernels")
except ImportError:
    logger.warning("vLLM fused ops not available - using fallback implementation")


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused residual addition + RMSNorm.

    Uses vLLM's optimized CUDA kernel when available, otherwise falls back to
    PyTorch implementation.

    Args:
        x: Input tensor [batch, hidden]
        residual: Residual tensor [batch, hidden] (modified in-place)
        weight: RMSNorm weight [hidden]
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (normalized output, updated residual)
    """
    if _vllm_fused_available:
        # vLLM's fused_add_rms_norm modifies residual in-place:
        # residual = residual + x
        # returns normalized residual
        _vllm_fused_add_rms_norm(x, residual, weight, eps)
        # After this call, residual contains (x + old_residual)
        # We need to compute the normalized version
        out = torch.empty_like(x)
        _vllm_rms_norm(out, residual, weight, eps)
        return out, residual
    else:
        # Fallback: PyTorch implementation
        residual = residual + x
        variance = residual.pow(2).mean(-1, keepdim=True)
        out = residual * torch.rsqrt(variance + eps) * weight
        return out, residual


def vllm_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMS normalization using vLLM's optimized kernel.

    Args:
        x: Input tensor [..., hidden]
        weight: RMSNorm weight [hidden]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    if _vllm_fused_available:
        out = torch.empty_like(x)
        _vllm_rms_norm(out, x, weight, eps)
        return out
    else:
        # Fallback: PyTorch implementation
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight


def rms_norm_and_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm followed by Linear projection.

    This is a "logical fusion" - uses optimized RMSNorm + optimized GEMM
    separately but minimizes intermediate memory.

    Args:
        x: Input tensor [batch, hidden]
        norm_weight: RMSNorm weight [hidden]
        linear_weight: Linear weight [out_features, hidden]
        eps: Epsilon for numerical stability

    Returns:
        Output tensor [batch, out_features]
    """
    # Use vLLM's optimized RMSNorm
    x_norm = vllm_rms_norm(x, norm_weight, eps)
    # Use PyTorch's optimized GEMM (backed by cuBLAS)
    return torch.nn.functional.linear(x_norm, linear_weight)


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_fused_rmsnorm_linear(
    batch_sizes: list = [1, 4, 16, 64, 256],
    hidden_size: int = 4096,
    out_size: int = 4096,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """
    Benchmark fused vs unfused RMSNorm + Linear.

    Returns dict with timing comparisons.
    """
    import time

    device = 'cuda'
    dtype = torch.bfloat16
    eps = 1e-6

    results = {}

    for batch in batch_sizes:
        # Create test tensors
        x = torch.randn(batch, hidden_size, device=device, dtype=dtype)
        norm_weight = torch.randn(hidden_size, device=device, dtype=dtype)
        linear_weight = torch.randn(out_size, hidden_size, device=device, dtype=dtype)

        # Warmup
        for _ in range(warmup):
            # Unfused
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + eps) * norm_weight
            out_unfused = torch.nn.functional.linear(x_norm, linear_weight)

            # Fused
            out_fused = fused_rmsnorm_linear(x, norm_weight, linear_weight, eps)

        torch.cuda.synchronize()

        # Benchmark unfused
        start = time.perf_counter()
        for _ in range(iterations):
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + eps) * norm_weight
            out_unfused = torch.nn.functional.linear(x_norm, linear_weight)
        torch.cuda.synchronize()
        unfused_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark fused
        start = time.perf_counter()
        for _ in range(iterations):
            out_fused = fused_rmsnorm_linear(x, norm_weight, linear_weight, eps)
        torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / iterations * 1000

        # Verify correctness
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps) * norm_weight
        expected = torch.nn.functional.linear(x_norm, linear_weight)
        actual = fused_rmsnorm_linear(x, norm_weight, linear_weight, eps)
        max_diff = (expected - actual).abs().max().item()

        results[batch] = {
            'unfused_ms': unfused_time,
            'fused_ms': fused_time,
            'speedup': unfused_time / fused_time,
            'max_diff': max_diff,
            'correct': max_diff < 0.01,
        }

    return results


def print_benchmark_results(results: dict):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print("FUSED RMSNORM+LINEAR BENCHMARK")
    print("=" * 70)
    print(f"\n{'Batch':<10} {'Unfused (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10} {'Correct':<10}")
    print("-" * 70)

    for batch, data in sorted(results.items()):
        correct = "Yes" if data['correct'] else f"No ({data['max_diff']:.4f})"
        print(f"{batch:<10} {data['unfused_ms']:<15.4f} {data['fused_ms']:<15.4f} {data['speedup']:<10.2f}x {correct:<10}")


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_fused_rmsnorm_linear()
    print_benchmark_results(results)
