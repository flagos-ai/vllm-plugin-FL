#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.
"""
Activation Kernel Micro-benchmarks

Compare Triton vs CUDA implementations of:
- silu_and_mul
- mul_and_silu
- gelu_tanh_and_mul

Usage:
    python benchmark_activation.py
"""

import sys
import time
import torch
import triton

# Add triton_ops to path
sys.path.insert(0, "/root/kimi2.5/triton_ops")


def benchmark_silu_and_mul():
    """Benchmark silu_and_mul implementations."""
    from vllm._custom_ops import silu_and_mul as cuda_silu_and_mul

    try:
        from mul_and_silu import silu_and_mul as triton_silu_and_mul
    except ImportError:
        print("Warning: Triton silu_and_mul not found")
        triton_silu_and_mul = None

    print("=" * 70)
    print("SILU_AND_MUL BENCHMARK")
    print("=" * 70)

    # Test configurations: (num_tokens, hidden_dim)
    configs = [
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (32, 2048),
        (32, 4096),
        (32, 8192),
        (128, 4096),
        (512, 4096),
        (1024, 4096),
    ]

    print(f"\n{'Tokens':<10} {'Hidden':<10} {'CUDA(us)':<12} {'Triton(us)':<12} {'Speedup':<10}")
    print("-" * 60)

    for num_tokens, d in configs:
        # Create tensors
        x = torch.randn(num_tokens, 2 * d, dtype=torch.bfloat16, device='cuda')
        out_cuda = torch.empty(num_tokens, d, dtype=torch.bfloat16, device='cuda')
        out_triton = torch.empty(num_tokens, d, dtype=torch.bfloat16, device='cuda')

        # Warmup
        for _ in range(10):
            cuda_silu_and_mul(out_cuda, x)
        torch.cuda.synchronize()

        # Benchmark CUDA
        cuda_times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cuda_silu_and_mul(out_cuda, x)
            torch.cuda.synchronize()
            cuda_times.append((time.perf_counter() - start) * 1e6)

        cuda_avg = sum(cuda_times) / len(cuda_times)

        # Benchmark Triton (if available)
        if triton_silu_and_mul:
            for _ in range(10):
                triton_silu_and_mul(out_triton, x)
            torch.cuda.synchronize()

            triton_times = []
            for _ in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter()
                triton_silu_and_mul(out_triton, x)
                torch.cuda.synchronize()
                triton_times.append((time.perf_counter() - start) * 1e6)

            triton_avg = sum(triton_times) / len(triton_times)
            speedup = cuda_avg / triton_avg
            print(f"{num_tokens:<10} {d:<10} {cuda_avg:<12.1f} {triton_avg:<12.1f} {speedup:<10.2f}x")
        else:
            print(f"{num_tokens:<10} {d:<10} {cuda_avg:<12.1f} {'N/A':<12} {'N/A':<10}")

    print("=" * 70)


def main():
    benchmark_silu_and_mul()


if __name__ == "__main__":
    main()
