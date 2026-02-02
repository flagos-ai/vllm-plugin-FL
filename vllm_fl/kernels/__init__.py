# Copyright (c) 2026 BAAI. All rights reserved.

"""
Custom fused kernels for vLLM-FL.
"""

from .fused_ops import (
    # Triton-based fusions (experimental)
    fused_rmsnorm_linear,
    fused_silu_mul,
    # vLLM-optimized operations (recommended)
    fused_add_rmsnorm,
    vllm_rms_norm,
    rms_norm_and_linear,
    # Benchmarking
    benchmark_fused_rmsnorm_linear,
    print_benchmark_results,
)

__all__ = [
    # Triton-based fusions
    'fused_rmsnorm_linear',
    'fused_silu_mul',
    # vLLM-optimized operations
    'fused_add_rmsnorm',
    'vllm_rms_norm',
    'rms_norm_and_linear',
    # Benchmarking
    'benchmark_fused_rmsnorm_linear',
    'print_benchmark_results',
]
