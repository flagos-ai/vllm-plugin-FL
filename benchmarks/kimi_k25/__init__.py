# Copyright (c) 2025 BAAI. All rights reserved.
"""
Kimi-K2.5 Benchmark Suite for vLLM-FL

This package provides benchmarking tools for:
1. End-to-end throughput comparison (SELECTIVE vs FP8)
2. CUDA Graph optimization analysis
3. Operator-level profiling
4. Triton kernel micro-benchmarks

Usage:
    # Run main benchmark
    python -m benchmarks.kimi_k25.benchmark_e2e --help

    # Run profiler
    python -m benchmarks.kimi_k25.profile_ops --help
"""

__version__ = "1.0.0"
