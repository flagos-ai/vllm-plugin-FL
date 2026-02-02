#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.
"""
Operator Profiling for Kimi-K2.5

Profile CUDA kernel execution times to identify optimization opportunities.

Output categories:
- GEMM (cuBLAS/cutlass): Matrix multiplications
- Attention: FlashAttention, MLA
- MoE: Mixture of Experts kernels
- Activation: silu, gelu, relu
- Norm: RMSNorm, LayerNorm
- Memory: copy, cache operations

Usage:
    python profile_ops.py --model /path/to/model
    python profile_ops.py --input-len 512 --output-len 64
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set environment before imports
os.environ["VLLM_PLATFORM_PLUGIN"] = "fl"
os.environ["USE_FLAGGEMS"] = "True"
os.environ["GEMS_MODE"] = "SELECTIVE"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import logging
logging.disable(logging.INFO)


def profile_model(model_path: str, input_len: int, output_len: int, num_iters: int = 3):
    """Profile model inference and analyze kernel distribution."""
    import torch
    from vllm import LLM, SamplingParams

    print("=" * 80)
    print("KIMI-K2.5 OPERATOR PROFILER")
    print("=" * 80)
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_path}")
    print(f"Input: {input_len} tokens, Output: {output_len} tokens")
    print(f"Iterations: {num_iters}")

    # Initialize model
    print("\nLoading model...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        load_format="dummy",
        enforce_eager=True,  # Required for profiling
    )

    tokenizer = llm.get_tokenizer()
    base_text = "The quick brown fox jumps over the lazy dog. " * 64
    tokens = tokenizer.encode(base_text)[:input_len]
    prompt = tokenizer.decode(tokens)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = llm.generate([prompt], sampling_params)
    torch.cuda.synchronize()

    # Profile
    print("Profiling...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        for _ in range(num_iters):
            _ = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()

    # Analyze results
    print("\n" + "=" * 80)
    print("TOP 25 CUDA KERNELS BY TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25))

    # Categorize kernels
    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_type == torch.profiler.DeviceType.CUDA]
    total_time = sum(e.cuda_time_total for e in cuda_events)

    categories = {
        'GEMM': [],
        'Attention': [],
        'MoE': [],
        'Activation': [],
        'Norm': [],
        'Memory': [],
        'Other': []
    }

    for e in cuda_events:
        n = e.key.lower()
        if any(x in n for x in ['gemm', 'cublas', 'cutlass', 'matmul', 'mm_']):
            categories['GEMM'].append(e)
        elif any(x in n for x in ['flash', 'attention', 'attn', 'softmax']):
            categories['Attention'].append(e)
        elif any(x in n for x in ['moe', 'expert', 'fused_moe']):
            categories['MoE'].append(e)
        elif any(x in n for x in ['silu', 'gelu', 'relu']):
            categories['Activation'].append(e)
        elif any(x in n for x in ['rms', 'layer_norm', 'norm']):
            categories['Norm'].append(e)
        elif any(x in n for x in ['copy', 'cache', 'memcpy']):
            categories['Memory'].append(e)
        else:
            categories['Other'].append(e)

    print("\n" + "=" * 80)
    print("KERNEL CATEGORY ANALYSIS")
    print("=" * 80)
    print(f"\n{'Category':<20} {'Time(ms)':<12} {'%Total':<10} {'Kernels':<8}")
    print("-" * 55)

    for cat, evts in sorted(categories.items(), key=lambda x: -sum(e.cuda_time_total for e in x[1])):
        t = sum(e.cuda_time_total for e in evts) / 1000
        p = t / (total_time / 1000) * 100 if total_time > 0 else 0
        print(f'{cat:<20} {t:<12.2f} {p:<10.1f} {len(evts):<8}')

    print("=" * 80)

    # Recommendations
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    gemm_pct = sum(e.cuda_time_total for e in categories['GEMM']) / total_time * 100 if total_time > 0 else 0
    moe_pct = sum(e.cuda_time_total for e in categories['MoE']) / total_time * 100 if total_time > 0 else 0
    activation_pct = sum(e.cuda_time_total for e in categories['Activation']) / total_time * 100 if total_time > 0 else 0

    print(f"""
    GEMM ({gemm_pct:.1f}% of time):
    - Primary optimization target via FP8 quantization
    - Use quantization='fp8' for cutlass FP8 kernels

    MoE ({moe_pct:.1f}% of time):
    - vLLM's fused_moe already optimized
    - FP8 MoE available with quantization='fp8'

    Activation ({activation_pct:.1f}% of time):
    - Triton silu_and_mul provides 3x kernel speedup
    - Low overall impact due to small percentage

    General:
    - CUDA Graph mode provides +30-37% improvement
    - FP8 + CUDA Graph provides best performance
    """)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Kimi-K2.5 Operator Profiler")
    parser.add_argument("--model", default="/root/kimi2.5/kimi_k25_dummy_2layer",
                       help="Model path")
    parser.add_argument("--input-len", type=int, default=256, help="Input length")
    parser.add_argument("--output-len", type=int, default=32, help="Output length")
    parser.add_argument("--num-iters", type=int, default=3, help="Profile iterations")
    args = parser.parse_args()

    profile_model(args.model, args.input_len, args.output_len, args.num_iters)


if __name__ == "__main__":
    main()
