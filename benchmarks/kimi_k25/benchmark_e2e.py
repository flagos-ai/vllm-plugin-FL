#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.
"""
End-to-End Benchmark: SELECTIVE vs FP8 Strategy

This is the canonical benchmark for comparing vLLM-FL optimization strategies
on Kimi-K2.5 models.

Test Matrix:
- Input/Output lengths: 1k/1k, 2k/1k, 4k/1k
- Batch sizes: 1, 2, 4
- Configurations: BASELINE, SELECTIVE, FP8 (all with CUDA graph)

Usage:
    # Full benchmark
    python benchmark_e2e.py

    # Quick test (single configuration)
    python benchmark_e2e.py --quick

    # Specific mode only
    python benchmark_e2e.py --modes baseline fp8
"""

import os
import sys
import subprocess
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class BenchResult:
    """Benchmark result container."""
    mode: str
    input_len: int
    output_len: int
    batch_size: int
    prefill_tps: float
    decode_tps: float
    total_tps: float
    latency_ms: float


# Default model path (can be overridden via --model)
DEFAULT_MODEL_PATH = "/root/kimi2.5/kimi_k25_dummy_2layer"

# vLLM-FL environment configuration
FL_ENV = {
    "VLLM_PLATFORM_PLUGIN": "fl",
    "USE_FLAGGEMS": "True",
    "GEMS_MODE": "SELECTIVE",
    "VLLM_FL_PREFER": "vendor",
    "VLLM_FL_ALLOW_VENDORS": "triton_optimized,cuda",
}


def run_benchmark(
    mode: str,
    env_vars: dict,
    use_fp8: bool,
    input_len: int,
    output_len: int,
    batch_size: int,
    model_path: str,
) -> Optional[BenchResult]:
    """Run benchmark in subprocess with isolation."""

    env_setup = '\n'.join(f'os.environ["{k}"] = "{v}"' for k, v in env_vars.items())
    quantization_arg = '"fp8"' if use_fp8 else 'None'

    script = f'''
import os
import sys
import time

{env_setup}
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

model_path = "{model_path}"

llm_kwargs = dict(
    model=model_path,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=8192,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    load_format="dummy",
    enforce_eager=False,
    compilation_config=CompilationConfig(level=2, cache_dir=""),
)

# FP8 quantization
if {quantization_arg} is not None:
    llm_kwargs["quantization"] = {quantization_arg}

llm = LLM(**llm_kwargs)
tokenizer = llm.get_tokenizer()

# Create prompts with specified input length
base_text = "The quick brown fox jumps over the lazy dog. " * 512
tokens = tokenizer.encode(base_text)[:{input_len}]
prompt = tokenizer.decode(tokens)

# Create batch
prompts = [prompt] * {batch_size}

sampling_params = SamplingParams(temperature=0.0, max_tokens={output_len}, ignore_eos=True)

# Warmup
for _ in range(3):
    _ = llm.generate(prompts, sampling_params)
torch.cuda.synchronize()

# Benchmark
times = []
total_input_tokens = []
total_output_tokens = []
num_iters = 5

for _ in range(num_iters):
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    times.append(elapsed)

    # Count tokens
    input_toks = sum(len(tokenizer.encode(p)) for p in prompts)
    output_toks = sum(len(out.outputs[0].token_ids) for out in outputs)
    total_input_tokens.append(input_toks)
    total_output_tokens.append(output_toks)

avg_time = sum(times) / len(times)
avg_input = sum(total_input_tokens) / len(total_input_tokens)
avg_output = sum(total_output_tokens) / len(total_output_tokens)

# Calculate metrics
prefill_tps = avg_input / avg_time
decode_tps = avg_output / avg_time
total_tps = (avg_input + avg_output) / avg_time

print(f"RESULT:{{avg_input:.0f}},{{avg_output:.0f}},{{prefill_tps:.1f}},{{decode_tps:.1f}},{{total_tps:.1f}},{{avg_time * 1000:.2f}}")
'''

    env = os.environ.copy()
    env.update(env_vars)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )

        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                parts = line[7:].split(',')
                return BenchResult(
                    mode=mode,
                    input_len=input_len,
                    output_len=output_len,
                    batch_size=batch_size,
                    prefill_tps=float(parts[2]),
                    decode_tps=float(parts[3]),
                    total_tps=float(parts[4]),
                    latency_ms=float(parts[5]),
                )

        # Debug on failure
        if "Error" in result.stderr or "error" in result.stderr.lower():
            print(f"    Error: {result.stderr[-500:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    Timeout!")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    import torch

    parser = argparse.ArgumentParser(description="SELECTIVE vs FP8 Strategy Benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--modes", nargs="+", default=["baseline", "selective", "fp8"],
                       choices=["baseline", "selective", "fp8"], help="Modes to test")
    args = parser.parse_args()

    print("=" * 90)
    print("SELECTIVE vs FP8 STRATEGY BENCHMARK")
    print("=" * 90)
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Model: {args.model}")

    # Test configurations: (name, env_vars, use_fp8)
    all_configs = {
        "baseline": ("BASELINE", {}, False),
        "selective": ("SELECTIVE", FL_ENV, False),
        "fp8": ("FP8", FL_ENV, True),
    }
    configs = [(all_configs[m][0], all_configs[m][1], all_configs[m][2])
               for m in args.modes if m in all_configs]

    # Test matrix
    if args.quick:
        test_cases = [(1024, 1024, 1)]
    else:
        test_cases = [
            # (input_len, output_len, batch_size)
            (1024, 1024, 1),
            (1024, 1024, 2),
            (1024, 1024, 4),
            (2048, 1024, 1),
            (2048, 1024, 2),
            (2048, 1024, 4),
            (4096, 1024, 1),
            (4096, 1024, 2),
        ]

    all_results: Dict[str, List[BenchResult]] = {name: [] for name, _, _ in configs}

    print("\n" + "-" * 90)
    print("Running benchmarks...")
    print("-" * 90)

    for input_len, output_len, batch_size in test_cases:
        print(f"\n[Input={input_len}, Output={output_len}, Batch={batch_size}]")

        for config_name, env_vars, use_fp8 in configs:
            print(f"  {config_name}...", end=" ", flush=True)
            result = run_benchmark(
                mode=config_name,
                env_vars=env_vars,
                use_fp8=use_fp8,
                input_len=input_len,
                output_len=output_len,
                batch_size=batch_size,
                model_path=args.model,
            )
            if result:
                all_results[config_name].append(result)
                print(f"Prefill={result.prefill_tps:.0f}, Decode={result.decode_tps:.0f}, Total={result.total_tps:.0f} tps")
            else:
                print("FAILED")

    # Detailed results table
    print("\n" + "=" * 90)
    print("DETAILED RESULTS")
    print("=" * 90)

    print(f"\n{'Config':<12} {'In/Out':<12} {'Batch':<6} {'Prefill':<10} {'Decode':<10} {'Total TPS':<10} {'Latency':<10}")
    print("-" * 80)

    for config_name in [c[0] for c in configs]:
        for r in all_results.get(config_name, []):
            print(f"{r.mode:<12} {r.input_len}/{r.output_len:<6} {r.batch_size:<6} "
                  f"{r.prefill_tps:<10.0f} {r.decode_tps:<10.0f} {r.total_tps:<10.0f} {r.latency_ms:<10.1f}ms")

    # Summary by configuration
    print("\n" + "=" * 90)
    print("SUMMARY BY CONFIGURATION (AVERAGES)")
    print("=" * 90)

    summary = {}
    for config_name, results in all_results.items():
        if results:
            avg_prefill = sum(r.prefill_tps for r in results) / len(results)
            avg_decode = sum(r.decode_tps for r in results) / len(results)
            avg_total = sum(r.total_tps for r in results) / len(results)
            avg_latency = sum(r.latency_ms for r in results) / len(results)
            summary[config_name] = {
                "prefill": avg_prefill,
                "decode": avg_decode,
                "total": avg_total,
                "latency": avg_latency,
                "count": len(results),
            }

    baseline_total = summary.get("BASELINE", {}).get("total", 0)

    print(f"\n{'Configuration':<15} {'Avg Prefill':<12} {'Avg Decode':<12} {'Avg Total':<12} {'Avg Latency':<12} {'vs BASELINE':<12}")
    print("-" * 85)

    for config_name in [c[0] for c in configs]:
        if config_name in summary:
            s = summary[config_name]
            speedup = ""
            if baseline_total and config_name != "BASELINE":
                speedup = f"{(s['total']/baseline_total - 1)*100:+.1f}%"
            print(f"{config_name:<15} {s['prefill']:<12.0f} {s['decode']:<12.0f} {s['total']:<12.0f} {s['latency']:<12.1f}ms {speedup:<12}")

    # Conclusion
    if "BASELINE" in summary and len(summary) > 1:
        print("\n" + "=" * 90)
        print("CONCLUSION")
        print("=" * 90)

        for config_name, s in summary.items():
            if config_name != "BASELINE":
                vs_baseline = (s["total"] / summary["BASELINE"]["total"] - 1) * 100
                print(f"  {config_name}: {vs_baseline:+.1f}% vs BASELINE")

    print("=" * 90)


if __name__ == "__main__":
    main()
