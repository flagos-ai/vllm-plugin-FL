# Kimi-K2.5 Benchmark Suite

Benchmarking tools for evaluating vLLM-FL optimization strategies on Kimi-K2.5 models.

## Quick Start

```bash
# Full benchmark (BASELINE vs SELECTIVE vs FP8)
./run_benchmark.sh

# Quick test
./run_benchmark.sh --quick

# Profile operators
./run_benchmark.sh --profile

# Specific modes only
./run_benchmark.sh --modes "baseline fp8"
```

## Benchmark Results Summary

| Configuration | Avg TPS | vs BASELINE |
|---------------|---------|-------------|
| BASELINE (BF16 + CUDA Graph) | 1806 | - |
| SELECTIVE (BF16 + Triton + Graph) | 1810 | +0.2% |
| FP8 (FP8 + CUDA Graph) | 1885 | **+4.4%** |

*Tested on NVIDIA A100-SXM4-40GB*

## Test Matrix

- **Input lengths**: 1024, 2048, 4096 tokens
- **Output length**: 1024 tokens
- **Batch sizes**: 1, 2, 4
- **All tests use CUDA Graph mode** (`enforce_eager=False`)

## Files

| File | Description |
|------|-------------|
| `benchmark_e2e.py` | Main end-to-end benchmark comparing strategies |
| `profile_ops.py` | CUDA kernel profiler for optimization analysis |
| `run_benchmark.sh` | Unified entry point script |
| `kernels/benchmark_activation.py` | Activation kernel micro-benchmarks |

## Configuration Explained

### BASELINE
- Pure vLLM with BF16 dtype
- CUDA Graph enabled
- No vLLM-FL plugin

### SELECTIVE
- vLLM-FL plugin active
- BF16 dtype with Triton-optimized kernels
- CUDA Graph enabled
- Environment:
  ```bash
  export VLLM_PLATFORM_PLUGIN=fl
  export USE_FLAGGEMS=True
  export GEMS_MODE=SELECTIVE
  export VLLM_FL_PREFER=vendor
  export VLLM_FL_ALLOW_VENDORS=triton_optimized,cuda
  ```

### FP8
- vLLM-FL plugin active
- FP8 quantization (`quantization='fp8'`)
- CUDA Graph enabled
- Uses cutlass FP8 kernels for GEMM
- ~2x memory reduction

## Profiling Analysis

Run the profiler to see kernel time distribution:

```bash
python profile_ops.py --input-len 512 --output-len 64
```

### Typical Results (Kimi-K2.5)

| Category | Time % | Primary Kernels |
|----------|--------|-----------------|
| GEMM | 81.4% | ampere_bf16_gemm, cutlass |
| MoE | 16.8% | fused_moe_kernel |
| Attention | 3.5% | unified_mla_attention |
| Norm | 0.6% | fused_add_rms_norm |
| Other | 1.7% | silu_and_mul, copy |

**Key Finding**: GEMM dominates at 81.4%. FP8 quantization directly targets this via cutlass FP8 kernels.

## Triton-Optimized Kernels

14 kernels registered in `vllm_fl.dispatch.backends.vendor.triton_optimized`:

| Operator | Speedup | Priority |
|----------|---------|----------|
| swap_blocks | 40x | 100 |
| fused_residual_add_rmsnorm | 1.75-2.71x | 100 |
| fused_silu_mul_residual | 1.25-3.49x | 100 |
| merge_attn_states | 5.0-5.9x | 95 |
| concat_and_cache_ds_mla | 3.4x | 95 |
| silu_and_mul | 2.0-4.0x | 90 |
| gelu_tanh_and_mul | 2.1-2.2x | 90 |
| gelu_new | 3.0-4.3x | 90 |
| static_scaled_fp8_quant | 2.2-2.7x | 90 |
| dynamic_per_token_scaled_fp8_quant | 1.7x | 90 |

## Recommendations

### Production Configuration

```python
from vllm import LLM
from vllm.config import CompilationConfig

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype="bfloat16",
    quantization="fp8",           # Enable FP8 for +4% throughput
    enforce_eager=False,          # Enable CUDA Graph for +30%
    compilation_config=CompilationConfig(level=2, cache_dir=""),
)
```

### Environment Variables

```bash
export VLLM_PLATFORM_PLUGIN=fl
export USE_FLAGGEMS=True
export GEMS_MODE=SELECTIVE
export VLLM_FL_PREFER=vendor
export VLLM_FL_ALLOW_VENDORS=triton_optimized,cuda
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

### Optimization Priority

1. **CUDA Graphs**: +30-37% (highest impact)
2. **FP8 Quantization**: +2-10% on A100, +50-100% on H100
3. **Triton Kernels**: ~1-2% (marginal due to GEMM dominance)

## A100 vs H100

| Feature | A100 (SM80) | H100 (SM89+) |
|---------|-------------|--------------|
| FP8 Hardware | Limited | Full |
| Expected FP8 Speedup | +2-10% | +50-100% |
| Recommendation | FP8 for memory | FP8 for speed + memory |

## FP8 Compatibility Fix

vLLM-FL includes a fix for FP8 compatibility with OOT (Out-of-Tree) platforms.

**Root Cause**: `_POSSIBLE_FP8_KERNELS` dictionary in vLLM doesn't include `PlatformEnum.OOT`.

**Location**: `vllm/model_executor/layers/quantization/kernels/scaled_mm/__init__.py:152`

**Solution**: `_register_oot_quantization_kernels()` in `vllm_fl/platform.py` patches the kernel mapping at load time.

---

*Generated: 2026-02-02*
*Device: NVIDIA A100-SXM4-40GB*
*vLLM Version: 0.15.0*
