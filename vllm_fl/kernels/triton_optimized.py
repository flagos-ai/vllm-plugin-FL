"""
Triton-Optimized Kernels Integration for Kimi-K2.5.

This module integrates the optimized Triton kernels from triton_ops/
into the vLLM plugin-PL dispatch system for throughput optimization.

Critical operators and their Triton speedups (with CUDA graphs):
- concat_and_cache_ds_mla: 3.4x faster
- gather_and_maybe_dequant_cache: 1.09x faster
- silu_and_mul: 1.56x faster
- gelu_tanh_and_mul: 1.38x faster
- dynamic_per_token_scaled_fp8_quant: 1.72x faster
- static_scaled_fp8_quant: 2.24x faster
- per_token_group_quant_fp8: 1.30x faster
- merge_attn_states: 1.0x (parity)
"""

import os
import sys
import logging
from typing import Callable, Dict, Optional, Tuple, Any
from functools import wraps

import torch

logger = logging.getLogger(__name__)

# Add triton_ops to path
TRITON_OPS_PATH = "/root/kimi2.5/triton_ops"
if TRITON_OPS_PATH not in sys.path:
    sys.path.insert(0, TRITON_OPS_PATH)


# =============================================================================
# Triton Kernel Wrappers
# =============================================================================

class TritonKernelRegistry:
    """Registry for optimized Triton kernels."""

    _instance = None
    _kernels: Dict[str, Callable] = {}
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "TritonKernelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """Initialize and load all Triton kernels."""
        if self._initialized:
            return

        logger.info("Initializing Triton-optimized kernels...")

        # Load activation kernels
        self._load_activation_kernels()

        # Load FP8 quantization kernels
        self._load_fp8_kernels()

        # Load MLA cache kernels
        self._load_mla_kernels()

        # Load attention kernels
        self._load_attention_kernels()

        self._initialized = True
        logger.info(f"Loaded {len(self._kernels)} Triton-optimized kernels")

    def _load_activation_kernels(self):
        """Load activation function kernels."""
        try:
            from mul_and_silu import mul_and_silu, silu_and_mul
            self._kernels['silu_and_mul'] = silu_and_mul
            self._kernels['mul_and_silu'] = mul_and_silu
            logger.debug("Loaded mul_and_silu/silu_and_mul kernels")
        except ImportError as e:
            logger.warning(f"Failed to load mul_and_silu: {e}")

        try:
            from gelu_tanh_and_mul import gelu_tanh_and_mul
            self._kernels['gelu_tanh_and_mul'] = gelu_tanh_and_mul
            logger.debug("Loaded gelu_tanh_and_mul kernel")
        except ImportError as e:
            logger.warning(f"Failed to load gelu_tanh_and_mul: {e}")

        try:
            from fatrelu_and_mul import fatrelu_and_mul
            self._kernels['fatrelu_and_mul'] = fatrelu_and_mul
            logger.debug("Loaded fatrelu_and_mul kernel")
        except ImportError as e:
            logger.warning(f"Failed to load fatrelu_and_mul: {e}")

        try:
            from gelu_new import gelu_new
            self._kernels['gelu_new'] = gelu_new
            logger.debug("Loaded gelu_new kernel")
        except ImportError as e:
            logger.warning(f"Failed to load gelu_new: {e}")

    def _load_fp8_kernels(self):
        """Load FP8 quantization kernels."""
        try:
            from static_scaled_fp8_quant_v2 import static_scaled_fp8_quant
            self._kernels['static_scaled_fp8_quant'] = static_scaled_fp8_quant
            logger.debug("Loaded static_scaled_fp8_quant kernel")
        except ImportError as e:
            logger.warning(f"Failed to load static_scaled_fp8_quant: {e}")

        try:
            from dynamic_per_token_scaled_fp8_quant import dynamic_per_token_scaled_fp8_quant
            self._kernels['dynamic_per_token_scaled_fp8_quant'] = dynamic_per_token_scaled_fp8_quant
            logger.debug("Loaded dynamic_per_token_scaled_fp8_quant kernel")
        except ImportError as e:
            logger.warning(f"Failed to load dynamic_per_token_scaled_fp8_quant: {e}")

        try:
            from per_token_group_quant_fp8 import per_token_group_quant_fp8
            self._kernels['per_token_group_quant_fp8'] = per_token_group_quant_fp8
            logger.debug("Loaded per_token_group_quant_fp8 kernel")
        except ImportError as e:
            logger.warning(f"Failed to load per_token_group_quant_fp8: {e}")

        try:
            from dynamic_scaled_fp8_quant import dynamic_scaled_fp8_quant_v4
            self._kernels['dynamic_scaled_fp8_quant'] = dynamic_scaled_fp8_quant_v4
            logger.debug("Loaded dynamic_scaled_fp8_quant kernel")
        except ImportError as e:
            logger.warning(f"Failed to load dynamic_scaled_fp8_quant: {e}")

    def _load_mla_kernels(self):
        """Load MLA (Multi-head Latent Attention) cache kernels."""
        try:
            from concat_and_cache_ds_mla_final import concat_and_cache_ds_mla
            self._kernels['concat_and_cache_ds_mla'] = concat_and_cache_ds_mla
            logger.debug("Loaded concat_and_cache_ds_mla kernel (3.4x speedup)")
        except ImportError as e:
            logger.warning(f"Failed to load concat_and_cache_ds_mla: {e}")

        try:
            from gather_and_maybe_dequant_cache_v5 import gather_and_maybe_dequant_cache_auto
            self._kernels['gather_and_maybe_dequant_cache'] = gather_and_maybe_dequant_cache_auto
            logger.debug("Loaded gather_and_maybe_dequant_cache kernel (1.09x speedup)")
        except ImportError as e:
            logger.warning(f"Failed to load gather_and_maybe_dequant_cache: {e}")

        try:
            from merge_attn_states_best import merge_attn_states_triton
            self._kernels['merge_attn_states'] = merge_attn_states_triton
            logger.debug("Loaded merge_attn_states kernel")
        except ImportError as e:
            logger.warning(f"Failed to load merge_attn_states: {e}")

    def _load_attention_kernels(self):
        """Load attention kernels."""
        try:
            from paged_attention_best import paged_attention_best
            self._kernels['paged_attention'] = paged_attention_best
            logger.debug("Loaded paged_attention kernel (1.34x speedup for long seqs)")
        except ImportError as e:
            logger.warning(f"Failed to load paged_attention: {e}")

        try:
            from swap_blocks import swap_blocks_auto
            self._kernels['swap_blocks'] = swap_blocks_auto
            logger.debug("Loaded swap_blocks kernel (40x speedup for 200 mappings)")
        except ImportError as e:
            logger.warning(f"Failed to load swap_blocks: {e}")

    def get_kernel(self, name: str) -> Optional[Callable]:
        """Get a Triton kernel by name."""
        self.initialize()
        return self._kernels.get(name)

    def list_kernels(self) -> list:
        """List all available kernels."""
        self.initialize()
        return list(self._kernels.keys())


# Global registry instance
_registry = TritonKernelRegistry.get_instance()


def get_triton_kernel(name: str) -> Optional[Callable]:
    """Get a Triton-optimized kernel by name."""
    return _registry.get_kernel(name)


def list_triton_kernels() -> list:
    """List all available Triton-optimized kernels."""
    return _registry.list_kernels()


# =============================================================================
# Wrapped Kernel Functions for vLLM Integration
# =============================================================================

def triton_silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """
    Triton-optimized SiLU and Mul activation.

    Performance: 1.56x faster than CUDA with CUDA graphs.
    """
    kernel = _registry.get_kernel('silu_and_mul')
    if kernel is not None:
        kernel(out, x)
    else:
        # Fallback to PyTorch
        d = x.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])


def triton_gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """
    Triton-optimized GELU (tanh approximation) and Mul activation.

    Performance: 1.38x faster than CUDA with CUDA graphs.
    """
    kernel = _registry.get_kernel('gelu_tanh_and_mul')
    if kernel is not None:
        kernel(out, x)
    else:
        # Fallback to PyTorch
        d = x.shape[-1] // 2
        gelu_out = torch.nn.functional.gelu(x[..., :d], approximate='tanh')
        out.copy_(gelu_out * x[..., d:])


def triton_gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    """
    Triton-optimized NewGELU activation.

    Performance: 4.32x faster than CUDA with CUDA graphs.
    """
    kernel = _registry.get_kernel('gelu_new')
    if kernel is not None:
        kernel(out, x)
    else:
        # Fallback to PyTorch
        out.copy_(torch.nn.functional.gelu(x, approximate='tanh'))


def triton_static_scaled_fp8_quant(
    out: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    group_shape: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Triton-optimized static-scaled FP8 quantization.

    Performance: 2.24x faster than CUDA with CUDA graphs.
    """
    kernel = _registry.get_kernel('static_scaled_fp8_quant')
    if kernel is not None:
        kernel(out, x, scale, group_shape)
    else:
        # Fallback: Simple scaling + clamp
        scaled = x / scale.view(1, -1) if scale.dim() > 0 else x / scale
        out.copy_(scaled.clamp(-448.0, 448.0).to(torch.uint8))


def triton_dynamic_per_token_scaled_fp8_quant(
    out: torch.Tensor,
    x: torch.Tensor,
    scales: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
) -> None:
    """
    Triton-optimized dynamic per-token scaled FP8 quantization.

    Performance: 1.72x faster than CUDA with CUDA graphs.
    """
    kernel = _registry.get_kernel('dynamic_per_token_scaled_fp8_quant')
    if kernel is not None:
        kernel(out, x, scales, scale_ub)
    else:
        # Fallback
        absmax = x.abs().amax(dim=-1, keepdim=True)
        scales_computed = absmax / 448.0
        if scale_ub is not None:
            scales_computed = torch.minimum(scales_computed, scale_ub)
        scales.copy_(scales_computed.squeeze(-1))
        out.copy_((x / scales_computed).clamp(-448.0, 448.0).to(torch.uint8))


# =============================================================================
# Dispatch Integration
# =============================================================================

class TritonOptimizedMode:
    """
    Triton-optimized dispatch mode for Kimi-K2.5.

    This mode selectively replaces critical operators with
    Triton-optimized implementations that leverage CUDA graphs.
    """

    # Operators that benefit most from Triton optimization
    CRITICAL_OPS = {
        'concat_and_cache_ds_mla': 3.4,    # 3.4x speedup
        'static_scaled_fp8_quant': 2.24,    # 2.24x speedup
        'dynamic_per_token_scaled_fp8_quant': 1.72,  # 1.72x speedup
        'silu_and_mul': 1.56,               # 1.56x speedup
        'gelu_tanh_and_mul': 1.38,          # 1.38x speedup
        'per_token_group_quant_fp8': 1.30,  # 1.30x speedup
        'gather_and_maybe_dequant_cache': 1.09,  # 1.09x speedup
    }

    @classmethod
    def get_optimized_ops(cls) -> Dict[str, float]:
        """Get list of operators optimized in this mode with their speedups."""
        return cls.CRITICAL_OPS.copy()

    @classmethod
    def should_use_triton(cls, op_name: str, batch_size: int = 1, is_decode: bool = True) -> bool:
        """
        Determine if Triton kernel should be used for an operator.

        Args:
            op_name: Operator name
            batch_size: Current batch size
            is_decode: Whether in decode phase (where Triton excels with CUDA graphs)

        Returns:
            True if Triton kernel should be used
        """
        if op_name not in cls.CRITICAL_OPS:
            return False

        # Triton kernels excel in decode phase due to CUDA graph integration
        if is_decode:
            return True

        # For prefill, Triton is competitive for large batches
        if batch_size >= 256:
            return True

        # For prefill with small batches, CUDA might be faster
        # due to Triton's launch overhead
        if op_name in ['concat_and_cache_ds_mla', 'static_scaled_fp8_quant']:
            # These have high speedups even without CUDA graphs
            return True

        return False


def register_triton_backends():
    """
    Register Triton-optimized kernels as dispatch backends.

    This integrates with vllm_fl's dispatch manager.
    """
    try:
        from vllm_fl.dispatch.registry import OpRegistry
        from vllm_fl.dispatch.types import OpImpl, BackendImplKind

        registry = OpRegistry()

        # Initialize kernel registry
        _registry.initialize()

        # Register each available kernel
        for op_name in _registry.list_kernels():
            kernel = _registry.get_kernel(op_name)
            if kernel is None:
                continue

            impl = OpImpl(
                impl_id=f"triton.optimized.{op_name}",
                fn=kernel,
                kind=BackendImplKind.VENDOR,
                vendor="triton",
                priority=100,  # High priority to prefer over default
                is_available=lambda: True,
            )

            registry.register(op_name, impl)
            logger.debug(f"Registered Triton backend for {op_name}")

        logger.info(f"Registered {len(_registry.list_kernels())} Triton-optimized backends")

    except ImportError as e:
        logger.warning(f"Could not register Triton backends: {e}")


# =============================================================================
# Performance Metrics
# =============================================================================

class TritonPerformanceMetrics:
    """Track performance of Triton-optimized kernels."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = {}
        return cls._instance

    def record(self, op_name: str, time_us: float, batch_size: int):
        """Record kernel execution time."""
        if op_name not in self._metrics:
            self._metrics[op_name] = {
                'calls': 0,
                'total_time_us': 0.0,
                'batch_sizes': [],
            }

        self._metrics[op_name]['calls'] += 1
        self._metrics[op_name]['total_time_us'] += time_us
        self._metrics[op_name]['batch_sizes'].append(batch_size)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        for op_name, data in self._metrics.items():
            avg_time = data['total_time_us'] / max(data['calls'], 1)
            avg_batch = sum(data['batch_sizes']) / max(len(data['batch_sizes']), 1)
            summary[op_name] = {
                'calls': data['calls'],
                'avg_time_us': avg_time,
                'total_time_us': data['total_time_us'],
                'avg_batch_size': avg_batch,
            }
        return summary

    def print_report(self):
        """Print performance report."""
        summary = self.get_summary()
        print("\n" + "=" * 70)
        print("TRITON-OPTIMIZED KERNEL PERFORMANCE REPORT")
        print("=" * 70)
        print(f"\n{'Operator':<40} {'Calls':<10} {'Avg(us)':<12} {'Total(ms)':<12}")
        print("-" * 74)
        for op_name, data in sorted(summary.items(), key=lambda x: -x[1]['total_time_us']):
            print(f"{op_name:<40} {data['calls']:<10} {data['avg_time_us']:<12.1f} {data['total_time_us']/1000:<12.2f}")
        print("=" * 70)


# Initialize on import
_metrics = TritonPerformanceMetrics()


def get_performance_metrics() -> TritonPerformanceMetrics:
    """Get the global performance metrics instance."""
    return _metrics
