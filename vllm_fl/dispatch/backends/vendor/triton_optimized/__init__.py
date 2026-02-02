"""
Triton-optimized vendor backend for vLLM-FL.

Provides high-performance Triton implementations of critical operators
for Kimi-K2.5/DeepSeekV3 inference.

Performance improvements:
- silu_and_mul: 2.0-4.0x faster
- merge_attn_states: 5.0-5.9x faster
- static_scaled_fp8_quant: 2.2-2.7x faster
- concat_and_cache_ds_mla: 3.4x faster with CUDA graphs
"""

from .register_ops import register_builtins

__all__ = ['register_builtins']
