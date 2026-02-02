"""
Register Triton-optimized operator implementations.

This module registers optimized Triton kernels as vendor implementations
in the vLLM-FL dispatch system, enabling them to be selected via the
SELECTIVE or TIERED dispatch modes.

To enable Triton-optimized kernels EXCLUSIVELY:
    export VLLM_FL_PREFER=vendor
    export VLLM_FL_ALLOW_VENDORS=triton_optimized

To enable Triton-optimized kernels with CUDA fallback:
    export VLLM_FL_PREFER=vendor
    export VLLM_FL_ALLOW_VENDORS=triton_optimized,cuda

Priority system (higher = preferred):
- vendor.cuda: priority=100 (vLLM's default CUDA kernels)
- triton_optimized: priority=90-100 (Triton kernels)
- To select Triton over CUDA, use VLLM_FL_ALLOW_VENDORS whitelist

Performance characteristics:
- Best for decode phase with CUDA graphs
- silu_and_mul: 2.0-4.0x faster than PyTorch
- merge_attn_states: 5.0-5.9x faster
- concat_and_cache_ds_mla: 3.4x faster with CUDA graphs
- swap_blocks: 40x faster than original
"""

import os
import sys
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Add triton_ops to path
TRITON_OPS_PATH = "/root/kimi2.5/triton_ops"
if TRITON_OPS_PATH not in sys.path:
    sys.path.insert(0, TRITON_OPS_PATH)


def _is_triton_available() -> bool:
    """Check if Triton is available."""
    try:
        import triton
        return True
    except ImportError:
        return False


def _load_triton_kernel(module_name: str, fn_name: str) -> Optional[Callable]:
    """Load a Triton kernel from triton_ops directory."""
    try:
        module = __import__(module_name)
        return getattr(module, fn_name, None)
    except Exception as e:
        logger.debug(f"Failed to load {module_name}.{fn_name}: {e}")
        return None


# =========================================================================
# Wrapper functions to adapt Triton kernels to vLLM's calling conventions
# =========================================================================

def _wrap_silu_and_mul(triton_fn):
    """
    Wrap Triton silu_and_mul to match vLLM's signature.

    vLLM: silu_and_mul(x: Tensor) -> Tensor
    Triton: silu_and_mul(out: Tensor, input: Tensor) -> None
    """
    import torch

    def wrapper(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        triton_fn(out, x)
        return out

    return wrapper


def _wrap_merge_attn_states(triton_fn):
    """
    Wrap Triton merge_attn_states to match vLLM's signature.
    """
    import torch

    def wrapper(output: torch.Tensor, prefix_output: torch.Tensor,
                suffix_output: torch.Tensor, prefix_lse: torch.Tensor,
                suffix_lse: torch.Tensor) -> None:
        triton_fn(output, prefix_output, suffix_output, prefix_lse, suffix_lse)

    return wrapper


def register_builtins(registry) -> None:
    """
    Register Triton-optimized operator implementations.

    This function is called automatically by the dispatch system's
    vendor backend discovery mechanism.

    Args:
        registry: OpRegistry instance to register into
    """
    from vllm_fl.dispatch.types import OpImpl, BackendImplKind

    if not _is_triton_available():
        logger.warning("Triton not available, skipping triton_optimized backend")
        return

    registered_count = 0

    # =========================================================================
    # Activation Functions
    # =========================================================================

    # silu_and_mul - 2.0-4.0x faster
    silu_and_mul_raw = _load_triton_kernel("mul_and_silu", "silu_and_mul")
    if silu_and_mul_raw is not None:
        # Wrap to match vLLM's signature: silu_and_mul(x) -> Tensor
        silu_and_mul_fn = _wrap_silu_and_mul(silu_and_mul_raw)
        try:
            registry.register_impl(OpImpl(
                op_name="silu_and_mul",
                impl_id="triton_optimized.silu_and_mul",
                fn=silu_and_mul_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,  # High priority but below CUDA (100)
            ))
            registered_count += 1
            logger.debug("Registered triton_optimized.silu_and_mul (2.0-4.0x speedup)")
        except ValueError as e:
            logger.debug(f"silu_and_mul already registered: {e}")

    # mul_and_silu - alternate form (also needs wrapping)
    mul_and_silu_raw = _load_triton_kernel("mul_and_silu", "mul_and_silu")
    if mul_and_silu_raw is not None:
        mul_and_silu_fn = _wrap_silu_and_mul(mul_and_silu_raw)  # Same wrapper works
        try:
            registry.register_impl(OpImpl(
                op_name="mul_and_silu",
                impl_id="triton_optimized.mul_and_silu",
                fn=mul_and_silu_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,
            ))
            registered_count += 1
        except ValueError as e:
            logger.debug(f"mul_and_silu already registered: {e}")

    # gelu_tanh_and_mul - 2.1-2.2x faster
    gelu_tanh_fn = _load_triton_kernel("gelu_tanh_and_mul", "gelu_tanh_and_mul")
    if gelu_tanh_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="gelu_tanh_and_mul",
                impl_id="triton_optimized.gelu_tanh_and_mul",
                fn=gelu_tanh_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,
            ))
            registered_count += 1
            logger.debug("Registered triton_optimized.gelu_tanh_and_mul (2.1-2.2x speedup)")
        except ValueError as e:
            logger.debug(f"gelu_tanh_and_mul already registered: {e}")

    # gelu_new - 3.0-4.3x faster
    gelu_new_fn = _load_triton_kernel("gelu_new", "gelu_new")
    if gelu_new_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="gelu_new",
                impl_id="triton_optimized.gelu_new",
                fn=gelu_new_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,
            ))
            registered_count += 1
        except ValueError as e:
            logger.debug(f"gelu_new already registered: {e}")

    # fatrelu_and_mul - 1.9x faster
    fatrelu_fn = _load_triton_kernel("fatrelu_and_mul", "fatrelu_and_mul")
    if fatrelu_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="fatrelu_and_mul",
                impl_id="triton_optimized.fatrelu_and_mul",
                fn=fatrelu_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,
            ))
            registered_count += 1
        except ValueError as e:
            logger.debug(f"fatrelu_and_mul already registered: {e}")

    # =========================================================================
    # FP8 Quantization
    # =========================================================================

    # static_scaled_fp8_quant - 2.2-2.7x faster
    static_fp8_fn = _load_triton_kernel("static_scaled_fp8_quant_v2", "static_scaled_fp8_quant")
    if static_fp8_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="static_scaled_fp8_quant",
                impl_id="triton_optimized.static_scaled_fp8_quant",
                fn=static_fp8_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,
            ))
            registered_count += 1
            logger.debug("Registered triton_optimized.static_scaled_fp8_quant (2.2-2.7x speedup)")
        except ValueError as e:
            logger.debug(f"static_scaled_fp8_quant already registered: {e}")

    # dynamic_per_token_scaled_fp8_quant - 1.7x faster
    dynamic_fp8_fn = _load_triton_kernel("dynamic_per_token_scaled_fp8_quant", "dynamic_per_token_scaled_fp8_quant")
    if dynamic_fp8_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="dynamic_per_token_scaled_fp8_quant",
                impl_id="triton_optimized.dynamic_per_token_scaled_fp8_quant",
                fn=dynamic_fp8_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,
            ))
            registered_count += 1
        except ValueError as e:
            logger.debug(f"dynamic_per_token_scaled_fp8_quant already registered: {e}")

    # per_token_group_quant_fp8 - 1.3x faster
    group_fp8_fn = _load_triton_kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")
    if group_fp8_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="per_token_group_quant_fp8",
                impl_id="triton_optimized.per_token_group_quant_fp8",
                fn=group_fp8_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,
            ))
            registered_count += 1
        except ValueError as e:
            logger.debug(f"per_token_group_quant_fp8 already registered: {e}")

    # =========================================================================
    # Attention/MLA Operations
    # =========================================================================

    # merge_attn_states - 5.0-5.9x faster!
    merge_attn_fn = _load_triton_kernel("merge_attn_states_best", "merge_attn_states_triton")
    if merge_attn_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="merge_attn_states",
                impl_id="triton_optimized.merge_attn_states",
                fn=merge_attn_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=95,  # Higher priority due to significant speedup
            ))
            registered_count += 1
            logger.debug("Registered triton_optimized.merge_attn_states (5.0-5.9x speedup)")
        except ValueError as e:
            logger.debug(f"merge_attn_states already registered: {e}")

    # concat_and_cache_ds_mla - 3.4x faster with CUDA graphs
    concat_cache_fn = _load_triton_kernel("concat_and_cache_ds_mla_final", "concat_and_cache_ds_mla")
    if concat_cache_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="concat_and_cache_ds_mla",
                impl_id="triton_optimized.concat_and_cache_ds_mla",
                fn=concat_cache_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=95,  # Higher priority for CUDA graph scenarios
            ))
            registered_count += 1
            logger.debug("Registered triton_optimized.concat_and_cache_ds_mla (3.4x speedup with graphs)")
        except ValueError as e:
            logger.debug(f"concat_and_cache_ds_mla already registered: {e}")

    # gather_and_maybe_dequant_cache - 1.09x faster
    gather_cache_fn = _load_triton_kernel("gather_and_maybe_dequant_cache_v5", "gather_and_maybe_dequant_cache_auto")
    if gather_cache_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="gather_and_maybe_dequant_cache",
                impl_id="triton_optimized.gather_and_maybe_dequant_cache",
                fn=gather_cache_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=90,
            ))
            registered_count += 1
        except ValueError as e:
            logger.debug(f"gather_and_maybe_dequant_cache already registered: {e}")

    # paged_attention - 1.34x faster for long sequences
    paged_attn_fn = _load_triton_kernel("paged_attention_best", "paged_attention_best")
    if paged_attn_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="paged_attention",
                impl_id="triton_optimized.paged_attention",
                fn=paged_attn_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=85,  # Lower than CUDA for attention (CUDA is well-optimized)
            ))
            registered_count += 1
        except ValueError as e:
            logger.debug(f"paged_attention already registered: {e}")

    # =========================================================================
    # KV Cache Operations
    # =========================================================================

    # swap_blocks - 40x faster!
    swap_blocks_fn = _load_triton_kernel("swap_blocks", "swap_blocks_auto")
    if swap_blocks_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="swap_blocks",
                impl_id="triton_optimized.swap_blocks",
                fn=swap_blocks_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=100,  # Highest priority - 40x speedup!
            ))
            registered_count += 1
            logger.debug("Registered triton_optimized.swap_blocks (40x speedup)")
        except ValueError as e:
            logger.debug(f"swap_blocks already registered: {e}")

    # =========================================================================
    # Fused Operations (eliminate kernel launch overhead)
    # =========================================================================

    # fused_residual_add_rmsnorm - 1.75-2.71x faster
    fused_residual_rmsnorm_fn = _load_triton_kernel("fused_ops", "fused_residual_add_rmsnorm")
    if fused_residual_rmsnorm_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="fused_residual_add_rmsnorm",
                impl_id="triton_optimized.fused_residual_add_rmsnorm",
                fn=fused_residual_rmsnorm_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=100,  # Highest priority for fused ops
            ))
            registered_count += 1
            logger.debug("Registered triton_optimized.fused_residual_add_rmsnorm (1.75-2.71x speedup)")
        except ValueError as e:
            logger.debug(f"fused_residual_add_rmsnorm already registered: {e}")

    # fused_silu_mul_residual - 1.25-3.49x faster
    fused_silu_fn = _load_triton_kernel("fused_ops", "fused_silu_mul_residual")
    if fused_silu_fn is not None:
        try:
            registry.register_impl(OpImpl(
                op_name="fused_silu_mul_residual",
                impl_id="triton_optimized.fused_silu_mul_residual",
                fn=fused_silu_fn,
                kind=BackendImplKind.VENDOR,
                vendor="triton_optimized",
                priority=100,
            ))
            registered_count += 1
            logger.debug("Registered triton_optimized.fused_silu_mul_residual (1.25-3.49x speedup)")
        except ValueError as e:
            logger.debug(f"fused_silu_mul_residual already registered: {e}")

    # =========================================================================
    # Summary
    # =========================================================================

    if registered_count > 0:
        logger.info(f"Registered {registered_count} triton_optimized operators")
    else:
        logger.warning("No triton_optimized operators were registered")
