# Copyright (c) 2025 BAAI. All rights reserved.
#
# Register torch.ops._C op schemas so that vllm compilation passes can
# reference them for pattern matching even when the native vllm._C extension
# is not compiled for this platform.

import logging

import torch

logger = logging.getLogger(__name__)


# Fallback implementations for query ops
_QUERY_OP_IMPLS = [
    ("cutlass_scaled_mm_supports_fp8", lambda cap: cap >= 89),
    ("cutlass_scaled_mm_supports_block_fp8", lambda cap: cap >= 100),
    ("cutlass_group_gemm_supported", lambda cap: cap >= 90),
    ("cutlass_scaled_mm_supports_fp4", lambda cap: cap >= 100),
    ("weak_ref_tensor", lambda t: t),
    ("get_cuda_view_from_cpu_tensor", lambda t: t),
]


def _apply_repetition_penalties_impl(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Pure-torch fallback for _C::apply_repetition_penalties_."""
    rp = repetition_penalties.unsqueeze(dim=1).repeat(1, logits.size(1))
    penalties = torch.where(prompt_mask | output_mask, rp, 1.0)
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits.mul_(scaling)


def _dynamic_scaled_int8_quant_impl(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    azp: torch.Tensor | None = None,
) -> None:
    """Triton fallback for _C::dynamic_scaled_int8_quant (symmetric only).

    Delegates to vLLM's triton per_token_quant_int8, writing into the
    caller-provided out/scale buffers to match the _C op's signature.
    """
    assert azp is None, (
        "asymmetric dynamic int8 quant is not supported by the triton fallback"
    )
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )

    x_q, x_s = per_token_quant_int8(input)
    result.copy_(x_q)
    scale.copy_(x_s.view(scale.shape))


def _silu_and_mul_impl(result: torch.Tensor, input: torch.Tensor) -> None:
    """FlagGems fallback for _C::silu_and_mul.

    input is [..., 2*d] gate||up; result is [..., d].
    """
    from flag_gems.modules.activation import gems_silu_and_mul

    d = input.shape[-1] // 2
    result.copy_(gems_silu_and_mul(input[..., :d], input[..., d:]))


def _fused_dsv4_qnorm_rope_kv_quant_insert_impl(
    q_in: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_head_padded: int,
    eps: float,
    cache_block_size: int,
) -> torch.Tensor:
    """FlagGems-backed impl of the 0.24 9-arg quant-insert schema.

    The FlagGems kernel follows the older vendor contract: 8 args,
    normalizes/ropes q IN PLACE, no head padding. The 0.24 schema takes
    q_in read-only, pads heads to q_head_padded (zero-filled slots for
    FlashMLA's 64/128-head requirement), and returns the padded q.
    """
    from flag_gems.fused import fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

    num_tokens, num_heads, head_dim = q_in.shape
    if q_head_padded == num_heads:
        q = q_in.contiguous()
    else:
        q = torch.zeros(
            (num_tokens, q_head_padded, head_dim),
            dtype=q_in.dtype,
            device=q_in.device,
        )
        q[:, :num_heads].copy_(q_in)
    # The FlagGems kernel RMSNorms every head slot; zero-filled padding rows
    # stay zero under RMSNorm (0/sqrt(eps)) and RoPE, so running it over the
    # padded tensor matches the reference kernel's zero-fill semantics.
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv.contiguous(),
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
    )
    return q


def _fused_dsv4_qnorm_rope_kv_quant_insert_meta(
    q_in: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_head_padded: int,
    eps: float,
    cache_block_size: int,
) -> torch.Tensor:
    return torch.empty(
        (q_in.shape[0], q_head_padded, q_in.shape[2]),
        dtype=q_in.dtype,
        device=q_in.device,
    )


# Ops that need a CUDA dispatch because vLLM calls them directly
# (not routed through FL's call_op) and only has _C kernel + torch fallback
# gated behind is_cuda checks.
_CUDA_FALLBACK_IMPLS = [
    ("apply_repetition_penalties_", _apply_repetition_penalties_impl),
    ("dynamic_scaled_int8_quant", _dynamic_scaled_int8_quant_impl),
    ("silu_and_mul", _silu_and_mul_impl),
    (
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        _fused_dsv4_qnorm_rope_kv_quant_insert_impl,
    ),
]

# (name, fn) pairs registered with the Meta dispatch key so fake tensors /
# torch.compile can infer output shapes for value-returning fallback ops.
_META_IMPLS = [
    (
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        _fused_dsv4_qnorm_rope_kv_quant_insert_meta,
    ),
]


def register_op_schemas():
    """Register _C op schemas if not already present."""
    if getattr(register_op_schemas, "_lib", None) is not None:
        return

    try:
        import vllm._C  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    # Pre-load mcoplib._C (MetaX) so its TORCH_LIBRARY registrations land
    # before our FRAGMENT definitions.  The hasattr check below will then
    # skip any ops already registered by mcoplib, avoiding c10::Error.
    import importlib.util
    if importlib.util.find_spec("mcoplib") is not None:
        try:
            import mcoplib._C  # noqa: F401
        except ImportError:
            logger.warning("Failed to import mcoplib._C")

    from vllm_fl.ops._C_ops_schemas import SCHEMAS as schemas

    if not schemas:
        logger.warning("No op schemas found; torch.compile may not work.")
        return

    lib = torch.library.Library("_C", "FRAGMENT")

    for schema in schemas:
        full_name = schema.split("(")[0]
        op_name = full_name.split(".")[0]
        overload = full_name.split(".")[1] if "." in full_name else "default"
        if hasattr(torch.ops._C, op_name) and hasattr(
            getattr(torch.ops._C, op_name), overload
        ):
            continue
        try:
            lib.define(schema)
        except Exception as e:
            logger.debug("Failed to register _C op schema '%s': %s", full_name, e)

    for name, fn in _QUERY_OP_IMPLS:
        try:
            lib.impl(name, fn, "CompositeImplicitAutograd")
        except Exception:
            pass

    for name, fn in _CUDA_FALLBACK_IMPLS:
        try:
            lib.impl(name, fn, "CUDA")
        except Exception:
            pass

    for name, fn in _META_IMPLS:
        try:
            lib.impl(name, fn, "Meta")
        except Exception:
            pass

    register_op_schemas._lib = lib
