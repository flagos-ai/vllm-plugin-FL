# Copyright (c) 2025 BAAI. All rights reserved.
#
# Register torch.ops._C op schemas so that vllm compilation passes can
# reference them for pattern matching even when the native vllm._C extension
# is not compiled for this platform.

import logging
from contextlib import suppress

import torch

logger = logging.getLogger(__name__)

# These optional kernels use schema presence as an upstream capability probe.
# They are not needed by compilation pattern imports. Defining only a schema
# would suppress the real Triton/Torch fallback despite having no CUDA kernel.
_OPTIONAL_NATIVE_ONLY_OPS = frozenset(
    {
        "dsv3_fused_a_gemm",
        "fused_gdn_decode_post_conv_mtp",
        "fused_kda_decode",
        "fused_kda_chunk",
        "hadacore_transform",
        "kimi_k3_attn_res",
    }
)


def _unavailable_cuda_view(tensor):
    raise RuntimeError(
        "vLLM empty builds do not provide a mapped CUDA view of CPU memory; "
        "this backend must provide an implementation or avoid mapped CPU buffers"
    )


def _unavailable_weak_ref(tensor):
    raise RuntimeError(
        "CUDA graph weak references require the vllm_fl._C extension; "
        "install the plugin with VLLM_VENDOR=cuda"
    )


# A capable GPU does not supply the absent CUTLASS kernels. Keep upstream
# selection on implementations that an EMPTY wheel actually provides.
_QUERY_OP_IMPLS = [
    ("cutlass_scaled_mm_supports_fp8", lambda cap: False),
    ("cutlass_scaled_mm_supports_block_fp8", lambda cap: False),
    ("cutlass_group_gemm_supported", lambda cap: False),
    ("cutlass_scaled_mm_supports_fp4", lambda cap: False),
    ("weak_ref_tensor", _unavailable_weak_ref),
    ("get_cuda_view_from_cpu_tensor", _unavailable_cuda_view),
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


# Ops that need a CUDA dispatch because vLLM calls them directly
# (not routed through FL's call_op) and only has _C kernel + torch fallback
# gated behind is_cuda checks.
_CUDA_FALLBACK_IMPLS = [
    ("apply_repetition_penalties_", _apply_repetition_penalties_impl),
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
        if op_name in _OPTIONAL_NATIVE_ONLY_OPS:
            continue
        overload = full_name.split(".")[1] if "." in full_name else "default"
        if hasattr(torch.ops._C, op_name) and hasattr(
            getattr(torch.ops._C, op_name), overload
        ):
            continue
        try:
            lib.define(schema)
        except Exception as e:
            raise RuntimeError(
                f"Failed to register the vLLM 0.32 schema _C::{full_name}"
            ) from e

    for name, fn in _QUERY_OP_IMPLS:
        with suppress(RuntimeError):
            lib.impl(name, fn, "CompositeImplicitAutograd")

    for name, fn in _CUDA_FALLBACK_IMPLS:
        with suppress(RuntimeError):
            lib.impl(name, fn, "CUDA")

    register_op_schemas._lib = lib
