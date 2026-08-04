# Copyright (c) 2025 BAAI. All rights reserved.
#
# Register torch.ops._C op schemas so that vllm compilation passes can
# reference them for pattern matching even when the native vllm._C extension
# is not compiled for this platform.

import importlib
import logging

import torch

logger = logging.getLogger(__name__)

_LEGACY_C_EXTENSION = "vllm._C"
_STABLE_C_EXTENSION = "vllm._C_stable_libtorch"


def _import_extension(module_name: str) -> None:
    importlib.import_module(module_name)


def load_vllm_native_extensions() -> bool:
    """Load native vLLM extensions before defining fallback ``_C`` schemas.

    vLLM wheels may split native operators between ``vllm._C`` and
    ``vllm._C_stable_libtorch`` on any supported platform. Some versions ship
    only one of them, so each extension must be attempted independently.
    """
    module_names = (
        _LEGACY_C_EXTENSION,
        _STABLE_C_EXTENSION,
    )
    loaded = False
    for module_name in module_names:
        try:
            _import_extension(module_name)
        except (ImportError, OSError) as exc:
            logger.debug("Could not import %s: %s", module_name, exc)
        else:
            logger.debug("Loaded native vLLM extension %s", module_name)
            loaded = True
    return loaded


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

    if load_vllm_native_extensions():
        return

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

    register_op_schemas._lib = lib
