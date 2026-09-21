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


# Ops that need a CUDA dispatch because vLLM calls them directly
# (not routed through FL's call_op) and only has _C kernel + torch fallback
# gated behind is_cuda checks.
_CUDA_FALLBACK_IMPLS = [
    ("apply_repetition_penalties_", _apply_repetition_penalties_impl),
]


def _is_metax_backend() -> bool:
    """True when vLLM resolved the MetaX (maca) platform.

    On that backend ``mcoplib._C`` already owns ``torch.ops._C``.  Importing
    vLLM's own ``vllm._C`` .so would dlopen a second copy of the same operator
    names and raise a duplicate-registration ``c10::Error``, so every
    ``import vllm._C`` below is gated on this check.
    """
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    is_metax = getattr(current_platform, "is_metax", None)
    if not callable(is_metax):
        return False
    try:
        return bool(is_metax())
    except Exception:
        return False


def _C_ops_present() -> bool:
    """True when torch.ops._C already carries ops this module would define."""
    ops = getattr(torch.ops, "_C", None)
    if ops is None:
        return False
    names = [name for name, _ in _QUERY_OP_IMPLS]
    names += [name for name, _ in _CUDA_FALLBACK_IMPLS]
    return any(hasattr(ops, name) for name in names)


def register_op_schemas():
    """Register _C op schemas if not already present."""
    if getattr(register_op_schemas, "_lib", None) is not None:
        return

    # Bail out before touching vllm._C when the active backend already owns _C.
    if _is_metax_backend() or _C_ops_present():
        logger.debug(
            "Skipping vllm._C import: _C ops are already provided by the active backend"
        )
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

    register_op_schemas._lib = lib
