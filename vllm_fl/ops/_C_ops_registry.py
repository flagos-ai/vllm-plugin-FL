# Copyright (c) 2025 BAAI. All rights reserved.
#
# Register namespaced vLLM op schemas so that vllm compilation passes can
# reference them for pattern matching even when the native vllm._C extension
# is not compiled for this platform.

import logging
import os
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


def _is_metax_backend() -> bool:
    """True when vLLM resolved the MetaX (maca) platform.

    On that backend ``mcoplib._C`` already owns ``torch.ops._C``.  Importing
    vLLM's own ``vllm._C`` .so would dlopen a second copy of the same operator
    names and raise a duplicate-registration ``c10::Error``, so every
    ``import vllm._C`` below is gated on this check.
    """
    # Non-reentrant by construction: this function is reached from the
    # ``vllm.platform_plugins`` entry point (vllm_fl:register) while
    # vllm.platforms is still resolving ``current_platform``.  A plain
    # ``from vllm.platforms import current_platform`` here would re-enter that
    # resolution, recursing until RecursionError and leaving vLLM with
    # UnspecifiedPlatform.  Consult only an *already resolved* platform and
    # otherwise fall back to a hardware/environment probe.
    import importlib.util
    import sys

    resolved = getattr(sys.modules.get("vllm.platforms"), "_current_platform", None)
    if resolved is not None:
        # Look the hook up on the class: ``Platform.__getattr__`` logs a
        # "does not have this attribute" warning via ``vllm.distributed``, which
        # is a circular import while the plugin is still registering.
        is_metax = getattr(type(resolved), "is_metax", None)
        if callable(is_metax):
            try:
                return bool(is_metax(resolved))
            except Exception:
                pass
    if os.environ.get("GEMS_VENDOR", "").strip().lower() == "metax":
        return True
    try:
        return importlib.util.find_spec("mcoplib") is not None
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
    """Register missing vLLM schemas in their upstream namespaces."""
    if getattr(register_op_schemas, "_libs", None) is not None:
        return

    # Bail out before touching vllm._C when the active backend already owns _C.
    # Note: _C_ops_present() is intentionally NOT checked here — the per-op
    # hasattr guard at schema-definition time already skips already-registered ops,
    # and an early exit here would prevent _libs from being set, breaking tests
    # that rely on the attribute to confirm initialization.
    if _is_metax_backend():
        logger.debug(
            "Skipping vllm._C import: _C ops are already provided by the active backend"
        )
        register_op_schemas._libs = {}
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

    libs = {}

    for namespace, schema in schemas:
        full_name = schema.split("(")[0]
        op_name = full_name.split(".")[0]
        if namespace == "_C" and op_name in _OPTIONAL_NATIVE_ONLY_OPS:
            continue
        if namespace not in libs:
            libs[namespace] = torch.library.Library(namespace, "FRAGMENT")
        lib = libs[namespace]
        ops = getattr(torch.ops, namespace)
        overload = full_name.split(".")[1] if "." in full_name else "default"
        if hasattr(ops, op_name) and hasattr(getattr(ops, op_name), overload):
            continue
        try:
            lib.define(schema)
        except Exception as e:
            raise RuntimeError(
                f"Failed to register the vLLM 0.32 schema {namespace}::{full_name}"
            ) from e

    c_lib = libs.get("_C")
    if c_lib is not None:
        for name, fn in _QUERY_OP_IMPLS:
            with suppress(RuntimeError):
                c_lib.impl(name, fn, "CompositeImplicitAutograd")

        for name, fn in _CUDA_FALLBACK_IMPLS:
            with suppress(RuntimeError):
                c_lib.impl(name, fn, "CUDA")

    # Library registrations are removed when the Library is garbage-collected.
    register_op_schemas._libs = libs
