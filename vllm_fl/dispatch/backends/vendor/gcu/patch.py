# Copyright (c) 2026 BAAI. All rights reserved.

import functools
import logging

import torch

from .impl.apply_moe_activation_gcu import apply_moe_activation_gcu_patch
from .impl.fp8_config import apply_fp8_config_gcu_patch
from .impl.bilinear_pos_embed import apply_bilinear_pos_embed_gcu_patch
from .impl.chunk_delta_h import apply_chunk_delta_h_gcu_patch
from .impl.fused_recurrent_packed_decode import (
    apply_fused_recurrent_packed_decode_gcu_patch,
)
from .impl.moe_align_block_size import apply_moe_align_block_size_gcu_patch
from .impl.moe_sum import apply_moe_sum_gcu_patch
from .impl.per_token_group_quant_fp8 import (
    apply_per_token_group_quant_fp8_gcu_patch,
)
from .impl.quant_fp8 import apply_quant_fp8_gcu_patch
from .impl.fused_moe import apply_fused_moe_triton_kernel_gcu_patch
from .impl.w8a8_block_scaled_mm import apply_w8a8_block_scaled_mm_gcu_patch

logger = logging.getLogger(__name__)
_patches_applied = False


def _patch_gcu_device_properties_gcn_arch() -> None:
    """Add ``gcnArchName`` attribute to GCU device properties class.

    ``torch._inductor.codecache.CacheBase.get_system()`` reads
    ``device_properties.gcnArchName``, but ``torch_gcu._C._GcuDeviceProperties``
    doesn't expose this attribute.  We add it as a class-level attribute
    so inductor's cache hash computation doesn't crash.
    """
    try:
        import torch_gcu
        from torch_gcu._C import _GcuDeviceProperties

        if not hasattr(_GcuDeviceProperties, "gcnArchName"):
            try:
                name = torch.gcu.get_device_name(0)
            except Exception:
                name = "GCU"
            _GcuDeviceProperties.gcnArchName = name
            logger.debug("Patched _GcuDeviceProperties.gcnArchName = %s", name)
    except Exception:
        pass


def _patch_has_triton_for_gcu() -> None:
    """Make torch dynamo recognize Triton kernels on GCU.

    ``torch.utils._triton.has_triton()`` only returns True for cuda/xpu/cpu/mtia
    devices. On GCU it returns False, which causes Dynamo to inline the Triton
    JIT runtime (``JITFunction.run``) instead of treating the kernel as an opaque
    higher-order op (``TritonKernelVariable``). Inlining reaches GCU builtins
    such as ``torch.gcu.current_device()`` / ``_gcu_isInBadFork`` (torch.* ops
    returning non-Tensors), which Dynamo cannot trace, breaking fullgraph
    capture and failing engine init.

    Patching ``has_triton`` to return True on GCU makes Dynamo wrap triton
    kernels as ``triton_kernel_wrapper_mutation`` HOPs, which inductor lowers
    to ``UserDefinedTritonKernel`` and launches via the triton_gcu backend.
    """
    import torch.utils._triton as _triton_mod

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return
    if getattr(_triton_mod.has_triton, "_gcu_patched", False):
        return

    orig_has_triton = _triton_mod.has_triton

    @functools.cache
    def has_triton_gcu() -> bool:
        if orig_has_triton():
            return True
        try:
            import triton  # noqa: F401
        except ImportError:
            return False
        g = getattr(torch, "gcu", None)
        return g is not None and g.is_available()

    has_triton_gcu._gcu_patched = True
    _triton_mod.has_triton = has_triton_gcu

    # Also update the module-level binding in torch._dynamo.utils (imported via
    # `from torch.utils._triton import has_triton` at module load).
    try:
        import torch._dynamo.utils as _dynamo_utils
        _dynamo_utils.has_triton = has_triton_gcu
    except Exception:
        pass

    logger.info(
        "Patched torch.utils._triton.has_triton for GCU (triton HOP support)"
    )


def apply_gcu_patches() -> None:
    """Apply all GCU-specific kernel / model monkey-patches."""
    global _patches_applied
    if _patches_applied:
        return
    
    apply_bilinear_pos_embed_gcu_patch()
    apply_chunk_delta_h_gcu_patch()
    apply_fused_recurrent_packed_decode_gcu_patch()
    apply_moe_activation_gcu_patch()
    apply_moe_align_block_size_gcu_patch()
    apply_moe_sum_gcu_patch()
    apply_per_token_group_quant_fp8_gcu_patch()
    apply_fp8_config_gcu_patch()
    apply_quant_fp8_gcu_patch()
    apply_fused_moe_triton_kernel_gcu_patch()
    apply_w8a8_block_scaled_mm_gcu_patch()

    # Inductor compatibility patches (gcnArchName etc.)
    _patch_gcu_device_properties_gcn_arch()
    _patch_has_triton_for_gcu()

    _patches_applied = True


def apply_op_kernel_patches() -> None:
    """Alias kept for callers that use the older name."""
    apply_gcu_patches()
