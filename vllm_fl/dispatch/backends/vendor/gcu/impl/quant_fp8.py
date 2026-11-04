# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU dispatch for ``QuantFP8`` (FP8 input-activation quantization).

On GCU (out-of-tree platform) ``CustomOp.dispatch_forward`` routes to
``forward_oot``.  Blanket-redirecting it to ``forward_cuda`` is wrong here:
``forward_cuda`` funnels the static / per-token / per-tensor modes into
``ops.scaled_fp8_quant``, which ends up in
``torch.ops._C.static_scaled_fp8_quant`` /
``_C.dynamic_scaled_fp8_quant`` / ``_C.dynamic_per_token_scaled_fp8_quant``.
``vllm_fl`` only registers the **schemas** of those ops for ``PrivateUse1``
(``vllm_fl/ops/_C_ops_schemas.py``) and ships no implementation, so every FP8
mode except dynamic per-group quant crashes on GCU.

This patch installs a GCU-aware ``forward_oot``:

* dynamic per-group quant -> Triton ``per_token_group_quant_fp8``
  (already rebased on the GCU Fixed-Grid implementation by
  ``per_token_group_quant_fp8_gcu_patch``);
* static / per-token / per-tensor -> pure-PyTorch ``forward_native``, which
  never touches a ``_C`` CUDA op.

Patching the class attribute on ``QuantFP8`` is sufficient in theory (all
importers share the same class object), but we also re-bind in every known
importer module as a defensive measure, following the same pattern used by
``per_token_group_quant_fp8_gcu_patch``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_patched = False


def _is_gcu_platform() -> bool:
    """True only when vLLM actually selected the GCU platform.

    ``torch.gcu`` may be importable on machines where vLLM runs CUDA/ROCm, and
    the patch below rewrites a shared vLLM class, so both checks are required.
    """
    try:
        from vllm.platforms import current_platform

        return current_platform.device_type == "gcu"
    except Exception:
        return False

# All modules known to import QuantFP8 via
# ``from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8``
# or ``from ...input_quant_fp8 import QuantFP8`` (lazy import).
_IMPORTERS: list[str] = [
    "vllm.model_executor.layers.quantization.input_quant_fp8",  # definition site
    "vllm.compilation.passes.fusion.matcher_utils",
    "vllm.model_executor.kernels.linear.mixed_precision.cutlass",
    "vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel",
    "vllm.model_executor.kernels.linear.scaled_mm.cutlass",
    "vllm.model_executor.kernels.linear.scaled_mm.deep_gemm",
    "vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel",
    "vllm.model_executor.layers.attention.attention",
    "vllm.model_executor.layers.attention.mla_attention",
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w4a8_fp8",
    "vllm.model_executor.layers.quantization.utils.marlin_utils",
]


def _gcu_forward_oot(
    self,
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_triton: bool = False,
):
    """GCU replacement for ``QuantFP8.forward_oot``.

    Mirrors the split vLLM 0.20.2 makes inside ``forward_cuda``, but never
    reaches ``torch.ops._C.*`` (schema-only on ``PrivateUse1``).
    """
    # -- dynamic per-group quant: GCU Triton kernel -------------------------
    if self.is_group_quant and not self.static:
        from vllm.model_executor.layers.quantization.utils import fp8_utils

        assert scale is None, "Dynamic group quantization does not use scale"
        return fp8_utils.per_token_group_quant_fp8(
            x,
            group_size=self.group_size,
            column_major_scales=self.column_major_scales,
            tma_aligned_scales=self.tma_aligned_scales,
            dtype=None,
            use_ue8m0=self.use_ue8m0,
        )

    # -- static / per-token / per-tensor: pure torch, no CUDA ``_C`` op -----
    return self.forward_native(x, scale, scale_ub)


def apply_quant_fp8_gcu_patch() -> None:
    """Install the GCU-aware ``QuantFP8.forward_oot`` and re-bind importers."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available() or not _is_gcu_platform():
        return

    try:
        patched_classes = 0
        for module_name in _IMPORTERS:
            try:
                mod = __import__(module_name, fromlist=["QuantFP8"])
            except ImportError:
                continue
            quant_fp8 = getattr(mod, "QuantFP8", None)
            if quant_fp8 is None:
                continue
            # Class-level attribute change: affects every reference to the
            # same class object, wherever it was imported from.
            if quant_fp8.forward_oot is not _gcu_forward_oot:
                quant_fp8.forward_oot = _gcu_forward_oot
                patched_classes += 1

        _patched = True
        logger.info(
            "Patched QuantFP8.forward_oot -> GCU dispatch "
            "(group=dynamic Triton, else forward_native) in %d modules",
            patched_classes,
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch QuantFP8.forward_oot for GCU: %s",
            exc,
        )
