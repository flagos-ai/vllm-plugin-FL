# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix for QuantFP8.forward_oot → forward_cuda.

On GCU (out-of-tree platform), ``CustomOp.dispatch_forward`` routes to
``forward_oot``, which by default calls ``forward_native`` (a pure-PyTorch
implementation).  However, the CUDA implementation (``forward_cuda``) works
correctly on GCU hardware and offers better performance.

This patch replaces ``QuantFP8.forward_oot`` with ``QuantFP8.forward_cuda``
so that the CUDA fast-path is used on GCU devices.

Patching only the class attribute on ``QuantFP8`` is sufficient in theory
(since all importers share the same class object), but we also re-bind in
every known importer module as a defensive measure, following the same
pattern used by ``per_token_group_quant_fp8_gcu_patch``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False

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


def apply_quant_fp8_gcu_patch() -> None:
    """Patch QuantFP8.forward_oot to delegate to forward_cuda on GCU
    and re-bind in every known importer module.
    """
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    try:
        for module_name in _IMPORTERS:
            try:
                mod = __import__(module_name, fromlist=["QuantFP8"])
            except ImportError:
                continue
            if hasattr(mod, "QuantFP8"):
                # Replace forward_oot → forward_cuda on the class itself.
                # This is a class-level attribute change so it affects all
                # references to the same class object everywhere.
                mod.QuantFP8.forward_oot = mod.QuantFP8.forward_cuda

        _patched = True
        logger.info(
            "Patched QuantFP8.forward_oot → forward_cuda for GCU "
            "in %d modules",
            len(_IMPORTERS),
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch QuantFP8.forward_oot for GCU: %s",
            exc,
        )
