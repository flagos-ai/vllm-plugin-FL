# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix: enable the Torch FP8 ScaledMM linear kernel for non-block FP8.

Upstream ``TorchFP8ScaledMMLinearKernel.is_supported`` requires compute
capability >= 89 (Ada+) because ``torch._scaled_mm`` is only fused there, and
``_POSSIBLE_FP8_KERNELS`` is cloned from CUDA for ``PlatformEnum.OOT``
(``vllm_fl/quantization/quant_linear.py``).  GCU reports capability 40, so any
FP8 checkpoint whose scales are per-tensor / per-channel (e.g.
``quant_method: fp8`` + ``activation_scheme: static``) fails while creating
weights:

    ValueError: Failed to find a kernel that can implement the ScaledMM linear
    layer. Reasons:
    MarlinFP8ScaledMMLinearKernel requires CUDA..
    FlashInferFP8ScaledMMLinearKernel requires CUDA..
    CutlassFP8ScaledMMLinearKernel requires CUDA..
    PerTensorTorchFP8ScaledMMLinearKernel requires compute capability 89 ...
    ChannelWiseTorchFP8ScaledMMLinearKernel requires compute capability 89 ...

``torch._scaled_mm`` *is* implemented on GCU (topsaten) and accepts exactly the
call shape vLLM uses: padded M, column-major ``B`` (``process_weights_after_loading``
already does ``weight = weight.t()``) and an optional bias.  Verified on device
for M=1/17/32, K=2048, N=512, with and without bias, against a dequantized
reference.

Block-wise (group) FP8 is untouched: it still resolves to
``FlagGemsFp8BlockScaledMMLinearKernel`` from ``_POSSIBLE_FP8_BLOCK_KERNELS``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def _is_gcu_platform() -> bool:
    """True only when vLLM actually selected the GCU platform.

    Both patches below rewrite shared vLLM kernel classes, so the GCU guard has
    to include the platform check (``torch.gcu`` alone is not enough).
    """
    try:
        from vllm.platforms import current_platform

        return current_platform.device_type == "gcu"
    except Exception:
        return False


def _gcu_is_supported(cls, compute_capability: int | None = None):
    """``TorchFP8ScaledMMLinearKernel.is_supported`` without the Ada gate.

    The upstream body is kept (platform check) but the ``capability >= 89``
    requirement is dropped: GCU implements ``torch._scaled_mm`` in its own
    backend and never reports an Ada-class capability.
    """
    from vllm.platforms import current_platform

    if not (current_platform.is_cuda_alike() or current_platform.is_cpu()):
        return False, "requires ROCm, CUDA or CPU."
    return True, None


def _gcu_get_output_padding(self) -> None:
    """GCU: never pad the activation to 17 tokens.

    The padding only exists to make ``torch._scaled_mm`` faster for batch > 16,
    but it materialises ``F.pad`` on an **fp8** tensor.  FlagGems' generated pad
    kernel lowers ``arith.truncf f32 -> f8E4M3FN`` into ``llvm.fptrunc`` with an
    i8 result, which triton_gcu400 rejects:

        error: 'llvm.fptrunc' op result #0 must be floating point LLVM type ...
        %11 = "arith.truncf"(%arg25) : (f32) -> f8E4M3FN

    Padding is a pure performance hint, so dropping it is safe.
    """
    return None


def apply_fp8_scaled_mm_gcu_patch() -> None:
    """Allow per-tensor / per-channel Torch FP8 scaled-MM kernels on GCU."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available() or not _is_gcu_platform():
        return

    if not hasattr(torch, "_scaled_mm"):
        logger.warning(
            "Skipped FP8 ScaledMM kernel patch: torch._scaled_mm is not available"
        )
        return

    try:
        from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
            TorchFP8ScaledMMLinearKernel,
        )

        TorchFP8ScaledMMLinearKernel.is_supported = classmethod(_gcu_is_supported)
        TorchFP8ScaledMMLinearKernel.get_output_padding = _gcu_get_output_padding
        _patched = True
        logger.info(
            "Patched TorchFP8ScaledMMLinearKernel.is_supported for GCU "
            "(per-tensor / per-channel FP8 linear via torch._scaled_mm)"
        )
    except Exception as exc:  # pragma: no cover - deferred import
        logger.debug("fp8 scaled_mm patch deferred (will retry): %s", exc)


__all__ = ["apply_fp8_scaled_mm_gcu_patch"]
