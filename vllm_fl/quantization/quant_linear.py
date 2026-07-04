# Copyright (c) 2025 BAAI. All rights reserved.
# All imports are lazy (inside functions) to avoid circular import
# during early plugin registration (vllm_fl.register()).

<<<<<<< HEAD
import logging
logger = logging.getLogger(__name__)
=======
from vllm.platforms import PlatformEnum, current_platform
>>>>>>> upstream/main


def _patch_triton_int8_is_supported() -> None:
    """
    TritonInt8ScaledMMLinearKernel.is_supported() checks is_cuda_alike(),
    which returns False on OOT platforms like HCU.
    Patch it to also accept OOT so the kernel can be selected.
    """
    try:
        from vllm.model_executor.kernels.linear import TritonInt8ScaledMMLinearKernel
        from vllm.platforms import PlatformEnum, current_platform

        original_is_supported = TritonInt8ScaledMMLinearKernel.__dict__.get("is_supported")
        if original_is_supported is None:
            return  # already patched or not classmethod descriptor

        @classmethod  # type: ignore[misc]
        def patched_is_supported(cls, compute_capability=None):
            if current_platform._enum == PlatformEnum.OOT:
                return True, None
            return original_is_supported.__func__(cls, compute_capability)

        TritonInt8ScaledMMLinearKernel.is_supported = patched_is_supported
        logger.debug("TritonInt8ScaledMMLinearKernel.is_supported patched for OOT")
    except Exception as e:
        logger.warning(f"_patch_triton_int8_is_supported failed (non-fatal): {e}")


def add_oot_quant_kernel() -> None:
    """
    Register OOT linear kernel classes to be considered in kernel selection.

    Copies the kernel candidate list from the matching upstream platform
    (CUDA / ROCM / CPU) into PlatformEnum.OOT. Each kernel's own
    is_supported() / can_implement() will filter at runtime.

    All imports are lazy to avoid circular imports during early plugin init.
    """
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_KERNELS,
        _POSSIBLE_FP8_BLOCK_KERNELS,
    )
<<<<<<< HEAD
    from vllm.platforms import PlatformEnum, current_platform

    def _resolve_source_platform() -> PlatformEnum:
        if current_platform.is_cuda_alike():
            return PlatformEnum.CUDA
        if current_platform.is_rocm():
            return PlatformEnum.ROCM
        if current_platform.is_cpu():
            return PlatformEnum.CPU
        return PlatformEnum.CUDA  # fallback

=======
>>>>>>> upstream/main
    source = _resolve_source_platform()

    if PlatformEnum.OOT not in _POSSIBLE_KERNELS:
        _POSSIBLE_KERNELS[PlatformEnum.OOT] = list(
            _POSSIBLE_KERNELS.get(source, [])
        )

    if PlatformEnum.OOT not in _POSSIBLE_INT8_KERNELS:
        _POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = list(
            _POSSIBLE_INT8_KERNELS.get(source, [])
        )

    if PlatformEnum.OOT not in _POSSIBLE_FP8_KERNELS:
        _POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = list(
            _POSSIBLE_FP8_KERNELS.get(source, [])
        )

    if PlatformEnum.OOT not in _POSSIBLE_FP8_BLOCK_KERNELS:
        _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT] = list(
            _POSSIBLE_FP8_BLOCK_KERNELS.get(source, [])
        )

    # Patch TritonInt8 is_supported to accept OOT platform
    _patch_triton_int8_is_supported()

    logger.debug(
        "add_oot_quant_kernel: registered OOT kernels cloned from %s", source
    )
