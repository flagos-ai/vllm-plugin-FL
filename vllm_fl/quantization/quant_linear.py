# Copyright (c) 2025 BAAI. All rights reserved.

from vllm.platforms import PlatformEnum, current_platform

from .fp8 import FlagGemsFp8BlockScaledMMLinearKernel
from vllm_fl.utils import use_flaggems_op

FLAGGEMS_FP8_BLOCK_GEMM_OP = "flaggems_fp8_block_gemm"


def _merge_candidates(registry, candidates) -> None:
    destination = registry.setdefault(PlatformEnum.OOT, [])
    for kernel in candidates:
        if kernel not in destination:
            destination.append(kernel)


def _resolve_source_platform() -> PlatformEnum | None:
    """
    Determine which upstream platform's kernel list to clone for OOT.

    Uses current_platform runtime checks so that:
    - nvidia, metax, musa, etc. (cuda_alike) -> CUDA kernels
    - rocm-alike OOT                         -> ROCM kernels
    - cpu-alike OOT                          -> CPU kernels
    - unknown                                -> no cloned kernels
    """
    if current_platform.is_rocm():
        return PlatformEnum.ROCM
    if current_platform.is_cpu():
        return PlatformEnum.CPU
    if current_platform.is_cuda_alike():
        return PlatformEnum.CUDA
    # Copying CUDA kernels to an unrelated platform can pass registration and
    # fail much later during model loading. Unknown platforms must provide
    # explicit FL kernels instead.
    return None


def add_oot_quant_kernel() -> None:
    """
    Register OOT linear kernel classes to be considered in kernel selection.

    Copies the kernel candidate list from the matching upstream platform
    (CUDA / ROCM / CPU) into PlatformEnum.OOT. Each kernel's own
    is_supported() / can_implement() will filter at runtime.
    """
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_FP8_BLOCK_KERNELS,
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_KERNELS,
    )

    source = _resolve_source_platform()
    source_kernels = _POSSIBLE_KERNELS.get(source, []) if source else []
    source_int8 = _POSSIBLE_INT8_KERNELS.get(source, []) if source else []
    source_fp8 = _POSSIBLE_FP8_KERNELS.get(source, []) if source else []
    source_fp8_block = _POSSIBLE_FP8_BLOCK_KERNELS.get(source, []) if source else []

    _merge_candidates(_POSSIBLE_KERNELS, source_kernels)
    _merge_candidates(_POSSIBLE_INT8_KERNELS, source_int8)
    _merge_candidates(_POSSIBLE_FP8_KERNELS, source_fp8)
    _merge_candidates(_POSSIBLE_FP8_BLOCK_KERNELS, source_fp8_block)

    from .wna16.linear import register_fl_wna16_linear_kernel

    register_fl_wna16_linear_kernel(_POSSIBLE_KERNELS)

    if (
        current_platform.supports_fp8()
        and use_flaggems_op(FLAGGEMS_FP8_BLOCK_GEMM_OP)
        and FlagGemsFp8BlockScaledMMLinearKernel
        not in _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT]
    ):
        _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT].insert(
            0, FlagGemsFp8BlockScaledMMLinearKernel
        )

    from .compressed_tensors import register_compressed_tensors_oot

    register_compressed_tensors_oot()
