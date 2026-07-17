# Copyright (c) 2025 BAAI. All rights reserved.

import os
from vllm.platforms import PlatformEnum, current_platform
from vllm_fl.utils import use_flaggems_op

from .fp8 import FlagGemsFp8BlockScaledMMLinearKernel


FLAGGEMS_FP8_BLOCK_GEMM_OP = "flaggems_fp8_block_gemm"


def _resolve_source_platform() -> PlatformEnum:
    """
    Determine which upstream platform's kernel list to clone for OOT.

    Uses current_platform runtime checks so that:
    - nvidia, metax, musa, etc. (cuda_alike) -> CUDA kernels
    - rocm-alike OOT                         -> ROCM kernels
    - cpu-alike OOT                          -> CPU kernels
    - fallback                               -> CUDA kernels
    """
    if current_platform.is_cuda_alike():
        return PlatformEnum.CUDA
    if current_platform.is_rocm():
        return PlatformEnum.ROCM
    if current_platform.is_cpu():
        return PlatformEnum.CPU
    # Fallback: try CUDA as the most common case
    return PlatformEnum.CUDA


def add_oot_quant_kernel() -> None:
    """
    Register OOT linear kernel classes to be considered in kernel selection.

    Copies the kernel candidate list from the matching upstream platform
    (CUDA / ROCM / CPU) into PlatformEnum.OOT. Each kernel's own
    is_supported() / can_implement() will filter at runtime.
    """
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_KERNELS,
        _POSSIBLE_FP8_BLOCK_KERNELS,
    )
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

    if (
        current_platform.supports_fp8()
        and use_flaggems_op(FLAGGEMS_FP8_BLOCK_GEMM_OP)
        and FlagGemsFp8BlockScaledMMLinearKernel
        not in _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT]
    ):
        _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT].insert(
            0, FlagGemsFp8BlockScaledMMLinearKernel
        )
