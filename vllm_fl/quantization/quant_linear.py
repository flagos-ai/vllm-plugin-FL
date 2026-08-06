# Copyright (c) 2025 BAAI. All rights reserved.

from vllm.platforms import PlatformEnum, current_platform


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
        _POSSIBLE_FP8_BLOCK_KERNELS,
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_KERNELS,
    )

    source = _resolve_source_platform()

    if PlatformEnum.OOT not in _POSSIBLE_KERNELS:
        _POSSIBLE_KERNELS[PlatformEnum.OOT] = list(_POSSIBLE_KERNELS.get(source, []))

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

    # compressed-tensors selects its Linear and MoE implementations while the
    # model is being constructed. Install the FL adapters only after the OOT
    # platform has inherited vLLM's native INT8 kernel candidates.
    from .w8a8.linear import register_fl_w8a8_linear_kernel
    from .w8a8.moe import install_fl_w8a8_moe_selector
    from .w8a8.packed import install_packed_w8a8_scheme

    register_fl_w8a8_linear_kernel(_POSSIBLE_INT8_KERNELS)
    install_packed_w8a8_scheme()
    install_fl_w8a8_moe_selector()
