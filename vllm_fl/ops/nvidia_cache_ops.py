# Copyright (c) 2026 BAAI. All rights reserved.

"""Executable CUDA cache operations for the kernel-free vLLM 0.32 wheel.

Keep imports of platform-dependent Triton/FlagGems code inside the kernels:
registration runs during platform discovery, before current_platform exists.
"""

import torch

from vllm_fl.nvidia_compat import has_native_vllm_kernels

_CACHE_SCHEMAS = {
    "reshape_and_cache_flash": (
        "reshape_and_cache_flash(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, "
        "Tensor k_scale, Tensor v_scale) -> ()"
    ),
    "concat_and_cache_mla": (
        "concat_and_cache_mla(Tensor kv_c, Tensor k_pe, Tensor! kv_cache, "
        "Tensor slot_mapping, str kv_cache_dtype, Tensor scale) -> ()"
    ),
}
_NATIVE_DTYPES = {"auto", "float16", "half", "bfloat16", "float32", "float"}
_FP8_DTYPES = {
    "fp8": "fp8",
    "fp8_e4m3": "fp8e4m3",
    "fp8_e5m2": "fp8e5m2",
}
_lib: torch.library.Library | None = None


def _reshape_and_cache_flash(
    key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
) -> None:
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    # Upstream's Triton writer uses the platform's E4M3 format for every
    # quantized dtype. Do not silently treat E5M2/NVFP4 as that format.
    if kv_cache_dtype not in _NATIVE_DTYPES | {"fp8", "fp8_e4m3"}:
        raise NotImplementedError(
            "NVIDIA empty reshape_and_cache_flash supports native dtypes and "
            f"FP8 E4M3, not {kv_cache_dtype}"
        )
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("Flash attention caches must have [B, N, H, D] views")
    if key_cache.stride() != value_cache.stride():
        raise ValueError("Key/value cache views must have identical strides")
    if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("Cache head dimensions must be contiguous")
    if key.stride(-1) != 1 or key.stride(-2) != key.shape[-1]:
        raise ValueError("Key token rows must be contiguous")
    if value.stride(-1) != 1 or value.stride(-2) != value.shape[-1]:
        raise ValueError("Value token rows must be contiguous")
    if slot_mapping.numel() == 0:
        return
    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def _concat_and_cache_mla(
    kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
) -> None:
    from flag_gems import concat_and_cache_mla

    if kv_cache_dtype in _NATIVE_DTYPES:
        gems_dtype = "auto"
    elif kv_cache_dtype in _FP8_DTYPES:
        gems_dtype = _FP8_DTYPES[kv_cache_dtype]
        # FlagGems stores encoded FP8 bytes. Reinterpret typed FP8 caches
        # without copying, preserving the cache's block/entry strides.
        if kv_cache.dtype != torch.uint8:
            if kv_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise ValueError("FP8 MLA cache must use uint8 or FP8 storage")
            kv_cache = kv_cache.view(torch.uint8)
    else:
        raise NotImplementedError(
            f"NVIDIA empty MLA cache does not support {kv_cache_dtype}"
        )
    if kv_cache.ndim != 3:
        raise ValueError("MLA cache must have shape [B, N, kv_lora_rank + rope_dim]")
    if any(tensor.stride(-1) != 1 for tensor in (kv_c, k_pe, kv_cache)):
        raise ValueError("MLA cache and input head dimensions must be contiguous")
    rank = kv_c.shape[-1]
    block_size = min(rank, 512)
    if block_size <= 0 or block_size & (block_size - 1):
        raise NotImplementedError(
            "FlagGems MLA cache requires a power-of-two kv_lora_rank below 512"
        )
    if any(
        tensor.device != kv_c.device for tensor in (k_pe, kv_cache, slot_mapping, scale)
    ):
        raise ValueError("MLA cache tensors and scale must be on the same CUDA device")
    if slot_mapping.numel() == 0:
        return
    concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, gems_dtype, scale)


def register_nvidia_cache_ops() -> bool:
    """Register real CUDA kernels; never replace an installed native build."""
    global _lib
    if has_native_vllm_kernels():
        return False
    if _lib is not None:
        return True
    lib = torch.library.Library("_C_cache_ops", "FRAGMENT")
    kernels = {
        "reshape_and_cache_flash": _reshape_and_cache_flash,
        "concat_and_cache_mla": _concat_and_cache_mla,
    }
    for name, kernel in kernels.items():
        if not hasattr(torch.ops._C_cache_ops, name):
            lib.define(_CACHE_SCHEMAS[name])
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"_C_cache_ops::{name}", "CUDA"
        ):
            lib.impl(name, kernel, "CUDA")
    _lib = lib
    return True
