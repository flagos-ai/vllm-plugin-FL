# Adapted for FlagOS from MetaX-MACA/vLLM-metax @ 440ec38e9508 (vLLM 0.32.0).
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch
from packaging.version import InvalidVersion, Version

from vllm.v1.attention.backends.fa_utils import logger
from vllm.platforms import current_platform

from ..ops._kernel_utils import has_cuda_or_composite_kernel

_flaggems_cache_state = {"eligible": None, "writer": None}


def _get_flaggems_cache_writer():
    """Resolve the block-stride-aware bridge, retrying partial imports."""
    state = _flaggems_cache_state
    if state["writer"] is not None:
        return state["writer"]
    if state["eligible"] is None:
        try:
            state["eligible"] = Version(version("flaggems-vllm")) >= Version(
                "0.3.0.dev1"
            )
        except (PackageNotFoundError, InvalidVersion):
            state["eligible"] = False
    if not state["eligible"]:
        return None
    try:
        module = importlib.import_module("flaggems_vllm")
    except (ImportError, OSError) as exc:
        if "partially initialized" not in str(exc):
            state["eligible"] = False
        return None
    if getattr(getattr(module, "__spec__", None), "_initializing", False):
        # vLLM plugin discovery can encounter the bridge's import cycle. The
        # version decision stays cached, but this temporary state is retried.
        return None
    writer = getattr(module, "reshape_and_cache_flash", None)
    if callable(writer):
        state["writer"] = writer
    else:
        state["eligible"] = False
    return state["writer"]


def _flaggems_cache_layout_supported(key_cache, value_cache) -> bool:
    # Bridge 0.3 supports gaps between blocks, but its kernel assumes dense
    # [N, H, D] contents within each block. vLLM 0.32's packed K+V views have
    # head/page gaps as well and must use the fully stride-aware local kernel.
    for cache in (key_cache, value_cache):
        if cache.ndim != 4:
            return False
        heads, dim = cache.shape[2:]
        if (
            cache.stride(3) != 1
            or cache.stride(2) != dim
            or cache.stride(1) != heads * dim
        ):
            return False
    return True


if current_platform.is_out_of_tree():
    from vllm import _custom_ops as ops
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401

    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]

    # MACA doesn't use scheduler metadata (FA3 feature), provide stub
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    if has_cuda_or_composite_kernel("_C_cache_ops::reshape_and_cache_flash"):
        return ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    writer = (
        _get_flaggems_cache_writer()
        if _flaggems_cache_layout_supported(key_cache, value_cache)
        else None
    )
    if writer is not None:
        return writer(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    # Official FlagOS images can omit mcoplib. The empty wheel still defines
    # the native schema, so hasattr(torch.ops...) cannot select an executable
    # implementation. Keep upstream's stride-aware kernel with a C550-safe
    # launch: its default 16 CUDA warps exceed MACA's 512-thread block limit.
    from ..ops.reshape_and_cache import (
        triton_reshape_and_cache_flash,
    )

    return triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    requires_softcap: bool = False,
    kv_cache_block_size: int | None = None,
    supports_fa4_hd256: bool = False,
) -> int | None:
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )

    # Note: In maca this need to be None since
    # metax flash_attn api does not have parameter
    # for `fa_version`.
    return 2


def uses_fa4_hd256_kernel(
    head_size: int | None, head_size_v: int | None = None
) -> bool:
    return False


def is_fa_version_supported(fa_version: int) -> bool:
    return fa_version == 2


def flash_attn_supports_kv_cache_dtype(
    kv_cache_dtype: str = "fp8_e4m3",
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    requires_softcap: bool = False,
    kv_cache_block_size: int | None = None,
    supports_fa4_hd256: bool = False,
) -> bool:
    # Maca does not support quantized kv for fa2 (but soon in fa3 in the future)
    # Currently set to false
    return False


def flash_attn_supports_quant_query_input() -> bool:
    return False


def flash_attn_supports_sinks() -> bool:
    # maca fa2 supports sinks only for gemma model
    return True


def flash_attn_supports_mla():
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True
