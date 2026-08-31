"""Ported verbatim from the T-Head vendor fork: vllm/utils/ppu_deep_gemm.py.
Only change: VLLM_PPU_DENSE_BACKEND env accessed via getattr (absent upstream).
"""
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

import ast
import functools
import importlib
import json
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, NoReturn

import torch

import vllm.envs as envs
from vllm.logger import logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv


best_configs = None
MAX_DECODE_BS = 1025
CANDIDATE_Ms = [1, 2, 4, 8] + list(range(16, MAX_DECODE_BS, 16))


@functools.cache
def get_deep_gemm_best_configs():
    config_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "ppu_deepgemm_configs"
    )
    if not os.path.exists(config_dir):
        return None
    configs = os.listdir(config_dir)

    config_dict_in_all = dict()

    device_name = current_platform.get_device_name().replace(" ", "_")
    device_name_signature = f",device_name={device_name}-"
    logger.info_once("ppu deepgemm start loading config......")

    for config in configs:
        if device_name_signature not in config:
            logger.debug_once(f"ppu deepgemm skipping device_name_signature {config}")
            continue
        logger.info_once(f"ppu deepgemm loading device_name_signature {config}")

        config_file_path = os.path.join(config_dir, config)
        config_dict = dict()

        try:
            with open(config_file_path) as f:
                config_dict = json.load(f)
            config_dict = dict(
                [((x["M"], x["N"], x["K"], x["num_groups"]), x) for x in config_dict]
            )
        except FileNotFoundError:
            logger.warning_once(f"Empty config found in {config_file_path}.")

        if config_dict:
            config_dict_in_all.update(config_dict)

    if not config_dict_in_all:
        logger.warning_once("No ppu deep gemm config found")

    return config_dict_in_all


@functools.cache
def get_deep_gemm_config(M, N, K, num_groups):
    best_configs = get_deep_gemm_best_configs()
    if best_configs is not None:
        # find neariest config
        config = None
        if (M, N, K, num_groups) in best_configs:
            config = ast.literal_eval(best_configs[(M, N, K, num_groups)]["config"])
            logger.debug_once("directly hit best config")
        else:

            def find_closest(lst, x):
                # As a dividing line of P and D
                if M > MAX_DECODE_BS:
                    return -1
                else:
                    return min(lst, key=lambda y: abs(y - x))

            candidate_Ms = CANDIDATE_Ms
            M_candidate = find_closest(candidate_Ms, M)
            if (M_candidate, N, K, num_groups) in best_configs:
                config = ast.literal_eval(best_configs[(M_candidate, N, K, num_groups)]["config"])
                logger.debug_once("find closest best config")
        if config is None:
            logger.debug_once(
                f"No config matched for M={M} N={N} K={K} num_groups = {num_groups}"
            )
            return None

        num_sms = config["num_min_sms"]
        block_m = config["best_block_m"]
        block_n = config["best_block_n"]
        block_k = config["block_k"]
        warp_m = config["warp_m"]
        warp_n = config["warp_n"]
        num_stages = config["best_num_stages"]
        smem_config = config["best_smem_config"]
        return (
            num_sms,
            block_m,
            block_n,
            block_k,
            warp_m,
            warp_n,
            num_stages,
            smem_config,
        )
    else:
        return None

class DeepGemmQuantScaleFMT(Enum):
    # Float32 scales in Float32 tensor
    FLOAT32 = 0
    # Compute float32 scales and ceil the scales to UE8M0.
    # Keep the scales in Float32 tensor.
    FLOAT32_CEIL_UE8M0 = 1
    # Compute float32 scales and ceil the scales to UE8M0.
    # Pack the scales into a int32 tensor where each int32
    # element contains 4 scale values.
    UE8M0 = 2

    @classmethod
    def init_oracle_cache(cls) -> None:
        """Initialize the oracle decision and store it in the class cache"""
        cached = getattr(cls, "_oracle_cache", None)
        if cached is not None:
            return

        use_e8m0 = (
            envs.VLLM_USE_DEEP_GEMM_E8M0
            and is_deep_gemm_supported()
            and (_fp8_gemm_nt_impl is not None)
        )
        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return

        cls._oracle_cache = (  # type: ignore
            cls.UE8M0
            if current_platform.is_device_capability_family(100)
            else cls.FLOAT32_CEIL_UE8M0
        )

    @classmethod
    def from_oracle(cls) -> "DeepGemmQuantScaleFMT":
        """Return the pre-initialized oracle decision"""
        cached = getattr(cls, "_oracle_cache", None)
        assert cached is not None, "DeepGemmQuantScaleFMT oracle cache not initialized"
        return cached


@functools.cache
def is_deep_gemm_supported() -> bool:
    """Return `True` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    # PlatformFL has no is_ppu(); thead vendor == PPU hardware.
    is_supported_arch = getattr(current_platform, 'vendor_name', None) == 'thead'
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    """Return `True` if vLLM is configured to use DeepGEMM "
    "E8M0 scale on a Hopper or Blackwell-class GPU.
    """
    logger.info_once(
        "PPU DeepGEMM do not support E8M0", scope="local"
    )
    return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "PPU DeepGEMM backend is not available or outdated. Please install or "
        "update the `deep_gemm` to a newer version."
    )

# dense gemm and gourp gemm
_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_einsum_impl: Callable[..., Any] | None = None
_fp8_grouped_nopad_impl: Callable[..., Any] | None = None
_grouped_masked_impl: Callable[..., Any] | None = None
_int8_gemm_nt_impl: Callable[..., Any] | None = None
_int8_grouped_nopad_impl: Callable[..., Any] | None = None
_int8_grouped_masked_impl: Callable[..., Any] | None = None
_bf16_grouped_nopad_impl: Callable[..., Any] | None = None
_bf16_grouped_masked_impl: Callable[..., Any] | None = None
_fp4_grouped_nopad_impl: Callable[..., Any] | None = None
_fp4_grouped_masked_impl: Callable[..., Any] | None = None

# mqa logtiss
_fp8_mqa_logits_impl: Callable[..., Any] | None = None
_fp8_paged_mqa_logits_impl: Callable[..., Any] | None = None
_int8_mqa_logits_impl: Callable[..., Any] | None = None
_int8_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_paged_mqa_logits_metadata_impl: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None

# auxiliary
_get_mn_major_tma_aligned_tensor_impl: Callable[..., Any] | None = None
_get_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = None
_transform_sf_into_required_layout_impl: Callable[..., Any] | None = None
_set_compile_mode_impl: Callable[..., Any] | None = None
_get_compile_mode_impl: Callable[..., Any] | None = None


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""

    global _fp8_gemm_nt_impl, _fp8_einsum_impl
    global _grouped_masked_impl, _fp8_grouped_nopad_impl
    global _int8_gemm_nt_impl, _int8_grouped_nopad_impl, _int8_grouped_masked_impl
    global _bf16_grouped_nopad_impl, _bf16_grouped_masked_impl
    global _fp4_grouped_nopad_impl, _fp4_grouped_masked_impl
    global _set_compile_mode_impl, _get_compile_mode_impl
    global _fp8_mqa_logits_impl, _fp8_paged_mqa_logits_impl
    global _int8_mqa_logits_impl, _int8_paged_mqa_logits_impl
    global _get_paged_mqa_logits_metadata_impl
    global _tf32_hc_prenorm_gemm_impl
    global _get_mn_major_tma_aligned_tensor_impl
    global _get_mk_alignment_for_contiguous_layout_impl
    global _transform_sf_into_required_layout_impl

    # fast path
    if (
        _fp8_gemm_nt_impl is not None
        or _fp8_einsum_impl is not None
        or _grouped_masked_impl is not None
        or _fp8_grouped_nopad_impl is not None
        or _int8_gemm_nt_impl is not None
        or _int8_grouped_nopad_impl is not None
        or _int8_grouped_masked_impl is not None
        or _bf16_grouped_nopad_impl is not None
        or _bf16_grouped_masked_impl is not None
        or _fp4_grouped_nopad_impl is not None
        or _fp4_grouped_masked_impl is not None
        or _fp8_mqa_logits_impl is not None
        or _fp8_paged_mqa_logits_impl is not None
        or _int8_mqa_logits_impl is not None
        or _int8_paged_mqa_logits_impl is not None
        or _get_paged_mqa_logits_metadata_impl is not None
        or _tf32_hc_prenorm_gemm_impl is not None
        or _get_mk_alignment_for_contiguous_layout_impl is not None
        or _transform_sf_into_required_layout_impl is not None
    ):
        return

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm"
        )

    _dg = importlib.import_module("deep_gemm")

    _fp8_gemm_nt_impl = getattr(_dg, "gemm_fp8_fp8_bf16_nt", None)
    _fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)
    _fp8_grouped_nopad_impl = getattr(_dg, "m_grouped_gemm_fp8_fp8_bf16_nt_nopad", None)
    _grouped_masked_impl = getattr(_dg, "fp8_m_grouped_gemm_nt_masked", None)
    _fp8_mqa_logits_impl = getattr(_dg, "fp8_mqa_logits", None)
    _fp8_paged_mqa_logits_impl = getattr(_dg, "fp8_paged_mqa_logits", None)
    _int8_mqa_logits_impl = getattr(_dg, "int8_mqa_logits", None)
    _int8_paged_mqa_logits_impl = getattr(_dg, "int8_paged_mqa_logits", None)
    _get_paged_mqa_logits_metadata_impl = getattr(
        _dg, "get_paged_mqa_logits_metadata", None
    )
    _tf32_hc_prenorm_gemm_impl = getattr(_dg, "tf32_hc_prenorm_gemm", None)
    _get_mn_major_tma_aligned_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_tensor", None
    )
    _get_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_mk_alignment_for_contiguous_layout", None
    )
    _transform_sf_into_required_layout_impl = getattr(
        _dg, "transform_sf_into_required_layout", None
    )
    _int8_gemm_nt_impl = getattr(_dg, "gemm_int8_int8_bf16_nt", None)
    _int8_grouped_nopad_impl = getattr(
        _dg, "m_grouped_gemm_int8_int8_bf16_nt_nopad", None
    )
    _int8_grouped_masked_impl = getattr(
        _dg, "m_grouped_gemm_int8_int8_bf16_nt_masked", None
    )
    _bf16_grouped_nopad_impl = getattr(
        _dg, "m_grouped_gemm_bf16_bf16_bf16_nt_nopad", None
    )
    _bf16_grouped_masked_impl = getattr(
        _dg, "m_grouped_gemm_bf16_bf16_bf16_nt_masked", None
    )
    _fp4_grouped_nopad_impl = getattr(
        _dg, "m_grouped_gemm_fp4_fp4_bf16_nt_nopad", None
    )
    _fp4_grouped_masked_impl = getattr(
        _dg, "m_grouped_gemm_fp4_fp4_bf16_nt_masked", None
    )
    _get_compile_mode_impl = getattr(_dg, "get_compile_mode", None)
    _set_compile_mode_impl = getattr(_dg, "set_compile_mode", None)
    DeepGemmQuantScaleFMT.init_oracle_cache()


def get_num_sms() -> int:
    _lazy_init()
    _dg = importlib.import_module("deep_gemm")
    return int(_dg.get_num_sms())


@functools.cache
def get_mk_alignment_for_contiguous_layout(
    is_blockwise: bool | None = False,
    is_mxfp4: bool | None = False,
) -> list[int]:
    # ppu bf16 and int8 don't need 128 align
    # ppu mxfp4 need 16 align
    if is_mxfp4:
        block = 16
    elif is_blockwise:
        block = 128
    else:
        block = 1
    return [1, block]


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for DeepGEMM's get_mn_major_tma_aligned_tensor"""
    _lazy_init()
    if _get_mn_major_tma_aligned_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_tensor_impl(x)


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_gemm_nt_impl(*args, **kwargs)


def m_grouped_fp8_gemm_nt_nopad(*args, **kwargs):
    _lazy_init()
    if _fp8_grouped_nopad_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_grouped_nopad_impl(*args, **kwargs)


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_masked_impl(*args, **kwargs)


def int8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _int8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    return _int8_gemm_nt_impl(*args, **kwargs)

def fp8_einsum(*args, **kwargs):
    _lazy_init()
    if _fp8_einsum_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_einsum_impl(*args, **kwargs)

def m_grouped_int8_gemm_nt_nopad(*args, **kwargs):
    _lazy_init()
    if _int8_grouped_nopad_impl is None:
        return _missing(*args, **kwargs)
    return _int8_grouped_nopad_impl(*args, **kwargs)


def int8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _int8_grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _int8_grouped_masked_impl(*args, **kwargs)


def m_grouped_bf16_gemm_nt_nopad(*args, **kwargs):
    _lazy_init()
    if _bf16_grouped_nopad_impl is None:
        return _missing(*args, **kwargs)
    return _bf16_grouped_nopad_impl(*args, **kwargs)


def m_grouped_fp4_gemm_nt_nopad(*args, **kwargs):
    _lazy_init()
    if _fp4_grouped_nopad_impl is None:
        return _missing(*args, **kwargs)
    return _fp4_grouped_nopad_impl(*args, **kwargs)


def fp4_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _fp4_grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _fp4_grouped_masked_impl(*args, **kwargs)


def bf16_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _bf16_grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _bf16_grouped_masked_impl(*args, **kwargs)


def transform_sf_into_required_layout(*args, **kwargs):
    _lazy_init()
    if _transform_sf_into_required_layout_impl is None:
        return _missing(*args, **kwargs)
    return _transform_sf_into_required_layout_impl(*args, **kwargs)


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N])
            with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    _lazy_init()
    if _fp8_mqa_logits_impl is None:
        return _missing()
    return _fp8_mqa_logits_impl(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
    )


def int8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute INT8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.int8` by caller.
        kv: Tuple `(k_int8, k_scales)` where `k_int8` has shape [N, D] with
            dtype `torch.int8` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    _lazy_init()
    if _int8_mqa_logits_impl is None:
        return _missing()
    return _int8_mqa_logits_impl(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
    )


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor:
    """Build scheduling metadata for paged MQA logits.

    Args:
        context_lens: Tensor of shape [B], dtype int32; effective context length
            per batch element.
        block_size: KV-cache block size in tokens (e.g., 64).
        num_sms: Number of SMs available. 132 for Hopper

    Returns:
        Backend-specific tensor consumed by `fp8_paged_mqa_logits` to
        schedule work across SMs.
    """
    _lazy_init()
    if _get_paged_mqa_logits_metadata_impl is None:
        return _missing()
    return _get_paged_mqa_logits_metadata_impl(context_lens, block_size, num_sms)


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    _lazy_init()
    if _fp8_paged_mqa_logits_impl is None:
        return _missing()
    return _fp8_paged_mqa_logits_impl(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )


def int8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute INT8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    _lazy_init()
    if _int8_paged_mqa_logits_impl is None:
        return _missing()
    return _int8_paged_mqa_logits_impl(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """
    Perform the following computation:
        out = x.float() @ fn.T
        sqrsum = x.float().square().sum(-1)

    See the caller function for shape requirement
    """
    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )


def get_compile_mode(*args, **kwargs):
    _lazy_init()
    if _get_compile_mode_impl is None:
        return _missing(*args, **kwargs)
    return _get_compile_mode_impl(*args, **kwargs)


def set_compile_mode(*args, **kwargs):
    _lazy_init()
    if _set_compile_mode_impl is None:
        return _missing(*args, **kwargs)
    return _set_compile_mode_impl(*args, **kwargs)


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _align(x: int, y: int) -> int:
    return cdiv(x, y) * y


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/csrc/utils/math.hpp#L19
def get_tma_aligned_size(x: int, element_size: int) -> int:
    return _align(x, 16 // element_size)


DEFAULT_BLOCK_SIZE = [128, 128]


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/dd6ed14acbc7445dcef224248a77ab4d22b5f240/deep_gemm/utils/math.py#L38
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def per_block_cast_to_fp8(
    x: torch.Tensor, block_size: list[int] = DEFAULT_BLOCK_SIZE, use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size
    x_padded = torch.zeros(
        (_align(m, block_m), _align(n, block_n)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    _, fp8_max = get_fp8_min_max()
    sf = x_amax / fp8_max
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(fp8_dtype)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing `torch.testing.assert_close` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report `1 - sim`.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight_shape: tuple[int, int],
    supports_deep_gemm: bool | None = None,
):
    if (getattr(envs, 'VLLM_PPU_DENSE_BACKEND', None)
        and getattr(envs, 'VLLM_PPU_DENSE_BACKEND', None) != "deepgemm"):
        return False

    if supports_deep_gemm is None:
        supports_deep_gemm = is_deep_gemm_supported()

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    return (
        supports_deep_gemm
        and output_dtype == torch.bfloat16
        and weight_shape[0] % N_MULTIPLE == 0
        and weight_shape[1] % K_MULTIPLE == 0
    )


def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging (CUDA fallback).

    This is a pure PyTorch fallback for CUDA when DeepGEMM is not available.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    kv_fp8, scale = kv
    seq_len_kv = kv_fp8.shape[0]
    k = kv_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache (CUDA fallback).

    This is a pure PyTorch fallback for CUDA when DeepGEMM is not available.
    Handles head_dim = 132 (128 + 4 for RoPE).

    Args:
        q: Query tensor of shape [B, next_n, H, D].
        kv_cache: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, heads, dim = q.size()
    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_blocks, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i].item()
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_idx in range(cdiv(context_len, block_size)):
            block_id = block_tables[i][block_idx]
            qx, kx = q[i], kv_cache[block_id]
            k_offsets = torch.arange(
                block_idx * block_size, (block_idx + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_idx * block_size : (block_idx + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


__all__ = [
    "calc_diff",
    "DeepGemmQuantScaleFMT",
    "fp8_gemm_nt",
    "fp8_einsum",
    "m_grouped_fp8_gemm_nt_nopad",
    "fp8_m_grouped_gemm_nt_masked",
    "int8_gemm_nt",
    "m_grouped_int8_gemm_nt_nopad",
    "int8_m_grouped_gemm_nt_masked",
    "m_grouped_bf16_gemm_nt_nopad",
    "bf16_m_grouped_gemm_nt_masked",
    "m_grouped_fp4_gemm_nt_nopad",
    "fp4_m_grouped_gemm_nt_masked",
    "fp8_mqa_logits",
    "fp8_mqa_logits_torch",
    "fp8_paged_mqa_logits",
    "fp8_paged_mqa_logits_torch",
    "int8_mqa_logits",
    "int8_paged_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "per_block_cast_to_fp8",
    "is_deep_gemm_e8m0_used",
    "is_deep_gemm_supported",
    "get_num_sms",
    "should_use_deepgemm_for_fp8_linear",
    "get_col_major_tma_aligned_tensor",
    "get_mk_alignment_for_contiguous_layout",
]