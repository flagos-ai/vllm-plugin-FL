# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backends.fa_utils import logger
from vllm.platforms import current_platform


if current_platform.is_out_of_tree():
    from vllm import _custom_ops as ops
    from vllm.logger import init_logger as _init_logger
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_cuda

    _logger = _init_logger(__name__)

    class _dummy_ops:
        @staticmethod
        def get_scheduler_metadata(*args, **kwargs) -> None:
            _logger.warning_once(
                "get_scheduler_metadata is not implemented for maca ops, returning None."
            )
            return None

    get_scheduler_metadata = _dummy_ops.get_scheduler_metadata
    reshape_and_cache_flash = ops.reshape_and_cache_flash

    def decode_into(q, k_cache, v_cache, lengths, pages, out, scale):
        """Write single-token GQA decode directly into ``out`` via grouped strided views."""
        batch, _, heads, dim = q.shape
        kv_heads = k_cache.shape[2]
        grouped_q = q.reshape(batch, kv_heads, heads // kv_heads, dim).transpose(1, 2)
        target = out.view(batch, kv_heads, heads // kv_heads, dim).transpose(1, 2)
        result, _ = flash_attn_cuda.fwd_kvcache(
            grouped_q,
            k_cache,
            v_cache,
            None,
            None,
            lengths,
            None,
            None,
            None,
            None,
            pages,
            None,
            target,
            scale,
            False,
            -1,
            -1,
            0.0,
            True,
            0,
            None,
        )
        if result.data_ptr() != target.data_ptr():
            target.copy_(result)

    def varlen_into(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        causal,
        block_table=None,
    ):
        """Write varlen prefill attention directly into ``out`` without temporary allocation."""
        res = flash_attn_cuda.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            block_table,
            None,
            int(max_seqlen_q),
            int(max_seqlen_k),
            0.0,
            float(scale),
            False,
            bool(causal),
            -1,
            -1,
            0.0,
            False,
            None,
            None,
            False,
        )
        if res[0].data_ptr() != out.data_ptr():
            out.copy_(res[0])


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )

    # Note: In maca this need to be None since
    # metax flash_attn api does not have parameter
    # for `fa_version`.
    return 2


def is_fa_version_supported(fa_version: int) -> bool:
    return fa_version == 2


def flash_attn_supports_fp8() -> bool:
    logger.info_once(
        "Using Maca version of flash attention, which does not support FP8"
    )
    return False


def flash_attn_supports_quant_query_input() -> bool:
    return False


def flash_attn_supports_sinks() -> bool:
    # maca fa2 supports sinks
    return True


def flash_attn_supports_mla():
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True
