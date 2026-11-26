# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MetaX MACA FlashAttention-2 binding resolution, direct C++ fast-path wrappers, and KV cache helpers."""

from vllm.v1.attention.backends.fa_utils import logger
from vllm.platforms import current_platform


# Out-of-tree `PlatformFL` on MetaX C500 (`xcore1000`) resolves MACA's `flash_attn`
# C++ extension (`flash_attn_cuda`) and `vllm._custom_ops` instead of upstream CUDA FA3 ops.
if current_platform.is_out_of_tree():
    import torch
    from vllm import _custom_ops as ops
    from vllm.logger import init_logger as _init_logger
    from flash_attn import (
        flash_attn_varlen_func as _orig_flash_attn_varlen_func,
        flash_attn_with_kvcache as _orig_flash_attn_with_kvcache,
    )
    from flash_attn.flash_attn_interface import flash_attn_cuda

    _logger = _init_logger(__name__)

    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        softcap=0.0,
        block_table=None,
        s_aux=None,
        return_max_logit=False,
        out=None,
    ):
        """Variable-length FlashAttention-2 wrapper writing directly into caller-allocated `out`.

        Serves initial prefill (`block_table=None`), paged varlen prefill (`block_table!=None`),
        and MetaX's fused contiguous-KV prefill (`prefill_mode == 5/6`, where paged KV blocks are
        gathered into linear `_KV_CONTIG_WORKSPACE` inside `_fused_qk_rope_and_cache_kv_kernel`).
        """
        # Fast path: bypass Python `autograd.Function` dispatch, temporary output allocation,
        # and extra `out.copy_()` when `out` has contiguous head_dim (`stride(-1) == 1`).
        if (
            out is not None
            and not return_attn_probs
            and not return_max_logit
            and out.stride(-1) == 1
        ):
            if q.stride(-1) != 1:
                q = q.contiguous()
            if k.stride(-1) != 1:
                k = k.contiguous()
            if v.stride(-1) != 1:
                v = v.contiguous()
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            flash_attn_cuda.varlen_fwd(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                None,
                None,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                False,
                causal,
                window_size[0],
                window_size[1],
                softcap,
                False,
                None,
                s_aux,
                False,
            )
            return out
        # Fallback path when no pre-allocated contiguous `out` buffer is supplied or debug outputs are requested.
        res = _orig_flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            softcap=softcap,
            block_table=block_table,
            s_aux=s_aux,
            return_max_logit=return_max_logit,
        )
        if out is not None and not return_attn_probs and not return_max_logit:
            out.copy_(res)
            return out
        return res

    def flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=None,
        cache_batch_idx=None,
        cache_leftpad=None,
        block_table=None,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        rotary_interleaved=True,
        alibi_slopes=None,
        num_splits=0,
        softcap=0.0,
        s_aux=None,
        out=None,
    ):
        """Paged KV-cache FlashAttention-2 wrapper with Split-KV and GQA zero-copy output view."""
        # Fast path: invoke `flash_attn_cuda.fwd_kvcache` directly into `out` (CUDAGraph-safe,
        # zero intermediate tensor allocation) when `out` is contiguous along `head_dim`.
        if out is not None and out.stride(-1) == 1:
            if q.stride(-1) != 1:
                q = q.contiguous()
            if k is not None and k.stride(-1) != 1:
                k = k.contiguous()
            if v is not None and v.stride(-1) != 1:
                v = v.contiguous()
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            if cache_seqlens is not None and isinstance(cache_seqlens, int):
                cache_seqlens = torch.full(
                    (k_cache.shape[0],),
                    cache_seqlens,
                    dtype=torch.int32,
                    device=k_cache.device,
                )
            if cache_batch_idx is not None and cache_batch_idx.stride(-1) != 1:
                cache_batch_idx = cache_batch_idx.contiguous()
            if block_table is not None and block_table.stride(-1) != 1:
                block_table = block_table.contiguous()
            nheads_q = q.shape[2]
            nheads_k = k_cache.shape[2]
            # For single-token GQA decode (`seqlen_q == 1`, `nheads_q > nheads_k`, no ALiBi),
            # `fwd_kvcache` packs the GQA group ratio `G = nheads_q // nheads_k` into the
            # query-length axis internally (`[B, G, H_kv, D]`). Passing a strided transpose
            # view of `out` lets the C++ kernel write directly into `[B, 1, H_q, D]` memory
            # without a post-kernel transpose copy.
            if (
                out.shape[1] == 1
                and alibi_slopes is None
                and nheads_q > nheads_k
                and nheads_q % nheads_k == 0
                and out.is_contiguous()
            ):
                out_arg = out.view(
                    out.shape[0], nheads_k, nheads_q // nheads_k, out.shape[3]
                ).transpose(1, 2)
            else:
                out_arg = out
            flash_attn_cuda.fwd_kvcache(
                q,
                k_cache,
                v_cache,
                k,
                v,
                cache_seqlens,
                rotary_cos,
                rotary_sin,
                cache_batch_idx,
                cache_leftpad,
                block_table,
                alibi_slopes,
                out_arg,
                softmax_scale,
                causal,
                window_size[0],
                window_size[1],
                softcap,
                rotary_interleaved,
                num_splits,
                s_aux,
            )
            return out
        res = _orig_flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            rotary_interleaved=rotary_interleaved,
            alibi_slopes=alibi_slopes,
            num_splits=max(0, num_splits),
            softcap=softcap,
            s_aux=s_aux,
        )
        if out is not None:
            out.copy_(res)
            return out
        return res

    class _dummy_ops:
        @staticmethod
        def get_scheduler_metadata(*args, **kwargs) -> None:
            """Stub for FA3 persistent tile-scheduler metadata (unused by MACA FA2)."""
            _logger.warning_once(
                "get_scheduler_metadata is not implemented for maca ops, returning None."
            )
            return None

    get_scheduler_metadata = _dummy_ops.get_scheduler_metadata

    def reshape_and_cache_flash(*args, **kwargs):
        """Scatter K/V tokens into paged `[num_blocks, block_size, num_kv_heads, head_size]` cache
        on fallback paths where fused `_fused_qk_rope_and_cache_kv_kernel` does not run.
        """
        return ops.reshape_and_cache_flash(*args, **kwargs)


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    """Return FlashAttention-2 (`2`) as the hardware-supported version on MetaX MACA."""
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )

    # Note: MACA's `flash_attn_cuda` C++ entrypoint does not accept an `fa_version`
    # argument; callers inspect this return value (`2`) to disable FA3 AOT scheduling.
    return 2


def is_fa_version_supported(fa_version: int) -> bool:
    return fa_version == 2


def flash_attn_supports_fp8() -> bool:
    """MetaX C500 (`xcore1000`) lacks native FP8 MMA instructions; KV cache stays BF16/FP16."""
    logger.info_once(
        "Using Maca version of flash attention, which does not support FP8"
    )
    return False


def flash_attn_supports_quant_query_input() -> bool:
    return False


def flash_attn_supports_sinks() -> bool:
    # MACA FA2 C++ kernels natively accept `s_aux` attention sink logits.
    return True


def flash_attn_supports_mla():
    # MLA models dispatch through the dedicated `mla/` backend rather than `MacaFlashAttentionBackend`.
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True
