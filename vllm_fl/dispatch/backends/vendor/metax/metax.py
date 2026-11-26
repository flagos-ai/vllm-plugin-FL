# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX backend implementation.

This backend provides operator implementations for METAX GPUs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


# Register attention backends for MACA
def register_attention_backends():
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.flashmla.MacaFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn.MacaFlashAttentionBackend",
    )


class MacaBackend(Backend):
    """
    METAX backend for operator implementations.

    This backend uses MACA libraries to provide high-performance
    operator implementations for METAX GPUs.
    """

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "maca"

    @property
    def vendor(self) -> Optional[str]:
        return "metax"

    def is_available(self) -> bool:
        """Check if Metax hardware and libraries are available."""
        if MacaBackend._available is None:
            try:
                # Check if Metax device is available
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    MacaBackend._available = True
                else:
                    MacaBackend._available = False
            except Exception:
                MacaBackend._available = False
        return MacaBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_maca

        return silu_and_mul_maca(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_maca

        return gelu_and_mul_maca(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization using Maca's CUDA implementation.
        """
        from .impl.layernorm import rms_norm_maca

        return rms_norm_maca(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding using vLLM's CUDA implementation.
        """
        from .impl.rotary_embedding import rotary_embedding_maca

        return rotary_embedding_maca(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for CUDA.

        Supports:
        - FLASH_ATTN (default)
        - TRITON_ATTN (when use_flaggems_op("triton_attn") is True)
        - FLASHMLA_SPARSE (when use_mla and use_sparse are both True)

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        # register before selection
        register_attention_backends()

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        # Default to FLASH_ATTN
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        from .impl.fused_moe import topk_softmax_maca

        return topk_softmax_maca(
            topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
        )

    def invoke_fused_moe_triton_kernel(
        self,
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    ):
        from .impl.fused_moe import invoke_fused_moe_triton_kernel_maca

        invoke_fused_moe_triton_kernel_maca(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=B_bias,
        )


# _metax_paged_attn_opt_hook
import logging as _logging
import os as _os
if "logger" not in globals():
    logger = _logging.getLogger(__name__)
_DECODE_WORKSPACES = {}
_ATTN_STATS = {"decode": 0, "prefill": 0, "fallback": 0}
_ATTN_FB = {}
_ATTN_CALLS = 0
_ATTN_DUMPED = False
_OPT_MOD = None
_ATTN_STATS_ON = _os.environ.get("FL_ATTN_STATS", "0") not in ("0", "false", "False")


def _attn_stat(branch, fb_reason=None):
    if not _ATTN_STATS_ON:
        return
    global _ATTN_CALLS
    _ATTN_STATS[branch] += 1
    if fb_reason is not None:
        _ATTN_FB[fb_reason] = _ATTN_FB.get(fb_reason, 0) + 1
    _ATTN_CALLS += 1
    if _ATTN_CALLS % 200 == 0:
        logger.info(
            "[FL_ATTN_STATS] calls=%d decode=%d prefill=%d fallback=%d fb_reasons=%s sampler=%s",
            _ATTN_CALLS, _ATTN_STATS["decode"], _ATTN_STATS["prefill"],
            _ATTN_STATS["fallback"], dict(_ATTN_FB), getattr(_OPT_MOD, "_SAMPLER_STATS", {}),
        )


def patch_triton_unified_attention_for_metax() -> None:
    """Route MiniCPM5-2B GQA attention to page-aligned BLOCK_Q=8 prefill and split-KV."""
    try:
        from . import paged_attn_opt as _opt
    except Exception as exc:
        logger.warning("patch_triton_unified_attention_for_metax skipped: %s", exc)
        return

    if __package__ and __package__.endswith(".metax"):
        try:
            import sys as _sys
            _pkg = _sys.modules.get(__package__)
            if _pkg is not None and "MacaBackend" in globals():
                setattr(_pkg, "MacaBackend", globals()["MacaBackend"])
            import vllm._custom_ops  # noqa: F401
            import vllm_fl.dispatch.backends.vendor.metax.patches  # noqa: F401
        except Exception:
            pass

    globals()["_OPT_MOD"] = _opt
    _opt.install(['block_pool', 'sampler_cast', 'sampler_gumbel', 'sched_admission'])
    logger.info(
        "MetaX C500: vLLM core components installed; Triton attention "
        "kernels stay off until `lab.probe` proves the unified path runs.",
    )
    return
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _tua
    except Exception as exc:
        logger.warning("patch_triton_unified_attention_for_metax skipped: %s", exc)
        return
    if getattr(_tua, "_metax_paged_attn_patched", False):
        return

    _orig_unified = _tua.unified_attention

    def _unified_attention_opt(
        q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
        softmax_scale, causal, window_size, block_table, softcap, *args, **kwargs
    ):
        # Forward-compatible shim: vLLM 0.24.0 passes extra keyword args
        # (use_alibi_sqrt, kv_quant_mode, k/v_scale_cache, chunk_lookback, use_td).
        # Anything we do not explicitly handle is captured in *args/**kwargs and
        # forwarded verbatim to the original on the fallback path.
        def _fallback(reason="?"):
            _attn_stat("fallback", reason)
            return _orig_unified(
                q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                softmax_scale, causal, window_size, block_table, softcap,
                *args, **kwargs,
            )

        if _ATTN_STATS_ON and not _ATTN_DUMPED:
            globals()["_ATTN_DUMPED"] = True
            try:
                logger.info(
                    "[FL_ATTN_KWARGS] mq=%s mk=%s qshape=%s seqused=%s kshape=%s kstride=%s "
                    "mm_prefix=%s kv_quant=%s q_descale=%s k_descale=%s v_descale=%s "
                    "k_scale_cache=%s chunk_lookback=%s causal=%s window=%s",
                    max_seqlen_q, max_seqlen_k, tuple(q.shape), tuple(seqused_k.shape),
                    tuple(k.shape), tuple(k.stride()), kwargs.get("mm_prefix_range") is not None,
                    getattr(kwargs.get("kv_quant_mode"), "name", kwargs.get("kv_quant_mode")),
                    kwargs.get("q_descale") is not None, kwargs.get("k_descale") is not None,
                    kwargs.get("v_descale") is not None, kwargs.get("k_scale_cache") is not None,
                    kwargs.get("chunk_lookback"), causal, window_size,
                )
            except Exception:
                pass

        # Only handle the plain causal BF16/FP16/FP32 GQA case. Fall back for
        # PrefixLM, KV/logits quantization, and chunked-attention lookback, whose
        # masking/scaling our kernels do not implement.
        # NOTE: vLLM 0.24.0 ALWAYS passes k_descale/v_descale as (1.0) scale tensors
        # even for bf16/fp16, so descale non-None is NOT a quantization signal.
        # Quantization is gated solely by kv_quant_mode (== NONE for unquantized).
        if kwargs.get("mm_prefix_range") is not None:
            return _fallback("mm_prefix_range")
        _km = kwargs.get("kv_quant_mode")
        if _km is not None and getattr(_km, "name", None) not in (None, "NONE"):
            return _fallback("kv_quant_mode=%s" % getattr(_km, "name", None))
        if kwargs.get("k_scale_cache") is not None or kwargs.get("v_scale_cache") is not None:
            return _fallback("kv_scale_cache")
        if kwargs.get("chunk_lookback", -1) not in (None, -1):
            return _fallback("chunk_lookback=%s" % kwargs.get("chunk_lookback"))
        if not _opt.can_use_specialized_attention(
            q, k, v, causal=causal, window_size=window_size,
            alibi_slopes=kwargs.get("alibi_slopes"), softcap=softcap,
            qq_bias=kwargs.get("qq_bias"), sinks=kwargs.get("sinks"),
            output_scale=kwargs.get("output_scale"),
        ):
            return _fallback("specialized_guard")
        dev_key = (q.device.type, q.device.index)
        ws = _DECODE_WORKSPACES.get(dev_key)
        if ws is None:
            ws = _opt.SplitKVWorkspace(
                max_batch=64, q_heads=q.shape[1], max_splits=8, dim=q.shape[2]
            )
            _DECODE_WORKSPACES[dev_key] = ws
        max_k = int(max_seqlen_k) if max_seqlen_k and int(max_seqlen_k) > 1 else 4096
        if int(max_seqlen_q) <= 1 and q.shape[0] == seqused_k.shape[0]:
            _attn_stat("decode")
            splits = _opt.select_num_splits(q.shape[0], max_k, kv_heads=k.shape[2])
            _opt.launch_split_kv_decode(
                q, k, v, block_table=block_table, seq_lens=seqused_k,
                workspace=ws, splits=splits, scale=softmax_scale, out=out,
            )
            return
        _attn_stat("prefill")
        _opt.launch_paged_prefill(
            q, k, v, block_table=block_table, seq_lens=seqused_k,
            cu_seqlens_q=cu_seqlens_q, scale=softmax_scale, out=out,
            block_q=16, workspace=ws, max_seqlen_k=max_k,
        )

    _tua.unified_attention = _unified_attention_opt
    _tua._metax_paged_attn_patched = True
    try:
        import vllm.v1.attention.backends.triton_attn as _ta
        if hasattr(_ta, "unified_attention"):
            _ta.unified_attention = _unified_attention_opt
    except Exception:
        pass
    logger.info(
        "Patched unified_attention with MetaX C500 page-aligned BLOCK_Q=16 + split-KV (stats=%s).",
        _ATTN_STATS_ON,
    )


patch_triton_unified_attention_for_metax()
