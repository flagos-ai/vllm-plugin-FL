"""PPU dense MLA: native Triton decode with the PPU FA3 prefill ABI."""

from __future__ import annotations

import importlib.util

from vllm.config import get_current_vllm_config
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend, TritonMLAImpl


class TheadMLAPrefillBackend(MLAPrefillBackend):
    """Use the existing PPU varlen wrapper for new tokens and cached context."""

    @staticmethod
    def get_name():
        return "THEAD_FA3"

    @classmethod
    def is_available(cls):
        return importlib.util.find_spec("flash_attn_3") is not None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .fa3 import _thead_flash_attn_varlen_func

        self.flash_attn_varlen_func = _thead_flash_attn_varlen_func

    def _run(
        self,
        q,
        k,
        v,
        *,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal,
        return_softmax_lse,
        out=None,
        output_scale=None,
    ):
        if output_scale is not None:
            raise NotImplementedError(
                "PPU MLA prefill does not support quantized output"
            )
        return self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            out=out,
        )

    def run_prefill_new_tokens(
        self, q, k, v, return_softmax_lse, out=None, output_scale=None
    ):
        meta = self._prefill_metadata
        return self._run(
            q,
            k,
            v,
            cu_q=meta.query_start_loc,
            cu_k=meta.query_start_loc,
            max_q=meta.max_query_len,
            max_k=meta.max_query_len,
            causal=True,
            return_softmax_lse=return_softmax_lse,
            out=out,
            output_scale=output_scale,
        )

    def run_prefill_context_chunk(self, chunk_idx, q, k, v):
        meta = self._prefill_metadata
        chunk = meta.chunked_context
        assert chunk is not None
        return self._run(
            q,
            k,
            v,
            cu_q=meta.query_start_loc,
            cu_k=chunk.cu_seq_lens[chunk_idx],
            max_q=meta.max_query_len,
            max_k=chunk.max_seq_lens[chunk_idx],
            causal=False,
            return_softmax_lse=True,
        )


class TheadMLAImpl(TritonMLAImpl):
    def __init__(self, *args, **kwargs):
        config = get_current_vllm_config()
        if config.attention_config.mla_prefill_backend is None:
            register_mla_prefill_backend(
                MLAPrefillBackendEnum.CUSTOM, __name__ + ".TheadMLAPrefillBackend"
            )
            config.attention_config.mla_prefill_backend = MLAPrefillBackendEnum.CUSTOM
        # Chunked/prefix prefill calls the common vLLM gather ABI. Install
        # the existing portable reader only when that native op is absent.
        import torch

        if not hasattr(torch.ops._C_cache_ops, "gather_and_maybe_dequant_cache"):
            from vllm import _custom_ops as ops

            from vllm_fl.dispatch.backends.mla_cache import (
                torch_gather_and_maybe_dequant_cache,
            )

            ops.gather_and_maybe_dequant_cache = torch_gather_and_maybe_dequant_cache
        super().__init__(*args, **kwargs)

    def do_kv_cache_update(
        self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
    ):
        if kv_cache.numel() == 0:
            return
        # PPU does not provide the CUDA _C_cache_ops ABI. Use the same
        # public FlagGems writer as the existing Sunrise MLA adapter.
        from flag_gems import concat_and_cache_mla

        concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
        )


class TheadMLABackend(TritonMLABackend):
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_impl_cls():
        return TheadMLAImpl
