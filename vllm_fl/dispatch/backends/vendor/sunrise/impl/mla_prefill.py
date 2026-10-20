# Copyright (c) 2026 BAAI. All rights reserved.
"""Sunrise MLA prefill backend (vLLM 0.24 ``MLAPrefillBackend`` interface).

vLLM 0.24 moved MLA prefill out of ``MLACommonImpl`` into a dedicated
``MLAPrefillBackend`` selected by ``get_mla_prefill_backend`` and stored as
``self.prefill_backend``. On PTPU, ``get_device_capability()`` returns ``None``,
so the selector short-circuits to ``MLAPrefillBackendEnum.FLASH_ATTN`` *without*
consulting the user config -- and the stock ``FlashAttnPrefillBackend`` asserts
on ``vllm_flash_attn`` which PTPU does not ship. We therefore override the
``FLASH_ATTN`` registry entry (ptpu-only) with this FlagGems-backed backend.

Decode-side MQA / KV-cache update still live in ``SunriseMLAImpl`` (see
``impl/mla.py``); this file only owns the prefill MHA path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from flag_gems import flash_attn_varlen_func
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class SunriseMLAPrefillBackend(MLAPrefillBackend):
    """FlagGems-backed MLA prefill backend for Sunrise/PTPU.

    Ports the prefill MHA logic that used to live in
    ``SunriseMLAImpl._flash_attn_varlen_diff_headdims`` onto the 0.24
    ``MLAPrefillBackend`` interface. FlagGems' ``flash_attn_varlen_func`` does
    not support differing q/v head dims, so V is zero-padded up to the qk head
    dim and the output is unpadded back to ``v_head_dim``.
    """

    @staticmethod
    def get_name() -> str:
        return "SUNRISE_MLA"

    @classmethod
    def is_available(cls) -> bool:
        # FlagGems flash_attn_varlen_func is always importable on PTPU builds.
        return True

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # FlagGems FA has no diff-headdim kernel: pad V to the qk head dim.
        self.requires_v_padding = self.qk_head_dim != v_head_dim
        logger.info_once(
            "Sunrise MLA prefill backend active (FlagGems flash_attn_varlen_func, "
            "v_padding=%s).",
            self.requires_v_padding,
        )

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None,
        causal: bool,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # PTPU FlagGems FA does not support fused quantized output.
        assert output_scale is None, (
            "Sunrise MLA prefill does not support fused output quantization"
        )

        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        attn_out = flash_attn_varlen_func(
            q,
            k,
            maybe_padded_v,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            out=out,
        )

        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Unpad output back to v_head_dim if we padded V.
        if self.requires_v_padding:
            attn_out = attn_out[..., : v.shape[-1]]

        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
            out=out,
            output_scale=output_scale,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_metadata.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )
