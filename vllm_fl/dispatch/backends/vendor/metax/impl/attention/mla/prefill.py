# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from MetaX-MACA/vLLM-metax @ 440ec38e9508 (vLLM 0.32.0).
"""MACA FA2 implementation of vLLM's independent MLA prefill interface."""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)

from ..utils.fa_utils import (
    flash_attn_varlen_func,
    is_flash_attn_varlen_func_available,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


class MacaFlashAttnPrefillBackend(MLAPrefillBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def is_available(cls) -> bool:
        return is_flash_attn_varlen_func_available()

    @classmethod
    def supports_mla_dimensions(cls, mla_dimensions: MLADimensions) -> bool:
        return mla_dimensions in [
            MLADimensions(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128),
            MLADimensions(qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256),
            MLADimensions(qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128),
            MLADimensions(qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256),
        ]

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
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # MACA FA2 handles DeepSeek's 192/128 pair natively. Other unequal
        # dimensions need padded V; the native MLA layer slices the output.
        self.requires_v_padding = not (
            v_head_dim == qk_head_dim or (v_head_dim == 128 and qk_head_dim == 192)
        )

    def supports_quant_output(self, quant_key: "QuantKey") -> bool:
        return False

    def supports_out(self) -> bool:
        # flash_attn's MACA API has neither the vLLM ``out`` argument nor
        # fused output quantization. Let the MLA layer own its output copy.
        return False

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if output_scale is not None:
            raise NotImplementedError("MACA FA2 does not support quantized MLA output")
        if self.requires_v_padding:
            v = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)
        result = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_attn_probs=return_softmax_lse,
            **kwargs,
        )
        if isinstance(result, tuple):
            attn_out, lse = result[:2]
        else:
            attn_out, lse = result, None
        if out is not None:
            out.copy_(attn_out[..., : self.v_head_dim])
            attn_out = out
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
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=chunk.query_start_loc,
            cu_seqlens_k=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len,
            max_seqlen_k=chunk.max_seq_len,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
            out=out,
        )
