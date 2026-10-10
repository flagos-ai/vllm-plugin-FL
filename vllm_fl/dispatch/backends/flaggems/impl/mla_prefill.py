# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlagGems MLA prefill backend for GLM5 empty-build bring-up.

The vLLM 0.24 Hopper selector only considers its FlashAttention C extension
for MLA prefill.  Empty-build wheels intentionally omit that extension, while
FlagGems already exposes the compatible varlen FlashAttention entry point.
Keep this adapter scoped to the GLM5 portable provider instead of changing the
process-wide vLLM selector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class FlagGemsMLAPrefillBackend(MLAPrefillBackend):
    """Use FlagGems' varlen FlashAttention for MLA prefill."""

    @staticmethod
    def get_name() -> str:
        return "FLAGGEMS"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flag_gems import flash_attn_varlen_func  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: VllmConfig,
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
        from flag_gems import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.requires_v_padding = (
            v_head_dim != qk_nope_head_dim + qk_rope_head_dim
        )

    def _run_flash_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if output_scale is not None:
            raise NotImplementedError(
                "FlagGems MLA prefill does not support quantized output"
            )

        original_v = v
        if self.requires_v_padding:
            v = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]])

        # FlagGems uses the same varlen argument names and LSE convention as
        # vLLM's FlashAttention backend.  ``fa_version=2`` is the portable
        # Triton path; no vLLM native CUDA extension is required.
        result = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            # The caller's output has the unpadded value width. Let FlagGems
            # allocate when padding is needed, then copy the trimmed result.
            out=out if v is original_v else None,
            fa_version=2,
        )
        lse = None
        if isinstance(result, tuple):
            result, lse = result[0], result[1]
        if v is not original_v:
            result = result[..., : original_v.shape[-1]]
            if out is not None:
                out.copy_(result)
                result = out
        if return_softmax_lse:
            if lse is None:
                raise RuntimeError(
                    "FlagGems MLA prefill did not return softmax LSE"
                )
            return result, lse
        return result

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._run_flash_attn(
            q,
            k,
            v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
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
        chunked = self._prefill_metadata.chunked_context
        result = self._run_flash_attn(
            q,
            k,
            v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=chunked.cu_seq_lens[chunk_idx],
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=chunked.max_seq_lens[chunk_idx],
            causal=False,
            return_softmax_lse=True,
        )
        assert isinstance(result, tuple)
        return result


__all__ = ["FlagGemsMLAPrefillBackend"]
