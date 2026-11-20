# SPDX-License-Identifier: Apache-2.0
"""FlagGems dense MLA prefill adapter with vLLM-owned metadata and outputs."""

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend


class FlagGemsMLAPrefillBackend(MLAPrefillBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._varlen_operator = self._operator_factory()

    @staticmethod
    def _operator_factory():
        from vllm_fl.dispatch.manager import get_default_manager

        return get_default_manager().resolve("flash_attn_varlen_func")

    def _operator(self):
        return self._varlen_operator

    @staticmethod
    def get_name() -> str:
        return "FLAGGEMS_MLA_PREFILL"

    @classmethod
    def is_available(cls) -> bool:
        from vllm_fl.dispatch.backends.flaggems.impl.mla_ops import is_available

        return is_available("flash_attn_varlen_func")

    @classmethod
    def validate_configuration(cls, device_capability, selector_config):
        dims = selector_config.mla_dimensions
        qk = dims.qk_nope_head_dim + dims.qk_rope_head_dim
        reasons = []
        if not cls.is_available():
            reasons.append("FlagGems varlen attention unavailable or disabled")
        if not (0 < dims.v_head_dim <= qk <= 256):
            reasons.append("requires 0 < v_head_dim <= qk_head_dim <= 256")
        if selector_config.dtype not in (torch.float16, torch.bfloat16):
            reasons.append("requires FP16/BF16")
        return reasons

    def _flash_attn_varlen(
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Dense prefill uses the expanded attention heads (HY4: q/k=256,
        # v=256), not the 576/512 compressed sparse-MLA representation.
        # Pad a smaller value head to match q/k for the FlagGems API.
        maybe_padded_v = v
        if v.shape[-1] != q.shape[-1]:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        result = self._operator()(
            q=q,
            k=k,
            v=maybe_padded_v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=self.scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            # ``out`` has the unpadded value width.  Let the Triton
            # wrapper allocate when padding is needed, then copy below.
            out=out if maybe_padded_v is v else None,
        )
        lse = None
        if isinstance(result, tuple):
            result, lse = result[0], result[1]
        if maybe_padded_v is not v:
            result = result[..., : v.shape[-1]]
            if out is not None:
                out.copy_(result)
                result = out
        if return_softmax_lse:
            assert lse is not None
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
        if output_scale is not None:
            raise NotImplementedError(
                "HY4 FlagGems MLA prefill does not support quantized output"
            )
        metadata = self._prefill_metadata
        return self._flash_attn_varlen(
            q,
            k,
            v,
            cu_seqlens_q=metadata.query_start_loc,
            cu_seqlens_k=metadata.query_start_loc,
            max_seqlen_q=metadata.max_query_len,
            max_seqlen_k=metadata.max_query_len,
            causal=True,
            return_softmax_lse=return_softmax_lse,
            out=out,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self._prefill_metadata
        assert metadata.chunked_context is not None
        chunked = metadata.chunked_context
        result = self._flash_attn_varlen(
            q,
            k,
            v,
            cu_seqlens_q=metadata.query_start_loc,
            cu_seqlens_k=chunked.cu_seq_lens[chunk_idx],
            max_seqlen_q=metadata.max_query_len,
            max_seqlen_k=chunked.max_seq_lens[chunk_idx],
            causal=False,
            return_softmax_lse=True,
        )
        assert isinstance(result, tuple)
        return result


def register_flaggems_mla_prefill():
    from vllm.v1.attention.backends.mla.prefill import (
        MLAPrefillBackendEnum,
        register_mla_prefill_backend,
    )

    if not MLAPrefillBackendEnum.CUSTOM.is_overridden():
        register_mla_prefill_backend(
            MLAPrefillBackendEnum.CUSTOM,
            "vllm_fl.attention.mla_prefill.FlagGemsMLAPrefillBackend",
        )


def make_flaggems_mla_prefill_backend(implementation):
    """Bind a provider selected before model allocation, including explicit plans."""

    class BoundFlagGemsMLAPrefillBackend(FlagGemsMLAPrefillBackend):
        _operator_factory = staticmethod(lambda: implementation)

    return BoundFlagGemsMLAPrefillBackend
