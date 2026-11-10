# SPDX-License-Identifier: Apache-2.0
"""FlagGems dense MLA prefill adapter with vLLM-owned metadata and outputs."""

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)
from vllm_fl.dispatch import CachedOp, resolve_op

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
        from vllm.platforms import current_platform

        from vllm_fl.dispatch.capabilities import get_flaggems_vllm_operator
        from vllm_fl.utils import use_flaggems_op

        return (
            current_platform.is_cuda()
            and use_flaggems_op("flash_attn_varlen_func")
            and callable(get_flaggems_vllm_operator("flash_attn_varlen_func"))
        )

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


_BACKEND_PATH = "vllm_fl.attention.mla_prefill.MLAPrefillBackendFL"


def configure_mla_prefill(vllm_config):
    """Register the generic backend while preserving explicit selections."""
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    # Hybrid GLM contains MLA layers without opting into global MLA caching.
    text_config = getattr(model_config, "hf_text_config", None)
    hybrid_mla = getattr(text_config, "model_type", None) == "glm5_next_text"
    if not hybrid_mla:
        return False

    backend = MLAPrefillBackendEnum.CUSTOM
    attention_config = vllm_config.attention_config
    if attention_config.mla_prefill_backend not in (None, backend):
        return False
    if backend.is_overridden() and backend.get_path() != _BACKEND_PATH:
        # Spawn serializes CUSTOM without its process-local class registration.
        # Rebind our built-in default for this model; preserve external CUSTOM paths.
        mainline_path = (
            "vllm_fl.attention.mla_prefill.FlagGemsMLAPrefillBackend"
        )
        if backend.get_path() != mainline_path:
            return False
    try:
        resolve_op("mla_prefill")
    except (RuntimeError, KeyError, ImportError):
        # Keep vLLM's native selection on platforms without a plugin kernel.
        return False

    register_mla_prefill_backend(backend, _BACKEND_PATH)
    if attention_config.mla_prefill_backend is None:
        attention_config.mla_prefill_backend = backend
    return True


class MLAPrefillBackendFL(MLAPrefillBackend):
    """Dense MLA prefill whose kernel is selected by platform dispatch."""

    @staticmethod
    def get_name():
        return "MLA_PREFILL_FL"

    @classmethod
    def is_available(cls):
        try:
            resolve_op("mla_prefill")
        except (RuntimeError, KeyError, ImportError):
            return False
        return True

    @classmethod
    def validate_configuration(cls, device_capability, selector_config):
        reasons = super().validate_configuration(device_capability, selector_config)
        dims = selector_config.mla_dimensions
        qk_dim = dims.qk_nope_head_dim + dims.qk_rope_head_dim
        if not (0 < dims.v_head_dim <= qk_dim <= 256 and qk_dim % 8 == 0):
            reasons.append(
                "MLA prefill requires 0 < V dim <= QK dim <= 256 and "
                "QK dim divisible by 8"
            )
        return reasons

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefill = CachedOp("mla_prefill")
        resolve_op("mla_prefill")

    def _run(
        self, q, k, v, cu_q, cu_k, max_q, max_k, causal, return_lse, out=None
    ):
        if (
            q.dtype not in self.supported_dtypes
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            raise ValueError("MLA prefill requires matching FP16/BF16 Q, K and V")
        if (
            q.device != k.device
            or q.device != v.device
            or q.device.type != current_platform.device_type
        ):
            raise ValueError("MLA prefill requires Q, K and V on the same accelerator")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("MLA prefill requires [tokens, heads, dim] tensors")
        if (
            q.shape[-1] != k.shape[-1]
            or k.shape[:2] != v.shape[:2]
            or q.shape[1] % k.shape[1] != 0
            or not 0 < v.shape[-1] <= q.shape[-1] <= 256
            or q.shape[-1] % 8
        ):
            raise ValueError("Unsupported MLA prefill shape")
        for cu in (cu_q, cu_k):
            if cu.dtype != torch.int32 or cu.device != q.device or cu.ndim != 1:
                raise ValueError("Cumulative lengths must be accelerator int32 vectors")
        if (
            cu_q.shape != cu_k.shape
            or not isinstance(max_q, int)
            or not isinstance(max_k, int)
        ):
            raise ValueError(
                "Prefill requires matching batches and host integer length maxima"
            )
        if out is not None and (
            out.shape != (*q.shape[:2], v.shape[-1])
            or out.dtype != q.dtype
            or out.device != q.device
        ):
            raise ValueError("Invalid MLA prefill output buffer")

        v_dim = v.shape[-1]
        padded_v = (
            F.pad(v, (0, q.shape[-1] - v_dim))
            if v_dim != q.shape[-1]
            else v
        )
        result = self._prefill(
            q=q,
            k=k,
            v=padded_v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.scale,
            causal=causal,
            return_softmax_lse=return_lse,
        )
        output, lse = result if return_lse else (result, None)
        output = output[..., :v_dim]
        if out is not None:
            out.copy_(output)
            output = out
        if return_lse:
            if lse.shape != (q.shape[1], q.shape[0]):
                raise ValueError("MLA prefill LSE must have shape [heads, total_q]")
            return output, lse
        return output

    def run_prefill_new_tokens(
        self, q, k, v, return_softmax_lse, out=None, output_scale=None
    ):
        if output_scale is not None:
            raise NotImplementedError("Quantized MLA prefill output is not supported")
        meta = self._prefill_metadata
        return self._run(
            q, k, v, meta.query_start_loc, meta.query_start_loc,
            meta.max_query_len, meta.max_query_len, True,
            return_softmax_lse, out,
        )

    def run_prefill_context_chunk(self, chunk_idx, q, k, v):
        meta = self._prefill_metadata
        chunk = meta.chunked_context
        if chunk is None:
            raise ValueError("Context prefill requires chunk metadata")
        return self._run(
            q, k, v, meta.query_start_loc, chunk.cu_seq_lens[chunk_idx],
            meta.max_query_len, chunk.max_seq_lens[chunk_idx], False, True,
        )
