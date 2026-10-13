# Copyright (c) 2026 BAAI. All rights reserved.
"""vLLM 0.24 MLA prefill integration through the plugin dispatcher."""

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)

from vllm_fl.dispatch import CachedOp, resolve_op

_BACKEND_PATH = "vllm_fl.attention.mla_prefill.MLAPrefillBackendFL"


def configure_mla_prefill(vllm_config):
    """Register the generic backend while preserving explicit selections."""
    model_config = vllm_config.model_config
    if model_config is None or not model_config.use_mla:
        return False

    backend = MLAPrefillBackendEnum.CUSTOM
    attention_config = vllm_config.attention_config
    if attention_config.mla_prefill_backend not in (None, backend):
        return False
    if backend.is_overridden() and backend.get_path() != _BACKEND_PATH:
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
