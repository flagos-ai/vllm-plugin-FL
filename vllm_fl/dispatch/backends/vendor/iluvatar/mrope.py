"""Keep multimodal RoPE inside an opaque, graph-capturable operation."""

import os
from functools import wraps

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _mrope_implementation() -> str:
    implementation = os.environ.get("VLLM_FL_ILUVATAR_MROPE_IMPL", "opaque-v1")
    if implementation not in ("opaque-v1", "native"):
        raise ValueError("VLLM_FL_ILUVATAR_MROPE_IMPL must be 'opaque-v1' or 'native'")
    return implementation


def _wrap_mrope_forward(original):
    if getattr(original, "_iluvatar_mrope_patched", False):
        return original

    @wraps(original)
    def forward(self, positions, query, key=None, offsets=None):
        if (
            positions.ndim != 2
            or positions.shape[0] != 3
            or key is None
            or offsets is not None
            or not self.mrope_section
            or len(self.mrope_section) != 3
            or query.device.type != "cuda"
        ):
            return original(self, positions, query, key, offsets)
        # The opaque op mutates Q/K in place; forward_native must leave its
        # inputs unchanged. Keep per-call outputs: a shared scratch buffer would
        # alias live results across concurrent calls or CUDA graph captures.
        query_out = query.clone(memory_format=torch.contiguous_format)
        key_out = key.clone(memory_format=torch.contiguous_format)
        torch.ops.vllm.mrope(
            positions,
            query_out,
            key_out,
            self._match_cos_sin_cache_dtype(query),
            self.head_size,
            self.rotary_dim,
            self.mrope_section[0],
            self.mrope_section[1],
            self.mrope_section[2],
            self.mrope_interleaved,
            self.is_neox_style,
        )
        return query_out, key_out

    forward._iluvatar_mrope_patched = True
    return forward


def patch_mrope() -> None:
    import vllm.envs
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", "") != "iluvatar":
        return

    vllm.envs.environment_variables["VLLM_FL_ILUVATAR_MROPE_IMPL"] = (
        _mrope_implementation
    )
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    forward = MRotaryEmbedding.forward_native
    if _mrope_implementation() == "native":
        if getattr(forward, "_iluvatar_mrope_patched", False):
            MRotaryEmbedding.forward_native = forward.__wrapped__
        return
    MRotaryEmbedding.forward_native = _wrap_mrope_forward(forward)
    logger.info("Iluvatar M-RoPE uses the opaque vLLM operation")
