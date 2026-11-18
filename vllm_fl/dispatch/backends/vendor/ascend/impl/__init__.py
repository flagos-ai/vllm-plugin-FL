# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend operator implementations.
https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu_list.md
"""

from importlib import import_module

from .activation import silu_and_mul_ascend
from .attention_mask import AttentionMaskBuilder, get_attention_mask_builder
from .normalization import rms_norm_ascend
from .rotary import rotary_embedding_ascend

_ATTENTION_EXPORTS = (
    "AscendAttentionBackend",
    "AscendAttentionBackendImpl",
    "AscendAttentionMetadataBuilder",
    "AscendMetadata",
    "AscendAttentionState",
    "AscendMLABackend",
    "is_torch_npu_available",
)

__all__ = [
    "silu_and_mul_ascend",
    "rms_norm_ascend",
    "rotary_embedding_ascend",
    *_ATTENTION_EXPORTS,
    "AttentionMaskBuilder",
    "get_attention_mask_builder",
]


def __getattr__(name):
    if name in _ATTENTION_EXPORTS:
        # Lightweight op registration does not depend on vLLM's attention
        # classes or graph runtime.
        value = getattr(import_module(".attention", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
