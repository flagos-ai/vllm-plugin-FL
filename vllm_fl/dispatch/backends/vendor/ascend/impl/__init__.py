# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend operator implementations.
https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu_list.md
"""

from .activation import silu_and_mul_ascend
from .attention import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
    AscendMLABackend,
    AscendMLAMetadataBuilder,
    AscendMLAMetadata,
    AscendMLAImpl,
    is_torch_npu_available,
)
from .attention_mask import AttentionMaskBuilder, get_attention_mask_builder
from .normalization import rms_norm_ascend
from .rotary import rotary_embedding_ascend

__all__ = [
    "silu_and_mul_ascend",
    "rms_norm_ascend",
    "rotary_embedding_ascend",
    "AscendAttentionBackend",
    "AscendAttentionBackendImpl",
    "AscendAttentionMetadataBuilder",
    "AscendMetadata",
    "AscendAttentionState",
    "AscendMLABackend",
    "AscendMLAMetadataBuilder",
    "AscendMLAMetadata",
    "AscendMLAImpl",
    "is_torch_npu_available",
    "AttentionMaskBuilder",
    "get_attention_mask_builder",
]
