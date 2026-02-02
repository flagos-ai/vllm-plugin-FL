# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend operator implementations.
"""

from .activation import silu_and_mul_ascend
from .normalization import rmsnorm_ascend
from .rotary import rotary_embedding_ascend
from .attention import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
    AscendAttentionState,
    AscendMLABackend,
    is_torch_npu_available,
)
from .attention_mask import (
    AttentionMaskBuilder,
    get_attention_mask_builder,
)

__all__ = [
    "silu_and_mul_ascend",
    "rmsnorm_ascend",
    "rotary_embedding_ascend",
    "AscendAttentionBackend",
    "AscendAttentionBackendImpl",
    "AscendAttentionMetadataBuilder",
    "AscendMetadata",
    "AscendAttentionState",
    "AscendMLABackend",
    "is_torch_npu_available",
    "AttentionMaskBuilder",
    "get_attention_mask_builder",
]
