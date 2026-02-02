# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems operator implementations.
"""

from .activation import silu_and_mul_flaggems
from .normalization import rmsnorm_flaggems
from .rotary import rotary_embedding_flaggems
from .attention import (
    AttentionFLBackend,
    AttentionFLMetadata,
    AttentionFLMetadataBuilder,
    AttentionFLImpl,
)
from .mla import (
    MLAFLBackend,
    MLAFLImpl,
)
from .custom_attention import register_attention

__all__ = [
    "silu_and_mul_flaggems",
    "rmsnorm_flaggems",
    "rotary_embedding_flaggems",
    "AttentionFLBackend",
    "AttentionFLMetadata",
    "AttentionFLMetadataBuilder",
    "AttentionFLImpl",
    "MLAFLBackend",
    "MLAFLImpl",
    "register_attention",
]
