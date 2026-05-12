# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""
Kunlunxin operator implementations.
"""

from .activation import silu_and_mul_kunlunxin
from .normalization import rms_norm_kunlunxin
from .rotary import rotary_embedding_kunlunxin
from .attention import (
    KunlunxinAttentionBackend,
    KunlunxinAttentionBackendImpl,
    KunlunxinAttentionMetadataBuilder,
    KunlunxinMetadata,
    is_kunlunxin_ops_available,
)
from .causal_conv1d import causal_conv1d_fn_kunlunxin, causal_conv1d_update_kunlunxin
from .fused_gdn_gating import fused_gdn_gating_kunlunxin

__all__ = [
    "silu_and_mul_kunlunxin",
    "rms_norm_kunlunxin",
    "rotary_embedding_kunlunxin",
    "KunlunxinAttentionBackend",
    "KunlunxinAttentionBackendImpl",
    "KunlunxinAttentionMetadataBuilder",
    "KunlunxinMetadata",
    "is_kunlunxin_ops_available",
    "causal_conv1d_fn_kunlunxin",
    "causal_conv1d_update_kunlunxin",
    "fused_gdn_gating_kunlunxin",
]
