# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems operator implementations.
"""

from importlib import import_module

from .activation import silu_and_mul_flaggems, gelu_and_mul_flaggems
from .normalization import rms_norm_flaggems
from .rotary import rotary_embedding_flaggems

# Attention backends depend on version-specific vLLM interfaces. Importing an
# unrelated operator must not load a backend that the selected device won't use.
_LAZY_IMPORTS = {
    "AttentionFLBackend": ".attention",
    "AttentionFLMetadata": ".attention",
    "AttentionFLMetadataBuilder": ".attention",
    "AttentionFLImpl": ".attention",
    "MLAFLBackend": ".mla",
    "MLAFLImpl": ".mla",
    "register_attention": ".custom_attention",
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_IMPORTS[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "silu_and_mul_flaggems",
    "gelu_and_mul_flaggems",
    "rms_norm_flaggems",
    "rotary_embedding_flaggems",
    "AttentionFLBackend",
    "AttentionFLMetadata",
    "AttentionFLMetadataBuilder",
    "AttentionFLImpl",
    "MLAFLBackend",
    "MLAFLImpl",
    "register_attention",
]
