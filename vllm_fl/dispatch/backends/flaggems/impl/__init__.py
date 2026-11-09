# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems operator implementations.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .activation import gelu_and_mul_flaggems, silu_and_mul_flaggems
    from .attention import (
        AttentionFLBackend,
        AttentionFLImpl,
        AttentionFLMetadata,
        AttentionFLMetadataBuilder,
    )
    from .custom_attention import register_attention
    from .mla import MLAFLBackend, MLAFLImpl
    from .normalization import rms_norm_flaggems
    from .rotary import rotary_embedding_flaggems

_EXPORTS = {
    "silu_and_mul_flaggems": (".activation", "silu_and_mul_flaggems"),
    "gelu_and_mul_flaggems": (".activation", "gelu_and_mul_flaggems"),
    "rms_norm_flaggems": (".normalization", "rms_norm_flaggems"),
    "rotary_embedding_flaggems": (".rotary", "rotary_embedding_flaggems"),
    "AttentionFLBackend": (".attention", "AttentionFLBackend"),
    "AttentionFLMetadata": (".attention", "AttentionFLMetadata"),
    "AttentionFLMetadataBuilder": (".attention", "AttentionFLMetadataBuilder"),
    "AttentionFLImpl": (".attention", "AttentionFLImpl"),
    "MLAFLBackend": (".mla", "MLAFLBackend"),
    "MLAFLImpl": (".mla", "MLAFLImpl"),
    "register_attention": (".custom_attention", "register_attention"),
}


def __getattr__(name: str):
    """Load optional implementation modules only when their exports are used."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "gelu_and_mul_flaggems",
    "silu_and_mul_flaggems",
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
