# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend operator implementations with side-effect-free lazy exports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "silu_and_mul_ascend": (".activation", "silu_and_mul_ascend"),
    "rms_norm_ascend": (".normalization", "rms_norm_ascend"),
    "rotary_embedding_ascend": (".rotary", "rotary_embedding_ascend"),
    "AscendAttentionBackend": (".attention", "AscendAttentionBackend"),
    "AscendAttentionBackendImpl": (".attention", "AscendAttentionBackendImpl"),
    "AscendAttentionMetadataBuilder": (".attention", "AscendAttentionMetadataBuilder"),
    "AscendMetadata": (".attention", "AscendMetadata"),
    "AscendAttentionState": (".attention", "AscendAttentionState"),
    "AscendMLABackend": (".attention", "AscendMLABackend"),
    "is_torch_npu_available": (".attention", "is_torch_npu_available"),
    "AttentionMaskBuilder": (".attention_mask", "AttentionMaskBuilder"),
    "get_attention_mask_builder": (".attention_mask", "get_attention_mask_builder"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
