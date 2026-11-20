# Copyright (c) 2026 BAAI. All rights reserved.

"""Preserve legacy FusedMoE targets when vLLM 0.32 creates RoutedExperts."""

from functools import wraps
from inspect import signature

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

_MARKER = "_fl_metax_legacy_fused_moe_target"


def prepare():
    original = CompressedTensorsConfig._add_fused_moe_to_target_scheme_map
    if getattr(original, _MARKER, False):
        return []

    if list(signature(original).parameters) != ["self"]:
        raise RuntimeError("unsupported compressed-tensors expert target contract")

    @wraps(original)
    def add_moe_target(self):
        schemes = self.target_scheme_map
        if "RoutedExperts" not in schemes and "FusedMoE" in schemes:
            schemes["RoutedExperts"] = schemes["FusedMoE"]
        return original(self)

    setattr(add_moe_target, _MARKER, True)
    return [
        (CompressedTensorsConfig, "_add_fused_moe_to_target_scheme_map", add_moe_target)
    ]
