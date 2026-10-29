# SPDX-License-Identifier: Apache-2.0
"""Model-local application hook for the BF16 shared-expert gate projection."""

from __future__ import annotations

from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_fl.dispatch import resolve_op


class Qwen38SmallProjectionMethod(UnquantizedLinearMethod):
    """Keep the existing unquantized loader; replace only bias-free apply."""

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise NotImplementedError("Qwen3.8 small projection is bias-free")
        return layer._qwen38_small_projection(x, layer.weight)


def use_qwen38_shared_expert_projection(layer, projection=None):
    """Change a gate instance in place, retaining its parameter and aliases."""
    if isinstance(layer.quant_method, Qwen38SmallProjectionMethod):
        return layer
    if type(layer.quant_method) is not UnquantizedLinearMethod:
        raise NotImplementedError("Qwen3.8 gate requires unquantized linear weights")
    if layer.weight.ndim != 2 or layer.weight.shape[0] != 1:
        raise ValueError("Qwen3.8 shared-expert gate must project to one value")
    if getattr(layer, "bias", None) is not None:
        raise NotImplementedError("Qwen3.8 shared-expert gate must be bias-free")
    layer._qwen38_small_projection = (
        resolve_op("qwen38_small_projection") if projection is None else projection
    )
    layer.quant_method = Qwen38SmallProjectionMethod()
    return layer
