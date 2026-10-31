# SPDX-License-Identifier: Apache-2.0
"""Qwen BF16 projection policy over the pinned MCTlass library."""

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod

_LINEAR = {
    (10240, 320): False,  # GR up: Universal
    (2560, 768): False,  # GDN output
    (2560, 80): False,  # shared-expert down
    (2048, 2560): True,  # GDN QKV/Z: one-expert identity GEMM
    (512, 2560): True,  # router
}


def supports_linear(x, weight):
    return (
        x.ndim == weight.ndim == 2
        and 1024 <= x.shape[0] <= 2048
        and x.device.type == "cuda"
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and tuple(weight.shape) in _LINEAR
    )


def linear_or_none(x, weight):
    if not supports_linear(x, weight):
        return None
    from flaggems_vllm.runtime.backend._metax.ops.mctlass_bf16 import linear

    result = linear(x, weight, identity=_LINEAR[tuple(weight.shape)])
    return result


def install_linear_methods(module):
    for child in module.modules():
        if type(getattr(child, "quant_method", None)) is not UnquantizedLinearMethod:
            continue
        weight = getattr(child, "weight", None)
        if weight is not None and weight.ndim == 2 and tuple(weight.shape) in _LINEAR:
            child.quant_method = Qwen38MctlassLinearMethod()


class Qwen38MctlassLinearMethod(UnquantizedLinearMethod):
    """Keep loader and processing hooks; only apply a scoped BF16 projection."""

    def apply(self, layer, x, bias=None):
        if bias is None:
            result = linear_or_none(x, layer.weight)
            if result is not None:
                return result
        return super().apply(layer, x, bias)
