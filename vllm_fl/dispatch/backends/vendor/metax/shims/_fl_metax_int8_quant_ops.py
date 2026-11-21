"""Bind vLLM's in-place INT8 quant ops to FlagGems-vllm on MetaX.

``vllm._custom_ops.scaled_int8_quant`` calls
``torch.ops._C.dynamic_scaled_int8_quant`` / ``static_scaled_int8_quant``,
which only exist in the compiled ``vllm._C``.  The empty wheel has no such
extension, so bind them to the FlagGems-vllm Triton kernels (metax backend).

Kill switch: ``FL_METAX_INT8_QUANT_OPS=0``.
"""

import os

import torch

from ._fl_metax_gate import metax_active

_SWITCH = "FL_METAX_INT8_QUANT_OPS"
_LIB = None

_DYNAMIC_SCHEMA = (
    "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
    "Tensor!? azp) -> ()"
)
_STATIC_SCHEMA = (
    "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale, "
    "Tensor? azp) -> ()"
)


def _quant(input, scale, azp, symmetric):
    from flaggems_vllm import scaled_int8_quant

    return scaled_int8_quant(input.contiguous(), scale, azp, symmetric)


def _dynamic_scaled_int8_quant(result, input, scale, azp=None):
    # azp is passed only for asymmetric quantization.
    q, s, z = _quant(input, None, None, symmetric=azp is None)
    result.copy_(q.view_as(result))
    scale.copy_(s.view_as(scale))
    if azp is not None and z is not None:
        azp.copy_(z.view_as(azp))


def _static_scaled_int8_quant(result, input, scale, azp=None):
    q, _, _ = _quant(input, scale, azp, symmetric=azp is None)
    result.copy_(q.view_as(result))


def patch() -> bool:
    if not metax_active():
        return False
    value = os.environ.get(_SWITCH, "1").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False

    global _LIB
    _LIB = torch.library.Library("_C", "FRAGMENT")
    if not hasattr(torch.ops._C, "dynamic_scaled_int8_quant"):
        _LIB.define(_DYNAMIC_SCHEMA)
    if not hasattr(torch.ops._C, "static_scaled_int8_quant"):
        _LIB.define(_STATIC_SCHEMA)
    _LIB.impl("dynamic_scaled_int8_quant", _dynamic_scaled_int8_quant, "CUDA")
    _LIB.impl("static_scaled_int8_quant", _static_scaled_int8_quant, "CUDA")
    print("[vllm_fl][metax] registered _C int8 quant ops (FlagGems-vllm)")
    return True


patch()
