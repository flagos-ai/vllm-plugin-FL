# SPDX-License-Identifier: Apache-2.0
"""Retire loaded MXFP4 buffers by aliasing already packed storage.

Only the FlagGems provider may interpret these aliased parameters afterwards.
The loader object/attributes and native parameter shapes are preserved when
there is no intermediate-size compaction. The original byte layout is gone.
"""
import torch
from types import MethodType


PARAMETER_NAMES = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")


def _signature(tensor):
    return (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), tensor.data_ptr(),
            tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())


def assert_retired_aliases(layer, packed):
    state = getattr(layer, "_fl_mxfp4_retirement", None)
    if state is None:
        raise RuntimeError("MXFP4 raw weights were not retired")
    for name, expected, target, packed_expected in zip(
            PARAMETER_NAMES, state["signatures"], packed, state["packed_signatures"]):
        param = getattr(layer, name)
        if (_signature(param) != expected or _signature(target) != packed_expected
                or param.untyped_storage().data_ptr() != target.untyped_storage().data_ptr()):
            raise RuntimeError("retired MXFP4 parameter moved/changed; rebuild the model instead of repacking")
    return state


def retire_raw_parameters(layer, packed, gems, *, compact_shapes=None):
    """Release raw storage without copying/changing any packed quantized value."""
    if getattr(layer, "_fl_mxfp4_retirement", None) is not None:
        return assert_retired_aliases(layer, packed)
    params = [getattr(layer, n) for n in PARAMETER_NAMES]
    original_shapes = [tuple(p.shape) for p in params]
    shapes = original_shapes if compact_shapes is None else compact_shapes
    views = []
    for param, target, shape in zip(params, packed, shapes):
        if target.device != param.device or not target.is_contiguous():
            raise RuntimeError("packed tensors must be contiguous on the loaded parameter device")
        count = 1
        for dim in shape:
            count *= dim
        # All original parameters use one byte per element (uint8/E8M0).
        if param.element_size() != 1:
            raise RuntimeError("retirement supports only original one-byte MXFP4 parameters")
        storage_bytes = target.view(torch.uint8).view(-1)
        if count > storage_bytes.numel():
            raise RuntimeError("packed storage cannot represent the requested metadata shape")
        views.append(storage_bytes[:count].view(param.dtype).view(shape))
    # Raw tensors are weak dictionary keys by object identity. Reusing their
    # former semantic cache entries after .data changes would be misleading.
    for param in params:
        for cache_name in ("_W_PACK_CACHE", "_SCALE_PACK_CACHE", "_SCALE_PACK_CACHE_E8M0",
                           "_SCALE_PACK_CACHE_E8M0_FOLD", "_E8M0_FOLD_SAFE_CACHE"):
            cache = getattr(gems, cache_name, None)
            if cache is not None:
                cache.pop(param, None)
    for param, view in zip(params, views):
        param.data = view  # Same Parameter identity and loader attributes.
        param._fl_mxfp4_retired = True
    state = {"original_shapes": original_shapes,
             "signatures": [_signature(p) for p in params],
             "packed_signatures": [_signature(p) for p in packed]}
    layer._fl_mxfp4_retirement = state
    if hasattr(layer, "_replace_quant_method"):
        retained_method = layer.quant_method
        def refuse_native_fallback(self, replacement):
            if replacement is retained_method:
                return
            raise RuntimeError("retired packed-only MXFP4 weights cannot switch quant methods; reload the model")
        layer._replace_quant_method = MethodType(refuse_native_fallback, layer)
    return state


def compact_raw_intermediate(layer, intermediate):
    """Remove loader-added gate/up/down padding without changing quantization.

    Call only before packing. Native MoE config stays intact; these parameters
    become the compact logical raw shapes and are immediately packed/retired.
    """
    if getattr(layer, "_fl_mxfp4_retirement", None) is not None:
        raise RuntimeError("cannot compact already retired/repacked weights")
    w1, w2, s1, s2 = [getattr(layer, n) for n in PARAMETER_NAMES]
    padded = w1.shape[1] // 2
    if intermediate <= 0 or intermediate > padded or intermediate % 64:
        raise ValueError("compact I must be a positive multiple of 64 within loader I")
    if intermediate == padded:
        return
    # Verify every discarded code is FP4 zero. Scales of zero-weight columns
    # can have any finite E8M0 exponent; no rounded values are discarded.
    tails = (w1[:, intermediate:padded], w1[:, padded+intermediate:],
             w2[:, :, intermediate//2:])
    if any(bool(((t & 0x77) != 0).any().item()) for t in tails):
        raise RuntimeError("refusing to compact nonzero expert weights")
    compact = (torch.cat((w1[:, :intermediate], w1[:, padded:padded+intermediate]), 1),
               w2[:, :, :intermediate//2].contiguous(),
               torch.cat((s1[:, :intermediate], s1[:, padded:padded+intermediate]), 1),
               s2[:, :, :intermediate//32].contiguous())
    layer._fl_mxfp4_before_compact_shapes = [tuple(p.shape) for p in (w1,w2,s1,s2)]
    for name, value in zip(PARAMETER_NAMES, compact):
        getattr(layer, name).data = value
