# SPDX-License-Identifier: Apache-2.0
"""Widen GDN state indices when a Triton element offset can exceed int32."""

import importlib
from functools import wraps

import torch

_INT32_MAX = (1 << 31) - 1


def widen_state_indices(inp, indices):
    """Use tensor metadata only; never inspect CUDA index values on the host."""
    if (
        inp.device.type != "cuda"
        or inp.ndim != 4
        or inp.dtype != torch.float32
        or len(indices) != 1
        or indices[0].ndim != 1
        or indices[0].dtype != torch.int32
        or any(size == 0 for size in inp.shape)
    ):
        return indices
    # The pinned generator multiplies each loaded index by strides without
    # widening it. Tensor.data_ptr already incorporates storage_offset.
    max_offset = sum(
        (size - 1) * stride for size, stride in zip(inp.shape, inp.stride())
    )
    if max_offset <= _INT32_MAX:
        return indices
    return [indices[0].to(dtype=torch.int64)]


def patch_qwen_gdn_state_indices(model_config):
    """Scope the workaround to the Qwen4 experimental architecture."""
    config = getattr(model_config, "hf_config", None)
    if getattr(config, "model_type", None) != "qwen4_exp":
        return False
    module = importlib.import_module("flag_gems.ops.index_put")
    original = module.IndexPutFunction.__call__
    if getattr(original, "_fl_wide_gdn_state_indices", False):
        return False

    @wraps(original)
    def call(self, inp, indices, values, accumulate):
        return original(
            self, inp, widen_state_indices(inp, indices), values, accumulate
        )

    call._fl_wide_gdn_state_indices = True
    module.IndexPutFunction.__call__ = call
    return True
