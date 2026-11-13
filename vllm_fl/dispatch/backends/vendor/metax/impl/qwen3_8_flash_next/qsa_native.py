# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pre-execution capability checks and native QSA numerical adapter."""

from functools import lru_cache
from importlib import import_module
from inspect import signature

import torch


@lru_cache(maxsize=16)
def native_available(device):
    """Check the actual paged API and extension entry, separately for each device."""
    if device.type != "cuda" or "C550" not in torch.cuda.get_device_name(device):
        return False
    try:
        api = import_module("flash_attn").flash_attn_varlen_func
        extension = import_module("flash_attn_2_cuda")
        parameters = signature(api).parameters
        return (
            callable(api)
            and callable(getattr(extension, "varlen_fwd", None))
            and {"block_table", "softmax_scale", "causal"}.issubset(parameters)
        )
    except (ImportError, OSError, AttributeError, TypeError, ValueError):
        return False


def native_eligible(q, k, v, indices, scale, out, gate, native_workspace):
    """Validate the native numerical contract before any metadata mutation."""
    return (
        q.device.type == "cuda"
        and q.shape[0] > 0
        and q.stride(-1) == 1
        and k.shape == v.shape
        and k.shape[0] > 0
        and k.stride(1) == v.stride(1) == 256
        and k.stride(-1) == v.stride(-1) == 1
        and k.stride(0) == v.stride(0)
        and k.stride(0) % 1024 == 0
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[1:] == (3, 256)
        and k.shape[2:] == (1, 256)
        and k.shape[1] in (16, 400)
        and indices.shape == (q.shape[0], 2051)
        and float(scale) == 0.0625
        and gate is not None
        and gate.dtype == q.dtype
        and gate.shape == q.shape
        and gate.device == q.device
        and out.shape == q.shape
        and out.dtype == q.dtype
        and out.device == q.device
        and native_workspace is not None
    )


def run_native(q, k, v, indices, table, mapping, out, gate, workspace):
    from flaggems_vllm.ops.qwen4.qsa_native import native_qsa_attention

    from vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa_native_metadata import (
        prepare_native_qsa_metadata,
    )

    prepare_native_qsa_metadata(indices, table, mapping, k, workspace)
    # No fallback or retry after metadata writes or a numerical launch.
    return native_qsa_attention(q, k, v, out, gate, workspace)
