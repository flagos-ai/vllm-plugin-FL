# Copyright (c) 2026 BAAI. All rights reserved.
"""Explicit MetaX policy selection for BGE-M3's FP32 encoder adapter."""

import copy
import math
from functools import lru_cache, partial
from importlib import import_module

import torch

import vllm.envs as vllm_envs
from vllm.config.vllm import get_current_vllm_config_or_none
from vllm.model_executor.parameter import BasevLLMParameter

import vllm_fl.envs as fl_envs
from vllm_fl.configs.bge_m3_metax import (
    LINEAR_EXTRA_CONFIGS,
    LINEAR_FEATURES,
    LINEAR_MAX_ROWS,
)
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def attention_backend(use_mla=False, use_sparse=False):
    config = get_current_vllm_config_or_none()
    model = getattr(config, "model_config", None)
    if (
        not use_mla
        and not use_sparse
        and model is not None
        and model.dtype == torch.float32
        and "BgeM3EmbeddingModelFL"
        in (getattr(model.hf_config, "architectures", None) or [])
    ):
        if not fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
            raise RuntimeError("BGE-M3 FP32 MetaX policy requires FlagGems-vllm")
        return "vllm_fl.models.bge_m3.attention.BgeM3AttentionBackend"

    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    return MacaBackend().attention_backend(use_mla=use_mla, use_sparse=use_sparse)


@lru_cache(maxsize=1)
def _linear_entry():
    """Bind model-owned candidates to the pinned FG kernel, without mutating FG."""
    import flag_gems
    import triton
    from flag_gems import runtime
    from flag_gems.utils import libentry, libtuner

    if flag_gems.vendor_name != "metax":
        raise RuntimeError("BGE-M3 MetaX Linear policy requires MetaX FlagGems")
    module = import_module(flag_gems.linear.__module__)
    kernel = module.linear_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    configs = copy.deepcopy(runtime.get_tuned_config("linear"))
    for item in LINEAR_EXTRA_CONFIGS:
        meta = dict(item)
        warps, stages = meta.pop("num_warps"), meta.pop("num_stages")
        configs.append(triton.Config(meta, num_warps=warps, num_stages=stages))
    tuned = libentry()(
        libtuner(
            configs=configs,
            key=["M", "N", "K"],
            strategy=["align32", "align32", "align32"],
            warmup=5,
            rep=10,
        )(kernel)
    )
    return tuned, module._disable_aabs_for_small_dims, runtime.torch_device_fn


def linear(native_apply, layer, x, bias=None):
    """Override launch parameters only inside the measured BGE inference scope."""
    rows = math.prod(x.shape[:-1])
    if not 0 < rows <= LINEAR_MAX_ROWS:
        return native_apply(layer, x, bias)
    weight = layer.weight
    grad_requested = torch.is_grad_enabled() and (
        x.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )
    if grad_requested:
        return native_apply(layer, x, bias)
    if isinstance(weight, BasevLLMParameter):
        # No copy: keep the layer's Parameter for loading, but avoid its Python
        # metadata dispatch in the FG launcher.
        weight = weight.as_subclass(torch.Tensor)
    if tuple(weight.shape) not in LINEAR_FEATURES:
        return native_apply(layer, x, bias)
    if not (
        x.device.type == "cuda"
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.float32
        and x.ndim >= 2
        and weight.ndim == 2
        and x.shape[-1] == weight.shape[1]
        and x.is_contiguous()
        and weight.is_contiguous()
        and (bias is None or (bias.device == x.device and bias.dtype == x.dtype))
        and not vllm_envs.VLLM_BATCH_INVARIANT
    ):
        return native_apply(layer, x, bias)
    import triton

    entry, aabs_guard, device_fn = _linear_entry()
    k, n = x.shape[-1], weight.shape[0]
    flat = x.view(rows, k)
    output = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_SIZE_M"]),
        triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    with device_fn.device(x.device), aabs_guard(rows, n, k):
        entry[grid](
            flat,
            weight,
            bias if bias is not None else weight,
            output,
            rows,
            n,
            k,
            flat.stride(0),
            flat.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            bias.stride(0) if bias is not None else 0,
            BIAS=bias is not None,
        )
    return output.view(*x.shape[:-1], n)


def register_bge_m3_operators(registry, *, is_platform_available):
    selected_backend = partial(attention_backend)
    selected_backend._is_available = is_platform_available
    selected_linear = partial(linear)
    selected_linear._is_available = is_platform_available
    registry.register_many(
        [
            OpImpl(
                "bge_m3_linear",
                "vendor.metax.bge_m3",
                BackendImplKind.VENDOR,
                selected_linear,
                vendor="metax",
                priority=BackendPriority.VENDOR,
            ),
            OpImpl(
                "attention_backend",
                "vendor.metax.bge_m3",
                BackendImplKind.VENDOR,
                selected_backend,
                vendor="metax",
                priority=BackendPriority.VENDOR,
            ),
        ]
    )
