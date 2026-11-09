# SPDX-License-Identifier: Apache-2.0
"""Common vendor dispatch for the optional upstream mHC broadcast API."""

from __future__ import annotations

import importlib
import inspect
from contextlib import suppress

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fl.models.glm5.indexer_backend import INDEXER_BACKEND
from vllm_fl.models.glm5.provider import use_nvidia_reference

_POSITIONAL_PARAMETERS = (
    "residual",
    "fn",
    "hc_scale",
    "hc_base",
    "rms_eps",
    "hc_pre_eps",
    "hc_sinkhorn_eps",
    "hc_post_mult_value",
    "sinkhorn_repeat",
)
_KEYWORD_PARAMETERS = ("norm_weight", "norm_eps", "fn_broadcast")


def _has_broadcast_api(candidate):
    if not callable(candidate):
        return False
    try:
        signature = inspect.signature(candidate)
        positional = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if tuple(positional[: len(_POSITIONAL_PARAMETERS)]) != _POSITIONAL_PARAMETERS:
            return False
        if not all(name in signature.parameters for name in _KEYWORD_PARAMETERS):
            return False
        signature.bind(
            *([None] * len(_POSITIONAL_PARAMETERS)),
            **dict.fromkeys(_KEYWORD_PARAMETERS),
        )
    except (TypeError, ValueError):
        return False
    return True


def _load_broadcast_impl():
    # Capability selection precedes optional compiler/vendor module imports.
    if not use_nvidia_reference() or not torch.cuda.is_available():
        raise ImportError("MHC broadcast requires the NVIDIA native backend")
    deep_gemm = importlib.import_module("vllm.utils.deep_gemm")
    supported = getattr(deep_gemm, "is_deep_gemm_supported", None)
    if not callable(supported) or not supported():
        raise ImportError("MHC broadcast requires the DeepGEMM TF32 projection")
    if not callable(getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)):
        raise ImportError("The DeepGEMM TF32 mHC projection API is unavailable")

    # FlashInfer binds CUDA before TileLang loads its runtime stub.
    with suppress(ImportError, OSError):
        importlib.import_module("flashinfer.comm")
    importlib.import_module("tilelang")

    try:
        upstream = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang")
    except (ImportError, OSError):
        upstream = None
    candidate = getattr(upstream, "mhc_pre_broadcast_tilelang", None)
    if _has_broadcast_api(candidate):
        return candidate

    backport = importlib.import_module("vllm_fl.dispatch.backends.tilelang.impl.mhc_broadcast")
    return backport.mhc_pre_broadcast_tilelang


_IMPL = _load_broadcast_impl()


def _native_broadcast(*args, **kwargs):
    return _IMPL(*args, **kwargs)


def preflight_mhc_broadcast():
    return INDEXER_BACKEND._call_flag(
        "mhc_pre_broadcast_tilelang",
        None,
        None,
        native=_native_broadcast,
        _preflight=True,
    )


def _mhc_pre_broadcast(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    norm_weight: torch.Tensor,
    norm_eps: float,
    fn_broadcast: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return INDEXER_BACKEND._call_flag(
        "mhc_pre_broadcast_tilelang",
        None,
        None,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        native=_native_broadcast,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        fn_broadcast=fn_broadcast,
    )


def _mhc_pre_broadcast_fake(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    *,
    norm_weight,
    norm_eps,
    fn_broadcast,
):
    tokens, hidden = residual.shape
    streams = fn.shape[1] // hidden
    return (
        residual.new_empty((tokens, streams, hidden)),
        residual.new_empty((tokens, streams, 1), dtype=torch.float32),
        residual.new_empty((tokens, streams, streams), dtype=torch.float32),
        torch.empty_strided(
            residual.shape,
            residual.stride(),
            dtype=residual.dtype,
            device=residual.device,
        )
        if tokens == 0
        else residual.new_empty((tokens, hidden)),
    )


direct_register_custom_op(
    op_name="glm5_mhc_pre_broadcast",
    op_func=_mhc_pre_broadcast,
    mutates_args=[],
    fake_impl=_mhc_pre_broadcast_fake,
)


def mhc_pre_broadcast_tilelang(*args, **kwargs):
    # Keep common-policy resolution and optional compiler host code outside
    # Dynamo tracing, like vLLM's other mHC entry points.
    return torch.ops.vllm.glm5_mhc_pre_broadcast(*args, **kwargs)
