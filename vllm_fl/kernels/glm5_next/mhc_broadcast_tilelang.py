# SPDX-License-Identifier: Apache-2.0
"""Common vendor dispatch for the optional upstream mHC broadcast API."""

from __future__ import annotations

import importlib
import inspect
from contextlib import suppress

import torch

from .indexer_backend import INDEXER_BACKEND
from .provider import use_nvidia_reference

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

    backport = importlib.import_module("vllm_fl._vendor.mhc_broadcast_tilelang")
    return backport.mhc_pre_broadcast_tilelang


_IMPL = _load_broadcast_impl()


def _native_broadcast(*args, **kwargs):
    return _IMPL(*args, **kwargs)


def mhc_pre_broadcast_tilelang(*args, **kwargs):
    return INDEXER_BACKEND._call_flag(
        "mhc_pre_broadcast_tilelang",
        None,
        None,
        *args,
        native=_native_broadcast,
        **kwargs,
    )
