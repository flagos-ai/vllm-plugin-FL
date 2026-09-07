# SPDX-License-Identifier: Apache-2.0
"""Explicit vLLM 0.24 dispatch for vendored Qwen4 math kernels.

The official image was built against a newer vLLM and therefore cannot be
selected implicitly by model registration.  The default ``fallback`` mode
keeps the already validated FlagOS implementation.  Setting
``QWEN4_HC_BACKEND=official`` opts into the vendored HC math only after the
runtime proves it is an NVIDIA CUDA/Triton path.  There is no silent second
dispatch path.

QSA uses the official image's fused pre-indexer and QSA math kernels. Only the
model-side vLLM 0.24 metadata/cache ABI glue remains local; there is no second
self-developed QSA implementation selected by this dispatch module.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch


def _requested_hc_backend() -> str:
    backend = os.getenv("QWEN4_HC_BACKEND", "fallback").strip().lower()
    if backend not in {"fallback", "official"}:
        raise ValueError(
            "QWEN4_HC_BACKEND must be exactly 'fallback' or 'official', "
            f"got {backend!r}"
        )
    return backend


def _is_nvidia_cuda_triton_path() -> bool:
    try:
        from vllm.platforms import current_platform
        from vllm.triton_utils import HAS_TRITON

        return bool(
            HAS_TRITON
            and current_platform.is_cuda()
            and not current_platform.is_rocm()
        )
    except (AttributeError, ImportError, RuntimeError):
        return False


def qwen4_hc_backend() -> str:
    """Return the one HC implementation selected for this process."""

    backend = _requested_hc_backend()
    if backend == "official" and not _is_nvidia_cuda_triton_path():
        raise RuntimeError(
            "QWEN4_HC_BACKEND=official requires the NVIDIA CUDA/Triton "
            "runtime; use QWEN4_HC_BACKEND=fallback on other devices"
        )
    return backend


def use_official_hc() -> bool:
    """Whether the explicit vLLM 0.24 HC adapter should load official math."""

    return qwen4_hc_backend() == "official"


def qwen4_qsa_pre_indexer_status() -> dict[str, Any]:
    """Describe the official QSA pre-indexer selected by the vLLM 0.24 adapter."""

    return {
        "backend": "official_qsa_pre_indexer",
        "official_kernel_vendored": True,
        "enabled": True,
        "reason": "official kernel called through the vLLM 0.24 metadata/cache adapter",
        "metadata_graph": "vllm024_compat",
    }


def dispatch_hc_math(
    official_call: Callable[..., torch.Tensor],
    fallback_call: Callable[..., torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run exactly one HC implementation under the explicit backend policy."""

    return (official_call if use_official_hc() else fallback_call)(*args, **kwargs)


__all__ = [
    "dispatch_hc_math",
    "qwen4_hc_backend",
    "qwen4_qsa_pre_indexer_status",
    "use_official_hc",
]
