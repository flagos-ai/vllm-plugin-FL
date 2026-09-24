# SPDX-License-Identifier: Apache-2.0
"""Explicit vLLM 0.24 dispatch for vendored Qwen4 math kernels.

The official image was built against a newer vLLM and therefore cannot be
selected implicitly by model registration.  The default ``fallback`` mode
keeps the already validated FlagOS implementation.  Setting
``QWEN4_HC_BACKEND=official`` opts into the vendored HC math only after the
runtime proves it is an NVIDIA CUDA/Triton path.  There is no silent second
dispatch path.

QSA runs the local gpu/ops/qsa.py composition. The vendored official QSA
modules are reference sources, not selected runtime implementations. Identity
reports resolve the actual local entry points and selected compression callable.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
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
            HAS_TRITON and current_platform.is_cuda() and not current_platform.is_rocm()
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


def _callable_identity(fn: Callable[..., Any]) -> dict[str, Any]:
    source = inspect.getsourcefile(fn)
    return {
        "callable": f"{fn.__module__}.{fn.__qualname__}",
        "source": source,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if source is not None
        else None,
    }


def qsa_runtime_status(
    *,
    compression_impl: Callable[..., Any] | None = None,
    select_all_tokens: bool = False,
) -> dict[str, Any]:
    """Describe executable entry points; this is selection, not a launch trace.

    An indexer instance supplies its shape-selected compression callable. Without
    an instance, list both candidates without claiming which branch ran. Device,
    shape and capture gates inside these entry points still require trace evidence.
    """
    from ...gpu.ops import qsa

    stages = {
        "metadata": qsa.build_qsa_forward_metadata,
        "mqa": qsa.qsa_mqa_paged,
        "topk": qsa._qsa_deterministic_block_topk,
        "selection": qsa.qsa_select_paged_tokens,
        "attention": qsa.qsa_sparse_paged_attention,
        "cache_store": qsa.qsa_store_cache_rows,
    }
    if select_all_tokens:
        stages = {
            "metadata": qsa.build_qsa_forward_metadata,
            "selection": qsa.qsa_select_all_paged_tokens,
            "attention": qsa.qsa_sparse_paged_attention,
        }
    elif compression_impl is not None:
        stages["compression"] = compression_impl
    return {
        "backend": "local_qsa_composition",
        "evidence": "selected_callables_not_launch_trace",
        "stages": {name: _callable_identity(fn) for name, fn in stages.items()},
        "compression_candidates": []
        if compression_impl is not None or select_all_tokens
        else [
            _callable_identity(qsa.qsa_compress_norm_mrope_store_groups),
            _callable_identity(qsa.qsa_compress_groups_with_ratio),
        ],
        "official_pre_indexer_enabled": False,
    }


def qwen4_qsa_pre_indexer_status() -> dict[str, Any]:
    """Compatibility status API: the vendored pre-indexer is not dispatched."""
    return {
        "backend": "local_qsa_composition",
        "official_kernel_vendored": True,
        "enabled": False,
        "reason": "local QSA composition is selected; vendor pre-indexer is reference-only",
        "metadata_graph": "vllm024_local",
        "runtime": qsa_runtime_status(),
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
    "qsa_runtime_status",
    "use_official_hc",
]
