# SPDX-License-Identifier: Apache-2.0
"""Native ABI and compiler capabilities for common GLM operator policy."""

from __future__ import annotations

import importlib
from functools import lru_cache

from vllm.logger import init_logger
from vllm.platforms import current_platform

_VLLM_NATIVE_EXTENSIONS = ("vllm._C", "vllm._C_stable_libtorch")

logger = init_logger(__name__)


def _has_vllm_native_extension() -> bool:
    """Return whether a vLLM native CUDA ABI can be imported.

    vLLM 0.24 deployments use either the legacy ``_C`` extension or the
    stable-ABI ``_C_stable_libtorch`` extension.  The empty-build wheel has
    neither one, even though it still provides the Python vLLM package.
    """
    for module_name in _VLLM_NATIVE_EXTENSIONS:
        try:
            importlib.import_module(module_name)
        except (ImportError, OSError, RuntimeError):
            continue
        return True
    return False


def _has_deep_gemm() -> bool:
    """Probe the vLLM DeepGEMM capability without failing model import."""
    try:
        from vllm.utils.deep_gemm import has_deep_gemm
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False

    try:
        return bool(has_deep_gemm())
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def _has_nvidia_reference_kernels() -> bool:
    """Check the native ABI and DeepGEMM prerequisites for GLM5's fast path."""
    return _has_vllm_native_extension() and _has_deep_gemm()


def use_nvidia_reference() -> bool:
    return current_platform.is_cuda() and _has_nvidia_reference_kernels()


__all__ = ["use_nvidia_reference"]


INDEXER_OPERATORS = (
    "per_token_group_quant_fp8",
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "pack_seq_triton",
    "unpack_seq_triton",
)
