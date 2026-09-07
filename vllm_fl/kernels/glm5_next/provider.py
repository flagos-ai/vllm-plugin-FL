# SPDX-License-Identifier: Apache-2.0
"""Process-start provider override for GLM5-Next A/B validation."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Literal, cast

from vllm.logger import init_logger
from vllm.platforms import current_platform

Provider = Literal["auto", "nvidia", "flaggems"]
ENV_NAME = "VLLM_FL_GLM5_PROVIDER"
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


@lru_cache(maxsize=1)
def get_glm5_provider() -> Provider:
    value = os.environ.get(ENV_NAME, "auto").strip().lower()
    if value not in ("auto", "nvidia", "flaggems"):
        raise ValueError(
            f"{ENV_NAME} must be one of auto|nvidia|flaggems, got {value!r}"
        )
    if value == "nvidia" and not current_platform.is_cuda():
        raise RuntimeError(f"{ENV_NAME}=nvidia requires an NVIDIA CUDA platform")
    return cast(Provider, value)


def use_nvidia_reference() -> bool:
    provider = get_glm5_provider()
    if provider == "nvidia":
        return True
    if provider != "auto" or not current_platform.is_cuda():
        return False

    if _has_nvidia_reference_kernels():
        return True

    logger.warning_once(
        "GLM5-Next auto provider could not find both a vLLM native CUDA ABI "
        "and DeepGEMM; using the FlagGems/Triton/portable implementation"
    )
    return False


__all__ = ["ENV_NAME", "get_glm5_provider", "use_nvidia_reference"]
