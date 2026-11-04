# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in MetaX support for the unchanged upstream vLLM 0.32 V2 runner."""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _has_native_mapped_cpu_view() -> bool:
    # The input is CPU: the real native registration is a CPU kernel. A schema
    # or the empty wheel's Composite placeholder cannot map host memory.
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::get_cuda_view_from_cpu_tensor", "CPU"
        )
    except RuntimeError:
        return False


def _no_uva() -> bool:
    return False


def prepare_metax_v2_runtime(vllm_config) -> bool:
    """Configure vendor compatibility before constructing upstream V2 buffers.

    Keep global pinned-memory/platform predicates unchanged. The upstream
    NonUvaBuffer retains its own copies, concurrency pool and lifetime rules.
    """
    if not (
        getattr(current_platform, "vendor_name", None) == "metax"
        and vllm_config.use_v2_model_runner
        and os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1"
    ):
        return False

    from vllm.utils import torch_utils
    from vllm.v1.worker.gpu import buffer_utils

    from vllm_fl.compilation.graph import _weak_ref_metax_tensor

    if not _has_native_mapped_cpu_view():
        buffer_utils.is_uva_available = _no_uva
        logger.info_once(
            "MetaX V2 uses upstream NonUvaBuffer: no native mapped CPU-view kernel."
        )
    # Native CUDAGraphWrapper keeps the upstream V2 profiling/cleanup registry.
    # Its imported weak_ref_tensors calls this module global at runtime. Reuse
    # the existing MetaX native-first primitive, including its borrowed-storage
    # fallback, rather than introducing another output lifetime mechanism.
    torch_utils.weak_ref_tensor = _weak_ref_metax_tensor
    return True
