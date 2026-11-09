# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the MetaX SDK startup route independently of model registration."""

import os
from importlib.metadata import version

from vllm.logger import init_logger

from vllm_fl.patches.metax_open import uses_open_metax_runtime

logger = init_logger(__name__)

_REGISTERED = False


def _register_open_runtime() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    if version("vllm").split("+", 1)[0] != "0.28.0":
        raise RuntimeError("The MACA/MCCL ABI adapters are pinned to vLLM 0.28.0")
    if os.environ.get("FLAGCX_PATH"):
        raise RuntimeError("The open MetaX runtime uses MCCL; FlagCX is not enabled")
    import vllm.envs as envs

    if envs.VLLM_USE_NCCL_SYMM_MEM:
        raise NotImplementedError("MCCL symmetric-memory windows are not supported")

    from vllm.distributed.device_communicators import (
        cuda_wrapper,
        pynccl,
        pynccl_wrapper,
    )
    from vllm.utils import nccl

    from .patches.abi import CudaRTLibrary, MCCLLibrary, find_mccl_library

    cuda_wrapper.CudaRTLibrary = CudaRTLibrary
    pynccl.NCCLLibrary = MCCLLibrary
    pynccl_wrapper.NCCLLibrary = MCCLLibrary
    nccl.find_nccl_library = find_mccl_library
    _REGISTERED = True


def register_runtime() -> None:
    if uses_open_metax_runtime():
        _register_open_runtime()
        return
    # Preserve the default vendor import sequence and failure reporting.
    try:
        import mcoplib._C  # noqa: F401
    except ImportError:
        logger.warning("Failed to import mcoplib._C")
    try:
        import mcoplib._moe_C  # noqa: F401
    except ImportError:
        logger.warning("Failed to import mcoplib._moe_C")
    try:
        import vllm_fl.dispatch.backends.vendor.metax.patches  # noqa: F401
    except Exception as error:
        logger.warning("Failed to import maca patches: %s", error)
