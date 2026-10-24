# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the permitted MACA/MCCL device-runtime ABI adapters once.

This module registers no model kernels. Missing model functionality must be
implemented through a reviewed open operator adapter, not a vendor backend.
"""

import os
from importlib.metadata import version

_REGISTERED = False


def register_runtime() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    if version("vllm").split("+", 1)[0] != "0.28.0":
        raise RuntimeError("The MACA/MCCL ABI adapters are pinned to vLLM 0.28.0")
    if os.environ.get("FLAGCX_PATH"):
        raise RuntimeError("The K3 MetaX runtime uses MCCL; FlagCX is not enabled")
    import vllm.envs as envs
    if envs.VLLM_USE_NCCL_SYMM_MEM:
        raise NotImplementedError("MCCL symmetric-memory windows are not supported")

    from vllm.distributed.device_communicators import (
        cuda_wrapper, pynccl, pynccl_wrapper,
    )
    from vllm.utils import nccl
    from .cuda_runtime import CudaRTLibrary
    from .mccl_runtime import MCCLLibrary
    from .runtime_library import find_mccl_library

    cuda_wrapper.CudaRTLibrary = CudaRTLibrary
    pynccl.NCCLLibrary = MCCLLibrary
    pynccl_wrapper.NCCLLibrary = MCCLLibrary
    nccl.find_nccl_library = find_mccl_library
    _REGISTERED = True
