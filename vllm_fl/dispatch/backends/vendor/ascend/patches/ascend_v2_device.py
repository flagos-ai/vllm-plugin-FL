# Copyright (c) 2026 BAAI. All rights reserved.

"""Route the upstream V2 runner's device helpers through the NPU API.

The upstream V2 modules use CUDA stream/event names even though their tensor
buffers accept an arbitrary device. Replace their local ``torch`` binding;
never mutate ``torch.cuda`` or the process-wide PyTorch module.
"""

import importlib
from types import ModuleType

import torch

_DEVICE_MODULES = (
    "vllm.v1.worker.gpu.async_utils",
    "vllm.v1.worker.gpu.pp_utils",
    "vllm.v1.worker.gpu.structured_outputs",
    "vllm.v1.worker.gpu.spec_decode.utils",
    "vllm.v1.worker.gpu.spec_decode.adaptive_verification",
    "vllm.v1.worker.gpu.cudagraph_utils",
    "vllm.v1.worker.gpu.ubatch_utils",
    "vllm.v1.worker.gpu.model_runner",
)


class _DeviceFacade:
    def __init__(self, device_api):
        self._device_api = device_api

    def __getattr__(self, name):
        # torch_npu uses NPUGraph for the platform-neutral graph interface.
        if name == "CUDAGraph":
            return self._device_api.NPUGraph
        return getattr(self._device_api, name)


class _TorchFacade(ModuleType):
    def __init__(self, device_api):
        super().__init__("torch")
        self.cuda = _DeviceFacade(device_api)

    def __getattr__(self, name):
        return getattr(torch, name)


def _no_uva() -> bool:
    # NPU pinned memory supports H2D copies, but not the CUDA-only accelerator
    # views used by UvaBuffer. Keep the upstream NonUvaBuffer copy fallback.
    return False


def install_ascend_v2_device_shims() -> bool:
    """Install NPU-local bindings before constructing an upstream V2 runner."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        return False

    facade = _TorchFacade(current_platform.torch_device_fn)
    for name in _DEVICE_MODULES:
        module = importlib.import_module(name)
        if module.torch is torch or isinstance(module.torch, _TorchFacade):
            module.torch = facade
        else:
            raise RuntimeError(f"V2 module {name} already owns a custom torch binding")

    buffers = importlib.import_module("vllm.v1.worker.gpu.buffer_utils")
    buffers.is_uva_available = _no_uva
    return True
