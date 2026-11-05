# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Sunrise adapters around the official vLLM 0.32 Runner V2.

The native V2 scheduler, buffer, attention and KV-cache contracts are retained.
Device API substitutions are local to V2 modules in a PTPU worker process.
"""

import sys
from types import ModuleType

import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu import model_runner as upstream

from vllm_fl.dispatch.io_common import managed_inference_mode
from vllm_fl.dispatch.io_dumper import (
    advance_io_step,
    init_io_dump_from_env,
    register_io_module_hooks,
)


class _DeviceRuntime:
    def __getattr__(self, name):
        if name == "OutOfMemoryError":
            return torch.OutOfMemoryError
        if name == "CUDAGraph":
            return current_platform.torch_device_fn.PTPUGraph
        return getattr(current_platform.torch_device_fn, name)


class _AcceleratorRuntime:
    _native_names = {
        "get_memory_info": "mem_get_info",
        "current_device_index": "current_device",
        "set_device_index": "set_device",
    }

    def __getattr__(self, name):
        native_name = self._native_names.get(name, name)
        runtime = current_platform.torch_device_fn
        if hasattr(runtime, native_name):
            return getattr(runtime, native_name)
        return getattr(torch.accelerator, name)


class _TorchRuntime:
    cuda = _DeviceRuntime()
    accelerator = _AcceleratorRuntime()

    def __getattr__(self, name):
        return getattr(torch, name)


def install_sunrise_v2_runtime():
    """Redirect V2 module-local streams/events/graphs without changing torch.cuda."""
    if current_platform.device_type != "ptpu":
        raise RuntimeError("SunriseV2ModelRunner requires the PTPU platform")
    # V2 imports its buffer, async-output, PP, structured-output and graph helpers
    # before constructing their device state. Each helper keeps its own torch
    # global; redirect these exact references before any CUDA API can execute.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("vllm.v1.worker.gpu."):
            continue
        if not isinstance(module, ModuleType):
            continue
        namespace = vars(module)
        if namespace.get("torch") is torch:
            namespace["torch"] = _TorchRuntime()


class SunriseV2ModelRunner(upstream.GPUModelRunner):
    """Native Runner V2 using Sunrise streams, graphs and FL operator dispatch."""

    def __init__(self, vllm_config, device):
        install_sunrise_v2_runtime()
        super().__init__(vllm_config, device)

    def load_model(self, *args, **kwargs):
        init_io_dump_from_env(getattr(self.model_config, "enforce_eager", False))
        install_sunrise_v2_runtime()
        result = super().load_model(*args, **kwargs)
        install_sunrise_v2_runtime()
        register_io_module_hooks(self.model)
        draft_model = self.get_draft_model()
        if draft_model is not None:
            register_io_module_hooks(draft_model)
        return result

    @managed_inference_mode()
    def execute_model(self, *args, **kwargs):
        # Strip only the outer torch.inference_mode wrapper. Preserve native
        # functional wrappers, such as V2's EPLB step hook, on inner callables.
        return upstream.GPUModelRunner.execute_model.__wrapped__(self, *args, **kwargs)

    @managed_inference_mode()
    def sample_tokens(self, *args, **kwargs):
        result = upstream.GPUModelRunner.sample_tokens.__wrapped__(
            self, *args, **kwargs
        )
        advance_io_step()
        return result

    @managed_inference_mode()
    def _dummy_run(self, *args, **kwargs):
        return upstream.GPUModelRunner._dummy_run.__wrapped__(self, *args, **kwargs)
