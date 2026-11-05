# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""PTPU adapters around the canonical vLLM 0.32 model runner.

Keep the upstream runner's KV-cache, sampling and scheduler contracts intact.
Only its module-local CUDA handles are redirected; torch.cuda itself is untouched.
"""

import inspect

import torch

from vllm.platforms import current_platform
from vllm.v1.worker import gpu_model_runner as upstream

from vllm_fl.compilation.graph import GraphWrapper
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
        return getattr(current_platform.torch_device_fn, name)


class _AcceleratorRuntime:
    def __getattr__(self, name):
        if name == "synchronize":
            return current_platform.torch_device_fn.synchronize
        return getattr(torch.accelerator, name)


class _TorchRuntime:
    cuda = _DeviceRuntime()
    accelerator = _AcceleratorRuntime()

    def __getattr__(self, name):
        return getattr(torch, name)


def install_sunrise_runner_runtime():
    """Adapt only the upstream runner module in a Sunrise worker process."""
    if current_platform.device_type != "ptpu":
        raise RuntimeError("SunriseModelRunner requires the PTPU platform")
    if getattr(upstream, "_fl_sunrise_runtime_installed", False):
        return
    upstream.torch = _TorchRuntime()
    upstream.CUDAGraphWrapper = GraphWrapper
    upstream._fl_sunrise_runtime_installed = True


class SunriseModelRunner(upstream.GPUModelRunner):
    """The native 0.32 runner with PTPU streams, graphs and FL IO hooks."""

    def __init__(self, vllm_config, device):
        install_sunrise_runner_runtime()
        super().__init__(vllm_config, device)

    def load_model(self, *args, **kwargs):
        init_io_dump_from_env(getattr(self.model_config, "enforce_eager", False))
        result = super().load_model(*args, **kwargs)
        register_io_module_hooks(self.model)
        drafter_model = getattr(getattr(self, "drafter", None), "model", None)
        if drafter_model is not None:
            register_io_module_hooks(drafter_model)
        return result

    @managed_inference_mode()
    def execute_model(self, *args, **kwargs):
        return inspect.unwrap(upstream.GPUModelRunner.execute_model)(
            self, *args, **kwargs
        )

    @managed_inference_mode()
    def sample_tokens(self, *args, **kwargs):
        result = inspect.unwrap(upstream.GPUModelRunner.sample_tokens)(
            self, *args, **kwargs
        )
        advance_io_step()
        return result

    @managed_inference_mode()
    def _dummy_run(self, *args, **kwargs):
        return inspect.unwrap(upstream.GPUModelRunner._dummy_run)(self, *args, **kwargs)

    @managed_inference_mode()
    def _dummy_sampler_run(self, *args, **kwargs):
        return inspect.unwrap(upstream.GPUModelRunner._dummy_sampler_run)(
            self, *args, **kwargs
        )

    @managed_inference_mode()
    def _dummy_pooler_run(self, *args, **kwargs):
        return inspect.unwrap(upstream.GPUModelRunner._dummy_pooler_run)(
            self, *args, **kwargs
        )
