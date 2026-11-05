# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for the isolated native vLLM 0.32 runner adapter."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.dispatch import io_common


@pytest.fixture
def runner_module(monkeypatch):
    def install(name, **attrs):
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is None:
                parent = install(parent_name)
        module = ModuleType(name)
        module.__path__ = []
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            monkeypatch.setattr(parent, child, module, raising=False)
        return module

    class NativeRunner:
        def __init__(self, config, device):
            self.model_config = config.model_config
            self.device = device
            self.model = object()

        @torch.inference_mode()
        def execute_model(self, *args, **kwargs):
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

        sample_tokens = execute_model
        _dummy_run = execute_model
        _dummy_sampler_run = execute_model
        _dummy_pooler_run = execute_model

        def load_model(self, *args, **kwargs):
            return "loaded"

    runtime = SimpleNamespace(Stream=Mock(), synchronize=Mock())
    platform = SimpleNamespace(device_type="ptpu", torch_device_fn=runtime)
    install("vllm.platforms", current_platform=platform)
    upstream = install(
        "vllm.v1.worker.gpu_model_runner", GPUModelRunner=NativeRunner, torch=torch
    )
    wrapper = type("GraphWrapper", (), {})
    install("vllm_fl.compilation.graph", GraphWrapper=wrapper)
    io = install(
        "vllm_fl.dispatch.io_dumper",
        advance_io_step=Mock(),
        init_io_dump_from_env=Mock(),
        register_io_module_hooks=Mock(),
    )
    monkeypatch.setattr(io_common, "_io_active", False)
    path = (
        Path(__file__).resolve().parents[3] / "vllm_fl/worker/sunrise_model_runner.py"
    )
    spec = importlib.util.spec_from_file_location("_sunrise_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        upstream=upstream,
        runtime=runtime,
        platform=platform,
        wrapper=wrapper,
        io=io,
    )


def make_runner(state):
    return state.module.SunriseModelRunner(
        SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True)), "ptpu:1"
    )


def test_only_the_runner_module_receives_device_adapters(runner_module):
    state = runner_module
    native_cuda = torch.cuda
    runner = make_runner(state)
    assert runner.device == "ptpu:1"
    assert torch.cuda is native_cuda
    assert state.upstream.torch is not torch
    assert state.upstream.torch.cuda.Stream is state.runtime.Stream
    assert state.upstream.torch.accelerator.synchronize is state.runtime.synchronize
    assert state.upstream.torch.empty is torch.empty
    assert state.upstream.torch.cuda.OutOfMemoryError is torch.OutOfMemoryError
    assert state.upstream.CUDAGraphWrapper is state.wrapper
    proxy = state.upstream.torch
    state.module.install_sunrise_runner_runtime()
    assert state.upstream.torch is proxy


@pytest.mark.parametrize("device", ["cuda", "cpu", "npu"])
def test_other_backends_cannot_install_the_ptpu_adapter(runner_module, device):
    state = runner_module
    state.platform.device_type = device
    with pytest.raises(RuntimeError, match="PTPU"):
        state.module.install_sunrise_runner_runtime()
    assert state.upstream.torch is torch


@pytest.mark.parametrize("io_active", [False, True])
@pytest.mark.parametrize(
    "method",
    [
        "execute_model",
        "sample_tokens",
        "_dummy_run",
        "_dummy_sampler_run",
        "_dummy_pooler_run",
    ],
)
def test_native_inference_decorator_respects_fl_io_scope(
    runner_module, io_active, method
):
    state = runner_module
    runner = make_runner(state)
    io_common.set_io_active(io_active)
    assert getattr(runner, method)() == (not io_active, False)
    assert state.io.advance_io_step.call_count == int(method == "sample_tokens")


def test_model_and_drafter_receive_io_hooks(runner_module):
    state = runner_module
    runner = make_runner(state)
    runner.drafter = SimpleNamespace(model=object())
    assert runner.load_model() == "loaded"
    state.io.init_io_dump_from_env.assert_called_once_with(True)
    assert state.io.register_io_module_hooks.call_count == 2
