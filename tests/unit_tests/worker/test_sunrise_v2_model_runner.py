# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Runner V2 device isolation and native execution hook contracts."""

import functools
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.dispatch import io_common


@pytest.fixture
def runner_v2(monkeypatch):
    events = []

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

    def eplb(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            events.append("eplb")
            return fn(*args, **kwargs)

        return wrapped

    class NativeV2:
        def __init__(self, config, device):
            self.model_config = config.model_config
            self.device = device
            self.model = object()

        @torch.inference_mode()
        def execute_model(self, *args, **kwargs):
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

        @torch.inference_mode()
        @eplb
        def sample_tokens(self, *args, **kwargs):
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

        _dummy_run = sample_tokens

        def load_model(self, *args, **kwargs):
            return "loaded"

        def get_draft_model(self):
            return None

    class NativeGraphManager:
        def _capture_stream(self, desc):
            return self.capture_stream

        def run_fullgraph(self, desc):
            events.append(("replay", desc))
            return desc

        def release_graphs(self):
            events.append("release")

    runtime = SimpleNamespace(
        Stream=Mock(),
        PTPUGraph=type("PTPUGraph", (), {}),
        synchronize=Mock(),
        mem_get_info=Mock(),
        set_device=Mock(),
    )
    platform = SimpleNamespace(device_type="ptpu", torch_device_fn=runtime)
    install("vllm.platforms", current_platform=platform)
    upstream = install(
        "vllm.v1.worker.gpu.model_runner",
        GPUModelRunner=NativeV2,
        ModelCudaGraphManager=NativeGraphManager,
        torch=torch,
    )
    helpers = [
        install("vllm.v1.worker.gpu." + name, torch=torch)
        for name in ("async_utils", "pp_utils", "cudagraph_utils", "structured_outputs")
    ]
    unrelated = install("_sunrise_v2_unrelated", torch=torch)
    io = install(
        "vllm_fl.dispatch.io_dumper",
        advance_io_step=Mock(),
        init_io_dump_from_env=Mock(),
        register_io_module_hooks=Mock(),
    )
    monkeypatch.setattr(io_common, "_io_active", False)
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/worker/sunrise_v2_model_runner.py"
    )
    spec = importlib.util.spec_from_file_location("_sunrise_v2_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        upstream=upstream,
        helpers=helpers,
        unrelated=unrelated,
        runtime=runtime,
        platform=platform,
        events=events,
        io=io,
    )


def make_runner(state, tp=1):
    return state.module.SunriseV2ModelRunner(
        SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        ),
        "ptpu:1",
    )


def test_v2_stream_graph_and_async_helpers_use_ptpu_locally(runner_v2):
    state = runner_v2
    native_cuda = torch.cuda
    make_runner(state)
    for module in [state.upstream, *state.helpers]:
        assert module.torch.cuda.Stream is state.runtime.Stream
        assert module.torch.cuda.CUDAGraph is state.runtime.PTPUGraph
        assert module.torch.accelerator.synchronize is state.runtime.synchronize
        assert module.torch.accelerator.get_memory_info is state.runtime.mem_get_info
        assert module.torch.empty is torch.empty
    assert state.unrelated.torch is torch
    assert torch.cuda is native_cuda
    first_proxy = state.upstream.torch
    state.module.install_sunrise_v2_runtime()
    assert state.upstream.torch is first_proxy


@pytest.mark.parametrize("device", ["cuda", "cpu", "npu"])
def test_v2_adapter_rejects_other_platforms(runner_v2, device):
    state = runner_v2
    state.platform.device_type = device
    with pytest.raises(RuntimeError, match="PTPU"):
        state.module.install_sunrise_v2_runtime()
    assert state.upstream.torch is torch


@pytest.mark.parametrize("io_active", [False, True])
@pytest.mark.parametrize("method", ["execute_model", "sample_tokens", "_dummy_run"])
def test_v2_io_scope_preserves_native_eplb_hook(runner_v2, io_active, method):
    state = runner_v2
    runner = make_runner(state)
    io_common.set_io_active(io_active)
    assert getattr(runner, method)() == (not io_active, False)
    assert state.events == ([] if method == "execute_model" else ["eplb"])
    assert state.io.advance_io_step.call_count == int(method == "sample_tokens")


def test_v2_installs_adapters_for_late_loaded_helpers(runner_v2):
    state = runner_v2
    runner = make_runner(state)
    late_helper = ModuleType("vllm.v1.worker.gpu._sunrise_late_helper_test")
    late_helper.torch = torch
    sys.modules[late_helper.__name__] = late_helper
    try:
        assert runner.load_model() == "loaded"
        assert late_helper.torch.cuda.Stream is state.runtime.Stream
        state.io.init_io_dump_from_env.assert_called_once_with(True)
        state.io.register_io_module_hooks.assert_called_once_with(runner.model)
    finally:
        sys.modules.pop(late_helper.__name__)


def test_sunrise_tp_replay_drains_capture_stream(runner_v2):
    state = runner_v2
    native_manager = state.upstream.ModelCudaGraphManager
    make_runner(state)
    assert state.upstream.ModelCudaGraphManager is native_manager
    make_runner(state, tp=2)
    manager = state.upstream.ModelCudaGraphManager()
    manager.capture_stream = SimpleNamespace(
        wait_stream=lambda stream: state.events.append(("capture_wait", stream))
    )
    caller = SimpleNamespace(
        wait_stream=lambda stream: state.events.append(("caller_wait", stream))
    )

    @contextmanager
    def use_stream(stream):
        state.events.append(("enter", stream))
        yield
        state.events.append(("exit", stream))

    state.runtime.current_stream = lambda: caller
    state.runtime.stream = use_stream
    state.runtime.synchronize.side_effect = lambda: state.events.append("sync")
    assert manager._capture_stream("decode") is manager.capture_stream
    assert manager.run_fullgraph("decode") == "decode"
    assert state.events == [
        ("capture_wait", caller),
        ("enter", manager.capture_stream),
        ("replay", "decode"),
        "sync",
        ("exit", manager.capture_stream),
        ("caller_wait", manager.capture_stream),
    ]
    manager.release_graphs()
    assert not manager._capture_streams
