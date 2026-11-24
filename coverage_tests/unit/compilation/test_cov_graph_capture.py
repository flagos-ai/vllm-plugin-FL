# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the successful GraphWrapper capture and debug replay paths.

``vllm_fl.compilation.graph`` resolves its graph backend at import from the
platform device type; the probe shim below imports it under the A100 value
and restores the real platform.  Capture is driven with a fake
``torch_device_fn.graph`` context manager, a fake graph factory and a fake
offloader module, so no device graph is instantiated.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)

_original_device_type = current_platform.device_type
if _original_device_type != "cuda":
    current_platform.device_type = "cuda"
import vllm_fl.compilation.graph as graph_module  # noqa: E402

current_platform.device_type = _original_device_type
if "device_type" in vars(current_platform):
    del current_platform.device_type

from vllm.compilation.counter import compilation_counter  # noqa: E402
from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    BatchDescriptor,
    ForwardContext,
    override_forward_context,
)


class _FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class _FakeDeviceFn:
    def __init__(self):
        self.graph_calls = []

    @contextmanager
    def graph(self, graph, pool=None, **kwargs):
        self.graph_calls.append((graph, pool, kwargs))
        yield


class _FakeRuntime:
    def __init__(self):
        self.capture_context = None
        self.replay_context = None

    @contextmanager
    def capture_scope(self, context):
        self.capture_context = context
        yield

    @contextmanager
    def replay_scope(self, context):
        self.replay_context = context
        yield


def _config(cudagraph_copy_inputs=False, tp=1):
    config = SimpleNamespace(
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_copy_inputs=cudagraph_copy_inputs),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(),
    )
    return config


def _context(mode, descriptor):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=None,
        slot_mapping=None,
        cudagraph_runtime_mode=mode,
        batch_descriptor=descriptor,
    )


def _install_offloader(monkeypatch, *, events, fail_import=False):
    module = types.ModuleType("vllm.model_executor.offloader.base")

    def get_offloader():
        if fail_import:
            raise ImportError("offloader unavailable")
        return SimpleNamespace(
            sync_prev_onload=lambda: events.append("sync"),
            join_after_forward=lambda: events.append("join"),
        )

    module.get_offloader = get_offloader
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture(autouse=True)
def _restore_capturing_flag():
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    set_cudagraph_capturing_enabled(True)
    yield
    set_cudagraph_capturing_enabled(True)


def _capture_environment(monkeypatch, *, device_type="cuda"):
    device_fn = _FakeDeviceFn()
    pool_calls = []
    monkeypatch.setattr(graph_module.current_platform, "device_type", device_type)
    monkeypatch.setattr(graph_module.current_platform, "torch_device_fn", device_fn)
    monkeypatch.setattr(
        graph_module.current_platform,
        "get_global_graph_pool",
        lambda: "GLOBAL-POOL",
    )
    monkeypatch.setattr(
        graph_module, "set_graph_pool_id", lambda pool: pool_calls.append(pool)
    )
    monkeypatch.setattr(graph_module, "weak_ref_tensors", lambda value: value)
    captured = []

    def factory():
        graph = _FakeGraph()
        captured.append(graph)
        return graph

    monkeypatch.setattr(graph_module.Graph, "graph", staticmethod(factory))
    monkeypatch.setattr(
        graph_module, "validate_cudagraph_capturing_enabled", lambda: None
    )
    return device_fn, pool_calls, captured


def test_capture_records_entry_output_and_inputs(monkeypatch):
    """Capture returns the runnable output and records addresses/tensors."""
    events = []
    _install_offloader(monkeypatch, events=events)
    device_fn, pools, captured = _capture_environment(monkeypatch)
    runtime = _FakeRuntime()
    seen = []
    output = None

    def runnable(*args, **kwargs):
        seen.append((args, kwargs))
        nonlocal output
        output = args[0] + 1
        return output

    wrapper = graph_module.GraphWrapper(
        runnable,
        _config(cudagraph_copy_inputs=True),
        CUDAGraphMode.PIECEWISE,
        cudagraph_options=graph_module.GraphOptions(gc_disable=True),
        graph_runtime=runtime,
    )
    descriptor = BatchDescriptor(num_tokens=4)
    tensor = torch.arange(4)
    before = compilation_counter.num_cudagraph_captured
    with override_forward_context(_context(CUDAGraphMode.PIECEWISE, descriptor)):
        result = wrapper(tensor)

    entry = wrapper.concrete_graph_entries[descriptor]
    assert result is output
    assert seen == [((tensor,), {})]
    assert entry.graph is captured[0]
    assert entry.output is output
    assert entry.input_addresses == [tensor.data_ptr()]
    assert entry.input_tensors == [tensor]
    assert device_fn.graph_calls == [(captured[0], "GLOBAL-POOL", {})]
    assert pools == ["GLOBAL-POOL"]
    assert runtime.capture_context.cudagraph_runtime_mode is CUDAGraphMode.PIECEWISE
    assert events == ["sync", "join"]
    assert compilation_counter.num_cudagraph_captured == before + 1


def test_capture_without_copy_inputs_keeps_only_addresses(monkeypatch):
    """Without the copy option only the input addresses are stored."""
    _install_offloader(monkeypatch, events=[])
    _capture_environment(monkeypatch)

    def runnable(tensor):
        return tensor * 2

    wrapper = graph_module.GraphWrapper(
        runnable, _config(cudagraph_copy_inputs=False), CUDAGraphMode.PIECEWISE
    )
    descriptor = BatchDescriptor(num_tokens=2)
    tensor = torch.ones(2)
    with override_forward_context(_context(CUDAGraphMode.PIECEWISE, descriptor)):
        wrapper(tensor)
    entry = wrapper.concrete_graph_entries[descriptor]
    assert entry.input_addresses == [tensor.data_ptr()]
    assert entry.input_tensors is None


def test_capture_uses_dynamic_pool_when_no_global_pool(monkeypatch):
    """A missing global pool falls back to the platform pool handle."""
    _install_offloader(monkeypatch, events=[])
    device_fn, pool_calls, _ = _capture_environment(monkeypatch)
    monkeypatch.setattr(
        graph_module.current_platform, "get_global_graph_pool", lambda: None
    )
    handles = []
    monkeypatch.setattr(
        graph_module.current_platform,
        "graph_pool_handle",
        lambda: handles.append(1) or "DYNAMIC-POOL",
    )
    wrapper = graph_module.GraphWrapper(
        lambda tensor: tensor,
        _config(),
        CUDAGraphMode.FULL,
    )
    descriptor = BatchDescriptor(num_tokens=1)
    with override_forward_context(_context(CUDAGraphMode.FULL, descriptor)):
        wrapper(torch.zeros(1))
    assert handles == [1]
    # The wrapper stores graph_pool=None; the dynamic handle is only handed to
    # set_graph_pool_id, while the capture context receives the stored pool.
    assert device_fn.graph_calls[0][1] is None
    assert pool_calls == ["DYNAMIC-POOL"]


def test_capture_survives_missing_offloader(monkeypatch):
    """Optional offloader hooks may raise ImportError without failing capture."""
    _install_offloader(monkeypatch, events=[], fail_import=True)
    _capture_environment(monkeypatch)
    wrapper = graph_module.GraphWrapper(
        lambda tensor: tensor,
        _config(),
        CUDAGraphMode.PIECEWISE,
    )
    descriptor = BatchDescriptor(num_tokens=1)
    tensor = torch.tensor([3.0])
    with override_forward_context(_context(CUDAGraphMode.PIECEWISE, descriptor)):
        result = wrapper(tensor)
    assert torch.equal(result, tensor)


def test_capture_txda_warms_up_and_relaxes_capture(monkeypatch):
    """txda runs one eager iteration and captures with relaxed error mode."""
    events = []
    _install_offloader(monkeypatch, events=events)
    device_fn, _, _ = _capture_environment(monkeypatch, device_type="txda")
    guard_calls = []
    patch_module = types.ModuleType("vllm_fl.dispatch.backends.vendor.txda.patch")

    def capture_guard_context():
        guard_calls.append(1)
        return nullcontext()

    patch_module.capture_guard_context = capture_guard_context
    monkeypatch.setitem(sys.modules, patch_module.__name__, patch_module)

    calls = []

    def runnable(tensor):
        calls.append(tensor)
        return tensor + 1

    wrapper = graph_module.GraphWrapper(runnable, _config(), CUDAGraphMode.PIECEWISE)
    descriptor = BatchDescriptor(num_tokens=3)
    tensor = torch.zeros(3)
    with override_forward_context(_context(CUDAGraphMode.PIECEWISE, descriptor)):
        wrapper(tensor)
    assert len(calls) == 2  # eager warmup + capture
    assert guard_calls == [1]
    assert device_fn.graph_calls[0][2] == {"capture_error_mode": "relaxed"}


def test_debug_replay_requires_unchanged_input_addresses(monkeypatch):
    """Debug mode asserts the replay addresses match the capture addresses."""
    _install_offloader(monkeypatch, events=[])
    monkeypatch.setattr(graph_module.envs, "VLLM_LOGGING_LEVEL", "DEBUG")
    wrapper = graph_module.GraphWrapper(MagicMock(), _config(), CUDAGraphMode.PIECEWISE)
    descriptor = BatchDescriptor(num_tokens=2)
    tensor = torch.ones(2)
    graph = _FakeGraph()
    wrapper.concrete_graph_entries[descriptor] = graph_module.GraphEntry(
        batch_descriptor=descriptor,
        graph=graph,
        output=tensor,
        input_addresses=[tensor.data_ptr()],
    )
    with override_forward_context(_context(CUDAGraphMode.PIECEWISE, descriptor)):
        assert wrapper(tensor) is tensor
    assert graph.replays == 1

    other = torch.ones(2)
    with (
        override_forward_context(_context(CUDAGraphMode.PIECEWISE, descriptor)),
        pytest.raises(AssertionError, match="Input addresses"),
    ):
        wrapper(other)


def test_debug_mode_getattr_reports_the_runnable(monkeypatch):
    """Debug mode includes the runnable string in a missing-attribute error."""
    monkeypatch.setattr(graph_module.envs, "VLLM_LOGGING_LEVEL", "DEBUG")
    wrapper = graph_module.GraphWrapper(
        SimpleNamespace(value=1), _config(), CUDAGraphMode.PIECEWISE
    )
    assert wrapper.value == 1
    with pytest.raises(AttributeError, match="not exists in the runnable"):
        wrapper.missing  # noqa: B018 - access is the trigger

    monkeypatch.setattr(graph_module.envs, "VLLM_LOGGING_LEVEL", "INFO")
    quiet = graph_module.GraphWrapper(
        SimpleNamespace(), _config(), CUDAGraphMode.PIECEWISE
    )
    with pytest.raises(AttributeError):
        quiet.missing  # noqa: B018 - access is the trigger


def test_wrapper_rejects_none_runtime_mode():
    """A wrapper is never constructed for CUDAGraphMode.NONE."""
    with pytest.raises(AssertionError):
        graph_module.GraphWrapper(MagicMock(), _config(), CUDAGraphMode.NONE)
