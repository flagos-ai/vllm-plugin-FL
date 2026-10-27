# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU regressions for Sunrise graph streams and real FlagCX call routing."""

import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def graph_runtime(monkeypatch):
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

    class Stream:
        def __init__(self, device=None):
            self.device = device
            self.wait_stream = Mock()
            self.ptpu_stream = id(self)

    default_stream = Stream()
    selected = SimpleNamespace(stream=default_stream)

    @contextmanager
    def stream_context(stream):
        previous = selected.stream
        selected.stream = stream
        events.append(("stream_enter", stream))
        try:
            yield
        finally:
            selected.stream = previous
            events.append(("stream_exit", stream))

    device_fn = SimpleNamespace(
        Stream=Mock(side_effect=Stream),
        current_stream=lambda: selected.stream,
        stream=stream_context,
        PTPUGraph=type("PTPUGraph", (), {}),
    )
    platform = SimpleNamespace(device_type="ptpu", torch_device_fn=device_fn)
    install("vllm")
    install("vllm.platforms", current_platform=platform)
    install("vllm.utils")
    torch_utils = install(
        "vllm.utils.torch_utils", current_stream=Mock(return_value=default_stream)
    )
    aiter = SimpleNamespace(is_enabled=lambda: False, get_aiter_allreduce=lambda: None)
    install("vllm._aiter_ops", rocm_aiter_ops=aiter)

    @dataclass
    class GraphCaptureContext:
        stream: object

    class GroupCoordinator:
        def __init__(self, name):
            self.name = name
            self.device_communicator = SimpleNamespace()

        @contextmanager
        def graph_capture(self, context=None):
            events.append(("original_group", self.name))
            yield context

    tp = GroupCoordinator("tp")
    pp = GroupCoordinator("pp")

    @contextmanager
    def original_graph_capture(device, graph_capture_context=None):
        events.append(("original_top", device))
        yield graph_capture_context

    parallel_state = install(
        "vllm.distributed.parallel_state",
        GroupCoordinator=GroupCoordinator,
        GraphCaptureContext=GraphCaptureContext,
        graph_capture=original_graph_capture,
        get_tp_group=lambda: tp,
        get_pp_group=lambda: pp,
    )
    alias = install(
        "_sunrise_cached_graph_aliases",
        graph_capture=original_graph_capture,
        current_stream=torch_utils.current_stream,
    )
    unrelated = install(
        "_sunrise_unrelated_graph_aliases", graph_capture=Mock(), current_stream=Mock()
    )
    unrelated_functions = (unrelated.graph_capture, unrelated.current_stream)
    root = Path(__file__).resolve().parents[3]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    patch_module = load(
        "_sunrise_graph_patch_test",
        root / "vllm_fl/dispatch/backends/vendor/sunrise/impl/graph_capture.py",
    )
    return SimpleNamespace(
        patch=patch_module,
        parallel_state=parallel_state,
        torch_utils=torch_utils,
        alias=alias,
        unrelated=unrelated,
        unrelated_functions=unrelated_functions,
        platform=platform,
        runtime=device_fn,
        default_stream=default_stream,
        selected=selected,
        tp=tp,
        pp=pp,
        context_cls=GraphCaptureContext,
        events=events,
        aiter=aiter,
        install=install,
        load=load,
        root=root,
    )


def test_capture_uses_one_ptpu_stream_for_tp_pp_and_cached_aliases(graph_runtime):
    state = graph_runtime
    state.patch.patch_graph_capture_runtime()
    seen = []
    for group in (state.tp, state.pp):
        group.device_communicator.ca_comm = SimpleNamespace(
            capture=lambda name=group.name: capture_hook(state, seen, name)
        )
    with state.alias.graph_capture("ptpu:1") as context:
        capture_stream = context.stream
        assert state.runtime.current_stream() is capture_stream
        assert state.alias.current_stream() is capture_stream
        assert state.torch_utils.current_stream() is capture_stream
        assert seen == ["tp", "pp"]
    assert state.runtime.current_stream() is state.default_stream
    state.runtime.Stream.assert_called_once_with(device="ptpu:1")
    capture_stream.wait_stream.assert_called_once_with(state.default_stream)
    assert seen == ["tp", "pp", "pp_exit", "tp_exit"]
    assert (
        state.unrelated.graph_capture,
        state.unrelated.current_stream,
    ) == state.unrelated_functions


@contextmanager
def capture_hook(state, seen, name):
    assert state.runtime.current_stream() is not state.default_stream
    seen.append(name)
    try:
        yield
    finally:
        seen.append(name + "_exit")


def test_explicit_capture_stream_is_reused_and_restored_on_error(graph_runtime):
    state = graph_runtime
    explicit = state.context_cls(state.runtime.Stream(device="ptpu:0"))
    state.runtime.Stream.reset_mock()
    state.patch.patch_graph_capture_runtime()
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        state.parallel_state.graph_capture("ptpu:0", explicit) as context,
    ):
        assert context is explicit
        assert state.alias.current_stream() is explicit.stream
        raise RuntimeError("capture failed")
    state.runtime.Stream.assert_not_called()
    assert state.runtime.current_stream() is state.default_stream


def test_non_ptpu_platforms_delegate_to_original_functions(graph_runtime):
    state = graph_runtime
    state.patch.patch_graph_capture_runtime()
    state.platform.device_type = "cuda"
    explicit = state.context_cls(object())
    with state.alias.graph_capture("cuda:0", explicit) as context:
        assert context is explicit
    with state.tp.graph_capture(explicit) as context:
        assert context is explicit
    assert state.alias.current_stream() is state.default_stream
    assert state.events == [("original_top", "cuda:0"), ("original_group", "tp")]
    state.runtime.Stream.assert_not_called()


def test_graph_patch_is_idempotent_and_does_not_activate_on_other_platforms(
    graph_runtime,
):
    state = graph_runtime
    state.platform.device_type = "npu"
    original = state.parallel_state.graph_capture
    state.patch.patch_graph_capture_runtime()
    assert state.parallel_state.graph_capture is original
    state.platform.device_type = "ptpu"
    state.patch.patch_graph_capture_runtime()
    installed = state.parallel_state.graph_capture
    state.patch.patch_graph_capture_runtime()
    assert state.parallel_state.graph_capture is installed


def test_real_flagcx_all_reduce_uses_the_live_capture_stream(
    graph_runtime, monkeypatch
):
    state = graph_runtime
    state.install(
        "vllm.distributed.utils",
        StatelessProcessGroup=type("StatelessProcessGroup", (), {}),
    )
    state.install(
        "plugin.interservice.flagcx_wrapper",
        FLAGCXLibrary=object,
        buffer_type=lambda pointer: pointer,
        flagcxComm_t=object,
        flagcxDataTypeEnum=SimpleNamespace(from_torch=lambda dtype: dtype),
        flagcxUniqueId=object,
        flagcxRedOpTypeEnum=SimpleNamespace(from_torch=lambda op: op),
    )
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    flagcx_module = state.load(
        "_sunrise_real_flagcx_test",
        state.root / "vllm_fl/distributed/device_communicators/flagcx.py",
    )
    communicator = object.__new__(flagcx_module.PyFlagcxCommunicator)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.comm = object()
    communicator.flagcx = SimpleNamespace(
        adaptor_stream_copy=Mock(side_effect=lambda stream: stream),
        flagcxAllReduce=Mock(),
        adaptor_stream_free=Mock(),
    )
    state.patch.patch_graph_capture_runtime()
    tensor = torch.ones(2)
    output = torch.empty_like(tensor)
    with state.parallel_state.graph_capture("ptpu:0") as context:
        assert communicator.all_reduce(tensor, output) is output
        communicator.flagcx.flagcxAllReduce.assert_called_once()
        assert communicator.flagcx.flagcxAllReduce.call_args.args[-1] is context.stream
        communicator.flagcx.adaptor_stream_copy.assert_called_once_with(context.stream)
        communicator.flagcx.adaptor_stream_free.assert_called_once_with(context.stream)


def load_graph_wrapper(state, monkeypatch):
    class CUDAGraphMode(Enum):
        NONE = 0
        FULL = 2

    state.install("vllm.envs")
    state.install(
        "vllm.config", CUDAGraphMode=CUDAGraphMode, VllmConfig=SimpleNamespace
    )
    state.install("vllm.compilation.counter", compilation_counter=SimpleNamespace())
    state.install(
        "vllm.compilation.monitor", validate_cudagraph_capturing_enabled=Mock()
    )
    state.install("vllm.compilation.cuda_graph", CUDAGraphStat=object)
    state.install(
        "vllm.distributed.device_communicators.pynccl_allocator",
        set_graph_pool_id=Mock(),
    )
    context = SimpleNamespace(
        batch_descriptor="batch:2", cudagraph_runtime_mode=CUDAGraphMode.FULL
    )
    state.install(
        "vllm.forward_context",
        BatchDescriptor=object,
        get_forward_context=lambda: context,
        is_forward_context_available=lambda: True,
    )
    logger = Mock()
    state.install("vllm.logger", init_logger=lambda name: logger)
    monkeypatch.setattr(torch, "ptpu", state.runtime, raising=False)
    module = state.load(
        "_sunrise_real_graph_wrapper_test", state.root / "vllm_fl/compilation/graph.py"
    )
    wrapper = object.__new__(module.GraphWrapper)
    wrapper.runtime_mode = CUDAGraphMode.FULL
    wrapper.is_debugging_mode = True
    output = torch.ones(2)
    graph = SimpleNamespace(replay=Mock())
    wrapper.concrete_graph_entries = {
        "batch:2": module.GraphEntry(
            batch_descriptor="batch:2",
            graph=graph,
            output=output,
            input_addresses=[],
        )
    }
    return module, wrapper, graph, output, logger


def test_graph_wrapper_uses_ptpu_stream_without_cuda_tls(graph_runtime, monkeypatch):
    state = graph_runtime
    module, _, _, _, _ = load_graph_wrapper(state, monkeypatch)
    explicit = object()
    state.selected.stream = explicit
    assert module._graph_capture_stream() is explicit


def test_success_marker_requires_a_real_successful_replay(graph_runtime, monkeypatch):
    module, wrapper, graph, output, logger = load_graph_wrapper(
        graph_runtime, monkeypatch
    )
    assert wrapper() is output
    graph.replay.assert_called_once_with()
    logger.debug.assert_called_once_with(
        "Sunrise graph replay completed runtime_mode=%s batch_descriptor=%s",
        "FULL",
        "batch:2",
    )
    logger.reset_mock()
    graph.replay.side_effect = RuntimeError("replay failed")
    with pytest.raises(RuntimeError, match="replay failed"):
        wrapper()
    logger.debug.assert_not_called()
