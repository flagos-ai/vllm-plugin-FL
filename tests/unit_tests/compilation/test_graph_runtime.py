# Copyright (c) 2025 BAAI. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode


def test_graph_class_resolution_is_centralized(monkeypatch):
    import vllm_fl.compilation.graph_runtime as graph_runtime

    cuda_graph = object()
    monkeypatch.setattr(
        graph_runtime.torch,
        "cuda",
        SimpleNamespace(CUDAGraph=cuda_graph),
    )

    assert graph_runtime.get_graph_class("cuda") is cuda_graph
    assert graph_runtime.get_graph_class("txda") is None
    with pytest.raises(NotImplementedError, match="unknown"):
        graph_runtime.get_graph_class("unknown")


def test_runtime_backend_selection():
    from vllm_fl.compilation.graph_runtime import (
        AscendGraphRuntimeBackend,
        GraphRuntimeBackend,
        get_graph_runtime_backend,
    )

    assert isinstance(
        get_graph_runtime_backend("npu"), AscendGraphRuntimeBackend
    )
    assert type(get_graph_runtime_backend("cuda")) is GraphRuntimeBackend


def test_prepare_model_compile_is_ascend_only(monkeypatch):
    import vllm_fl.dispatch as dispatch
    from vllm_fl.compilation.graph_runtime import (
        AscendGraphRuntimeBackend,
        GraphRuntimeBackend,
    )

    prewarm_calls = []
    monkeypatch.setattr(
        dispatch,
        "prewarm_cached_ops",
        lambda: prewarm_calls.append(True),
    )

    GraphRuntimeBackend().prepare_model_compile()
    assert prewarm_calls == []

    AscendGraphRuntimeBackend().prepare_model_compile()
    assert prewarm_calls == [True]


def test_ascend_synchronizes_before_replay(monkeypatch):
    import vllm_fl.compilation.graph_runtime as graph_runtime

    synchronize_calls = []
    monkeypatch.setattr(
        graph_runtime.current_platform,
        "torch_device_fn",
        SimpleNamespace(synchronize=lambda: synchronize_calls.append(True)),
    )

    graph_runtime.AscendGraphRuntimeBackend().before_replay()

    assert synchronize_calls == [True]


def test_ascend_capture_lifecycle_tracks_attention_tasks(monkeypatch):
    import vllm_fl.compilation.graph as graph
    from vllm_fl.compilation.graph_runtime import AscendGraphRuntimeBackend

    backend = AscendGraphRuntimeBackend()
    forward_context = SimpleNamespace()

    backend.prepare_forward_context(forward_context)
    assert forward_context.capturing is False

    backend.begin_capture(forward_context)
    assert forward_context.capturing is True
    assert graph.is_ascend_graph_capturing() is True

    backend.end_capture()
    assert graph.is_ascend_graph_capturing() is False


def test_ascend_updates_attention_params_after_graph_forward(monkeypatch):
    import vllm.forward_context as forward_context_module

    import vllm_fl.compilation.graph as graph
    from vllm_fl.compilation.graph_runtime import AscendGraphRuntimeBackend

    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        capturing=False,
        batch_descriptor=SimpleNamespace(num_tokens=4),
    )
    update_calls = []
    monkeypatch.setattr(
        forward_context_module,
        "get_forward_context",
        lambda: forward_context,
    )
    monkeypatch.setattr(
        graph,
        "update_ascend_full_graph_params",
        lambda stream, context, num_tokens: update_calls.append(
            (stream, context, num_tokens)
        ),
    )

    backend = AscendGraphRuntimeBackend()
    backend._update_stream = object()
    backend.after_model_forward(SimpleNamespace())

    assert update_calls == [
        (backend._update_stream, forward_context, 4)
    ]


def test_graph_capture_override_is_ascend_only(monkeypatch):
    import vllm_fl.compilation.graph_runtime as graph_runtime

    @contextmanager
    def default_capture(device):
        yield device

    monkeypatch.setattr(
        graph_runtime,
        "current_platform",
        SimpleNamespace(device_type="cuda", dist_backend="nccl"),
    )
    assert graph_runtime.get_graph_capture(default_capture) is default_capture

    monkeypatch.setattr(
        graph_runtime.current_platform,
        "device_type",
        "npu",
    )
    assert (
        graph_runtime.get_graph_capture(default_capture)
        is graph_runtime._ascend_graph_capture
    )
