# Copyright (c) 2025 BAAI. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest


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
