# Copyright (c) 2025 BAAI. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call

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


def test_ascend_capture_lifecycle_delegates_graph_state(monkeypatch):
    import vllm_fl.compilation.graph as graph
    from vllm_fl.compilation.graph_runtime import AscendGraphRuntimeBackend

    set_capturing = MagicMock()
    monkeypatch.setattr(graph, "set_ascend_graph_capturing", set_capturing)
    forward_context = SimpleNamespace()
    backend = AscendGraphRuntimeBackend()

    backend.prepare_forward_context(forward_context)
    assert forward_context.capturing is False
    backend.begin_capture(forward_context)
    assert forward_context.capturing is True
    backend.end_capture()

    assert set_capturing.call_args_list == [
        call(True),
        call(False),
    ]


def test_ascend_prepare_capture_uses_all_token_sizes(monkeypatch):
    import vllm_fl.compilation.graph as graph
    from vllm_fl.compilation.graph_runtime import AscendGraphRuntimeBackend

    set_graph_params = MagicMock()
    monkeypatch.setattr(graph, "set_ascend_graph_params", set_graph_params)
    capture_descs = [
        (CUDAGraphMode.FULL, [SimpleNamespace(num_tokens=8)]),
        (
            CUDAGraphMode.FULL_DECODE_ONLY,
            [SimpleNamespace(num_tokens=1), SimpleNamespace(num_tokens=4)],
        ),
    ]

    AscendGraphRuntimeBackend().prepare_capture(capture_descs)

    set_graph_params.assert_called_once_with([8, 1, 4])


@pytest.mark.parametrize(
    ("runtime_mode", "capturing", "expected_updates"),
    [
        pytest.param(CUDAGraphMode.FULL, False, 1, id="full-replay"),
        pytest.param(CUDAGraphMode.FULL, True, 0, id="full-capture"),
        pytest.param(CUDAGraphMode.NONE, False, 0, id="eager"),
    ],
)
def test_ascend_updates_tasks_only_after_full_replay(
    monkeypatch,
    runtime_mode,
    capturing,
    expected_updates,
):
    import vllm.forward_context as forward_context_module
    import vllm_fl.compilation.graph as graph
    import vllm_fl.compilation.graph_runtime as graph_runtime

    update_stream = object()
    monkeypatch.setattr(
        graph_runtime.torch,
        "npu",
        SimpleNamespace(Stream=lambda: update_stream),
        raising=False,
    )
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=runtime_mode,
        capturing=capturing,
        batch_descriptor=SimpleNamespace(num_tokens=4),
    )
    monkeypatch.setattr(
        forward_context_module,
        "get_forward_context",
        lambda: forward_context,
    )
    update_graph_params = MagicMock()
    monkeypatch.setattr(
        graph,
        "update_ascend_full_graph_params",
        update_graph_params,
    )
    config = MagicMock()
    backend = graph_runtime.AscendGraphRuntimeBackend()
    backend.prepare_graph_wrapper()

    backend.after_model_forward(config)

    assert update_graph_params.call_count == expected_updates
    if expected_updates:
        update_graph_params.assert_called_once_with(
            update_stream,
            forward_context,
            4,
            config,
        )


def test_ascend_extends_only_eligible_mrope_compile_range():
    from vllm_fl.compilation.graph_runtime import AscendGraphRuntimeBackend

    backend = AscendGraphRuntimeBackend()
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        compile_ranges_endpoints=[1, 8],
    )

    backend.adjust_compile_ranges(
        compilation_config,
        uses_mrope=True,
        max_num_tokens=8,
    )
    assert compilation_config.compile_ranges_endpoints == [1, 9]

    compilation_config.compile_ranges_endpoints = [1, 8]
    backend.adjust_compile_ranges(
        compilation_config,
        uses_mrope=False,
        max_num_tokens=8,
    )
    assert compilation_config.compile_ranges_endpoints == [1, 8]


def test_moe_workspace_reservation_is_an_ascend_graph_capability():
    from vllm_fl.compilation.graph_runtime import (
        AscendGraphRuntimeBackend,
        GraphRuntimeBackend,
    )

    graph_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
    )
    eager_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)

    assert AscendGraphRuntimeBackend().should_reserve_moe_workspace(
        graph_config
    )
    assert not AscendGraphRuntimeBackend().should_reserve_moe_workspace(
        eager_config
    )
    assert not GraphRuntimeBackend().should_reserve_moe_workspace(graph_config)
