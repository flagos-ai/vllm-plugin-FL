# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for compilation graph module.
"""

from importlib import import_module
from unittest.mock import MagicMock


def test_npu_weak_refs_are_recursive(monkeypatch):
    import torch

    from vllm.sequence import IntermediateTensors

    graph_module = import_module("vllm_fl.compilation.graph")

    weak_refs = {}

    def fake_weak_ref(tensor):
        weak_refs[id(tensor)] = object()
        return weak_refs[id(tensor)]

    monkeypatch.setattr(graph_module.current_platform, "device_type", "npu")
    monkeypatch.setattr(graph_module, "_weak_ref_npu_tensor", fake_weak_ref)

    first = torch.ones(1)
    second = torch.ones(2)
    third = torch.ones(3)
    empty = torch.ones(8)[:0]
    value = {
        "nested": [first, (second, empty)],
        "intermediate": IntermediateTensors({"hidden": third}),
        "metadata": None,
    }

    result = graph_module.weak_ref_tensors(value)

    assert result["nested"][0] is weak_refs[id(first)]
    assert result["nested"][1][0] is weak_refs[id(second)]
    assert result["nested"][1][1] is weak_refs[id(empty)]
    assert result["intermediate"].tensors["hidden"] is weak_refs[id(third)]
    assert result["metadata"] is None
    assert len(weak_refs) == 4


def test_npu_weak_ref_keeps_cpu_tensors(monkeypatch):
    import sys
    from types import SimpleNamespace

    import torch

    graph_module = import_module("vllm_fl.compilation.graph")

    def reject_cpu_tensor(_tensor):
        raise AssertionError("torch_npu weak refs must only receive NPU tensors")

    monkeypatch.setattr(graph_module.current_platform, "device_type", "npu")
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(_C=SimpleNamespace(_weak_ref_tensor=reject_cpu_tensor)),
    )
    tensor = torch.ones(2)

    result = graph_module.weak_ref_tensors({"cpu": tensor})

    assert result["cpu"] is tensor


def test_npu_graph_capture_uses_active_accelerator_stream(monkeypatch):
    from types import SimpleNamespace

    graph_module = import_module("vllm_fl.compilation.graph")

    active_stream = object()
    stale_vllm_stream = MagicMock(return_value=object())
    npu_current_stream = MagicMock(return_value=active_stream)
    monkeypatch.setattr(graph_module.current_platform, "device_type", "npu")
    monkeypatch.setattr(
        graph_module.current_platform,
        "torch_device_fn",
        SimpleNamespace(current_stream=npu_current_stream),
    )
    monkeypatch.setattr(graph_module, "current_stream", stale_vllm_stream)

    assert graph_module._graph_capture_stream() is active_stream
    npu_current_stream.assert_called_once_with()
    stale_vllm_stream.assert_not_called()


def test_non_npu_graph_capture_preserves_vllm_stream(monkeypatch):
    graph_module = import_module("vllm_fl.compilation.graph")

    expected_stream = object()
    vllm_current_stream = MagicMock(return_value=expected_stream)
    monkeypatch.setattr(graph_module.current_platform, "device_type", "cuda")
    monkeypatch.setattr(graph_module, "current_stream", vllm_current_stream)

    assert graph_module._graph_capture_stream() is expected_stream
    vllm_current_stream.assert_called_once_with()


class TestGraphOptions:
    """Test GraphOptions dataclass."""

    def test_default_values(self):
        from vllm_fl.compilation.graph import GraphOptions

        options = GraphOptions()

        assert options.debug_log_enable is True
        assert options.gc_disable is False
        assert options.weak_ref_output is True

    def test_custom_values(self):
        from vllm_fl.compilation.graph import GraphOptions

        options = GraphOptions(
            debug_log_enable=False,
            gc_disable=True,
            weak_ref_output=False,
        )

        assert options.debug_log_enable is False
        assert options.gc_disable is True
        assert options.weak_ref_output is False


class TestGraphEntry:
    """Test GraphEntry dataclass."""

    def test_default_values(self):
        from vllm_fl.compilation.graph import GraphEntry

        mock_batch_desc = MagicMock()

        entry = GraphEntry(batch_descriptor=mock_batch_desc)

        assert entry.batch_descriptor is mock_batch_desc
        assert entry.graph is None
        assert entry.output is None
        assert entry.input_addresses is None
