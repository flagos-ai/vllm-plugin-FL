# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for compilation graph module.
"""

from unittest.mock import MagicMock

import torch


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


def test_ascend_graph_params_are_initialized_per_capture_size(monkeypatch):
    import vllm_fl.compilation.graph as graph

    monkeypatch.setattr(graph, "_ascend_graph_params", None)
    from vllm_fl.compilation.graph import (
        get_ascend_graph_params,
        set_ascend_graph_params,
    )

    set_ascend_graph_params([4, 1, 4])
    params = get_ascend_graph_params()

    assert params is not None
    assert set(params.events) == {1, 4}
    assert set(params.workspaces) == {1, 4}
    assert set(params.handles) == {1, 4}
    assert set(params.attention_params) == {1, 4}


def test_weak_workspace_lifecycle_retains_each_task(monkeypatch):
    import vllm_fl.compilation.graph as graph

    monkeypatch.setattr(graph, "_ascend_graph_params", None)
    graph.set_ascend_graph_params([2])
    owned = [torch.empty(7), torch.empty(9), torch.empty(11)]
    graph.get_ascend_graph_params().workspaces[2] = owned
    weakened = []

    def weak_ref(value):
        weakened.append(value)
        return ("weak", id(value))

    monkeypatch.setattr(graph, "weak_ref_tensors", weak_ref)
    graph.weak_ref_ascend_graph_workspaces()
    assert len(weakened) == 3
    assert all(actual is expected for actual, expected in zip(weakened, owned))
    assert graph.get_ascend_graph_params().workspaces[2] == [
        ("weak", id(value)) for value in owned
    ]
