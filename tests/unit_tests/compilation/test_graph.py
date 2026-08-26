# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for compilation graph module.
"""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.config import CUDAGraphMode


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


class TestAscendGraphParams:
    """Tests for Ascend capture/update state initialization."""

    def test_params_are_initialized_for_unique_capture_sizes(self):
        from vllm_fl.compilation.graph import (
            get_ascend_graph_params,
            set_ascend_graph_params,
        )

        set_ascend_graph_params([8, 1, 4, 8])
        params = get_ascend_graph_params()

        assert params is not None
        assert set(params.events) == {1, 4, 8}
        assert set(params.workspaces) == {1, 4, 8}
        assert set(params.handles) == {1, 4, 8}
        assert set(params.attention_params) == {1, 4, 8}
        assert set(params.conv1d_events) == {1, 4, 8}
        assert set(params.conv1d_handles) == {1, 4, 8}
        assert set(params.conv1d_params) == {1, 4, 8}
        assert set(params.task_order) == {1, 4, 8}
        assert all(workspace is None for workspace in params.workspaces.values())

    def test_reinitialization_drops_profile_capture_state(self):
        from vllm_fl.compilation.graph import (
            get_ascend_graph_params,
            set_ascend_graph_params,
        )

        set_ascend_graph_params([1, 2])
        params = get_ascend_graph_params()
        assert params is not None
        params.handles[1].append(object())

        # Memory profiling captures temporary graphs. Runtime capture must
        # receive fresh task handles instead of retaining those temporary
        # handles after GraphWrapper.clear_all_graphs().
        set_ascend_graph_params([1, 2])
        refreshed = get_ascend_graph_params()
        assert refreshed is not None
        assert refreshed.handles[1] == []

    def test_workspace_setter_updates_only_initialized_shape(self):
        from vllm_fl.compilation.graph import (
            get_ascend_graph_params,
            set_ascend_graph_params,
            update_ascend_graph_params_workspace,
        )

        set_ascend_graph_params([1, 4])
        workspace = object()

        update_ascend_graph_params_workspace(4, workspace)

        params = get_ascend_graph_params()
        assert params is not None
        assert params.workspaces[1] is None
        assert params.workspaces[4] is workspace

    def test_task_descriptors_preserve_capture_order(self):
        from vllm_fl.compilation.graph import (
            AscendGraphTaskDescriptor,
            get_ascend_graph_params,
            record_ascend_graph_task,
            set_ascend_graph_params,
        )

        set_ascend_graph_params([4])
        record_ascend_graph_task(4, "conv1d", 0, "model.layers.0.self_attn")
        record_ascend_graph_task(4, "attention", 0, "model.layers.1.self_attn")

        params = get_ascend_graph_params()
        assert params is not None
        assert params.task_order[4] == [
            AscendGraphTaskDescriptor(
                "conv1d", 0, "model.layers.0.self_attn"
            ),
            AscendGraphTaskDescriptor(
                "attention", 0, "model.layers.1.self_attn"
            ),
        ]

    def test_workspaces_are_converted_to_weak_references(self, monkeypatch):
        import vllm_fl.compilation.graph as graph

        graph.set_ascend_graph_params([1, 4])
        params = graph.get_ascend_graph_params()
        assert params is not None
        params.workspaces[1] = "workspace"

        monkeypatch.setattr(
            graph,
            "weak_ref_tensors",
            lambda tensor: ("weak", tensor),
        )

        graph.weak_ref_ascend_graph_workspaces()

        assert params.workspaces[1] == ("weak", "workspace")
        assert params.workspaces[4] is None


def _graph_update_config(
    *,
    enabled=True,
    hybrid=True,
    speculative_config=None,
    mode=CUDAGraphMode.FULL_DECODE_ONLY,
):
    return SimpleNamespace(
        additional_config={
            "enable_interleaved_graph_task_update": enabled,
        },
        model_config=SimpleNamespace(is_hybrid=hybrid),
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    )


def _populate_graph_tasks(graph, descriptors):
    graph.set_ascend_graph_params([32])
    params = graph.get_ascend_graph_params()
    assert params is not None
    attention_layers = [
        "model.layers.3.self_attn",
        "model.layers.7.self_attn",
    ]
    conv1d_layers = [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
        "model.layers.4.self_attn",
        "model.layers.5.self_attn",
        "model.layers.6.self_attn",
    ]
    params.attention_params[32].extend((name,) for name in attention_layers)
    params.handles[32].extend(object() for _ in attention_layers)
    params.events[32].extend(object() for _ in attention_layers)
    params.conv1d_params[32].extend(
        tuple([None] * 9 + [name]) for name in conv1d_layers
    )
    params.conv1d_handles[32].extend(object() for _ in conv1d_layers)
    params.conv1d_events[32].extend(object() for _ in conv1d_layers)
    params.task_order[32].extend(descriptors(graph))
    return params


def _valid_descriptors(graph):
    descriptor = graph.AscendGraphTaskDescriptor
    return [
        descriptor("conv1d", 0, "model.layers.0.self_attn"),
        descriptor("conv1d", 1, "model.layers.1.self_attn"),
        descriptor("conv1d", 2, "model.layers.2.self_attn"),
        descriptor("attention", 0, "model.layers.3.self_attn"),
        descriptor("conv1d", 3, "model.layers.4.self_attn"),
        descriptor("conv1d", 4, "model.layers.5.self_attn"),
        descriptor("conv1d", 5, "model.layers.6.self_attn"),
        descriptor("attention", 1, "model.layers.7.self_attn"),
    ]


def _install_update_fakes(monkeypatch, calls):
    class FakeAttentionMetadata:
        pass

    class FakeGDNMetadata:
        pass

    class FakeAttentionImpl:
        @staticmethod
        def update_graph_params(*_args):
            calls.append("attention_all")

        @staticmethod
        def _update_graph_task(*args, **_kwargs):
            calls.append(f"attention:{args[-1]}")

    attention_module = ModuleType("fake_attention")
    attention_module.AscendAttentionBackendImpl = FakeAttentionImpl
    attention_module.AscendMetadata = FakeAttentionMetadata
    attention_module.using_paged_attention = lambda *_args: False
    gdn_module = ModuleType("fake_gdn")
    gdn_module.GDNAttentionMetadata = FakeGDNMetadata

    def update_conv1d_graph_params(*_args):
        calls.append("conv1d_all")

    def update_conv1d_graph_task(*args):
        calls.append(f"conv1d:{args[-1]}")

    gdn_module.update_conv1d_graph_params = update_conv1d_graph_params
    gdn_module._update_conv1d_graph_task = update_conv1d_graph_task
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.attention",
        attention_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.gdn",
        gdn_module,
    )
    return FakeAttentionMetadata, FakeGDNMetadata


def _complete_forward_context(graph, metadata_types):
    attention_type, conv1d_type = metadata_types
    return SimpleNamespace(
        attn_metadata={
            task.layer_name: (
                attention_type()
                if task.kind == "attention"
                else conv1d_type()
            )
            for task in _valid_descriptors(graph)
        }
    )


class TestInterleavedAscendGraphTaskUpdate:
    @pytest.mark.parametrize(
        ("device_type", "additional_config", "expected"),
        [
            ("npu", {}, True),
            (
                "npu",
                {"enable_interleaved_graph_task_update": False},
                False,
            ),
            ("cuda", {}, False),
        ],
    )
    def test_config_defaults_on_only_for_eligible_npu(
        self,
        monkeypatch,
        device_type,
        additional_config,
        expected,
    ):
        import vllm_fl.compilation.graph as graph

        config = _graph_update_config()
        config.additional_config = additional_config
        monkeypatch.setattr(
            graph,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )

        assert (
            graph._interleaved_graph_task_update_enabled(config) is expected
        )

    def test_explicit_off_keeps_established_update_path(
        self, monkeypatch
    ):
        import vllm_fl.compilation.graph as graph

        _populate_graph_tasks(graph, _valid_descriptors)
        calls = []
        _install_update_fakes(monkeypatch, calls)
        monkeypatch.setattr(
            graph,
            "_validated_interleaved_task_order",
            MagicMock(side_effect=AssertionError("must not validate")),
        )

        graph.update_ascend_full_graph_params(
            object(), object(), 32, _graph_update_config(enabled=False)
        )

        assert calls == ["attention_all", "conv1d_all"]

    def test_on_replays_tasks_in_capture_model_order(self, monkeypatch):
        import vllm_fl.compilation.graph as graph

        _populate_graph_tasks(graph, _valid_descriptors)
        calls = []
        metadata_types = _install_update_fakes(monkeypatch, calls)
        monkeypatch.setattr(
            graph, "current_platform", SimpleNamespace(device_type="npu")
        )
        monkeypatch.setattr(
            graph.torch,
            "npu",
            SimpleNamespace(stream=lambda _stream: nullcontext()),
            raising=False,
        )

        graph.update_ascend_full_graph_params(
            object(), _complete_forward_context(graph, metadata_types), 32,
            _graph_update_config()
        )

        assert calls == [
            "conv1d:0",
            "conv1d:1",
            "conv1d:2",
            "attention:0",
            "conv1d:3",
            "conv1d:4",
            "conv1d:5",
            "attention:1",
        ]

    def test_incomplete_runtime_metadata_falls_back_before_submit(
        self, monkeypatch
    ):
        import vllm_fl.compilation.graph as graph

        _populate_graph_tasks(graph, _valid_descriptors)
        calls = []
        _install_update_fakes(monkeypatch, calls)
        monkeypatch.setattr(
            graph, "current_platform", SimpleNamespace(device_type="npu")
        )

        graph.update_ascend_full_graph_params(
            object(),
            SimpleNamespace(attn_metadata={"model.layers.0.self_attn": object()}),
            32,
            _graph_update_config(),
        )

        assert calls == ["attention_all", "conv1d_all"]

    @pytest.mark.parametrize(
        "mutation", ["missing", "duplicate", "layer", "handle"]
    )
    def test_invalid_descriptor_falls_back(self, monkeypatch, mutation):
        import vllm_fl.compilation.graph as graph

        params = _populate_graph_tasks(graph, _valid_descriptors)
        if mutation == "missing":
            params.task_order[32].pop()
        elif mutation == "duplicate":
            params.task_order[32][-1] = params.task_order[32][0]
        else:
            if mutation == "layer":
                params.task_order[32][0] = graph.AscendGraphTaskDescriptor(
                    "conv1d", 0, "model.layers.9.self_attn"
                )
            else:
                params.conv1d_handles[32].pop()
        calls = []
        _install_update_fakes(monkeypatch, calls)
        monkeypatch.setattr(
            graph, "current_platform", SimpleNamespace(device_type="npu")
        )

        graph.update_ascend_full_graph_params(
            object(), object(), 32, _graph_update_config()
        )

        assert calls == ["attention_all", "conv1d_all"]

    @pytest.mark.parametrize(
        "config",
        [
            _graph_update_config(hybrid=False),
            _graph_update_config(speculative_config=object()),
            _graph_update_config(mode=CUDAGraphMode.FULL),
        ],
    )
    def test_ineligible_runtime_falls_back(self, monkeypatch, config):
        import vllm_fl.compilation.graph as graph

        _populate_graph_tasks(graph, _valid_descriptors)
        calls = []
        _install_update_fakes(monkeypatch, calls)
        monkeypatch.setattr(
            graph, "current_platform", SimpleNamespace(device_type="npu")
        )

        graph.update_ascend_full_graph_params(
            object(), object(), 32, config
        )

        assert calls == ["attention_all", "conv1d_all"]
