# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for compilation graph module.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch


def has_graph_module():
    try:
        from vllm_fl.compilation import graph  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not has_graph_module(), reason="vllm_fl.compilation.graph not available"
)


@dataclass(frozen=True)
class MockBatchDescriptor:
    num_tokens: int = 30
    uniform: bool = False


@pytest.fixture(autouse=True)
def reset_graph_params(monkeypatch):
    from vllm_fl.compilation import graph

    monkeypatch.setattr(graph, "_graph_params", None)
    monkeypatch.setattr(graph, "_draft_graph_params", None)


@pytest.fixture
def wrapper_context():
    from vllm.config import CUDAGraphMode, VllmConfig
    from vllm_fl.compilation.graph import GraphOptions

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = MagicMock()
    runnable = MagicMock(return_value="test_output")
    graph_pool = MagicMock(name="graph_pool")
    graph_options = GraphOptions(
        debug_log_enable=False,
        gc_disable=False,
        weak_ref_output=False,
    )
    batch_descriptor = MockBatchDescriptor()
    forward_context = MagicMock()
    forward_context.batch_descriptor = batch_descriptor
    forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    forward_context.capturing = False
    return {
        "vllm_config": vllm_config,
        "runnable": runnable,
        "graph_pool": graph_pool,
        "graph_options": graph_options,
        "batch_descriptor": batch_descriptor,
        "forward_context": forward_context,
    }


def _mock_graph_context():
    graph_context = MagicMock()
    graph_context.__enter__ = Mock(return_value=None)
    graph_context.__exit__ = Mock(return_value=None)
    return graph_context


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

        batch_descriptor = MockBatchDescriptor()

        entry = GraphEntry(batch_descriptor=batch_descriptor)

        assert entry.batch_descriptor == batch_descriptor
        assert entry.graph is None
        assert entry.output is None
        assert entry.input_addresses is None

    def test_custom_values(self):
        from vllm_fl.compilation.graph import GraphEntry

        batch_descriptor = MockBatchDescriptor()
        mock_graph = MagicMock()
        mock_output = MagicMock()
        input_addresses = [12345, 67890]

        entry = GraphEntry(
            batch_descriptor=batch_descriptor,
            graph=mock_graph,
            output=mock_output,
            input_addresses=input_addresses,
        )

        assert entry.batch_descriptor == batch_descriptor
        assert entry.graph is mock_graph
        assert entry.output is mock_output
        assert entry.input_addresses == input_addresses


class TestGraphWrapper:
    """Test GraphWrapper behavior without real device graph capture."""

    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_initialization_with_default_options(
        self, mock_envs, mock_current_platform, wrapper_context
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphOptions, GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
        )

        assert wrapper.runnable is wrapper_context["runnable"]
        assert wrapper.vllm_config is wrapper_context["vllm_config"]
        assert wrapper.graph_pool is wrapper_context["graph_pool"]
        assert wrapper.runtime_mode == CUDAGraphMode.FULL
        assert wrapper.is_debugging_mode is False
        assert isinstance(wrapper.graph_options, GraphOptions)
        assert wrapper.concrete_graph_entries == {}
        assert wrapper.enable_enpu is False
        assert wrapper.use_eagle is False

    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_initialization_with_custom_options(
        self, mock_envs, mock_current_platform, wrapper_context
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
            use_eagle=True,
            enable_enpu=True,
        )

        assert wrapper.runnable is wrapper_context["runnable"]
        assert wrapper.vllm_config is wrapper_context["vllm_config"]
        assert wrapper.graph_pool is wrapper_context["graph_pool"]
        assert wrapper.runtime_mode == CUDAGraphMode.FULL
        assert wrapper.is_debugging_mode is True
        assert wrapper.graph_options is wrapper_context["graph_options"]
        assert wrapper.concrete_graph_entries == {}
        assert wrapper.enable_enpu is True
        assert wrapper.use_eagle is True

    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_initialization_assertion_error(
        self, mock_envs, mock_current_platform, wrapper_context
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]

        with pytest.raises(AssertionError):
            GraphWrapper(
                runnable=wrapper_context["runnable"],
                vllm_config=wrapper_context["vllm_config"],
                runtime_mode=CUDAGraphMode.NONE,
            )

    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_with_none_runtime_mode(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        wrapper_context["forward_context"].cudagraph_runtime_mode = CUDAGraphMode.NONE
        mock_get_forward_context.return_value = wrapper_context["forward_context"]

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        result = wrapper("arg1", "arg2")

        wrapper_context["runnable"].assert_called_once_with("arg1", "arg2")
        assert result == "test_output"

    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_with_mismatched_runtime_mode(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        wrapper_context["forward_context"].cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        mock_get_forward_context.return_value = wrapper_context["forward_context"]

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        result = wrapper("arg1", "arg2")

        wrapper_context["runnable"].assert_called_once_with("arg1", "arg2")
        assert result == "test_output"

    @patch("vllm_fl.compilation.graph.compilation_counter")
    @patch("vllm_fl.compilation.graph.weak_ref_workspaces")
    @patch("vllm_fl.compilation.graph.weak_ref_tensors")
    @patch("vllm_fl.compilation.graph.set_graph_pool_id")
    @patch("vllm_fl.compilation.graph.validate_cudagraph_capturing_enabled")
    @patch("vllm_fl.compilation.graph.Graph.graph")
    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_capture_graph_first_time(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_graph_cls,
        mock_validate_capturing,
        mock_set_graph_pool_id,
        mock_weak_ref_tensors,
        mock_weak_ref_workspaces,
        mock_compilation_counter,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        mock_graph_context = _mock_graph_context()
        mock_current_platform.torch_device_fn.graph.return_value = mock_graph_context
        mock_get_forward_context.return_value = wrapper_context["forward_context"]
        mock_graph = MagicMock()
        mock_graph_cls.return_value = mock_graph
        mock_weak_ref_tensors.return_value = "weak_ref_output"
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )
        test_tensor = torch.tensor([1, 2, 3])

        result = wrapper(test_tensor, "arg2")

        mock_validate_capturing.assert_called_once()
        mock_graph_cls.assert_called_once()
        mock_set_graph_pool_id.assert_called_once_with(wrapper_context["graph_pool"])
        mock_current_platform.torch_device_fn.graph.assert_called_once_with(
            mock_graph, pool=wrapper_context["graph_pool"]
        )
        wrapper_context["runnable"].assert_called_once_with(test_tensor, "arg2")
        assert wrapper_context["forward_context"].capturing is True
        assert wrapper_context["batch_descriptor"] in wrapper.concrete_graph_entries
        entry = wrapper.concrete_graph_entries[wrapper_context["batch_descriptor"]]
        assert entry.graph is mock_graph
        assert entry.output == "weak_ref_output"
        assert mock_weak_ref_workspaces.call_count == 2
        assert mock_compilation_counter.num_cudagraph_captured == 1
        assert result == "test_output"

    @patch("vllm_fl.compilation.graph.compilation_counter")
    @patch("vllm_fl.compilation.graph.weak_ref_workspaces")
    @patch("vllm_fl.compilation.graph.weak_ref_tensors")
    @patch("vllm_fl.compilation.graph.set_graph_pool_id")
    @patch("vllm_fl.compilation.graph.validate_cudagraph_capturing_enabled")
    @patch("vllm_fl.compilation.graph.Graph.graph")
    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_replay_graph(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_graph_cls,
        mock_validate_capturing,
        mock_set_graph_pool_id,
        mock_weak_ref_tensors,
        mock_weak_ref_workspaces,
        mock_compilation_counter,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.device_type = "cuda"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        mock_graph_context = _mock_graph_context()
        mock_current_platform.torch_device_fn.graph.return_value = mock_graph_context
        mock_stream = MagicMock()
        mock_current_platform.torch_device_fn.current_stream.return_value = mock_stream
        mock_get_forward_context.return_value = wrapper_context["forward_context"]
        mock_graph = MagicMock()
        mock_graph_cls.return_value = mock_graph
        mock_weak_ref_tensors.return_value = "weak_ref_output"
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )
        test_tensor = torch.tensor([1, 2, 3])

        first_result = wrapper(test_tensor, "arg2")
        wrapper_context["runnable"].reset_mock()
        second_result = wrapper(test_tensor, "arg2")

        mock_validate_capturing.assert_called_once()
        mock_graph_cls.assert_called_once()
        mock_set_graph_pool_id.assert_called_once_with(wrapper_context["graph_pool"])
        wrapper_context["runnable"].assert_not_called()
        mock_stream.synchronize.assert_called_once()
        mock_graph.replay.assert_called_once()
        assert first_result == "test_output"
        assert second_result == "weak_ref_output"

    @patch("vllm_fl.compilation.graph.weak_ref_workspaces")
    @patch("vllm_fl.compilation.graph.weak_ref_tensors")
    @patch("vllm_fl.compilation.graph.set_graph_pool_id")
    @patch("vllm_fl.compilation.graph.validate_cudagraph_capturing_enabled")
    @patch("vllm_fl.compilation.graph.Graph.graph")
    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_with_debug_mode_input_address_check(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_graph_cls,
        mock_validate_capturing,
        mock_set_graph_pool_id,
        mock_weak_ref_tensors,
        mock_weak_ref_workspaces,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"
        mock_current_platform.device_type = "cuda"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        mock_current_platform.torch_device_fn.graph.return_value = _mock_graph_context()
        mock_current_platform.torch_device_fn.current_stream.return_value = MagicMock()
        mock_get_forward_context.return_value = wrapper_context["forward_context"]
        mock_graph_cls.return_value = MagicMock()
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )
        tensor = torch.tensor([1, 2, 3])

        wrapper(tensor, "arg2")
        wrapper(tensor, "arg2")

        assert mock_validate_capturing.call_count == 1

    @patch("vllm_fl.compilation.graph.weak_ref_workspaces")
    @patch("vllm_fl.compilation.graph.weak_ref_tensors")
    @patch("vllm_fl.compilation.graph.set_graph_pool_id")
    @patch("vllm_fl.compilation.graph.validate_cudagraph_capturing_enabled")
    @patch("vllm_fl.compilation.graph.Graph.graph")
    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_with_debug_mode_input_address_mismatch(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_graph_cls,
        mock_validate_capturing,
        mock_set_graph_pool_id,
        mock_weak_ref_tensors,
        mock_weak_ref_workspaces,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"
        mock_current_platform.device_type = "cuda"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        mock_current_platform.torch_device_fn.graph.return_value = _mock_graph_context()
        mock_current_platform.torch_device_fn.current_stream.return_value = MagicMock()
        mock_get_forward_context.return_value = wrapper_context["forward_context"]
        mock_graph_cls.return_value = MagicMock()
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        wrapper(torch.tensor([1, 2, 3]), "arg2")

        with pytest.raises(AssertionError, match="Input addresses for cudagraphs"):
            wrapper(torch.tensor([4, 5, 6]), "arg2")

    @patch("vllm_fl.compilation.graph.compilation_counter")
    @patch("vllm_fl.compilation.graph.weak_ref_workspaces")
    @patch("vllm_fl.compilation.graph.weak_ref_tensors")
    @patch("vllm_fl.compilation.graph.set_graph_pool_id")
    @patch("vllm_fl.compilation.graph.validate_cudagraph_capturing_enabled")
    @patch("vllm_fl.compilation.graph.Graph.graph")
    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_capture_graph_with_gc_disable(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_graph_cls,
        mock_validate_capturing,
        mock_set_graph_pool_id,
        mock_weak_ref_tensors,
        mock_weak_ref_workspaces,
        mock_compilation_counter,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.device_type = "cuda"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        mock_current_platform.torch_device_fn.graph.return_value = _mock_graph_context()
        mock_get_forward_context.return_value = wrapper_context["forward_context"]
        mock_graph_cls.return_value = MagicMock()
        mock_weak_ref_tensors.return_value = "weak_ref_output"
        mock_compilation_counter.num_cudagraph_captured = 0
        wrapper_context["graph_options"].gc_disable = True

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        with patch("vllm_fl.compilation.graph.patch") as mock_patch:
            mock_patch.return_value = MagicMock()
            result = wrapper(torch.tensor([1, 2, 3]), "arg2")

        assert mock_patch.call_count == 2
        assert result == "test_output"

    @patch("vllm_fl.compilation.graph.compilation_counter")
    @patch("vllm_fl.compilation.graph.weak_ref_workspaces")
    @patch("vllm_fl.compilation.graph.weak_ref_tensors")
    @patch("vllm_fl.compilation.graph.set_graph_pool_id")
    @patch("vllm_fl.compilation.graph.validate_cudagraph_capturing_enabled")
    @patch("vllm_fl.compilation.graph.Graph.graph")
    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_capture_graph_with_weak_ref_output(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_graph_cls,
        mock_validate_capturing,
        mock_set_graph_pool_id,
        mock_weak_ref_tensors,
        mock_weak_ref_workspaces,
        mock_compilation_counter,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        mock_current_platform.torch_device_fn.graph.return_value = _mock_graph_context()
        mock_get_forward_context.return_value = wrapper_context["forward_context"]
        mock_graph_cls.return_value = MagicMock()
        mock_weak_ref_tensors.side_effect = ["inner_output", "weak_ref_output"]
        mock_compilation_counter.num_cudagraph_captured = 0
        wrapper_context["graph_options"].weak_ref_output = True

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        result = wrapper(torch.tensor([1, 2, 3]), "arg2")

        assert mock_weak_ref_tensors.call_count == 2
        assert result == "inner_output"

    @patch("vllm_fl.compilation.graph.logger")
    @patch("vllm_fl.compilation.graph.weak_ref_workspaces")
    @patch("vllm_fl.compilation.graph.weak_ref_tensors")
    @patch("vllm_fl.compilation.graph.set_graph_pool_id")
    @patch("vllm_fl.compilation.graph.validate_cudagraph_capturing_enabled")
    @patch("vllm_fl.compilation.graph.Graph.graph")
    @patch("vllm_fl.compilation.graph.get_forward_context")
    @patch("vllm_fl.compilation.graph.current_platform")
    @patch("vllm_fl.compilation.graph.envs")
    def test_call_capture_graph_with_debug_log(
        self,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_graph_cls,
        mock_validate_capturing,
        mock_set_graph_pool_id,
        mock_weak_ref_tensors,
        mock_weak_ref_workspaces,
        mock_logger,
        wrapper_context,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = wrapper_context[
            "graph_pool"
        ]
        mock_current_platform.torch_device_fn.graph.return_value = _mock_graph_context()
        mock_get_forward_context.return_value = wrapper_context["forward_context"]
        mock_graph_cls.return_value = MagicMock()
        mock_weak_ref_tensors.return_value = "weak_ref_output"
        wrapper_context["graph_options"].debug_log_enable = True

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        wrapper(torch.tensor([1, 2, 3]), "arg2")

        mock_logger.debug.assert_called_once()

    def test_getattr_access_runnable_attributes(self, wrapper_context):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        wrapper_context["runnable"].test_attr = "test_value"

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        assert wrapper.test_attr == "test_value"

    def test_getattr_attribute_not_exists(self, wrapper_context):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        class EmptyRunnable:
            pass

        wrapper = GraphWrapper(
            runnable=EmptyRunnable(),
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        with pytest.raises(AttributeError, match="Attribute non_existent_attr not found"):
            _ = wrapper.non_existent_attr

    def test_unwrap_method(self, wrapper_context):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation.graph import GraphWrapper

        wrapper = GraphWrapper(
            runnable=wrapper_context["runnable"],
            vllm_config=wrapper_context["vllm_config"],
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=wrapper_context["graph_options"],
        )

        assert wrapper.unwrap() is wrapper_context["runnable"]

    @pytest.mark.parametrize(
        ("device_type", "enable_enpu", "use_eagle", "is_draft_model", "expected"),
        [
            ("cuda", False, False, False, True),
            ("npu", False, False, False, True),
            ("npu", True, False, False, False),
            ("npu", False, True, True, False),
        ],
    )
    def test_should_synchronize_before_replay(
        self,
        device_type,
        enable_enpu,
        use_eagle,
        is_draft_model,
        expected,
        wrapper_context,
        monkeypatch,
    ):
        from vllm.config import CUDAGraphMode
        from vllm_fl.compilation import graph
        from vllm_fl.compilation.graph import GraphWrapper

        current_platform = MagicMock()
        current_platform.device_type = device_type
        current_platform.get_global_graph_pool.return_value = wrapper_context["graph_pool"]
        monkeypatch.setattr(graph, "current_platform", current_platform)
        monkeypatch.setattr(graph.envs, "VLLM_LOGGING_LEVEL", "INFO")

        extra_ctx = MagicMock()
        extra_ctx.is_draft_model = is_draft_model
        with patch.dict(
            "sys.modules",
            {
                "vllm_ascend": MagicMock(),
                "vllm_ascend.ascend_forward_context": MagicMock(
                    _EXTRA_CTX=extra_ctx
                ),
            },
        ):
            wrapper = GraphWrapper(
                runnable=wrapper_context["runnable"],
                vllm_config=wrapper_context["vllm_config"],
                runtime_mode=CUDAGraphMode.FULL,
                cudagraph_options=wrapper_context["graph_options"],
                use_eagle=use_eagle,
                enable_enpu=enable_enpu,
            )

            assert wrapper._should_synchronize_before_replay() is expected


class TestWeakRefTensors:
    """Test platform-specific weak reference behavior."""

    def test_returns_original_tensor_for_non_cuda_platform(self, monkeypatch):
        from vllm_fl.compilation import graph
        from vllm_fl.compilation.graph import weak_ref_tensors

        mock_platform = MagicMock()
        mock_platform.device_type = "npu"
        monkeypatch.setattr(graph, "current_platform", mock_platform)
        tensor = torch.tensor([1, 2, 3])

        assert weak_ref_tensors(tensor) is tensor

    def test_delegates_to_vllm_for_cuda_platform(self, monkeypatch):
        from vllm_fl.compilation import graph
        from vllm_fl.compilation.graph import weak_ref_tensors

        mock_platform = MagicMock()
        mock_platform.device_type = "cuda"
        monkeypatch.setattr(graph, "current_platform", mock_platform)
        tensor = torch.tensor([1, 2, 3])

        with patch("vllm.utils.torch_utils.weak_ref_tensors") as mock_weak_ref:
            mock_weak_ref.return_value = "weak_ref_tensor"
            assert weak_ref_tensors(tensor) == "weak_ref_tensor"
            mock_weak_ref.assert_called_once_with(tensor)


class TestGraphParams:
    """Test graph parameter global state helpers."""

    def test_set_and_get_graph_params(self):
        from vllm_fl.compilation.graph import get_graph_params, set_graph_params

        set_graph_params([4, 8])
        graph_params = get_graph_params()

        assert graph_params is not None
        assert graph_params.events == {4: [], 8: []}
        assert graph_params.workspaces == {4: None, 8: None}
        assert graph_params.handles == {4: [], 8: []}
        assert graph_params.attn_params == {4: [], 8: []}

    def test_set_graph_params_raises_if_already_set(self):
        from vllm_fl.compilation.graph import set_graph_params

        set_graph_params([4])

        with pytest.raises(ValueError, match="Graph parameters have already been set"):
            set_graph_params([8])

    def test_update_graph_params_workspaces(self):
        from vllm_fl.compilation.graph import (
            get_graph_params,
            set_graph_params,
            update_graph_params_workspaces,
        )

        workspace = MagicMock()
        set_graph_params([4])
        update_graph_params_workspaces(4, workspace)

        assert get_graph_params().workspaces[4] is workspace

    def test_update_graph_params_workspaces_noop_before_set(self):
        from vllm_fl.compilation.graph import (
            get_graph_params,
            update_graph_params_workspaces,
        )

        update_graph_params_workspaces(4, MagicMock())

        assert get_graph_params() is None

    def test_set_and_get_draft_graph_params(self):
        from vllm_fl.compilation.graph import (
            get_draft_graph_params,
            set_draft_graph_params,
        )

        set_draft_graph_params([4])
        graph_params = get_draft_graph_params()

        assert graph_params is not None
        assert graph_params.events == {4: []}
        assert graph_params.workspaces == {4: None}
        assert graph_params.handles == {4: []}
        assert graph_params.attn_params == {4: []}

    def test_set_draft_graph_params_raises_if_already_set(self):
        from vllm_fl.compilation.graph import set_draft_graph_params

        set_draft_graph_params([4])

        with pytest.raises(ValueError, match="DraftGraph parameters have already been set"):
            set_draft_graph_params([8])

    def test_update_draft_graph_params_workspaces(self):
        from vllm_fl.compilation.graph import (
            get_draft_graph_params,
            set_draft_graph_params,
            update_draft_graph_params_workspaces,
        )

        workspace = MagicMock()
        set_draft_graph_params([4])
        update_draft_graph_params_workspaces(4, workspace)

        assert get_draft_graph_params().workspaces[4] is workspace

    def test_update_draft_graph_params_workspaces_noop_before_set(self):
        from vllm_fl.compilation.graph import (
            get_draft_graph_params,
            update_draft_graph_params_workspaces,
        )

        update_draft_graph_params_workspaces(4, MagicMock())

        assert get_draft_graph_params() is None

    def test_weak_ref_workspaces(self):
        from vllm_fl.compilation.graph import GraphParams, weak_ref_workspaces

        workspace = MagicMock()
        params = GraphParams(
            events={4: []},
            workspaces={4: workspace, 8: None},
            handles={4: []},
            attn_params={4: []},
        )

        with patch("vllm_fl.compilation.graph.weak_ref_tensors") as mock_weak_ref:
            mock_weak_ref.return_value = "weak_ref_workspace"
            weak_ref_workspaces(params)

        mock_weak_ref.assert_called_once_with(workspace)
        assert params.workspaces[4] == "weak_ref_workspace"
        assert params.workspaces[8] is None

    def test_weak_ref_workspaces_noop_for_none(self):
        from vllm_fl.compilation.graph import weak_ref_workspaces

        weak_ref_workspaces(None)


class TestUpdateFullGraphParams:
    """Test full graph parameter update delegation."""

    def test_update_full_graph_params_delegates_to_backend_impl(self):
        from vllm_fl.compilation.graph import update_full_graph_params


        impl_cls = MagicMock()
        attn_backend = MagicMock()
        attn_backend.get_impl_cls.return_value = impl_cls
        update_stream = MagicMock()
        forward_context = MagicMock()
        vllm_config = MagicMock()
        speculative_config = MagicMock()
        draft_attn_metadatas = MagicMock()

        update_full_graph_params(
            attn_backend,
            update_stream,
            forward_context,
            4,
            vllm_config,
            speculative_config,
            2,
            draft_attn_metadatas,
        )

        impl_cls.update_graph_params.assert_called_once_with(
            update_stream,
            forward_context,
            4,
            vllm_config,
            speculative_config,
            2,
            draft_attn_metadatas,
        )
