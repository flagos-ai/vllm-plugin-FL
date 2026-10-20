# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for compilation graph module.
"""

from unittest.mock import MagicMock

import pytest
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
        assert entry.input_tensors is None


class TestCopyGraphInputs:
    def test_copies_changed_addresses(self):
        from vllm_fl.compilation.graph import _copy_graph_inputs

        static = torch.zeros(4)
        runtime = torch.arange(4, dtype=static.dtype)

        _copy_graph_inputs([static], [runtime])

        torch.testing.assert_close(static, runtime)

    def test_rejects_metadata_changes(self):
        from vllm_fl.compilation.graph import _copy_graph_inputs

        with pytest.raises(RuntimeError, match="metadata changed"):
            _copy_graph_inputs([torch.zeros(4)], [torch.zeros(5)])


class TestGraphWrapperInputCopy:
    @pytest.mark.parametrize(
        (
            "device_type",
            "tensor_parallel_size",
            "runtime_mode",
            "global_copy",
            "expected",
        ),
        [
            ("musa", 2, "PIECEWISE", False, True),
            ("musa", 1, "PIECEWISE", False, False),
            ("musa", 2, "FULL", False, False),
            ("cuda", 2, "PIECEWISE", False, False),
            ("cuda", 1, "FULL", True, True),
        ],
    )
    def test_musa_tp_piecewise_copies_segment_inputs(
        self,
        monkeypatch,
        device_type,
        tensor_parallel_size,
        runtime_mode,
        global_copy,
        expected,
    ):
        from vllm.config import CUDAGraphMode

        import vllm_fl.compilation.graph as graph_module

        monkeypatch.setattr(
            graph_module.current_platform,
            "device_type",
            device_type,
        )
        monkeypatch.setattr(
            graph_module.current_platform,
            "get_global_graph_pool",
            lambda: None,
        )
        config = MagicMock()
        config.compilation_config.cudagraph_copy_inputs = global_copy
        config.parallel_config.tensor_parallel_size = tensor_parallel_size

        wrapper = graph_module.GraphWrapper(
            MagicMock(),
            config,
            getattr(CUDAGraphMode, runtime_mode),
        )

        assert wrapper._copy_inputs is expected
