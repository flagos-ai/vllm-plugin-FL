# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional tests for graph capture and replay.
Tests CUDA/NPU graph functionality for model optimization.

Note: Unit tests for GraphOptions, GraphEntry, and GraphWrapper are in
unit_tests/compilation/test_graph.py. This file only contains functional
tests that require actual GPU execution.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from vllm.platforms import current_platform

# Mark all tests as requiring GPU
pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Platform-aware graph helpers (mirrors vllm_fl/compilation/graph.py)
# ---------------------------------------------------------------------------


def _new_graph():
    """Create a platform-appropriate graph object."""
    if current_platform.device_type == "cuda":
        return torch.cuda.CUDAGraph()
    elif current_platform.device_type == "npu":
        return torch.npu.NPUGraph()
    else:
        pytest.skip(f"Graph capture not supported on {current_platform.device_type}")
        return None


@contextmanager
def _graph_capture(graph):
    """Platform-aware context manager for graph capture."""
    if current_platform.device_type == "cuda":
        with torch.cuda.graph(graph):
            yield
    elif current_platform.device_type == "npu":
        with torch.npu.graph(graph):
            yield
    else:
        pytest.skip(f"Graph capture not supported on {current_platform.device_type}")


class TestWeakRefTensors:
    """Test weak reference tensor functionality."""

    def test_weak_ref_tensors_function(self):
        """Test weak_ref_tensors function exists."""
        try:
            from vllm_fl.compilation.graph import weak_ref_tensors

            assert weak_ref_tensors is not None
        except Exception:
            pytest.skip("weak_ref_tensors not available")

    def test_weak_ref_tensors_with_cuda_tensor(self, device):
        """Test weak_ref_tensors with GPU tensor."""
        try:
            from vllm_fl.compilation.graph import weak_ref_tensors
        except Exception:
            pytest.skip("weak_ref_tensors not available")

        tensor = torch.randn(4, 8, device=device)
        result = weak_ref_tensors(tensor)
        # Result should be either the tensor or a weak reference
        assert result is not None


class TestGraphCaptureFlow:
    """Test the complete graph capture flow."""

    def test_graph_basic_capture(self, device):
        """Test basic graph capture and replay."""
        # Simple test without vllm_fl dependencies

        # Create a simple computation
        def computation(x):
            return x * 2 + 1

        # Create input tensor
        x = torch.randn(4, 8, device=device)

        # Warmup
        computation(x)

        # Capture graph
        g = _new_graph()
        with _graph_capture(g):
            y = computation(x)

        # Replay graph
        g.replay()

        # Verify output
        expected = x * 2 + 1
        assert torch.allclose(y, expected)

    def test_graph_with_different_inputs(self, device):
        """Test graph with different input values."""

        # Static input buffer
        static_input = torch.randn(4, 8, device=device)
        static_output = torch.empty(4, 8, device=device)

        def computation(x, out):
            out.copy_(x * 2)

        # Warmup
        computation(static_input, static_output)

        # Capture graph
        g = _new_graph()
        with _graph_capture(g):
            computation(static_input, static_output)

        # Test with new input values (copy to static buffer)
        new_input = torch.ones(4, 8, device=device)
        static_input.copy_(new_input)

        # Replay
        g.replay()

        expected = new_input * 2
        assert torch.allclose(static_output, expected)


class TestGraphCacheManagement:
    """Test graph cache management functionality."""

    def test_batch_descriptor_hashing(self):
        """Test that batch descriptors can be used as dict keys."""

        @dataclass(frozen=True)
        class MockBatchDescriptor:
            num_tokens: int
            max_num_reqs: int

        desc1 = MockBatchDescriptor(num_tokens=16, max_num_reqs=4)
        desc2 = MockBatchDescriptor(num_tokens=16, max_num_reqs=4)
        desc3 = MockBatchDescriptor(num_tokens=32, max_num_reqs=8)

        cache = {}
        cache[desc1] = "graph1"
        cache[desc3] = "graph3"

        # Same values should hash to same key
        assert cache[desc2] == "graph1"
        assert cache[desc3] == "graph3"

    def test_graph_entry_storage(self):
        """Test storing graph entries in cache."""
        try:
            from vllm_fl.compilation.graph import GraphEntry
        except Exception:
            pytest.skip("GraphEntry not available")

        @dataclass(frozen=True)
        class MockBatchDescriptor:
            num_tokens: int

        cache = {}
        desc = MockBatchDescriptor(num_tokens=16)

        entry = GraphEntry(batch_descriptor=desc)
        cache[desc] = entry

        assert cache[desc].batch_descriptor.num_tokens == 16
