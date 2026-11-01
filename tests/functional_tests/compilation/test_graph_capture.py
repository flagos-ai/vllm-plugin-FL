# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional tests for graph capture and replay.
Tests CUDA/NPU/GCU graph functionality for model optimization.

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
    elif current_platform.device_type == "gcu":
        return torch.gcu.GCUGraph()
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
    elif current_platform.device_type == "gcu":
        with torch.gcu.graph(graph):
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


@pytest.mark.parametrize("weak_ref_output", [False, True])
def test_enflame_native_v1_wrapper_captures_and_replays_on_side_stream(
    device, weak_ref_output, record_property
):
    from importlib.metadata import version

    from packaging.version import Version

    if device.type != "gcu" or Version(version("vllm")) < Version("0.32.0"):
        pytest.skip("Enflame native V1 integration requires vLLM 0.32")
    from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import CUDAGraphMode, VllmConfig
    from vllm.forward_context import BatchDescriptor, set_forward_context

    from vllm_fl.compilation.enflame_graph import EnflameGraphWrapper

    config = VllmConfig()
    descriptor = BatchDescriptor(num_tokens=2, num_reqs=2, uniform=True)
    calls = []

    def runnable(x):
        calls.append(None)
        return x * 2 + 1

    static_input = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
    runnable(static_input)
    calls.clear()
    wrapper = EnflameGraphWrapper(
        runnable,
        config,
        CUDAGraphMode.FULL,
        CUDAGraphOptions(weak_ref_output=weak_ref_output),
    )
    stream = torch.gcu.Stream()
    stream.wait_stream(torch.gcu.current_stream())
    set_cudagraph_capturing_enabled(True)
    try:
        with set_forward_context(
            None,
            config,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=descriptor,
        ):
            with torch.gcu.stream(stream):
                captured = wrapper(static_input)
            torch.gcu.current_stream().wait_stream(stream)
            torch.gcu.synchronize()
            entry = wrapper.concrete_graph_entries[descriptor]
            assert isinstance(entry.graph, torch.gcu.GCUGraph)
            assert wrapper in CUDAGraphWrapper._all_instances
            for revision in (1, 2):
                values = torch.full((2, 4), revision, dtype=torch.float32)
                static_input.copy_(values)
                captured.fill_(-999)
                result = wrapper(static_input)
                torch.gcu.synchronize()
                torch.testing.assert_close(result.cpu(), values * 2 + 1)
                assert len(calls) == 1
                assert wrapper.concrete_graph_entries[descriptor].graph is entry.graph
            record_property("enflame_native_wrapper_graph_replays", 2)
    finally:
        set_cudagraph_capturing_enabled(False)
        wrapper.clear_graphs()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gcu_shared_experts_capture_and_replay(device, dtype, monkeypatch):
    """The native MoE shared-expert path must allocate on the capture stream."""
    if current_platform.device_type != "gcu":
        pytest.skip("Requires TorchGCU graph capture")

    from types import SimpleNamespace

    import flag_gems

    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
        SharedExpertsOrder,
    )

    from vllm_fl.dispatch.backends.vendor.gcu.patch import (
        apply_shared_experts_gcu_patch,
    )

    monkeypatch.delenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", raising=False)
    apply_shared_experts_gcu_patch()
    config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            use_fi_nvl_two_sided_kernels=False,
        )
    )
    layer = torch.nn.Linear(64, 128, bias=False, device=device, dtype=dtype)
    shared = SharedExperts(layer, config, False, lambda: False)
    static_input = torch.ones(32, 64, device=device, dtype=dtype)
    assert shared._stream is None
    with flag_gems.use_gems():
        for _ in range(3):
            assert not shared.maybe_forward_async(static_input)
            shared(static_input, SharedExpertsOrder.NO_OVERLAP)
            _ = shared.output
        torch.gcu.synchronize()
        graph = _new_graph()
        with _graph_capture(graph):
            shared(static_input, SharedExpertsOrder.NO_OVERLAP)
            output = shared.output
        for value in (2, 3):
            static_input.fill_(value)
            graph.replay()
            torch.gcu.synchronize()
            expected = torch.nn.functional.linear(
                static_input.cpu().float(), layer.weight.cpu().float()
            )
            torch.testing.assert_close(
                output.cpu().float(), expected, atol=0.03, rtol=0.03
            )
