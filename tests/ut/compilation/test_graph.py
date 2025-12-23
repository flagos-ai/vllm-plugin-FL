import dataclasses
from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    from vllm_fl.compilation.graph import GraphEntry, GraphWrapper
except ImportError:
    pass

# --- Mocks ---


@dataclasses.dataclass
class MockBatchDescriptor:
    batch_size: int

    def __hash__(self):
        return hash(self.batch_size)


class MockEnumMember:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if hasattr(other, "name"):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"MockMode.{self.name}"


class MockCUDAGraphMode:
    NONE = MockEnumMember("NONE")
    FULL = MockEnumMember("FULL")


@dataclasses.dataclass
class MockForwardContext:
    batch_descriptor: MockBatchDescriptor
    cudagraph_runtime_mode: MockEnumMember


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def forward(self, x):
        self.counter += 1
        return x.clone() * 2 + 1


# --- Fixtures ---


@pytest.fixture(autouse=True)
def mock_dependencies():
    with (
        patch("vllm_fl.compilation.graph.envs") as mock_envs,
        patch("vllm_fl.compilation.graph.get_forward_context") as mock_get_ctx,
        patch("vllm_fl.compilation.graph.current_platform") as mock_platform,
        patch("vllm_fl.compilation.graph.set_graph_pool_id") as mock_set_pool,
        patch("vllm_fl.compilation.graph.CUDAGraphMode", MockCUDAGraphMode),
        patch("vllm_fl.compilation.graph.validate_cudagraph_capturing_enabled"),
        patch("vllm_fl.compilation.graph.weak_ref_tensors", side_effect=lambda x: x),
        patch("vllm_fl.compilation.graph.Graph") as MockGraph,
    ):

        MockGraph.graph = torch.cuda.CUDAGraph
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"

        mock_platform.device_type = "cuda"
        mock_platform.is_cuda = True
        mock_platform.get_global_graph_pool.return_value = None
        mock_platform.torch_device_fn.graph = torch.cuda.graph
        mock_platform.torch_device_fn.synchronize = torch.cuda.synchronize

        yield {"get_context": mock_get_ctx, "envs": mock_envs}


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.compilation_config.cudagraph_mode = MockCUDAGraphMode.FULL
    config.compilation_config.cudagraph_options = MagicMock()
    config.compilation_config.cudagraph_options.debug_log_enable = True
    config.compilation_config.cudagraph_options.gc_disable = True
    return config


# --- Tests ---


def test_cuda_graph_capture_and_replay(mock_dependencies, mock_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda:0"
    mock_get_context = mock_dependencies["get_context"]

    # 1. Setup
    model = SimpleModel().to(device)
    input_tensor = torch.ones(2, 2, device=device, dtype=torch.float32)

    wrapper = GraphWrapper(
        runnable=model, vllm_config=mock_config, runtime_mode=MockCUDAGraphMode.FULL
    )

    descriptor = MockBatchDescriptor(batch_size=2)
    mock_get_context.return_value = MockForwardContext(
        batch_descriptor=descriptor, cudagraph_runtime_mode=MockCUDAGraphMode.FULL
    )

    # --- Run 1: Capture Phase ---
    output1 = wrapper(input_tensor)
    torch.cuda.synchronize()

    assert model.counter == 1, "Capture phase should run Python code"
    assert descriptor in wrapper.concrete_graph_entries
    assert wrapper.concrete_graph_entries[descriptor].graph is not None

    # --- Run 2: Replay Phase ---
    input_tensor.fill_(2.0)
    expected_output_2 = input_tensor * 2 + 1  # 5.0

    output2 = wrapper(input_tensor)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output2, expected_output_2, rtol=1e-3, atol=1e-3, msg="Replay output mismatch"
    )

    assert model.counter == 1, "Replay should SKIP Python code execution"


def test_passthrough_when_mode_mismatch(mock_dependencies, mock_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mock_get_context = mock_dependencies["get_context"]
    model = SimpleModel().to("cuda")
    input_tensor = torch.randn(2, 2, device="cuda")

    wrapper = GraphWrapper(
        runnable=model, vllm_config=mock_config, runtime_mode=MockCUDAGraphMode.FULL
    )

    # Mode = NONE -> Passthrough
    mock_get_context.return_value = MockForwardContext(
        batch_descriptor=MockBatchDescriptor(batch_size=2),
        cudagraph_runtime_mode=MockCUDAGraphMode.NONE,
    )

    wrapper(input_tensor)
    assert model.counter == 1
    assert len(wrapper.concrete_graph_entries) == 0

    wrapper(input_tensor)
    assert model.counter == 2


def test_input_address_validation(mock_dependencies, mock_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mock_get_context = mock_dependencies["get_context"]
    model = SimpleModel().to("cuda")

    t1 = torch.zeros(2, device="cuda")
    t2 = torch.zeros(2, device="cuda")

    wrapper = GraphWrapper(
        runnable=model, vllm_config=mock_config, runtime_mode=MockCUDAGraphMode.FULL
    )

    descriptor = MockBatchDescriptor(batch_size=2)
    mock_get_context.return_value = MockForwardContext(
        batch_descriptor=descriptor, cudagraph_runtime_mode=MockCUDAGraphMode.FULL
    )

    wrapper(t1)  # Capture

    with pytest.raises(AssertionError, match="Input addresses .* different"):
        wrapper(t2)  # Replay with diff address


def test_multiple_batch_descriptors(mock_dependencies, mock_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mock_get_context = mock_dependencies["get_context"]
    model = SimpleModel().to("cuda")
    t = torch.zeros(2, device="cuda")

    wrapper = GraphWrapper(
        runnable=model, vllm_config=mock_config, runtime_mode=MockCUDAGraphMode.FULL
    )

    # Batch 2
    bd1 = MockBatchDescriptor(batch_size=2)
    mock_get_context.return_value = MockForwardContext(bd1, MockCUDAGraphMode.FULL)
    wrapper(t)
    assert model.counter == 1

    # Batch 4
    bd2 = MockBatchDescriptor(batch_size=4)
    mock_get_context.return_value = MockForwardContext(bd2, MockCUDAGraphMode.FULL)
    wrapper(t)

    assert model.counter == 2
    assert len(wrapper.concrete_graph_entries) == 2


if __name__ == "__main__":
    pytest.main([__file__])
