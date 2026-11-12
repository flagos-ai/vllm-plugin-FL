# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for model runner module.

This module follows a layered testing strategy:
- Layer 1: Pure functions and data classes (no external dependencies)
- Layer 2: Methods with mocked dependencies
- Layer 3: Integration tests (in functional_tests/, requires GPU)

Note: These tests require vllm >= 0.13.0 with full installation.
"""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# ======================================================================        )


# Test Utilities - Check availability before importing
# ======================================================================        )


def has_vllm_model_runner():
    """Check if vllm model runner dependencies are available."""
    try:
        from vllm_fl.worker.model_runner import ModelRunnerFL  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


# Skip all tests if vllm model runner is not available
pytestmark = pytest.mark.skipif(
    not has_vllm_model_runner(), reason="vllm_fl.worker.model_runner not available"
)


@pytest.mark.parametrize("reuse_context", [False, True])
def test_device_graph_capture_uses_accelerator_stream(monkeypatch, reuse_context):
    from contextlib import nullcontext
    from types import SimpleNamespace

    model_runner = import_module("vllm_fl.worker.model_runner")

    capture_stream = MagicMock()
    current_stream = object()
    device_api = SimpleNamespace(
        Stream=MagicMock(return_value=capture_stream),
        current_stream=MagicMock(return_value=current_stream),
        stream=MagicMock(return_value=nullcontext()),
    )
    monkeypatch.setattr(model_runner.current_platform, "torch_device_fn", device_api)
    supplied_context = SimpleNamespace(stream=capture_stream) if reuse_context else None

    with model_runner._device_graph_capture(
        torch.device("cpu"), graph_capture_context=supplied_context
    ) as capture_context:
        assert capture_context.stream is capture_stream

    if reuse_context:
        assert capture_context is supplied_context
        device_api.Stream.assert_not_called()
    else:
        device_api.Stream.assert_called_once_with(device=torch.device("cpu"))
    capture_stream.wait_stream.assert_called_once_with(current_stream)
    device_api.stream.assert_called_once_with(capture_stream)


@pytest.mark.parametrize("method", ["unknown", "future_draft_method"])
def test_propose_draft_token_ids_rejects_unsupported_method(method):
    from types import SimpleNamespace

    from vllm_fl.worker.model_runner import ModelRunnerFL

    config = SimpleNamespace(
        method=method,
        use_ngram_gpu=lambda: False,
        uses_extract_hidden_states=lambda: False,
        use_eagle=lambda: False,
        use_dflash=lambda: False,
        uses_draft_model=lambda: False,
    )
    runner = SimpleNamespace(speculative_config=config)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0, num_spec_tokens_to_schedule=0
    )
    with pytest.raises(
        ValueError, match=f"Unsupported speculative decoding method: {method}"
    ):
        ModelRunnerFL.propose_draft_token_ids(
            runner,
            scheduler_output=scheduler_output,
            sampled_token_ids=[],
            sampling_metadata=None,
            hidden_states=None,
            sample_hidden_states=None,
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata=None,
            slot_mappings=None,
        )


@pytest.mark.parametrize("disable_full", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_full_graph_exclusion_survives_dp_redispatch(
    monkeypatch, disable_full, dp_size
):
    from vllm.config import CUDAGraphMode

    module = import_module("vllm_fl.worker.model_runner")

    dispatcher = MagicMock()
    mode = CUDAGraphMode.NONE if disable_full else CUDAGraphMode.FULL
    dispatcher.dispatch.return_value = (mode, SimpleNamespace(num_tokens=1))
    parallel = SimpleNamespace(data_parallel_size=dp_size, data_parallel_rank=0)
    runner = SimpleNamespace(
        _is_uniform_decode=module.ModelRunnerFL._is_uniform_decode,
        uniform_decode_query_len=1,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        _pad_for_sequence_parallelism=lambda tokens: tokens,
        cudagraph_dispatcher=dispatcher,
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=parallel,
        vllm_config=SimpleNamespace(
            parallel_config=parallel,
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
    )
    monkeypatch.setattr(
        module,
        "coordinate_batch_across_dp",
        lambda **kwargs: (False, torch.ones(dp_size, dtype=torch.int32), mode.value),
    )
    result = module.ModelRunnerFL._determine_batch_execution_and_padding(
        runner,
        num_tokens=1,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([1]),
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
        disable_full=disable_full,
    )
    assert result[0] == mode
    assert dispatcher.dispatch.call_count == dp_size
    for call in dispatcher.dispatch.call_args_list:
        assert call.kwargs["invalid_modes"] == (
            {CUDAGraphMode.FULL} if disable_full else None
        )


# ======================================================================        )


# Layer 1: ExecuteModelState Data Structure Tests
# ======================================================================        )


class TestExecuteModelState:
    """Test ExecuteModelState NamedTuple behavior and contract."""

    def test_fields_match_expected_contract(self):
        """Verify ExecuteModelState has exact fields required by execute_model pipeline."""
        from vllm_fl.worker.model_runner import ExecuteModelState

        expected_fields = (
            "scheduler_output",
            "logits",
            "spec_decode_metadata",
            "spec_decode_common_attn_metadata",
            "hidden_states",
            "sample_hidden_states",
            "aux_hidden_states",
            "ec_connector_output",
            "cudagraph_stats",
            "slot_mappings",
        )
        assert ExecuteModelState._fields == expected_fields, (
            "ExecuteModelState fields changed - this may break execute_model consumers"
        )

    def test_immutability_prevents_accidental_mutation(self):
        """Ensure state cannot be mutated after creation (important for pipeline safety)."""
        from vllm_fl.worker.model_runner import ExecuteModelState

        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.randn(4, 1000),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.randn(4, 512),
            sample_hidden_states=torch.randn(4, 512),
            aux_hidden_states=None,
            ec_connector_output=None,
            cudagraph_stats=None,
            slot_mappings=None,
        )

        with pytest.raises(AttributeError):
            state.logits = torch.randn(4, 1000)

    def test_unpacking_for_downstream_processing(self):
        """Test that state can be unpacked correctly for downstream use."""
        from vllm_fl.worker.model_runner import ExecuteModelState

        mock_scheduler = MagicMock()
        mock_logits = torch.randn(4, 1000)

        state = ExecuteModelState(
            scheduler_output=mock_scheduler,
            logits=mock_logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=None,
            sample_hidden_states=None,
            aux_hidden_states=None,
            ec_connector_output=None,
            cudagraph_stats=None,
            slot_mappings=None,
        )

        # Simulate downstream unpacking
        scheduler, logits, *rest = state
        assert scheduler is mock_scheduler
        assert torch.equal(logits, mock_logits)


# ======================================================================        )


# Layer 2: _get_cumsum_and_arange Algorithm Tests
# ======================================================================        )


class TestGetCumsumAndArange:
    """Test _get_cumsum_and_arange method - critical for batch processing."""

    @pytest.fixture
    def mock_model_runner(self):
        """Create a minimal mock of ModelRunnerFL for testing."""
        from vllm_fl.worker.model_runner import ModelRunnerFL

        mock_runner = MagicMock(spec=ModelRunnerFL)
        mock_runner.arange_np = np.arange(10000)
        mock_runner._get_cumsum_and_arange = (
            ModelRunnerFL._get_cumsum_and_arange.__get__(mock_runner, ModelRunnerFL)
        )
        return mock_runner

    def test_multi_sequence_batch(self, mock_model_runner):
        """Test cumsum and per-sequence arange for typical multi-sequence batch."""
        num_tokens = np.array([2, 5, 3])
        arange_out = np.zeros(10, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(num_tokens, arange_out)

        # Cumsum: [2, 7, 10] - used for indexing into flattened batch
        np.testing.assert_array_equal(cu_num_tokens, np.array([2, 7, 10]))

        # Arange: per-sequence position indices [0,1 | 0,1,2,3,4 | 0,1,2]
        expected_arange = np.array([0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        np.testing.assert_array_equal(arange_out, expected_arange)

    def test_single_sequence(self, mock_model_runner):
        """Test with single sequence (common in generation phase)."""
        num_tokens = np.array([5])
        arange_out = np.zeros(5, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(num_tokens, arange_out)

        np.testing.assert_array_equal(cu_num_tokens, np.array([5]))
        np.testing.assert_array_equal(arange_out, np.array([0, 1, 2, 3, 4]))

    def test_all_single_token_sequences(self, mock_model_runner):
        """Test batch where each sequence has 1 token (decode phase)."""
        num_tokens = np.array([1, 1, 1, 1])
        arange_out = np.zeros(4, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(num_tokens, arange_out)

        np.testing.assert_array_equal(cu_num_tokens, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(arange_out, np.array([0, 0, 0, 0]))

    def test_large_sequences(self, mock_model_runner):
        """Test with larger sequences to verify correct boundary handling."""
        num_tokens = np.array([10, 20, 30])
        arange_out = np.zeros(60, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(num_tokens, arange_out)

        assert cu_num_tokens[-1] == 60
        # Verify boundaries: first seq 0-9, second seq 0-19, third seq 0-29
        np.testing.assert_array_equal(arange_out[:10], np.arange(10))
        np.testing.assert_array_equal(arange_out[10:30], np.arange(20))
        np.testing.assert_array_equal(arange_out[30:60], np.arange(30))

    def test_dtype_preservation(self, mock_model_runner):
        """Test that dtype is correctly applied to cumsum output."""
        num_tokens = np.array([2, 3])
        arange_out = np.zeros(5, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(
            num_tokens, arange_out, cumsum_dtype=np.int64
        )

        assert cu_num_tokens.dtype == np.int64


# ======================================================================        )


# Layer 2: _get_positions Routing Tests
# ======================================================================        )


class TestGetPositions:
    """Test position retrieval for different position encoding schemes."""

    @pytest.fixture
    def mock_model_runner(self):
        """Create mock model runner for position tests."""
        from vllm_fl.worker.model_runner import ModelRunnerFL

        mock_runner = MagicMock(spec=ModelRunnerFL)

        # Standard positions buffer (used directly, not .gpu)
        mock_runner.positions = torch.arange(100)

        # MRoPE positions (3D for temporal, height, width)
        mock_runner.mrope_positions = MagicMock()
        mock_runner.mrope_positions.gpu = torch.arange(300).reshape(3, 100)

        mock_runner.uses_mrope = False

        mock_runner._get_positions = ModelRunnerFL._get_positions.__get__(
            mock_runner, ModelRunnerFL
        )
        return mock_runner

    def test_standard_positions_with_int(self, mock_model_runner):
        """Standard RoPE: integer returns first N positions."""
        result = mock_model_runner._get_positions(10)
        torch.testing.assert_close(result, torch.arange(10))

    def test_standard_positions_with_indices(self, mock_model_runner):
        """Standard RoPE: tensor indices for selective position lookup."""
        indices = torch.tensor([0, 5, 10, 15])
        result = mock_model_runner._get_positions(indices)
        expected = mock_model_runner.positions[indices]
        torch.testing.assert_close(result, expected)

    def test_mrope_returns_3d_positions(self, mock_model_runner):
        """MRoPE (Qwen2-VL): returns [3, num_tokens] positions."""
        mock_model_runner.uses_mrope = True

        result = mock_model_runner._get_positions(10)

        expected = mock_model_runner.mrope_positions.gpu[:, :10]
        assert result.shape == (3, 10)
        torch.testing.assert_close(result, expected)

    def test_mrope_returns_indexed_positions(self, mock_model_runner):
        """0.32 routes indexed multimodal positions through MRoPE."""
        mock_model_runner.uses_mrope = True
        indices = torch.tensor([15, 0, 99, 5])
        result = mock_model_runner._get_positions(indices)
        expected = mock_model_runner.mrope_positions.gpu[:, indices]
        assert result.shape == (3, 4)
        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dense_hybrid", [False, True])
def test_cache_initialization_preserves_native_and_dense_binding(
    monkeypatch, dense_hybrid
):
    from contextlib import nullcontext

    module = import_module("vllm_fl.worker.model_runner")
    caches = {"layer": torch.zeros(1), "mamba": torch.zeros(1)}
    mapped = {"layer": torch.ones(1)}
    native_allocator = MagicMock(return_value=dict(caches))
    dense_allocator = MagicMock(return_value=dict(caches))
    mapper = MagicMock(return_value=mapped)
    binder = MagicMock()
    monkeypatch.setattr(module, "allocate_kv_cache", native_allocator)
    monkeypatch.setattr(module, "_allocate_dense_hybrid_cache", dense_allocator)
    monkeypatch.setattr(module, "map_kv_caches_to_kernel_blocks", mapper)
    monkeypatch.setattr(module, "bind_kv_cache", binder)
    runner = SimpleNamespace(
        use_ascend_hybrid_cache=dense_hybrid,
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(
            cache_dtype="auto", get_resolved_kv_cache_layout=lambda: "layout"
        ),
        attn_groups=[[]],
        shared_kv_cache_layers={"shared": "layer"},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_caches=[],
    )
    config = SimpleNamespace(kv_cache_groups=[])
    result = module.ModelRunnerFL.initialize_kv_cache_tensors(
        runner, config, [1], nullcontext()
    )
    assert result["shared"] is result["layer"]
    if dense_hybrid:
        dense_allocator.assert_called_once_with(
            config, runner.device, [[]], [1], "auto"
        )
        native_allocator.assert_not_called()
        mapper.assert_not_called()
        assert binder.call_args.kwargs["layer_kv_caches"] is result
    else:
        native_allocator.assert_called_once_with(config, runner.device, "layout", [1])
        dense_allocator.assert_not_called()
        mapper.assert_called_once_with(result, [])
        assert binder.call_args.kwargs["layer_kv_caches"] is mapped
