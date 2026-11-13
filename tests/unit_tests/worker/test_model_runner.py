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

# =============================================================================
# Test Utilities - Check availability before importing
# =============================================================================


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
        cudagraph_dispatcher=dispatcher,
        compilation_config=SimpleNamespace(pass_config=SimpleNamespace()),
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


@pytest.mark.parametrize("num_tokens", [1, 10, 15])
def test_batch_padding_ignores_tensor_parallel_size(monkeypatch, num_tokens):
    """vLLM 0.32 dropped sequence parallelism, so TP never pads num_tokens."""
    from vllm.config import CUDAGraphMode

    module = import_module("vllm_fl.worker.model_runner")

    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = (
        CUDAGraphMode.NONE,
        SimpleNamespace(num_tokens=num_tokens),
    )
    parallel = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank=0,
        tensor_parallel_size=4,
    )
    runner = SimpleNamespace(
        _is_uniform_decode=module.ModelRunnerFL._is_uniform_decode,
        uniform_decode_query_len=1,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        cudagraph_dispatcher=dispatcher,
        compilation_config=SimpleNamespace(pass_config=SimpleNamespace()),
        parallel_config=parallel,
        vllm_config=SimpleNamespace(
            parallel_config=parallel,
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
    )

    module.ModelRunnerFL._determine_batch_execution_and_padding(
        runner,
        num_tokens=num_tokens,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([num_tokens]),
        max_num_scheduled_tokens=num_tokens,
        use_cascade_attn=False,
    )

    # Before 0.32 the token count was rounded up to a multiple of TP when
    # sequence parallelism was on; the graph dispatch must now see it unchanged.
    assert dispatcher.dispatch.call_args.kwargs["num_tokens"] == num_tokens


def test_init_is_wrapped_by_jit_warmup_capture():
    """The V1 runner must open its JIT warmup scope through upstream's capture.

    Upstream decorates ``GPUModelRunner.__init__``; ``ModelRunnerFL`` does not
    inherit that class, so the decorator is applied here instead. Dropping it
    silently disables JIT warmup collection.
    """
    module = import_module("vllm_fl.worker.model_runner")

    init = module.ModelRunnerFL.__init__

    assert hasattr(init, "__wrapped__"), "ModelRunnerFL.__init__ is not decorated"
    assert init.__wrapped__.__name__ == "__init__"


# =============================================================================
# Layer 1: ExecuteModelState Data Structure Tests
# =============================================================================


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


# =============================================================================
# Layer 2: _get_cumsum_and_arange Algorithm Tests
# =============================================================================


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


# =============================================================================
# Layer 2: _get_positions Routing Tests
# =============================================================================


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


# =============================================================================
# Layer 2: DCP metadata field contract
# =============================================================================


class TestDcpAttentionMetadataContract:
    """Guard the vLLM 0.32 DCP field names used to build attention metadata.

    ``CommonAttentionMetadata`` is a plain dataclass, so assigning a field that
    no longer exists upstream silently creates a new attribute instead of
    raising. Every backend that runs DCP (Metax, GCU, Sunrise, ...) would then
    read ``None`` and fail much later, so pin the names here.
    """

    def test_upstream_defines_upper_bound_field(self):
        from vllm.v1.attention.backend import CommonAttentionMetadata

        names = set(getattr(CommonAttentionMetadata, "__annotations__", {}))
        assert "dcp_local_seq_lens_cpu_upper_bound" in names
        # vLLM 0.32 dropped the old name; re-introducing it would be a no-op.
        assert "dcp_local_seq_lens_cpu" not in names

    def test_runner_populates_upper_bound_field_only(self):
        import inspect

        from vllm_fl.worker.model_runner import ModelRunnerFL

        src = inspect.getsource(ModelRunnerFL._build_attention_metadata)
        assert "cm_base.dcp_local_seq_lens_cpu_upper_bound = " in src
        assert "cm_base.dcp_local_seq_lens_cpu = " not in src


# =============================================================================
# Layer 2: JIT warmup registry lifecycle
# =============================================================================


class TestJitWarmupRegistryCapture:
    """Registry must be active while ``__init__`` runs, not just constructed."""

    def test_init_is_wrapped_by_capture(self):
        import pytest

        pytest.importorskip("vllm.model_executor.warmup.jit_warmup")
        from vllm_fl.worker.model_runner import ModelRunnerFL

        assert hasattr(ModelRunnerFL.__init__, "__wrapped__"), (
            "ModelRunnerFL.__init__ must be wrapped by JitWarmupRegistry.capture; "
            "a bare registry instance never activates the registration ContextVar "
            "and every register_warmup() call is silently dropped."
        )

    def test_registry_is_active_while_init_body_runs(self, monkeypatch):
        import pytest

        module = pytest.importorskip("vllm.model_executor.warmup.jit_warmup")
        from vllm_fl.worker import model_runner as mr

        seen = {}

        def probe(self, vllm_config, device):
            seen["active"] = module.JitWarmupRegistry._active.get()
            seen["attached"] = getattr(self, "jit_warmup_registry", None)

        monkeypatch.setattr(
            mr.ModelRunnerFL, "__init__", module.JitWarmupRegistry.capture(probe)
        )
        instance = object.__new__(mr.ModelRunnerFL)
        mr.ModelRunnerFL.__init__(instance, "cfg", None)

        assert seen["active"] is not None
        assert seen["attached"] is seen["active"]
