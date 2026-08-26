# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for model runner module.

This module follows a layered testing strategy:
- Layer 1: Pure functions and data classes (no external dependencies)
- Layer 2: Methods with mocked dependencies
- Layer 3: Integration tests (in functional_tests/, requires GPU)

Note: These tests require vllm >= 0.13.0 with full installation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class TestAscendSlotMappingBuffers:
    def test_int32_conversion_buffer_has_stable_address_across_steps(self):
        from vllm_fl.worker import model_runner as model_runner_module

        source = torch.arange(16, dtype=torch.int64)
        runner = SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        kv_cache_spec=object(),
                        layer_names=["model.layers.0.self_attn"],
                    )
                ]
            ),
            input_batch=SimpleNamespace(
                block_table=[
                    SimpleNamespace(
                        slot_mapping=SimpleNamespace(gpu=source)
                    )
                ]
            ),
            device=torch.device("cpu"),
        )

        with patch.object(
            model_runner_module,
            "current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            first_by_group, _ = model_runner_module.ModelRunnerFL._get_slot_mappings(
                runner,
                num_tokens_padded=8,
                num_reqs_padded=1,
                num_tokens_unpadded=6,
            )
            first = first_by_group[0]
            first_address = first.data_ptr()

            source[:8].add_(100)
            second_by_group, _ = model_runner_module.ModelRunnerFL._get_slot_mappings(
                runner,
                num_tokens_padded=8,
                num_reqs_padded=1,
                num_tokens_unpadded=7,
            )
            second = second_by_group[0]

        assert first.dtype == torch.int32
        assert second.data_ptr() == first_address
        torch.testing.assert_close(
            second,
            torch.tensor(
                [100, 101, 102, 103, 104, 105, 106, -1],
                dtype=torch.int32,
            ),
        )


class TestDenseMambaStateViews:
    def test_dense_mamba_views_are_non_overlapping(self):
        from vllm_fl.worker.model_runner import _make_dense_mamba_state_views

        num_blocks = 4
        shapes = [(3, 8), (2, 8, 8)]
        dtypes = [torch.bfloat16, torch.float32]
        required_bytes = sum(
            num_blocks
            * int(np.prod(shape))
            * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in zip(shapes, dtypes)
        )
        raw = torch.empty(required_bytes + 64, dtype=torch.uint8)

        conv_state, ssm_state = _make_dense_mamba_state_views(
            raw, num_blocks, shapes, dtypes
        )

        assert conv_state.shape == (num_blocks, *shapes[0])
        assert ssm_state.shape == (num_blocks, *shapes[1])
        assert conv_state.stride(0) == int(np.prod(shapes[0]))
        assert ssm_state.stride(0) == int(np.prod(shapes[1]))

        conv_start = conv_state.storage_offset() * conv_state.element_size()
        conv_end = conv_start + conv_state.numel() * conv_state.element_size()
        ssm_start = ssm_state.storage_offset() * ssm_state.element_size()
        ssm_end = ssm_start + ssm_state.numel() * ssm_state.element_size()
        assert conv_start == 0
        assert conv_end == ssm_start
        assert ssm_end == required_bytes


class TestReserveModularMoeWorkspace:
    def test_reserves_maximum_modular_kernel_requirement(self):
        from vllm_fl.worker.model_runner import _reserve_modular_moe_workspace

        class FakeExperts:
            @staticmethod
            def workspace_shapes(
                tokens,
                intermediate_dim,
                hidden_dim,
                top_k,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            ):
                assert (global_num_experts, local_num_experts) == (256, 128)
                assert expert_tokens_meta is None
                assert activation == "silu"
                return (
                    (tokens, top_k, intermediate_dim // 2),
                    (tokens, top_k, max(intermediate_dim, hidden_dim)),
                    (tokens, hidden_dim),
                )

            @staticmethod
            def workspace_dtype(in_dtype):
                assert in_dtype == torch.bfloat16
                return torch.bfloat16

        moe_layer = torch.nn.Module()
        moe_layer.quant_method = MagicMock()
        moe_layer.quant_method.moe_kernel.is_monolithic = False
        moe_layer.quant_method.moe_kernel.impl.fused_experts = FakeExperts()
        moe_layer.moe_config = MagicMock(
            hidden_dim=1024,
            experts_per_token=8,
            num_experts=256,
            in_dtype=torch.bfloat16,
        )
        moe_layer.global_num_experts = 256
        moe_layer.w13_weight = torch.empty(128, 512, 1024, device="meta")
        moe_layer.activation = "silu"

        model = torch.nn.Module()
        model.add_module("moe", moe_layer)
        workspace_manager = MagicMock()

        num_kernels, reserved_bytes = _reserve_modular_moe_workspace(
            model, max_num_tokens=64, workspace_manager=workspace_manager
        )

        assert num_kernels == 1
        assert reserved_bytes == (64 * 8 * 256 + 64 * 8 * 1024) * 2
        workspace_manager.get_simultaneous.assert_called_once_with(
            ((64 * 8 * 256,), torch.bfloat16),
            ((64, 8, 1024), torch.bfloat16),
        )

    def test_skips_monolithic_moe_kernel(self):
        from vllm_fl.worker.model_runner import _reserve_modular_moe_workspace

        moe_layer = torch.nn.Module()
        moe_layer.quant_method = MagicMock()
        moe_layer.quant_method.moe_kernel.is_monolithic = True
        model = torch.nn.Module()
        model.add_module("moe", moe_layer)
        workspace_manager = MagicMock()

        result = _reserve_modular_moe_workspace(
            model, max_num_tokens=64, workspace_manager=workspace_manager
        )

        assert result == (0, 0)
        workspace_manager.get_simultaneous.assert_not_called()


# =============================================================================
# Layer 2: _pad_for_sequence_parallelism Logic Tests
# =============================================================================


class TestPadForSequenceParallelism:
    """Test sequence parallelism padding logic."""

    @pytest.fixture
    def mock_model_runner(self):
        """Create mock model runner for padding tests."""
        from vllm_fl.worker.model_runner import ModelRunnerFL

        mock_runner = MagicMock(spec=ModelRunnerFL)
        mock_runner.vllm_config = MagicMock()
        mock_runner.vllm_config.parallel_config = MagicMock()
        mock_runner.compilation_config = MagicMock()
        mock_runner.compilation_config.pass_config = MagicMock()
        mock_runner._pad_for_sequence_parallelism = (
            ModelRunnerFL._pad_for_sequence_parallelism.__get__(
                mock_runner, ModelRunnerFL
            )
        )
        return mock_runner

    def test_no_padding_when_sp_disabled(self, mock_model_runner):
        """SP disabled should return original token count."""
        mock_model_runner.vllm_config.parallel_config.tensor_parallel_size = 4
        mock_model_runner.compilation_config.pass_config.enable_sp = False

        assert mock_model_runner._pad_for_sequence_parallelism(10) == 10

    def test_no_padding_when_tp_size_1(self, mock_model_runner):
        """TP size 1 means no parallelism, no padding needed."""
        mock_model_runner.vllm_config.parallel_config.tensor_parallel_size = 1
        mock_model_runner.compilation_config.pass_config.enable_sp = True

        assert mock_model_runner._pad_for_sequence_parallelism(10) == 10

    @pytest.mark.parametrize(
        "num_tokens,tp_size,expected",
        [
            (10, 4, 12),  # 10 -> ceil to multiple of 4
            (8, 4, 8),  # 8 already multiple of 4
            (10, 8, 16),  # 10 -> ceil to multiple of 8
            (1, 4, 4),  # 1 -> ceil to multiple of 4
            (15, 8, 16),  # 15 -> ceil to multiple of 8
        ],
    )
    def test_padding_calculation(
        self, mock_model_runner, num_tokens, tp_size, expected
    ):
        """Verify padding rounds up to next multiple of tp_size."""
        mock_model_runner.vllm_config.parallel_config.tensor_parallel_size = tp_size
        mock_model_runner.compilation_config.pass_config.enable_sp = True

        result = mock_model_runner._pad_for_sequence_parallelism(num_tokens)

        assert result == expected
        assert result % tp_size == 0  # Must be divisible


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

        # XDRoPE positions (2D)
        mock_runner.xdrope_positions = MagicMock()
        mock_runner.xdrope_positions.gpu = torch.arange(200).reshape(2, 100)

        mock_runner.uses_mrope = False
        mock_runner.uses_xdrope_dim = 0

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

    def test_xdrope_returns_2d_positions(self, mock_model_runner):
        """XDRoPE: returns [2, num_tokens] positions."""
        mock_model_runner.uses_xdrope_dim = 64

        result = mock_model_runner._get_positions(10)

        expected = mock_model_runner.xdrope_positions.gpu[:, :10]
        assert result.shape == (2, 10)
        torch.testing.assert_close(result, expected)


class TestGraphRuntimeDelegation:
    def test_model_forward_delegates_graph_lifecycle(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        model_output = object()
        runner = MagicMock(spec=ModelRunnerFL)
        runner.model = MagicMock(return_value=model_output)
        runner.graph_runtime = MagicMock()
        runner.vllm_config = MagicMock()
        runner._model_forward = ModelRunnerFL._model_forward.__get__(
            runner, ModelRunnerFL
        )
        positions = torch.zeros((1,), dtype=torch.int64)

        result = runner._model_forward(positions=positions)

        assert result is model_output
        runner.model.assert_called_once_with(
            input_ids=None,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=None,
        )
        runner.graph_runtime.after_model_forward.assert_called_once_with(
            runner.vllm_config
        )
