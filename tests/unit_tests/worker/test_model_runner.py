# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for model runner module.

This module follows a layered testing strategy:
- Layer 1: Pure functions and data classes (no external dependencies)
- Layer 2: Methods with mocked dependencies
- Layer 3: Integration tests (in functional_tests/, requires GPU)

Note: These tests require vllm >= 0.13.0 with full installation.
"""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock


# =============================================================================
# Test Utilities - Check availability before importing
# =============================================================================

def has_vllm_model_runner():
    """Check if vllm model runner dependencies are available."""
    try:
        from vllm_fl.worker.model_runner import ModelRunnerFL
        return True
    except ImportError:
        return False


# Skip all tests if vllm model runner is not available
pytestmark = pytest.mark.skipif(
    not has_vllm_model_runner(),
    reason="vllm_fl.worker.model_runner not available"
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
            'scheduler_output', 'logits', 'spec_decode_metadata',
            'spec_decode_common_attn_metadata', 'hidden_states',
            'sample_hidden_states', 'aux_hidden_states',
            'ec_connector_output', 'cudagraph_stats'
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
        mock_runner._get_cumsum_and_arange = ModelRunnerFL._get_cumsum_and_arange.__get__(
            mock_runner, ModelRunnerFL
        )
        return mock_runner

    def test_multi_sequence_batch(self, mock_model_runner):
        """Test cumsum and per-sequence arange for typical multi-sequence batch."""
        num_tokens = np.array([2, 5, 3])

        cu_num_tokens, arange = mock_model_runner._get_cumsum_and_arange(num_tokens)

        # Cumsum: [2, 7, 10] - used for indexing into flattened batch
        np.testing.assert_array_equal(cu_num_tokens, np.array([2, 7, 10]))

        # Arange: per-sequence position indices [0,1 | 0,1,2,3,4 | 0,1,2]
        expected_arange = np.array([0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        np.testing.assert_array_equal(arange, expected_arange)

    def test_single_sequence(self, mock_model_runner):
        """Test with single sequence (common in generation phase)."""
        num_tokens = np.array([5])

        cu_num_tokens, arange = mock_model_runner._get_cumsum_and_arange(num_tokens)

        np.testing.assert_array_equal(cu_num_tokens, np.array([5]))
        np.testing.assert_array_equal(arange, np.array([0, 1, 2, 3, 4]))

    def test_all_single_token_sequences(self, mock_model_runner):
        """Test batch where each sequence has 1 token (decode phase)."""
        num_tokens = np.array([1, 1, 1, 1])

        cu_num_tokens, arange = mock_model_runner._get_cumsum_and_arange(num_tokens)

        np.testing.assert_array_equal(cu_num_tokens, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(arange, np.array([0, 0, 0, 0]))

    def test_large_sequences(self, mock_model_runner):
        """Test with larger sequences to verify correct boundary handling."""
        num_tokens = np.array([10, 20, 30])

        cu_num_tokens, arange = mock_model_runner._get_cumsum_and_arange(num_tokens)

        assert cu_num_tokens[-1] == 60
        assert len(arange) == 60
        # Verify boundaries: first seq 0-9, second seq 0-19, third seq 0-29
        np.testing.assert_array_equal(arange[:10], np.arange(10))
        np.testing.assert_array_equal(arange[10:30], np.arange(20))
        np.testing.assert_array_equal(arange[30:60], np.arange(30))

    def test_dtype_preservation(self, mock_model_runner):
        """Test that dtype is correctly applied to cumsum output."""
        num_tokens = np.array([2, 3])

        cu_num_tokens, _ = mock_model_runner._get_cumsum_and_arange(
            num_tokens, cumsum_dtype=np.int64
        )

        assert cu_num_tokens.dtype == np.int64


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

    @pytest.mark.parametrize("num_tokens,tp_size,expected", [
        (10, 4, 12),   # 10 -> ceil to multiple of 4
        (8, 4, 8),     # 8 already multiple of 4
        (10, 8, 16),   # 10 -> ceil to multiple of 8
        (1, 4, 4),     # 1 -> ceil to multiple of 4
        (15, 8, 16),   # 15 -> ceil to multiple of 8
    ])
    def test_padding_calculation(self, mock_model_runner, num_tokens, tp_size, expected):
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

        # Standard positions buffer
        mock_runner.positions = MagicMock()
        mock_runner.positions.gpu = torch.arange(100)

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
        expected = mock_model_runner.positions.gpu[indices]
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
