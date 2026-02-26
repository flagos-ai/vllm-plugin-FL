# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for layernorm ops.
"""

import pytest
import torch
from unittest.mock import patch


class TestRMSNormFL:
    """Test RMSNormFL class behavior."""

    @pytest.fixture
    def mock_call_op(self):
        with patch("vllm_fl.ops.layernorm.call_op") as mock:
            yield mock

    def test_init_creates_weight_parameter(self):
        """Test that initialization creates weight parameter with correct shape."""
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        eps = 1e-5
        layer = RMSNormFL(hidden_size=hidden_size, eps=eps)

        assert layer.variance_epsilon == eps
        assert layer.weight.shape == (hidden_size,)

    def test_forward_oot_dispatches_without_residual(self, mock_call_op):
        """Test forward_oot calls dispatch system correctly without residual."""
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        mock_call_op.return_value = torch.randn(2, hidden_size)

        layer = RMSNormFL(hidden_size=hidden_size)
        x = torch.randn(2, hidden_size)

        layer.forward_oot(x)

        mock_call_op.assert_called_once()
        call_args = mock_call_op.call_args
        assert call_args[0][0] == "rms_norm"
        assert call_args[0][1] is layer  # self
        assert torch.equal(call_args[0][2], x)
        assert call_args[0][3] is None  # residual should be None

    def test_forward_oot_dispatches_with_residual(self, mock_call_op):
        """Test forward_oot passes residual to dispatch system."""
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        mock_call_op.return_value = (torch.randn(2, hidden_size), torch.randn(2, hidden_size))

        layer = RMSNormFL(hidden_size=hidden_size)
        x = torch.randn(2, hidden_size)
        residual = torch.randn(2, hidden_size)

        layer.forward_oot(x, residual=residual)

        mock_call_op.assert_called_once()
        call_args = mock_call_op.call_args
        assert call_args[0][0] == "rms_norm"
        assert call_args[0][1] is layer  # self
        assert torch.equal(call_args[0][2], x)
        assert torch.equal(call_args[0][3], residual)
