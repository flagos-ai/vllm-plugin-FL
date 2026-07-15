# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for layernorm ops.
"""

from unittest.mock import patch

import pytest
import torch


class TestRMSNormFL:
    """Test RMSNormFL class behavior."""

    def __init__(self):
        from vllm.config import VllmConfig, set_current_vllm_config

        set_current_vllm_config(VllmConfig())

    @pytest.fixture
    def mock_cached_op(self):
        with patch("vllm_fl.ops.layernorm._rms_norm") as mock:
            yield mock

    def test_init_creates_weight_parameter(self):
        """Test that initialization creates weight parameter with correct shape."""
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        eps = 1e-5
        layer = RMSNormFL(hidden_size=hidden_size, eps=eps)

        assert layer.variance_epsilon == eps
        assert layer.weight.shape == (hidden_size,)

    def test_forward_oot_dispatches_without_residual(self, mock_cached_op):
        """Test forward_oot calls dispatch system correctly without residual."""
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        mock_cached_op.return_value = torch.randn(2, hidden_size)

        layer = RMSNormFL(hidden_size=hidden_size)
        x = torch.randn(2, hidden_size)

        layer.forward_oot(x)

        mock_cached_op.assert_called_once()
        call_args = mock_cached_op.call_args
        assert call_args[0][0] is layer  # self
        assert torch.equal(call_args[0][1], x)
        assert call_args[0][2] is None  # residual should be None

    def test_forward_oot_dispatches_with_residual(self, mock_cached_op):
        """Test forward_oot passes residual to dispatch system."""
        from vllm_fl.ops.layernorm import RMSNormFL

        hidden_size = 128
        mock_cached_op.return_value = (
            torch.randn(2, hidden_size),
            torch.randn(2, hidden_size),
        )

        layer = RMSNormFL(hidden_size=hidden_size)
        x = torch.randn(2, hidden_size)
        residual = torch.randn(2, hidden_size)

        layer.forward_oot(x, residual=residual)

        mock_cached_op.assert_called_once()
        call_args = mock_cached_op.call_args
        assert call_args[0][0] is layer  # self
        assert torch.equal(call_args[0][1], x)
        assert torch.equal(call_args[0][2], residual)
