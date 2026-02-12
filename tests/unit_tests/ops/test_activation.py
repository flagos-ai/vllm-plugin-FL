# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for activation ops.
"""

import pytest
import torch
from unittest.mock import patch


class TestSiluAndMulFL:
    """Test SiluAndMulFL class behavior."""

    @pytest.fixture
    def mock_call_op(self):
        with patch("vllm_fl.ops.activation.call_op") as mock:
            yield mock

    @pytest.fixture
    def mock_parent_init(self):
        with patch("vllm_fl.ops.activation.SiluAndMul.__init__", return_value=None):
            yield

    def test_forward_oot_dispatches_correctly(self, mock_parent_init, mock_call_op):
        """Test forward_oot calls dispatch system with correct op name and input."""
        from vllm_fl.ops.activation import SiluAndMulFL

        mock_call_op.return_value = torch.randn(2, 4)
        layer = SiluAndMulFL()
        x = torch.randn(2, 8)

        result = layer.forward_oot(x)

        mock_call_op.assert_called_once_with("silu_and_mul", layer, x)
        assert result.shape == (2, 4)
