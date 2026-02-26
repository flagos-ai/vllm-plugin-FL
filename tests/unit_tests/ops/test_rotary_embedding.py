# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for rotary embedding ops.
"""

import pytest
import torch
from unittest.mock import patch


class TestRotaryEmbeddingFL:
    """Test RotaryEmbeddingFL class behavior."""

    @pytest.fixture
    def mock_call_op(self):
        with patch("vllm_fl.ops.rotary_embedding.call_op") as mock:
            yield mock

    @pytest.fixture
    def mock_parent_init(self):
        with patch("vllm_fl.ops.rotary_embedding.RotaryEmbedding.__init__", return_value=None):
            yield

    def test_forward_oot_dispatches_correctly(self, mock_parent_init, mock_call_op):
        """Test forward_oot calls dispatch system with correct arguments."""
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        layer = RotaryEmbeddingFL(
            head_size=64,
            rotary_dim=32,
            max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )

        # Manually set attributes that parent __init__ would set
        layer.head_size = 64
        layer.rotary_dim = 32
        layer.is_neox_style = True
        layer.cos_sin_cache = torch.randn(2048, 64)

        mock_call_op.return_value = (torch.randn(4, 8, 32), torch.randn(4, 8, 32))

        positions = torch.tensor([0, 1, 2, 3])
        query = torch.randn(4, 8, 64)
        key = torch.randn(4, 8, 64)

        layer.forward_oot(positions, query, key)

        mock_call_op.assert_called_once()
        call_args = mock_call_op.call_args
        assert call_args[0][0] == "rotary_embedding"
