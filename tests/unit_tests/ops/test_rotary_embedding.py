# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for rotary embedding ops.
"""

from unittest.mock import patch

import pytest
import torch


class TestRotaryEmbeddingFL:
    """Test RotaryEmbeddingFL class behavior."""

    @pytest.fixture
    def mock_cached_op(self):
        with patch("vllm_fl.ops.rotary_embedding._rotary_embedding") as mock:
            yield mock

    @pytest.fixture
    def mock_parent_init(self):
        with patch(
            "vllm_fl.ops.rotary_embedding.RotaryEmbedding.__init__", return_value=None
        ):
            yield

    def test_forward_oot_dispatches_correctly(self, mock_parent_init, mock_cached_op):
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

        mock_cached_op.return_value = (
            torch.randn(4, 8, 32),
            torch.randn(4, 8, 32),
        )

        positions = torch.tensor([0, 1, 2, 3])
        query = torch.randn(4, 8, 64)
        key = torch.randn(4, 8, 64)
        matched_cache = torch.cat(
            (torch.full((2048, 32), 3.0), torch.full((2048, 32), 7.0)),
            dim=-1,
        )

        with patch.object(
            RotaryEmbeddingFL,
            "_match_cos_sin_cache_dtype",
            return_value=matched_cache,
        ) as match_cache:
            layer.forward_oot(positions, query, key)

        match_cache.assert_called_once_with(query)
        mock_cached_op.assert_called_once()
        call_args = mock_cached_op.call_args
        assert call_args[0][0] is layer
        torch.testing.assert_close(call_args[0][3], matched_cache[:, :32])
        torch.testing.assert_close(call_args[0][4], matched_cache[:, 32:])

    def test_forward_oot_preserves_upstream_key_none_contract(
        self, mock_parent_init, mock_cached_op
    ):
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        layer = RotaryEmbeddingFL(
            head_size=64,
            rotary_dim=32,
            max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )
        positions = torch.tensor([0, 1, 2, 3])
        query = torch.randn(4, 8, 64)
        expected = (query + 1, None)

        with patch.object(
            RotaryEmbeddingFL, "forward_native", return_value=expected
        ) as native:
            actual = layer.forward_oot(positions, query, None)

        assert actual is expected
        native.assert_called_once_with(positions, query, None)
        mock_cached_op.assert_not_called()
