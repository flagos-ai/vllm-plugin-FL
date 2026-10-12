# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for rotary embedding ops.
"""

from unittest.mock import patch

import pytest
import torch


def test_upstream_constructor_accepts_init_cache_false(monkeypatch):
    """OOT replacement must accept every vLLM 0.28 constructor argument."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

    config = VllmConfig()
    config.compilation_config.custom_ops = ["all"]
    monkeypatch.setitem(op_registry_oot, "RotaryEmbedding", RotaryEmbeddingFL)

    with set_current_vllm_config(config):
        layer = RotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=16,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
            init_cache=False,
        )

    assert type(layer) is RotaryEmbeddingFL
    assert not hasattr(layer, "cos_sin_cache")


@pytest.fixture
def cuda_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("CUDA dispatch test requires the NVIDIA test environment")

    config = VllmConfig()
    config.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(config):
        yield


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

    def test_forward_oot_supports_missing_key(self, mock_cached_op):
        """vLLM 0.28 permits key=None for cross-layer KV sharing."""
        from vllm.config import VllmConfig, set_current_vllm_config

        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        config = VllmConfig()
        config.compilation_config.custom_ops = ["all"]
        with set_current_vllm_config(config):
            layer = RotaryEmbeddingFL(
                head_size=8,
                rotary_dim=8,
                max_position_embeddings=16,
                base=10000.0,
                is_neox_style=True,
                dtype=torch.float32,
            )

        positions = torch.tensor([0, 1])
        query = torch.randn(2, 1, 8)

        result_query, result_key = layer.forward_oot(positions, query, key=None)

        assert result_query.shape == query.shape
        assert result_key is None
        mock_cached_op.assert_not_called()

    def test_cuda_layer_call_dispatches_to_cached_op(
        self,
        cuda_vllm_config,
        mock_cached_op,
    ):
        """The normal ``layer(...)`` path must use FL dispatch on CUDA."""
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        layer = RotaryEmbeddingFL(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=16,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )
        positions = torch.tensor([0, 1])
        query = torch.randn(2, 1, 8)
        key = torch.randn(2, 1, 8)
        expected_query = torch.randn_like(query)
        expected_key = torch.randn_like(key)
        mock_cached_op.return_value = (expected_query, expected_key)

        assert layer._forward_method.__func__ is RotaryEmbeddingFL.forward_cuda
        result_query, result_key = layer(positions, query, key)

        mock_cached_op.assert_called_once()
        assert mock_cached_op.call_args.args[0] is layer
        assert torch.equal(result_query, expected_query)
        assert torch.equal(result_key, expected_key)
