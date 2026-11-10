# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for activation ops.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class TestSiluAndMulFL:
    """Test SiluAndMulFL class behavior."""

    @pytest.fixture
    def mock_cached_op(self):
        with patch("vllm_fl.ops.activation._silu_and_mul") as mock:
            yield mock

    @pytest.fixture
    def mock_parent_init(self):
        with patch("vllm_fl.ops.activation.SiluAndMul.__init__", return_value=None):
            yield

    def test_forward_oot_dispatches_correctly(self, mock_parent_init, mock_cached_op):
        """Test forward_oot calls dispatch system with correct op name and input."""
        from vllm_fl.ops.activation import SiluAndMulFL

        mock_cached_op.return_value = torch.randn(2, 4)
        layer = SiluAndMulFL()
        x = torch.randn(2, 8)

        result = layer.forward_oot(x)

        mock_cached_op.assert_called_once_with(layer, x)
        assert result.shape == (2, 4)


def test_fused_moe_activation_tolerates_missing_optional_enum(monkeypatch):
    import vllm_fl.ops.fused_moe.activation as activation_module

    enum_without_uninterleave = SimpleNamespace(
        SILU=object(),
        GELU=object(),
        SWIGLUOAI=object(),
        SWIGLUSTEP=object(),
        SILU_NO_MUL=object(),
        GELU_NO_MUL=object(),
        RELU2_NO_MUL=object(),
    )
    unknown_activation = SimpleNamespace(is_gated=False, value="unknown")
    input_tensor = torch.randn(2, 4)
    output_tensor = torch.empty_like(input_tensor)
    monkeypatch.setattr(activation_module, "MoEActivation", enum_without_uninterleave)

    with pytest.raises(ValueError, match="Unsupported FusedMoe activation"):
        activation_module.apply_moe_activation(
            unknown_activation,
            output_tensor,
            input_tensor,
        )


def test_fused_moe_clamped_silu_uses_upstream_math(monkeypatch):
    import vllm_fl.ops.fused_moe.activation as activation_module

    input_tensor = torch.randn(2, 4)
    output_tensor = torch.empty(2, 2)
    with (
        patch.object(
            activation_module,
            "upstream_apply_moe_activation",
            return_value=output_tensor,
        ) as upstream,
        patch.object(activation_module, "_silu_and_mul") as ordinary_silu,
    ):
        result = activation_module.apply_moe_activation(
            activation_module.MoEActivation.SILU,
            output_tensor,
            input_tensor,
            clamp_limit=7.0,
        )

    assert result is output_tensor
    upstream.assert_called_once_with(
        activation_module.MoEActivation.SILU,
        output_tensor,
        input_tensor,
        clamp_limit=7.0,
    )
    ordinary_silu.assert_not_called()
