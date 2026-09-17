# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for activation ops.
"""

from unittest.mock import patch

import pytest
import torch


def test_upstream_factory_forwards_compile_native_false(monkeypatch):
    """OOT replacement must preserve the vLLM 0.28 factory contract."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.custom_op import CustomOp, op_registry_oot
    from vllm.model_executor.layers.activation import get_act_and_mul_fn

    import vllm_fl.ops.activation as activation

    config = VllmConfig()
    config.compilation_config.custom_ops = ["none"]
    monkeypatch.setitem(op_registry_oot, "SiluAndMul", activation.SiluAndMulFL)

    with (
        set_current_vllm_config(config),
        patch.object(
            CustomOp,
            "maybe_compile",
            autospec=True,
            side_effect=lambda _self, fn, *, enable: fn,
        ) as maybe_compile,
    ):
        layer = get_act_and_mul_fn("silu", compile_native=False)

    assert type(layer) is activation.SiluAndMulFL
    assert maybe_compile.call_args.kwargs["enable"] is False


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
        import vllm_fl.ops.activation as activation

        mock_cached_op.return_value = torch.randn(2, 4)
        layer = activation.SiluAndMulFL()
        x = torch.randn(2, 8)

        result = layer.forward_oot(x)

        mock_cached_op.assert_called_once_with(layer, x)
        assert result.shape == (2, 4)


@pytest.mark.parametrize(
    ("layer_name", "cached_op_name", "kwargs"),
    [
        ("SiluAndMulFL", "_silu_and_mul", {}),
        ("GeluAndMulFL", "_gelu_and_mul", {"approximate": "tanh"}),
    ],
)
def test_cuda_layer_call_dispatches_to_cached_op(
    cuda_vllm_config,
    layer_name,
    cached_op_name,
    kwargs,
):
    """The normal ``layer(...)`` path must use FL dispatch on CUDA."""
    import vllm_fl.ops.activation as activation

    layer_cls = getattr(activation, layer_name)
    expected = torch.randn(2, 4)
    x = torch.randn(2, 8)

    with patch.object(activation, cached_op_name, return_value=expected) as cached_op:
        layer = layer_cls(**kwargs)

        assert layer._forward_method.__func__ is layer_cls.forward_cuda
        result = layer(x)

    cached_op.assert_called_once_with(layer, x)
    assert result is expected
