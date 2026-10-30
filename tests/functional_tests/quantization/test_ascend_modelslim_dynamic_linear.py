# SPDX-License-Identifier: Apache-2.0
"""Small native NPU tests; no model checkpoint or custom extension required."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.device_type != "npu", reason="Requires Ascend NPU"
    ),
]


@pytest.fixture
def native(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.ascend.impl import quantization as ops
    from vllm_fl.quantization.w8a8 import modelslim as linear

    monkeypatch.setattr(linear, "_prepare_weight", ops.modelslim_w8a8_prepare_weight)
    monkeypatch.setattr(linear, "_linear", ops.modelslim_w8a8_dynamic_linear)
    return linear.AscendW8A8DynamicLinearMethod


def make_layer(method, dtype):
    layer = torch.nn.Module()

    def loader(*args):
        pass

    method.create_weights(layer, 128, [32, 32], 128, 64, dtype, weight_loader=loader)
    assert layer.weight.input_dim == 1
    assert layer.weight.output_dim == 0
    assert layer.weight_scale.output_dim == 0
    assert layer.weight_offset.output_dim == 0
    assert all(p.weight_loader is loader for p in layer.parameters())
    assert layer.weight_scale.shape == (64, 1)
    assert layer.weight_offset.shape == (64, 1)
    layer = layer.to("npu")
    with torch.no_grad():
        layer.weight.copy_(
            torch.randint(-64, 64, (64, 128), device="npu", dtype=torch.int8)
        )
        layer.weight_scale.fill_(0.015625)
        layer.weight_offset.zero_()
    checkpoint_weight = layer.weight.detach().clone()
    method.process_weights_after_loading(layer)
    assert layer.weight.shape == (128, 64)
    assert layer.weight_scale.shape == (64,)
    assert layer.weight_offset.shape == (64,)
    torch.testing.assert_close(layer.weight, checkpoint_weight.t())
    return layer, checkpoint_weight


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(3, 128), (2, 1, 128), (2, 3, 128)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_loader_and_native_reference(native, dtype, shape, with_bias):
    from vllm_fl.dispatch.backends.reference.impl.quantization import (
        modelslim_w8a8_dynamic_linear,
    )

    torch.manual_seed(17)
    method = native()
    layer, checkpoint_weight = make_layer(method, dtype)
    x = torch.randn(shape, dtype=dtype, device="npu")
    x.reshape(-1, 128)[0].zero_()
    bias = torch.randn(64, dtype=dtype, device="npu") if with_bias else None
    actual = method.apply(layer, x, bias)
    expected = modelslim_w8a8_dynamic_linear(
        x.reshape(-1, 128),
        checkpoint_weight.t().contiguous(),
        layer.weight_scale,
        bias,
    ).reshape(*shape[:-1], 64)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.reshape(-1, 64)[0],
        bias if with_bias else torch.zeros(64, dtype=dtype, device="npu"),
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fresh_input_graph_replay(native, dtype):
    import torch_npu  # noqa: F401

    method = native()
    layer, _ = make_layer(method, dtype)
    x = torch.randn(2, 1, 128, dtype=dtype, device="npu")
    for _ in range(3):
        method.apply(layer, x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = method.apply(layer, x)
    torch.npu.synchronize()
    previous = output.clone()
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        expected = method.apply(layer, x)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected)
        assert not torch.equal(output, previous)
        previous = output.clone()


def test_nonzero_offset_rejected(native):
    method = native(weight_nz_mode=0)
    layer = torch.nn.Module()
    method.create_weights(layer, 128, [64], 128, 64, torch.bfloat16)
    layer = layer.to("npu")
    with torch.no_grad():
        layer.weight_offset.fill_(1)
    with pytest.raises(NotImplementedError, match="Nonzero"):
        method.process_weights_after_loading(layer)


def test_default_dispatch_native():
    from vllm_fl.dispatch import get_default_manager, with_strict_mode
    from vllm_fl.quantization.w8a8.modelslim import AscendW8A8DynamicLinearMethod

    manager = get_default_manager()
    for op in ("modelslim_w8a8_prepare_weight", "modelslim_w8a8_dynamic_linear"):
        assert manager.get_selected_impl_id(op) == "vendor.ascend"
    with with_strict_mode():
        method = AscendW8A8DynamicLinearMethod()
        layer, _ = make_layer(method, torch.bfloat16)
        output = method.apply(
            layer, torch.ones(2, 128, device="npu", dtype=torch.bfloat16)
        )
    assert output.shape == (2, 64)
    assert torch.isfinite(output).all()


def test_scoped_reference_dispatch():
    from vllm_fl.dispatch import get_default_manager, with_preference
    from vllm_fl.quantization.w8a8.modelslim import AscendW8A8DynamicLinearMethod

    manager = get_default_manager()
    before = manager.get_selected_impl_id("modelslim_w8a8_dynamic_linear")
    with with_preference("reference"):
        for op in ("modelslim_w8a8_prepare_weight", "modelslim_w8a8_dynamic_linear"):
            assert manager.get_selected_impl_id(op) == "reference.torch"
        method = AscendW8A8DynamicLinearMethod(weight_nz_mode=0)
        layer, _ = make_layer(method, torch.float16)
        output = method.apply(
            layer, torch.ones(2, 128, device="npu", dtype=torch.float16)
        )
        assert output.dtype == torch.float16
        assert torch.isfinite(output).all()
    assert manager.get_selected_impl_id("modelslim_w8a8_dynamic_linear") == before
