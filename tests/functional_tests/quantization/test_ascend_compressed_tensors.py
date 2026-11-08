# SPDX-License-Identifier: Apache-2.0
"""Compressed-tensors matching, loading and dynamic Linear NPU contracts."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.device_type != "npu", reason="Requires Ascend NPU"
    ),
]


def _config(targets=None, ignore=None):
    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        AscendCompressedTensorsConfig,
    )

    result = AscendCompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "w8a8": {
                    "targets": targets
                    if targets is not None
                    else [
                        "layers.0.attn.wq_a",
                        "layers.0.attn.wq_b",
                        r"re:.*\.experts\.\d+\.(?:w1|w2|w3)$",
                        r"re:.*\.shared_experts\.(?:w1|w2|w3)$",
                    ],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                    "format": "int-quantized",
                }
            },
            "ignore": ignore or [],
        }
    )
    result.weight_nz_mode = 0
    configure_quant_config(result, DeepseekV4ForCausalLM)
    return result


@pytest.fixture
def config():
    return _config()


def _linear_layer():
    from vllm.model_executor.layers.linear import LinearBase

    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    return layer


def test_method_resolution(config):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_fl.quantization.w8a8.compressed_tensors import (
        AscendW8A8DynamicLinearMethod,
    )

    layer = _linear_layer()
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.wq_a"),
        AscendW8A8DynamicLinearMethod,
    )
    assert isinstance(
        config.get_quant_method(layer, "lm_head"), UnquantizedLinearMethod
    )
    assert config.get_quant_method(torch.nn.Identity(), "irrelevant") is None


def test_raw_regex_ignore_survives_model_mapper():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_fl.quantization.w8a8.compressed_tensors import (
        AscendW8A8DynamicLinearMethod,
    )

    config = _config([r"re:.*\.attn\..*"], [r"re:.*layers\.0\.attn\.compressor\.wkv$"])
    layer = _linear_layer()
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.compressor.wkv"),
        UnquantizedLinearMethod,
    )
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.wq_a"),
        AscendW8A8DynamicLinearMethod,
    )


def test_mixed_packed_rejected():
    config = _config(["layers.0.ffn.shared_experts.w1"])
    with pytest.raises(ValueError, match="Mixed"):
        config.get_quant_method(
            _linear_layer(), "model.layers.0.mlp.shared_experts.gate_up_proj"
        )


def test_mixed_expertbank_rejected():
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(num_experts=2)
    config = _config([r"re:.*\.experts\.0\.(?:w1|w2|w3)$"])
    with pytest.raises(ValueError, match="Mixed"):
        config.get_quant_method(layer, "model.layers.0.mlp.experts")


def test_registration_idempotent(monkeypatch):
    from functools import cache

    from vllm.model_executor.layers import quantization

    from vllm_fl.quantization.w8a8 import (
        compressed_tensors as compressed_tensors_config,
    )
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        AscendCompressedTensorsConfig,
    )

    calls = []

    def register(name):
        def decorator(config_class):
            calls.append((name, config_class))
            return config_class

        return decorator

    monkeypatch.setattr(quantization, "register_quantization_config", register)
    monkeypatch.setattr(
        compressed_tensors_config,
        "register_ascend_compressed_tensors",
        cache(compressed_tensors_config.register_ascend_compressed_tensors.__wrapped__),
    )
    compressed_tensors_config.register_ascend_compressed_tensors()
    compressed_tensors_config.register_ascend_compressed_tensors()
    assert calls == [("compressed-tensors", AscendCompressedTensorsConfig)]


@pytest.fixture
def native(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.ascend.impl import quantization as ops
    from vllm_fl.quantization.w8a8 import compressed_tensors as linear

    monkeypatch.setattr(linear, "_prepare_weight", ops.w8a8_prepare_weight)
    monkeypatch.setattr(linear, "_linear", ops.w8a8_dynamic_linear)
    return linear.AscendW8A8DynamicLinearMethod


def make_layer(method, dtype):
    layer = torch.nn.Module()

    def loader(*args):
        pass

    method.create_weights(layer, 128, [32, 32], 128, 64, dtype, weight_loader=loader)
    assert layer.weight.input_dim == 1
    assert layer.weight.output_dim == 0
    assert layer.weight_scale.output_dim == 0
    assert not hasattr(layer, "weight_offset")
    assert all(p.weight_loader is loader for p in layer.parameters())
    assert layer.weight_scale.shape == (64, 1)
    assert layer.weight_scale.dtype == torch.float32
    layer = layer.to("npu")
    with torch.no_grad():
        layer.weight.copy_(
            torch.randint(-64, 64, (64, 128), device="npu", dtype=torch.int8)
        )
        layer.weight_scale.fill_(0.015625)
    checkpoint_weight = layer.weight.detach().clone()
    method.process_weights_after_loading(layer)
    assert layer.weight.shape == (128, 64)
    assert layer.weight_scale.shape == (64,)
    torch.testing.assert_close(layer.weight, checkpoint_weight.t())
    return layer, checkpoint_weight


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(3, 128), (2, 1, 128), (2, 3, 128)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_loader_and_native_reference(native, dtype, shape, with_bias):
    from vllm_fl.dispatch.backends.reference.impl.quantization import (
        w8a8_dynamic_linear,
    )

    torch.manual_seed(17)
    method = native()
    layer, checkpoint_weight = make_layer(method, dtype)
    x = torch.randn(shape, dtype=dtype, device="npu")
    x.reshape(-1, 128)[0].zero_()
    bias = torch.randn(64, dtype=dtype, device="npu") if with_bias else None
    actual = method.apply(layer, x, bias)
    expected = w8a8_dynamic_linear(
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


def test_asymmetric_scheme_rejected():
    config = _config()
    for scheme in config.target_scheme_map.values():
        scheme["weights"].symmetric = False
    with pytest.raises(NotImplementedError, match="Unsupported"):
        config.get_quant_method(_linear_layer(), "model.layers.0.self_attn.wq_a")


def test_default_dispatch_native():
    from vllm_fl.dispatch import get_default_manager, with_strict_mode
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        AscendW8A8DynamicLinearMethod,
    )

    manager = get_default_manager()
    for op in ("w8a8_prepare_weight", "w8a8_dynamic_linear"):
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
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        AscendW8A8DynamicLinearMethod,
    )

    manager = get_default_manager()
    before = manager.get_selected_impl_id("w8a8_dynamic_linear")
    with with_preference("reference"):
        for op in ("w8a8_prepare_weight", "w8a8_dynamic_linear"):
            assert manager.get_selected_impl_id(op) == "reference.torch"
        method = AscendW8A8DynamicLinearMethod(weight_nz_mode=0)
        layer, _ = make_layer(method, torch.float16)
        output = method.apply(
            layer, torch.ones(2, 128, device="npu", dtype=torch.float16)
        )
        assert output.dtype == torch.float16
        assert torch.isfinite(output).all()
    assert manager.get_selected_impl_id("w8a8_dynamic_linear") == before


@pytest.mark.parametrize("groups", [1, 2])
def test_grouped_wo_a_dynamic_projection(native, groups):
    from vllm_fl.dispatch.backends.reference.impl.quantization import (
        w8a8_dynamic_linear,
    )

    method = native(weight_nz_mode=0)
    method.configure_grouped(groups)
    layer = torch.nn.Module()
    method.create_weights(layer, 128, [64 * groups], 128, 64 * groups, torch.bfloat16)
    layer = layer.to("npu")
    with torch.no_grad():
        layer.weight.copy_(
            torch.randint(-32, 32, layer.weight.shape, device="npu", dtype=torch.int8)
        )
        # Values deliberately not representable in BF16: verify lossless loading.
        scales = torch.linspace(0.010001, 0.020003, 64 * groups, device="npu").reshape(
            -1, 1
        )
        layer.weight_scale.copy_(scales)
    weight = layer.weight.detach().clone()
    torch.testing.assert_close(layer.weight_scale, scales, rtol=0, atol=0)
    assert layer.weight_scale.dtype == torch.float32
    method.process_weights_after_loading(layer)
    x = torch.randn(3, groups * 128, dtype=torch.bfloat16, device="npu")
    expected = torch.cat(
        [
            w8a8_dynamic_linear(
                x.reshape(3, groups, 128)[:, i].contiguous(),
                weight.chunk(groups)[i].t().contiguous(),
                scales.chunk(groups)[i].flatten(),
            )
            for i in range(groups)
        ],
        dim=-1,
    )
    actual = method.apply_grouped(layer, x)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)
