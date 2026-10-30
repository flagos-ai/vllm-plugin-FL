# SPDX-License-Identifier: Apache-2.0
"""ModelSlim configuration, weight loading and dynamic Linear NPU contracts."""

import json

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
def config(tmp_path):
    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM
    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

    description = {
        "embed.weight": "FLOAT",
        "head.weight": "FLOAT",
        "layers.0.attn.wq_a.weight": "W8A8_DYNAMIC",
        "layers.0.attn.wq_b.weight": "W8A8_DYNAMIC",
        "layers.0.attn.wo_a.weight": "FLOAT",
        "layers.0.ffn.gate.weight": "FLOAT",
    }
    for projection in ("w1", "w2", "w3"):
        for owner in ("experts.0", "shared_experts"):
            description[f"layers.0.ffn.{owner}.{projection}.weight"] = "W8A8_DYNAMIC"
    (tmp_path / "quant_model_description.json").write_text(json.dumps(description))
    result = AscendModelSlimConfig()
    result.maybe_update_config(str(tmp_path))
    configure_quant_config(result, DeepseekV4ForCausalLM)
    return result


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("model.layers.0.self_attn.wq_a", "W8A8_DYNAMIC"),
        ("model.layers.0.self_attn.wq_b", "W8A8_DYNAMIC"),
        ("model.layers.0.self_attn.wo_a", "FLOAT"),
        ("model.layers.0.mlp.gate", "FLOAT"),
        ("model.layers.0.mlp.experts", "W8A8_DYNAMIC"),
        ("model.layers.0.mlp.shared_experts.gate_up_proj", "W8A8_DYNAMIC"),
        ("model.embed_tokens", "FLOAT"),
        ("lm_head", "FLOAT"),
    ],
)
def test_descriptor_prefixes(config, prefix, expected):
    assert (
        config.get_layer_quant_type(prefix, is_moe=prefix.endswith(".experts"))
        == expected
    )


def test_method_resolution(config):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_fl.quantization.w8a8.modelslim import AscendW8A8DynamicLinearMethod

    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    method = config.get_quant_method(layer, "model.layers.0.self_attn.wq_a")
    assert isinstance(method, AscendW8A8DynamicLinearMethod)
    assert isinstance(
        config.get_quant_method(layer, "lm_head"), UnquantizedLinearMethod
    )
    assert config.get_quant_method(torch.nn.Identity(), "irrelevant") is None


def test_packed_mixed_and_missing(config):
    key = "model.layers.0.mlp.experts.0.down_proj.weight"
    config.quant_description[key] = "FLOAT"
    with pytest.raises(ValueError, match="Mixed"):
        config.get_layer_quant_type("model.layers.0.mlp.experts", is_moe=True)
    del config.quant_description[key]
    with pytest.raises(ValueError, match="Missing"):
        config.get_layer_quant_type("model.layers.0.mlp.experts", is_moe=True)


def test_framework_packing_preserves_descriptor_expert_aliases(config):
    from copy import deepcopy

    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM

    original = deepcopy(DeepseekV4ForCausalLM.packed_modules_mapping)
    configure_quant_config(config, DeepseekV4ForCausalLM)
    assert config.packed_modules_mapping is DeepseekV4ForCausalLM.packed_modules_mapping
    assert "experts" not in config.packed_modules_mapping
    assert (
        config.get_layer_quant_type("model.layers.0.mlp.experts", is_moe=True)
        == "W8A8_DYNAMIC"
    )
    assert (
        config.get_layer_quant_type("model.layers.0.mlp.shared_experts.gate_up_proj")
        == "W8A8_DYNAMIC"
    )

    class AlternatePacking(DeepseekV4ForCausalLM):
        packed_modules_mapping = {"shared_pair": ["gate_proj", "up_proj"]}

    alternate = deepcopy(AlternatePacking.packed_modules_mapping)
    configure_quant_config(config, AlternatePacking)
    assert (
        config.get_layer_quant_type("model.layers.0.mlp.shared_experts.shared_pair")
        == "W8A8_DYNAMIC"
    )
    assert (
        config.get_layer_quant_type("model.layers.0.mlp.experts", is_moe=True)
        == "W8A8_DYNAMIC"
    )
    assert AlternatePacking.packed_modules_mapping == alternate
    assert DeepseekV4ForCausalLM.packed_modules_mapping == original


def test_external_mapper_idempotence():
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

    config = AscendModelSlimConfig({"model.proj.weight": "W8A8_DYNAMIC"})
    mapper = WeightsMapper(orig_to_new_prefix={"model.": "wrapped.model."})
    config.apply_vllm_mapper(mapper)
    once = dict(config.quant_description)
    config.apply_vllm_mapper(mapper)
    assert config.quant_description == once
    assert config.get_layer_quant_type("wrapped.model.proj") == "W8A8_DYNAMIC"


def test_generic_dense_names_and_framework_packing():
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_fl.quantization.w8a8.modelslim import (
        AscendModelSlimConfig,
        AscendW8A8DynamicLinearMethod,
    )

    description = {
        "layers.0.attn.w1.weight": "W8A8_DYNAMIC",
        "layers.0.attn.w3.weight": "W8A8_DYNAMIC",
    }
    config = AscendModelSlimConfig(description)
    assert config.quant_description == description
    assert config.get_layer_quant_type("layers.0.attn.w1") == "W8A8_DYNAMIC"

    class DenseModel(torch.nn.Module):
        packed_modules_mapping = {"pair": ["w1", "w3"]}

    configure_quant_config(config, DenseModel)
    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    assert isinstance(
        config.get_quant_method(layer, "layers.0.attn.pair"),
        AscendW8A8DynamicLinearMethod,
    )
    assert DenseModel.packed_modules_mapping == {"pair": ["w1", "w3"]}
    assert config.quant_description == description


def test_non_idempotent_model_mapper_applies_only_to_descriptor():
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

    class DenseModel(torch.nn.Module):
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"model.": "model.language_model."}
        )

    config = AscendModelSlimConfig({"model.proj.weight": "W8A8_DYNAMIC"})
    configure_quant_config(config, DenseModel)
    configure_quant_config(config, DenseModel)
    assert config.quant_description == {
        "model.language_model.proj.weight": "W8A8_DYNAMIC"
    }
    assert config.get_layer_quant_type("model.language_model.proj") == "W8A8_DYNAMIC"


def test_dense_linear_named_experts_is_not_moe():
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_fl.quantization.w8a8.modelslim import (
        AscendModelSlimConfig,
        AscendW8A8DynamicLinearMethod,
    )

    config = AscendModelSlimConfig({"experts.weight": "W8A8_DYNAMIC"})
    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    assert config.get_layer_quant_type("experts") == "W8A8_DYNAMIC"
    assert isinstance(
        config.get_quant_method(layer, "experts"), AscendW8A8DynamicLinearMethod
    )


def test_generic_descriptor_file_load_other_model_type(tmp_path):
    from types import SimpleNamespace

    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

    description = {"layers.0.attn.w1.weight": "W8A8_DYNAMIC"}
    (tmp_path / "quant_model_description.json").write_text(json.dumps(description))
    config = AscendModelSlimConfig()
    config.maybe_update_config(str(tmp_path), SimpleNamespace(model_type="dense_probe"))
    assert config.quant_description == description
    assert config.get_layer_quant_type("layers.0.attn.w1") == "W8A8_DYNAMIC"


def test_deepseek_descriptor_mapper_loading_order(tmp_path):
    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM
    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

    description = {"layers.0.attn.wq_a.weight": "W8A8_DYNAMIC"}
    (tmp_path / "quant_model_description.json").write_text(json.dumps(description))
    before = AscendModelSlimConfig(description)
    configure_quant_config(before, DeepseekV4ForCausalLM)
    after = AscendModelSlimConfig()
    configure_quant_config(after, DeepseekV4ForCausalLM)
    after.maybe_update_config(str(tmp_path))
    assert (
        before.quant_description
        == after.quant_description
        == {"model.layers.0.self_attn.wq_a.weight": "W8A8_DYNAMIC"}
    )
    for config in (before, after):
        configure_quant_config(config, DeepseekV4ForCausalLM)
        assert config.quant_description == before.quant_description
        with pytest.raises(ValueError, match="Missing"):
            config.get_layer_quant_type("layers.0.attn.wq_a")
        assert (
            config.get_layer_quant_type("model.layers.0.self_attn.wq_a")
            == "W8A8_DYNAMIC"
        )


@pytest.mark.parametrize(
    "description",
    [
        {"a.weight": "W8A8"},
        {"a.weight": "W4A8_DYNAMIC"},
        {"kv_cache_type": "C8"},
        {"fa_quant_type": "W8A8_DYNAMIC"},
    ],
)
def test_unsupported_types(description):
    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

    with pytest.raises(NotImplementedError):
        AscendModelSlimConfig(description)


def test_registration_idempotent(monkeypatch):
    from functools import cache

    from vllm.model_executor.layers import quantization

    from vllm_fl.quantization.w8a8 import modelslim as modelslim_config
    from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimConfig

    calls = []

    def register(name):
        def decorator(config_class):
            calls.append((name, config_class))
            return config_class

        return decorator

    monkeypatch.setattr(quantization, "register_quantization_config", register)
    monkeypatch.setattr(
        modelslim_config,
        "register_modelslim_config",
        cache(modelslim_config.register_modelslim_config.__wrapped__),
    )
    modelslim_config.register_modelslim_config()
    modelslim_config.register_modelslim_config()
    assert calls == [("ascend", AscendModelSlimConfig)]


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
