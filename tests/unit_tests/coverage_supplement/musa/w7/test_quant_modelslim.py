# Coverage tests: vllm_fl/quantization/modelslim_w8a8.py (CPU only).

import sys
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm_fl.quantization import modelslim_w8a8 as modelslim


@pytest.fixture(autouse=True)
def _single_rank_tensor_parallel(monkeypatch):
    import vllm.model_executor.parameter as vllm_parameter

    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )


def _config(description=None):
    return modelslim.ModelSlimW8A8Config(description or {})


def test_class_metadata_and_from_config():
    assert modelslim.ModelSlimW8A8Config.get_name() == "fl_modelslim_w8a8"
    assert modelslim.ModelSlimW8A8Config.get_supported_act_dtypes() == [
        torch.bfloat16,
        torch.float16,
    ]
    assert modelslim.ModelSlimW8A8Config.get_min_capability() == 0
    assert modelslim.ModelSlimW8A8Config.get_config_filenames() == [
        "quant_model_description.json"
    ]

    config = modelslim.ModelSlimW8A8Config.from_config({"a.weight": "W8A8"})
    assert config.description == {"a.weight": "W8A8"}
    assert config.weight_quant.num_bits == 8
    assert config.input_quant.dynamic is True
    assert modelslim._ModelSlimW8A8StaticLinearScheme.get_min_capability() == 0


def test_root_prefixes_both_roots():
    config = _config()
    assert config._root_prefixes("model.layers.0") == ["model.layers.0", "layers.0"]
    assert config._root_prefixes("layers.0") == ["layers.0", "model.layers.0"]


@pytest.mark.parametrize(
    ("prefix", "root_groups"),
    [
        ("x.attn.fused_wqa_wkv", [("x.attn.wq_a", "x.attn.wkv")]),
        (
            "x.fused_qkv_a_proj",
            [("x.q_a_proj", "x.kv_a_proj_with_mqa")],
        ),
        (
            "x.experts",
            [
                ("x.experts.0.w1", "x.experts.0.w2", "x.experts.0.w3"),
                (
                    "x.experts.0.gate_proj",
                    "x.experts.0.up_proj",
                    "x.experts.0.down_proj",
                ),
            ],
        ),
        (
            "x.gate_up_proj",
            [("x.w1", "x.w3"), ("x.gate_proj", "x.up_proj")],
        ),
        ("x.down_proj", [("x.down_proj",), ("x.w2",)]),
        ("x.plain", [("x.plain",)]),
    ],
)
def test_checkpoint_prefix_groups(prefix, root_groups):
    config = _config()
    expected = list(root_groups)
    expected.extend(
        tuple(f"model.{name}" for name in group) for group in root_groups
    )
    assert config._checkpoint_prefix_groups(prefix) == expected


def test_checkpoint_prefix_groups_covers_both_roots():
    config = _config()
    groups = config._checkpoint_prefix_groups("model.layers.0.attn.fused_wqa_wkv")
    assert ("model.layers.0.attn.wq_a", "model.layers.0.attn.wkv") in groups
    assert ("layers.0.attn.wq_a", "layers.0.attn.wkv") in groups


def test_quant_kind_float_none_and_unsupported():
    assert _config({"x.weight": "FLOAT"})._quant_kind("x") == "FLOAT"
    assert _config({})._quant_kind("x") is None

    with pytest.raises(NotImplementedError, match="not supported"):
        _config({"x.weight": "INT4"})._quant_kind("x")


def test_quant_kind_mixed_group_raises():
    config = _config(
        {
            "x.attn.wq_a.weight": "W8A8",
            "x.attn.wkv.weight": "FLOAT",
        }
    )
    with pytest.raises(ValueError, match="mixes"):
        config._quant_kind("x.attn.fused_wqa_wkv")


def test_quant_kind_missing_group_entries_raise():
    config = _config({"x.attn.wq_a.weight": "W8A8"})
    with pytest.raises(ValueError, match="missing quantization entries"):
        config._quant_kind("x.attn.fused_wqa_wkv")


def test_dynamic_and_static_helpers():
    assert _config({"x.weight": "W8A8_DYNAMIC"})._is_dynamic_w8a8("x") is True
    assert _config({"x.weight": "W8A8_DYNAMIC"})._is_static_w8a8("x") is False
    assert _config({"x.weight": "W8A8"})._is_static_w8a8("x") is True
    assert _config({"x.weight": "W8A8"})._is_dynamic_w8a8("x") is False


class _DummyLinear:
    pass


class _DummyRouted:
    def __init__(self):
        self.moe_config = object()


def test_get_quant_method_for_linear(monkeypatch):
    monkeypatch.setattr(modelslim, "LinearBase", _DummyLinear)
    monkeypatch.setattr(modelslim, "RoutedExperts", _DummyRouted)
    monkeypatch.setattr(modelslim, "UnquantizedLinearMethod", lambda: "unquantized")
    monkeypatch.setattr(
        modelslim, "CompressedTensorsLinearMethod", lambda config: "ct-linear"
    )
    monkeypatch.setattr(
        modelslim, "_ModelSlimW8A8LinearScheme", lambda **kwargs: ("dyn", kwargs)
    )
    monkeypatch.setattr(
        modelslim, "_ModelSlimW8A8StaticLinearScheme", lambda: "static-scheme"
    )

    assert _config({}).get_quant_method(_DummyLinear(), "x") == "unquantized"
    assert (
        _config({"x.weight": "FLOAT"}).get_quant_method(_DummyLinear(), "x")
        == "unquantized"
    )

    static_layer = _DummyLinear()
    assert (
        _config({"x.weight": "W8A8"}).get_quant_method(static_layer, "x")
        == "ct-linear"
    )
    assert static_layer.scheme == "static-scheme"

    dynamic_layer = _DummyLinear()
    assert (
        _config({"x.weight": "W8A8_DYNAMIC"}).get_quant_method(dynamic_layer, "x")
        == "ct-linear"
    )
    assert dynamic_layer.scheme[0] == "dyn"


def test_get_quant_method_for_routed_experts(monkeypatch):
    monkeypatch.setattr(modelslim, "LinearBase", _DummyLinear)
    monkeypatch.setattr(modelslim, "RoutedExperts", _DummyRouted)
    monkeypatch.setattr(
        modelslim, "UnquantizedFusedMoEMethod", lambda config: ("unquantized", config)
    )
    monkeypatch.setattr(
        modelslim, "_ModelSlimW8A8MoEMethod", lambda *args, **kwargs: ("moe", args, kwargs)
    )

    layer = _DummyRouted()
    result = _config({}).get_quant_method(layer, "x")
    assert result[0] == "unquantized"

    with pytest.raises(NotImplementedError, match="static W8A8 routed experts"):
        _config({"x.weight": "W8A8"}).get_quant_method(_DummyRouted(), "x")

    config = _config({"x.weight": "W8A8_DYNAMIC"})
    result = config.get_quant_method(layer, "x")
    assert result[0] == "moe"
    assert result[2]["layer_name"] == "x"

    assert config.get_quant_method(object(), "x") is None


def test_linear_scheme_registers_weight_offset(monkeypatch):
    registered = []
    monkeypatch.setattr(
        modelslim.CompressedTensorsW8A8Int8,
        "create_weights",
        lambda self, layer, **kwargs: registered.append(layer),
    )
    layer = torch.nn.Module()

    scheme = modelslim._ModelSlimW8A8LinearScheme(
        strategy=QuantizationStrategy.CHANNEL,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    scheme.create_weights(
        layer,
        output_partition_sizes=[4, 4],
        weight_loader=lambda *args, **kwargs: None,
    )

    assert registered == [layer]
    assert layer.weight_offset.shape == (8, 1)
    assert layer.weight_offset.dtype == torch.float32


def test_moe_method_registers_offset_parameters(monkeypatch):
    monkeypatch.setattr(
        modelslim.CompressedTensorsW8A8Int8MoEMethod,
        "create_weights",
        lambda self, layer, *args, **kwargs: None,
    )
    attrs_seen = []
    monkeypatch.setattr(
        modelslim, "set_weight_attrs", lambda tensor, attrs: attrs_seen.append(attrs)
    )
    layer = torch.nn.Module()
    method = object.__new__(modelslim._ModelSlimW8A8MoEMethod)

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=8,
        params_dtype=torch.bfloat16,
    )

    assert layer.w13_weight_offset.shape == (2, 16, 1)
    assert layer.w2_weight_offset.shape == (2, 4, 1)
    assert layer.w13_weight_offset.dtype == torch.float32
    assert attrs_seen[0]["quant_method"] == modelslim.FusedMoeWeightScaleSupported.CHANNEL.value
    assert len(attrs_seen) == 2


def _static_layer(widths=(3, 5), input_size=4):
    scheme = modelslim._ModelSlimW8A8StaticLinearScheme()
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=list(widths),
        input_size_per_partition=input_size,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    return scheme, layer


def test_static_scheme_rejects_fp16():
    scheme = modelslim._ModelSlimW8A8StaticLinearScheme()
    with pytest.raises(ValueError, match="requires bfloat16"):
        scheme.create_weights(
            torch.nn.Module(),
            output_partition_sizes=[8],
            input_size_per_partition=4,
            params_dtype=torch.float16,
            weight_loader=lambda *args, **kwargs: None,
        )


def test_static_scheme_registers_and_prepares_parameters():
    scheme, layer = _static_layer()
    assert layer.logical_widths == [3, 5]
    assert layer.params_dtype is torch.bfloat16
    assert set(dict(layer.named_parameters())) == {
        "weight",
        "input_scale",
        "input_offset",
        "deq_scale",
        "quant_bias",
    }

    layer.weight.data.copy_(torch.arange(32, dtype=torch.int8).reshape(8, 4))
    layer.input_scale.data.copy_(torch.tensor([0.25, 0.5]))
    layer.input_offset.data.copy_(torch.tensor([1.0, -2.0], dtype=torch.float32))
    scheme.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 8)
    assert layer.aclnn_input_scale_reciprocal.dtype == torch.bfloat16
    assert torch.equal(
        layer.aclnn_input_scale_reciprocal,
        torch.tensor([[4.0] * 4, [2.0] * 4], dtype=torch.bfloat16),
    )


def test_static_scheme_rejects_scale_count_mismatch():
    scheme, layer = _static_layer()
    layer.weight.data.zero_()
    layer.input_scale = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    layer.input_offset.data.zero_()

    with pytest.raises(ValueError, match="one input scale per logical"):
        scheme.process_weights_after_loading(layer)


def test_static_scheme_quantize_and_matmul_reference(monkeypatch):
    calls = []
    module = SimpleNamespace(
        npu_quantize=lambda x, scale, offset, dtype, axis, sqrt_mode: calls.append(
            ("quant", scale, offset, dtype, axis, sqrt_mode)
        )
        or torch.zeros_like(x, dtype=torch.int8),
        npu_quant_matmul=lambda x, weight, deq_scale, *, bias, output_dtype: calls.append(
            ("matmul", weight.shape, bias, output_dtype)
        )
        or torch.zeros((x.shape[0], weight.shape[1]), dtype=output_dtype),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    scheme = modelslim._ModelSlimW8A8StaticLinearScheme()

    x = torch.ones((2, 4), dtype=torch.bfloat16)
    quantized = scheme._quantize(x, torch.ones((1, 4)), torch.zeros((1, 4)))
    assert quantized.dtype == torch.int8
    output = scheme._quant_matmul(
        quantized, torch.ones((4, 3), dtype=torch.int8), torch.ones(3), None, torch.bfloat16
    )
    assert output.shape == (2, 3)
    assert [call[0] for call in calls] == ["quant", "matmul"]


def _torch_npu_recorder(monkeypatch):
    quantized = torch.zeros((1, 1), dtype=torch.int8)

    def quantize(x, scale, offset, dtype, axis, sqrt_mode):
        return torch.full_like(x, 1, dtype=torch.int8)

    def quant_matmul(x, weight, deq_scale, *, bias, output_dtype):
        return torch.zeros((x.shape[0], weight.shape[1]), dtype=output_dtype)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_quantize=quantize, npu_quant_matmul=quant_matmul),
    )
    return quantized


def _prepared_static(widths, input_size):
    scheme, layer = _static_layer(widths, input_size)
    output = sum(widths)
    layer.weight.data.copy_(
        torch.arange(output * input_size, dtype=torch.int8).reshape(output, input_size)
    )
    layer.input_scale.data.fill_(0.5)
    layer.input_offset.data.zero_()
    layer.deq_scale.data.fill_(0.125)
    layer.quant_bias.data.zero_()
    scheme.process_weights_after_loading(layer)
    return scheme, layer


def test_static_apply_rejects_prequantized_multi_partition():
    scheme = modelslim._ModelSlimW8A8StaticLinearScheme()
    layer = SimpleNamespace(logical_widths=[3, 5])

    with pytest.raises(ValueError, match="single logical matrix"):
        scheme.apply_weights(layer, torch.zeros((2, 4), dtype=torch.int8), None)


def test_static_apply_single_and_fused_partitions(monkeypatch):
    _torch_npu_recorder(monkeypatch)

    single_scheme, single = _prepared_static([4], 4)
    single_out = single_scheme.apply_weights(
        single, torch.ones((2, 4), dtype=torch.bfloat16), None
    )
    assert single_out.shape == (2, 4)

    scheme, layer = _prepared_static([3, 5], 4)
    bias = torch.ones((8,), dtype=torch.bfloat16)
    fused = scheme.apply_weights(layer, torch.ones((2, 4), dtype=torch.bfloat16), bias)
    assert fused.shape == (2, 8)


def test_static_apply_row_parallel_skips_tp1_bias(monkeypatch):
    _torch_npu_recorder(monkeypatch)
    recorded_bias = []

    def matmul(x, weight, deq_scale, *, bias, output_dtype):
        recorded_bias.append(bias)
        return torch.zeros((x.shape[0], weight.shape[1]), dtype=output_dtype)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_quantize=lambda x, scale, offset, dtype, axis, sqrt_mode: torch.full_like(
                x, 1, dtype=torch.int8
            ),
            npu_quant_matmul=matmul,
        ),
    )

    class FakeRowParallel(torch.nn.Module):
        pass

    monkeypatch.setattr(modelslim, "RowParallelLinear", FakeRowParallel)
    scheme, base = _prepared_static([4], 4)
    layer = FakeRowParallel()
    layer.logical_widths = base.logical_widths
    layer.params_dtype = base.params_dtype
    layer.weight = base.weight
    layer.quant_bias = base.quant_bias
    layer.deq_scale = base.deq_scale
    layer.aclnn_input_scale_reciprocal = base.aclnn_input_scale_reciprocal
    layer.aclnn_input_offset = base.aclnn_input_offset
    layer.tp_rank = 1
    bias = torch.ones((4,), dtype=torch.bfloat16)

    scheme.apply_weights(layer, torch.ones((2, 4), dtype=torch.bfloat16), bias)

    assert recorded_bias == [None]


def test_static_apply_row_parallel_resolves_rank(monkeypatch):
    _torch_npu_recorder(monkeypatch)
    monkeypatch.setattr(modelslim, "get_tensor_model_parallel_rank", lambda: 0)

    class FakeRowParallel(torch.nn.Module):
        pass

    monkeypatch.setattr(modelslim, "RowParallelLinear", FakeRowParallel)
    scheme, base = _prepared_static([4], 4)
    layer = FakeRowParallel()
    layer.logical_widths = base.logical_widths
    layer.params_dtype = base.params_dtype
    layer.weight = base.weight
    layer.quant_bias = base.quant_bias
    layer.deq_scale = base.deq_scale
    layer.aclnn_input_scale_reciprocal = base.aclnn_input_scale_reciprocal
    layer.aclnn_input_offset = base.aclnn_input_offset
    layer.tp_rank = None

    scheme.apply_weights(layer, torch.ones((2, 4), dtype=torch.bfloat16), None)
