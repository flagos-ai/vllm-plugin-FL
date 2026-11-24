# Coverage tests for vllm_fl/quantization/arm_cpu_w4a8.py using fake vLLM
# collaborators, so the patch body can be exercised on any CPU host.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.quantization import arm_cpu_w4a8


def _parameter(data=None, **kwargs):
    del kwargs
    return torch.nn.Parameter(data, requires_grad=False)

def _build_classes():
    class FakeKernel(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config
            self.kwargs = kwargs
            self.w_q_name = kwargs.get("w_q_param_name", "weight")
            self.w_s_name = kwargs.get("w_s_param_name", "weight_scale")

        def process_weights_after_loading(self, layer):
            self.original_processed = layer

        def apply_weights(self, layer, input, bias=None):
            return torch.zeros((input.shape[0], 1), dtype=input.dtype)

    class FakeScheme:
        def __init__(
            self,
            strategy=None,
            num_bits=4,
            group_size=None,
            is_static_input_scheme=False,
            input_symmetric=True,
            packed=False,
        ):
            self.strategy = strategy
            self.num_bits = num_bits
            self.group_size = group_size
            self.is_static_input_scheme = is_static_input_scheme
            self.input_symmetric = input_symmetric
            self.packed = packed
            self.quant_type = "int"
            self.original_created = []

        def create_weights(self, layer, *args, **kwargs):
            self.original_created.append((layer, args, kwargs))

    class FakeConfig:
        def __init__(self):
            self.quant_format = None
            self._dynamic = False

        def _is_dynamic_token_w4a8_int(self, weight_quant, input_quant):
            return self._dynamic

        def _get_scheme_from_parts(
            self,
            weight_quant,
            input_quant,
            output_quant=None,
            format=None,
            layer_name=None,
        ):
            return ("original", weight_quant, input_quant)

    return FakeKernel, FakeScheme, FakeConfig

def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install(monkeypatch):
    FakeKernel, FakeScheme, FakeConfig = _build_classes()
    linear_mod = _module(
        "vllm.model_executor.kernels.linear",
        MPLinearLayerConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        choose_mp_linear_kernel=lambda config: FakeKernel,
    )
    linear_mod.__path__ = []
    mixed_mod = _module(
        "vllm.model_executor.kernels.linear.mixed_precision"
    )
    mixed_mod.__path__ = []
    dynamic_mod = _module(
        "vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit",
        Dynamic4bitLinearKernel=FakeKernel,
    )
    config_mod = _module("vllm...compressed_tensors", CompressedTensorsConfig=FakeConfig)
    schemes_pkg = _module("vllm...schemes")
    schemes_pkg.__path__ = []
    scheme_mod = _module("vllm...schemes.w4a8", CompressedTensorsW4A8Int=FakeScheme)
    schemes_pkg.compressed_tensors_w4a8_int = scheme_mod
    ct_pkg = _module("vllm...compressed_tensors.pkg")
    ct_pkg.__path__ = []
    ct_pkg.compressed_tensors = config_mod
    ct_pkg_outer = _module(
        "vllm.model_executor.layers.quantization.compressed_tensors"
    )
    ct_pkg_outer.__path__ = []
    ct_pkg_outer.compressed_tensors = config_mod
    ct_pkg_outer.schemes = schemes_pkg
    quant_utils = _module(
        "vllm.model_executor.layers.quantization.utils",
        replace_parameter=lambda layer, name, value: setattr(layer, name, value),
    )
    parameter_mod = _module(
        "vllm.model_executor.parameter",
        BasevLLMParameter=_parameter,
        ChannelQuantScaleParameter=_parameter,
        GroupQuantScaleParameter=_parameter,
        PackedvLLMParameter=_parameter,
    )

    calls = {"pack": [], "linear": []}

    def pack_rhs_qsi4c128p(weight, scales):
        calls["pack"].append((weight.clone(), scales.clone()))
        return torch.zeros((weight.shape[0], weight.shape[1] // 2), dtype=torch.uint8)

    def w4a8_g128_linear(input, weight, n, k):
        calls["linear"].append((input, weight, n, k))
        return torch.zeros((input.shape[0], n), dtype=torch.bfloat16)

    qlinear = _module(
        "flag_gems.quantized_linear",
        pack_rhs_qsi4c128p=pack_rhs_qsi4c128p,
        w4a8_g128_linear=w4a8_g128_linear,
    )

    monkeypatch.setitem(sys.modules, "vllm.model_executor.kernels.linear", linear_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.linear.mixed_precision",
        mixed_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit",
        dynamic_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors",
        ct_pkg_outer,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        config_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes",
        schemes_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a8_int",
        scheme_mod,
    )
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.quantization.utils", quant_utils
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.parameter", parameter_mod)
    monkeypatch.setitem(sys.modules, "flag_gems.quantized_linear", qlinear)

    assert arm_cpu_w4a8.install_arm_cpu_packed_w4a8() is True
    assert arm_cpu_w4a8.install_arm_cpu_packed_w4a8() is False
    return SimpleNamespace(
        config_cls=FakeConfig,
        scheme_cls=FakeScheme,
        kernel_cls=FakeKernel,
        calls=calls,
    )


def test_installer_marks_runtime_and_scheme_init(monkeypatch):
    ctx = _install(monkeypatch)

    scheme = ctx.scheme_cls(
        strategy="group", num_bits=4, group_size=128, packed=True
    )
    assert scheme._vllm_fl_checkpoint_packed is True
    assert scheme._vllm_fl_pack_factor == 8
    packed_scheme = ctx.scheme_cls(
        strategy="group", num_bits=8, group_size=64, packed=False
    )
    assert packed_scheme._vllm_fl_pack_factor == 4


def test_create_weights_non_packed_delegates(monkeypatch):
    ctx = _install(monkeypatch)
    scheme = ctx.scheme_cls(strategy="group", num_bits=4, group_size=128)
    layer = torch.nn.Module()

    ctx.scheme_cls.create_weights(
        scheme,
        layer,
        output_size=4,
        input_size=8,
        output_partition_sizes=[4],
        input_size_per_partition=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert len(scheme.original_created) == 1
    assert not hasattr(layer, "weight_packed")


def test_create_weights_packed_group_scale(monkeypatch):
    ctx = _install(monkeypatch)
    scheme = ctx.scheme_cls(
        strategy="group", num_bits=4, group_size=8, packed=True
    )
    layer = torch.nn.Module()

    ctx.scheme_cls.create_weights(
        scheme,
        layer,
        output_size=8,
        input_size=8,
        output_partition_sizes=[8],
        input_size_per_partition=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.weight_packed.shape == (8, 1)
    assert layer.weight_packed.dtype == torch.int32
    assert layer.weight_scale.shape == (8, 1)
    assert layer.weight_shape.shape == (2,)
    assert scheme.kernel._vllm_fl_arm_packed_checkpoint is True


def test_create_weights_packed_row_parallel_channel_scale(monkeypatch):
    ctx = _install(monkeypatch)
    scheme = ctx.scheme_cls(
        strategy="group", num_bits=4, group_size=-1, packed=True
    )
    layer = torch.nn.Module()

    ctx.scheme_cls.create_weights(
        scheme,
        layer,
        output_size=8,
        input_size=16,
        output_partition_sizes=[8],
        input_size_per_partition=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.weight_scale.shape == (8, 1)
    assert scheme.kernel.config.group_size == 8


def test_create_weights_packed_rejects_indivisible_group(monkeypatch):
    ctx = _install(monkeypatch)
    scheme = ctx.scheme_cls(
        strategy="group", num_bits=4, group_size=128, packed=True
    )
    with pytest.raises(ValueError, match="not divisible"):
        ctx.scheme_cls.create_weights(
            scheme,
            torch.nn.Module(),
            output_size=8,
            input_size=8,
            output_partition_sizes=[8],
            input_size_per_partition=8,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )


def test_get_scheme_from_parts_selects_packed_scheme(monkeypatch):
    ctx = _install(monkeypatch)
    config = ctx.config_cls()
    config.quant_format = "pack-quantized"
    config._dynamic = True
    weight_quant = SimpleNamespace(num_bits=4, strategy="group", group_size=128)
    input_quant = SimpleNamespace(symmetric=True)

    scheme = config._get_scheme_from_parts(
        weight_quant, input_quant, format=None, layer_name="linear"
    )

    assert scheme._vllm_fl_checkpoint_packed is True
    assert scheme.group_size == 128
    assert ctx.config_cls()._get_scheme_from_parts(weight_quant, input_quant)[0] == "original"


def test_process_weights_non_packed_delegates(monkeypatch):
    ctx = _install(monkeypatch)
    kernel = ctx.kernel_cls(
        SimpleNamespace(group_size=64, zero_points=False, partition_weight_shape=(8, 8)),
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
    )
    kernel._vllm_fl_arm_packed_checkpoint = False
    layer = torch.nn.Module()

    ctx.kernel_cls.process_weights_after_loading(kernel, layer)

    assert kernel.original_processed is layer


def test_process_weights_packed_unpacks_and_packs(monkeypatch):
    ctx = _install(monkeypatch)
    config = SimpleNamespace(
        group_size=128, zero_points=False, partition_weight_shape=(128, 8)
    )
    kernel = ctx.kernel_cls(
        config, w_q_param_name="weight_packed", w_s_param_name="weight_scale"
    )
    kernel._vllm_fl_arm_packed_checkpoint = True

    quantized = torch.zeros((8, 16), dtype=torch.int32)
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed", torch.nn.Parameter(quantized, requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((8, 1), dtype=torch.float32), requires_grad=False),
    )
    layer.register_parameter(
        "weight_shape",
        torch.nn.Parameter(
            torch.tensor([8, 128], dtype=torch.int64), requires_grad=False
        ),
    )
    layer.register_parameter(
        "weight", torch.nn.Parameter(quantized.detach(), requires_grad=False)
    )

    ctx.kernel_cls.process_weights_after_loading(kernel, layer)

    assert layer.weight_packed.dtype == torch.uint8
    assert layer.weight_scale is None
    assert layer.weight_shape is None
    assert kernel._vllm_fl_arm_w4a8_shape == (8, 128)
    assert layer.weight.numel() == 0
    assert ctx.calls["pack"][0][0].dtype == torch.int8
    assert ctx.calls["pack"][0][0].shape == (8, 128)


def test_process_weights_packed_rejects_bad_weight_shape(monkeypatch):
    ctx = _install(monkeypatch)
    kernel = ctx.kernel_cls(
        SimpleNamespace(group_size=128, zero_points=False, partition_weight_shape=(128, 8)),
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
    )
    kernel._vllm_fl_arm_packed_checkpoint = True
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.zeros((8, 16), dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((8, 1)), requires_grad=False),
    )
    layer.register_parameter(
        "weight_shape",
        torch.nn.Parameter(torch.tensor([8, 64], dtype=torch.int64), requires_grad=False),
    )

    with pytest.raises(RuntimeError, match="unexpected packed-Q4 weight_shape"):
        ctx.kernel_cls.process_weights_after_loading(kernel, layer)


def test_process_weights_rejects_bad_unpacked_parameter(monkeypatch):
    ctx = _install(monkeypatch)
    kernel = ctx.kernel_cls(
        SimpleNamespace(group_size=128, zero_points=False, partition_weight_shape=(128, 8)),
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
    )
    kernel._vllm_fl_arm_packed_checkpoint = True
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.zeros((8, 64), dtype=torch.int8), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((8, 1)), requires_grad=False),
    )

    with pytest.raises(RuntimeError, match="unexpected compressed-tensors"):
        ctx.kernel_cls.process_weights_after_loading(kernel, layer)


def test_process_weights_packed_falls_back_for_other_groups(monkeypatch):
    ctx = _install(monkeypatch)
    kernel = ctx.kernel_cls(
        SimpleNamespace(group_size=64, zero_points=False, partition_weight_shape=(8, 8)),
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
    )
    kernel._vllm_fl_arm_packed_checkpoint = True
    layer = torch.nn.Module()

    ctx.kernel_cls.process_weights_after_loading(kernel, layer)

    assert kernel.original_processed is layer


def test_apply_weights_unprepared_uses_original(monkeypatch):
    ctx = _install(monkeypatch)
    kernel = ctx.kernel_cls(SimpleNamespace(), w_q_param_name="weight_packed")
    result = ctx.kernel_cls.apply_weights(kernel, torch.nn.Module(), torch.ones((2, 4)))
    assert result.shape == (2, 1)


def test_apply_weights_prepared_converts_dtype_and_bias(monkeypatch):
    ctx = _install(monkeypatch)
    kernel = ctx.kernel_cls(SimpleNamespace(), w_q_param_name="weight_packed")
    kernel._vllm_fl_arm_w4a8_shape = (4, 8)
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.zeros((4, 4), dtype=torch.uint8), requires_grad=False),
    )

    input_f16 = torch.ones((2, 8), dtype=torch.float16)
    bias = torch.ones(4, dtype=torch.float16)
    output = ctx.kernel_cls.apply_weights(kernel, layer, input_f16, bias)

    assert output.shape == (2, 4)
    assert output.dtype == torch.float16
    assert ctx.calls["linear"][0][0].dtype == torch.bfloat16
    assert ctx.calls["linear"][0][2:] == (4, 8)

    input_bf16 = torch.ones((2, 8), dtype=torch.bfloat16)
    bf16_output = ctx.kernel_cls.apply_weights(kernel, layer, input_bf16, None)
    assert bf16_output.dtype == torch.bfloat16

