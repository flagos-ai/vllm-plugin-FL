# Copyright (c) 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coverage-focused unit tests for the quantization adapters.

Everything here runs on CPU small tensors and mocks the vendor kernels.  The
goal is to execute error branches, dtype dispatch and configuration parsing
without ever launching a real quantized GEMM/MoE kernel.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.quantization import arm_cpu_w4a8, modelslim_w8a8, quant_linear
from vllm_fl.quantization.w8a8 import moe_experts, packed, reference


# ---------------------------------------------------------------------------
# ARM CPU packed W4A8: a self-contained fake vLLM kernel/scheme tree.
# ---------------------------------------------------------------------------


class _FakeParameter(torch.nn.Parameter):
    def __new__(cls, data=None, *args, **kwargs):
        if data is None:
            data = torch.empty(0)
        obj = super().__new__(cls, data, requires_grad=False)
        obj.fl_kwargs = dict(kwargs)
        return obj

    def __init__(self, *args, **kwargs):
        pass


class PackedvLLMParameter(_FakeParameter):
    pass


class ChannelQuantScaleParameter(_FakeParameter):
    pass


class GroupQuantScaleParameter(_FakeParameter):
    pass


class BasevLLMParameter(_FakeParameter):
    pass


class _MPConfig:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _LaunchedKernel:
    def __init__(
        self,
        config,
        w_q_param_name=None,
        w_s_param_name=None,
        w_zp_param_name=None,
        w_gidx_param_name=None,
    ):
        self.config = config
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        self.w_zp_name = w_zp_param_name
        self.w_gidx_name = w_gidx_param_name


class _Dynamic4bitLinearKernel:
    def __init__(self, *args, **kwargs):
        self.config = None
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self._orig_process_calls = 0
        self._orig_apply_calls = 0

    def process_weights_after_loading(self, layer):
        self._orig_process_calls += 1

    def apply_weights(self, layer, input, bias=None):
        self._orig_apply_calls += 1
        return "stock-apply"


class _FakeW4A8Scheme:
    def __init__(
        self,
        strategy=None,
        num_bits=None,
        group_size=None,
        is_static_input_scheme=False,
        input_symmetric=True,
    ):
        self.strategy = strategy
        self.num_bits = num_bits
        self.group_size = group_size
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric
        self.quant_type = torch.int8
        self.original_create_weights_calls = 0

    def create_weights(self, *args, **kwargs):
        self.original_create_weights_calls += 1


class _FakeCompressedTensorsConfig:
    def __init__(self, *args, **kwargs):
        self.quant_format = None
        self._dynamic_token = True
        self.original_get_scheme_calls = 0

    def _is_dynamic_token_w4a8_int(self, weight_quant, input_quant):
        return self._dynamic_token

    def _get_scheme_from_parts(self, *args, **kwargs):
        self.original_get_scheme_calls += 1
        return "stock-scheme"


_ARM_PATCHED_ATTRS = {
    _FakeW4A8Scheme: (
        "__init__",
        "create_weights",
        "_vllm_fl_arm_packed_w4a8",
        "_vllm_fl_arm_original_init",
        "_vllm_fl_arm_original_create_weights",
    ),
    _FakeCompressedTensorsConfig: (
        "_get_scheme_from_parts",
        "_vllm_fl_arm_original_get_scheme",
    ),
    _Dynamic4bitLinearKernel: (
        "process_weights_after_loading",
        "apply_weights",
        "_vllm_fl_arm_w4a8",
        "_vllm_fl_arm_original_process",
        "_vllm_fl_arm_original_apply",
    ),
}


@pytest.fixture(autouse=True)
def _restore_arm_fake_classes():
    """Undo the class-level monkeypatching performed by the ARM installer."""
    snapshots = {
        cls: {name: (name in cls.__dict__, cls.__dict__.get(name)) for name in names}
        for cls, names in _ARM_PATCHED_ATTRS.items()
    }
    yield
    for cls, state in snapshots.items():
        for name, (present, value) in state.items():
            if present:
                setattr(cls, name, value)
            elif name in cls.__dict__:
                delattr(cls, name)


@pytest.fixture(autouse=True)
def _single_rank_tensor_parallel(monkeypatch):
    """Keep vLLM parameter constructors on a deterministic single rank."""
    import vllm.model_executor.parameter as vllm_parameter

    monkeypatch.setattr(
        vllm_parameter,
        "get_tensor_model_parallel_rank",
        lambda: 0,
        raising=False,
    )
    monkeypatch.setattr(
        vllm_parameter,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
        raising=False,
    )


class _ArmFakes(SimpleNamespace):
    pass


def _install_fake_arm_modules(monkeypatch):
    def replace_parameter(layer, name, value):
        if name in getattr(layer, "_parameters", {}):
            layer._parameters[name] = value
        else:
            setattr(layer, name, value)

    def choose_mp_linear_kernel(config):
        return _LaunchedKernel

    modules = {}

    def reg(name):
        module = ModuleType(name)
        modules[name] = module
        return module

    reg("vllm")
    reg("vllm.model_executor")
    kernels = reg("vllm.model_executor.kernels")
    kernels_linear = reg("vllm.model_executor.kernels.linear")
    mixed = reg("vllm.model_executor.kernels.linear.mixed_precision")
    dynamic = reg(
        "vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit"
    )
    layers = reg("vllm.model_executor.layers")
    quant = reg("vllm.model_executor.layers.quantization")
    ct = reg("vllm.model_executor.layers.quantization.compressed_tensors")
    ct_mod = reg(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors"
    )
    ct_schemes = reg(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes"
    )
    ct_w4 = reg(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a8_int"
    )
    q_utils = reg("vllm.model_executor.layers.quantization.utils")
    params = reg("vllm.model_executor.parameter")

    kernels.linear = kernels_linear
    kernels_linear.mixed_precision = mixed
    mixed.dynamic_4bit = dynamic
    layers.quantization = quant
    quant.compressed_tensors = ct
    ct.compressed_tensors = ct_mod
    ct.schemes = ct_schemes
    ct_schemes.compressed_tensors_w4a8_int = ct_w4
    quant.utils = q_utils

    kernels_linear.MPLinearLayerConfig = _MPConfig
    kernels_linear.choose_mp_linear_kernel = choose_mp_linear_kernel
    dynamic.Dynamic4bitLinearKernel = _Dynamic4bitLinearKernel
    ct_mod.CompressedTensorsConfig = _FakeCompressedTensorsConfig
    ct_w4.CompressedTensorsW4A8Int = _FakeW4A8Scheme
    q_utils.replace_parameter = replace_parameter
    params.BasevLLMParameter = BasevLLMParameter
    params.ChannelQuantScaleParameter = ChannelQuantScaleParameter
    params.GroupQuantScaleParameter = GroupQuantScaleParameter
    params.PackedvLLMParameter = PackedvLLMParameter

    try:
        from compressed_tensors.compressors.pack_quantized import (
            helpers as ct_pack_helpers,
        )
    except Exception:  # pragma: no cover - older layouts keep the real helper
        ct_pack_helpers = None
    if ct_pack_helpers is not None:

        def fake_unpack_from_int32(packed, num_bits, shape):
            del packed, num_bits
            return torch.zeros(tuple(shape), dtype=torch.int8)

        monkeypatch.setattr(
            ct_pack_helpers,
            "unpack_from_int32",
            fake_unpack_from_int32,
            raising=False,
        )

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return _ArmFakes(
        scheme=_FakeW4A8Scheme,
        config=_FakeCompressedTensorsConfig,
        kernel=_Dynamic4bitLinearKernel,
    )


def _install_fake_flag_gems(monkeypatch, packed_value=None, linear_value=None):
    flag_gems = ModuleType("flag_gems")
    quantized_linear = ModuleType("flag_gems.quantized_linear")
    calls = {"pack": [], "linear": []}

    def pack_rhs_qsi4c128p(quantized_nk, scales):
        calls["pack"].append((quantized_nk, scales))
        if packed_value is not None:
            return packed_value
        rows, cols = quantized_nk.shape
        return torch.zeros((rows, cols // 8), dtype=torch.uint8)

    def w4a8_g128_linear(input_bf16, weight, n, k):
        calls["linear"].append((input_bf16, weight, n, k))
        if linear_value is not None:
            return linear_value
        return torch.zeros((input_bf16.shape[0], n), dtype=input_bf16.dtype)

    quantized_linear.pack_rhs_qsi4c128p = pack_rhs_qsi4c128p
    quantized_linear.w4a8_g128_linear = w4a8_g128_linear
    flag_gems.quantized_linear = quantized_linear
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setitem(sys.modules, "flag_gems.quantized_linear", quantized_linear)
    return calls


def _packed_scheme(fakes, *, group_size=128):
    return fakes.scheme(
        strategy="group",
        num_bits=4,
        group_size=group_size,
        is_static_input_scheme=False,
        input_symmetric=True,
        packed=True,
    )


def test_arm_installer_marks_classes_and_is_idempotent(monkeypatch):
    _install_fake_arm_modules(monkeypatch)

    assert arm_cpu_w4a8.install_arm_cpu_packed_w4a8() is True
    assert _FakeW4A8Scheme._vllm_fl_arm_packed_w4a8 is True
    assert _Dynamic4bitLinearKernel._vllm_fl_arm_w4a8 is True
    assert _FakeW4A8Scheme._vllm_fl_arm_original_init is not None

    assert arm_cpu_w4a8.install_arm_cpu_packed_w4a8() is False


def test_arm_installer_rejects_incomplete_markers(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    monkeypatch.setattr(_FakeW4A8Scheme, "_vllm_fl_arm_packed_w4a8", True)
    monkeypatch.setattr(_Dynamic4bitLinearKernel, "_vllm_fl_arm_w4a8", False)
    with pytest.raises(RuntimeError, match="incomplete ARM W4A8"):
        arm_cpu_w4a8.install_arm_cpu_packed_w4a8()


def test_arm_scheme_init_records_pack_factor(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    packed_scheme = _FakeW4A8Scheme("group", 4, 128, packed=True)
    assert packed_scheme._vllm_fl_checkpoint_packed is True
    assert packed_scheme._vllm_fl_pack_factor == 8
    assert packed_scheme.group_size == 128

    unpacked_scheme = _FakeW4A8Scheme("group", 4, 128)
    assert unpacked_scheme._vllm_fl_checkpoint_packed is False


def test_arm_create_weights_unpacked_delegates_to_stock(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    scheme = _FakeW4A8Scheme("group", 4, 128)
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_size=32,
        input_size=128,
        output_partition_sizes=[32],
        input_size_per_partition=128,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert scheme.original_create_weights_calls == 1
    assert not hasattr(layer, "weight_packed")


@pytest.mark.parametrize(
    ("group_size", "input_size", "input_partition", "out", "effective", "scale_cls"),
    [
        (-1, 128, 64, 32, 64, ChannelQuantScaleParameter),
        (-1, 128, 128, 32, 128, GroupQuantScaleParameter),
        (128, 256, 256, 32, 128, GroupQuantScaleParameter),
    ],
)
def test_arm_create_weights_packed_group_and_scale_branches(
    monkeypatch,
    group_size,
    input_size,
    input_partition,
    out,
    effective,
    scale_cls,
):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    scheme = _packed_scheme(
        _ArmFakes(scheme=_FakeW4A8Scheme),
        group_size=group_size,
    )
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_size=out,
        input_size=input_size,
        output_partition_sizes=[out],
        input_size_per_partition=input_partition,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )

    assert layer.weight_packed.shape == (out, input_partition // 8)
    assert layer.weight_packed.dtype == torch.int32
    assert layer.weight_scale.shape == (out, input_partition // effective)
    assert isinstance(layer.weight_scale, scale_cls)
    assert layer.weight_shape.shape == (2,)
    assert scheme.kernel.config.group_size == effective
    assert scheme.kernel.config.zero_points is False
    assert scheme.kernel._vllm_fl_arm_packed_checkpoint is True


def test_arm_create_weights_rejects_indivisible_partition(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    scheme = _FakeW4A8Scheme("group", 4, 128, packed=True)
    with pytest.raises(ValueError, match="not divisible"):
        scheme.create_weights(
            torch.nn.Module(),
            output_size=32,
            input_size=130,
            output_partition_sizes=[32],
            input_size_per_partition=130,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )


def test_arm_get_scheme_from_parts_selects_packed(monkeypatch):
    from compressed_tensors.config import CompressionFormat

    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    config = _FakeCompressedTensorsConfig()
    config.quant_format = CompressionFormat.pack_quantized.value
    config._dynamic_token = True
    scheme = config._get_scheme_from_parts(
        SimpleNamespace(num_bits=4, strategy="group", group_size=128),
        SimpleNamespace(symmetric=True),
        format=None,
        layer_name="model.linear",
    )
    assert isinstance(scheme, _FakeW4A8Scheme)
    assert scheme._vllm_fl_checkpoint_packed is True
    assert scheme.strategy == "group"

    explicit = config._get_scheme_from_parts(
        SimpleNamespace(num_bits=4, strategy="group", group_size=128),
        SimpleNamespace(symmetric=True),
        format=CompressionFormat.pack_quantized.value,
        layer_name="model.linear",
    )
    assert isinstance(explicit, _FakeW4A8Scheme)


def test_arm_get_scheme_from_parts_falls_back(monkeypatch):
    from compressed_tensors.config import CompressionFormat

    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    config = _FakeCompressedTensorsConfig()
    config.quant_format = CompressionFormat.pack_quantized.value
    config._dynamic_token = False
    result = config._get_scheme_from_parts(
        object(),
        object(),
        output_quant=object(),
        format=None,
        layer_name="model.linear",
    )
    assert result == "stock-scheme"
    assert config.original_get_scheme_calls == 1


def _make_packed_layer(scheme, n=32, k=256):
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_size=n,
        input_size=k,
        output_partition_sizes=[n],
        input_size_per_partition=k,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    layer.weight_packed.data.zero_()
    layer.weight_scale.data.fill_(0.5)
    layer.weight_shape.data.copy_(torch.tensor([n, k]))
    return layer


def test_arm_process_delegates_when_kernel_unmarked(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    kernel = _Dynamic4bitLinearKernel()
    _Dynamic4bitLinearKernel.process_weights_after_loading(kernel, torch.nn.Module())
    assert kernel._orig_process_calls == 1


def test_arm_process_delegates_for_unsupported_config(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    kernel = _Dynamic4bitLinearKernel()
    kernel._vllm_fl_arm_packed_checkpoint = True
    kernel.config = SimpleNamespace(group_size=64, zero_points=False)
    _Dynamic4bitLinearKernel.process_weights_after_loading(kernel, torch.nn.Module())
    assert kernel._orig_process_calls == 1


def test_arm_process_packed_checkpoint_packs_and_clears(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()
    _install_fake_flag_gems(monkeypatch)

    scheme = _packed_scheme(_ArmFakes(scheme=_FakeW4A8Scheme))
    layer = _make_packed_layer(scheme, n=32, k=256)
    kernel = scheme.kernel
    assert kernel._vllm_fl_arm_packed_checkpoint is True

    _Dynamic4bitLinearKernel.process_weights_after_loading(kernel, layer)

    assert kernel._vllm_fl_arm_w4a8_shape == (32, 256)
    assert layer.weight_scale is None
    assert layer.weight_shape is None
    assert layer.weight_packed.dtype == torch.uint8


def test_arm_process_rejects_bad_weight_shape(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()
    _install_fake_flag_gems(monkeypatch)

    scheme = _packed_scheme(_ArmFakes(scheme=_FakeW4A8Scheme))
    layer = _make_packed_layer(scheme, n=32, k=256)
    layer.weight_shape.data.copy_(torch.tensor([32, 255]))

    with pytest.raises(RuntimeError, match="unexpected packed-Q4 weight_shape"):
        _Dynamic4bitLinearKernel.process_weights_after_loading(scheme.kernel, layer)


def test_arm_process_accepts_none_weight_shape(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()
    _install_fake_flag_gems(monkeypatch)

    scheme = _packed_scheme(_ArmFakes(scheme=_FakeW4A8Scheme))
    layer = _make_packed_layer(scheme, n=32, k=256)
    layer.weight_shape = None

    _Dynamic4bitLinearKernel.process_weights_after_loading(scheme.kernel, layer)
    assert scheme.kernel._vllm_fl_arm_w4a8_shape == (32, 256)


def test_arm_process_nonpacked_int8_path(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()
    _install_fake_flag_gems(monkeypatch)

    scheme = _packed_scheme(_ArmFakes(scheme=_FakeW4A8Scheme))
    layer = _make_packed_layer(scheme, n=32, k=256)
    layer.weight_packed.data = torch.zeros((32, 256), dtype=torch.int8)

    _Dynamic4bitLinearKernel.process_weights_after_loading(scheme.kernel, layer)
    assert scheme.kernel._vllm_fl_arm_w4a8_shape == (32, 256)


def test_arm_process_rejects_wrong_scale_shape(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()
    _install_fake_flag_gems(monkeypatch)

    scheme = _packed_scheme(_ArmFakes(scheme=_FakeW4A8Scheme))
    layer = _make_packed_layer(scheme, n=32, k=256)
    layer.weight_packed.data = torch.zeros((32, 256), dtype=torch.int8)
    layer.weight_scale.data = torch.zeros((32, 1), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="unexpected compressed-tensors G128"):
        _Dynamic4bitLinearKernel.process_weights_after_loading(scheme.kernel, layer)


def test_arm_process_clears_source_alias_storage(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()
    _install_fake_flag_gems(monkeypatch)

    scheme = _packed_scheme(_ArmFakes(scheme=_FakeW4A8Scheme))
    layer = _make_packed_layer(scheme, n=32, k=256)
    layer.register_parameter(
        "weight", torch.nn.Parameter(layer.weight_packed.data, requires_grad=False)
    )

    _Dynamic4bitLinearKernel.process_weights_after_loading(scheme.kernel, layer)
    assert layer.weight.numel() == 0
    assert layer.weight.dtype == torch.int32


def test_arm_apply_delegates_without_shape(monkeypatch):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()

    kernel = _Dynamic4bitLinearKernel()
    output = _Dynamic4bitLinearKernel.apply_weights(
        kernel, torch.nn.Module(), torch.zeros((1, 8)), None
    )
    assert output == "stock-apply"
    assert kernel._orig_apply_calls == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [True, False])
def test_arm_apply_weights_casts_and_adds_bias(monkeypatch, dtype, with_bias):
    _install_fake_arm_modules(monkeypatch)
    arm_cpu_w4a8.install_arm_cpu_packed_w4a8()
    calls = _install_fake_flag_gems(monkeypatch)

    scheme = _packed_scheme(_ArmFakes(scheme=_FakeW4A8Scheme))
    layer = _make_packed_layer(scheme, n=32, k=256)
    kernel = scheme.kernel
    kernel._vllm_fl_arm_w4a8_shape = (32, 256)

    x = torch.ones((2, 256), dtype=dtype)
    bias = torch.ones((32,), dtype=dtype) if with_bias else None
    output = _Dynamic4bitLinearKernel.apply_weights(kernel, layer, x, bias)

    assert calls["linear"]
    passed = calls["linear"][0][0]
    assert passed.dtype == torch.bfloat16
    assert output.shape == (2, 32)
    assert output.dtype == dtype


# ---------------------------------------------------------------------------
# w8a8 reference helpers: CPU tensors only.
# ---------------------------------------------------------------------------


def test_unpack_uint8b128_rejects_bad_layout_and_range():
    with pytest.raises(ValueError, match="2D int32"):
        reference.unpack_uint8b128_int32(torch.zeros((4,), dtype=torch.int32))
    with pytest.raises(ValueError, match="2D int32"):
        reference.unpack_uint8b128_int32(torch.zeros((2, 4), dtype=torch.int8))

    packed = torch.zeros((2, 2), dtype=torch.int32)
    with pytest.raises(ValueError, match="incompatible"):
        reference.unpack_uint8b128_int32(packed, in_features=9)
    with pytest.raises(ValueError, match="incompatible"):
        reference.unpack_uint8b128_int32(packed, in_features=-1)


def test_unpack_uint8b128_truncates_to_in_features():
    values = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
    codes = (values.to(torch.int16) + 128).to(torch.uint8)
    packed = codes.contiguous().view(torch.int32)

    unpacked = reference.unpack_uint8b128_int32(packed, in_features=4)
    assert torch.equal(unpacked, values[:, :4])


def test_dynamic_quant_rejects_non_float_tensor():
    with pytest.raises(TypeError, match="floating point"):
        reference.dynamic_per_token_quant_int8(torch.ones((2, 4), dtype=torch.int8))


def test_w8a8_linear_reference_rejects_bad_weight_and_bias():
    with pytest.raises(ValueError, match="2D int8"):
        reference.w8a8_linear_reference(
            torch.ones(1, 4), torch.ones(4, dtype=torch.int8), torch.ones(1)
        )
    with pytest.raises(ValueError, match="incompatible input"):
        reference.w8a8_linear_reference(
            torch.ones(1, 5),
            torch.ones((2, 4), dtype=torch.int8),
            torch.ones(2),
        )
    with pytest.raises(ValueError, match="one value per output channel"):
        reference.w8a8_linear_reference(
            torch.ones(1, 4),
            torch.ones((2, 4), dtype=torch.int8),
            torch.ones(2),
            bias=torch.ones(3),
        )


def test_w8a8_linear_reference_runs_with_bias():
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
    weight = torch.tensor([[127, -128, 127, -128]], dtype=torch.int8)
    scale = torch.tensor([0.01], dtype=torch.float32)
    bias = torch.tensor([0.5], dtype=torch.float32)
    output = reference.w8a8_linear_reference(x, weight, scale, bias)
    assert output.shape == (1, 1)
    assert output.dtype == torch.float32


# ---------------------------------------------------------------------------
# int8_mode selection helpers.
# ---------------------------------------------------------------------------


def test_is_packed_int8_weight_handles_none_and_non_pack():
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm_fl.quantization.w8a8 import int8_mode

    assert int8_mode.is_packed_int8_weight(None, "pack-quantized") is False
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    assert int8_mode.is_packed_int8_weight(args, "dense") is False
    assert int8_mode.is_packed_int8_weight(args, "pack-quantized") is True
    assert int8_mode.is_dynamic_token_int8(None) is False


# ---------------------------------------------------------------------------
# packed.py scheme installation and creation guards.
# ---------------------------------------------------------------------------


def _packed_quant_args():
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    return (
        QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        ),
        QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        ),
    )


def test_packed_scheme_create_weights_rejects_indivisible_input(monkeypatch):
    scheme = object.__new__(packed.CompressedTensorsPackedW8A8Int8)
    with pytest.raises(ValueError, match="divisible by 4"):
        scheme.create_weights(
            torch.nn.Module(),
            output_partition_sizes=[4],
            input_size_per_partition=6,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )


def test_install_packed_w8a8_scheme_wraps_selector(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as ct_module,
    )

    weight_quant, input_quant = _packed_quant_args()
    records = {"current": [], "constructed": []}

    def fake_current(self, wq, iq, output_quant=None, format=None, layer_name=None):
        records["current"].append((wq, iq, output_quant, format, layer_name))
        return "current-scheme"

    monkeypatch.setattr(
        ct_module.CompressedTensorsConfig, "_get_scheme_from_parts", fake_current
    )
    monkeypatch.setattr(packed, "should_use_packed_w8a8", lambda *a, **k: True)
    monkeypatch.setattr(
        packed,
        "CompressedTensorsPackedW8A8Int8",
        lambda layer_name=None: records["constructed"].append(layer_name)
        or SimpleNamespace(layer_name=layer_name),
    )

    assert packed.install_packed_w8a8_scheme() is True
    selector = ct_module.CompressedTensorsConfig._get_scheme_from_parts
    assert getattr(selector, packed._PATCH_MARKER) is True

    result = selector(
        SimpleNamespace(quant_format="pack-quantized"),
        weight_quant,
        input_quant,
        layer_name="model.linear",
    )
    assert result.layer_name == "model.linear"
    assert records["constructed"] == ["model.linear"]

    # A second install must short-circuit on the marker.
    assert packed.install_packed_w8a8_scheme() is True


def test_packed_selector_falls_back_when_not_packed(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as ct_module,
    )

    weight_quant, input_quant = _packed_quant_args()
    records = []

    def fake_current(self, wq, iq, output_quant=None, format=None, layer_name=None):
        records.append((wq, iq, output_quant, format, layer_name))
        return "current-scheme"

    monkeypatch.setattr(
        ct_module.CompressedTensorsConfig, "_get_scheme_from_parts", fake_current
    )
    monkeypatch.setattr(packed, "should_use_packed_w8a8", lambda *a, **k: False)

    assert packed.install_packed_w8a8_scheme() is True
    selector = ct_module.CompressedTensorsConfig._get_scheme_from_parts
    result = selector(
        SimpleNamespace(quant_format="pack-quantized"),
        weight_quant,
        input_quant,
        output_quant=object(),
        format="dense",
        layer_name="model.linear",
    )
    assert result == "current-scheme"
    assert records and records[0][4] == "model.linear"


def test_packed_install_short_circuits_when_already_marked(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as ct_module,
    )

    def marked(self, *a, **k):
        return "pre-marked"

    setattr(marked, packed._PATCH_MARKER, True)
    monkeypatch.setattr(
        ct_module.CompressedTensorsConfig, "_get_scheme_from_parts", marked
    )

    assert packed.install_packed_w8a8_scheme() is True
    assert (
        ct_module.CompressedTensorsConfig._get_scheme_from_parts is marked
    )


# ---------------------------------------------------------------------------
# quant_linear.py platform resolution and registry inheritance.
# ---------------------------------------------------------------------------


def _fake_platform(*, cuda_alike=False, rocm=False, cpu=False):
    return SimpleNamespace(
        is_cuda_alike=lambda: cuda_alike,
        is_rocm=lambda: rocm,
        is_cpu=lambda: cpu,
    )


@pytest.mark.parametrize(
    ("platform", "expected_name"),
    [
        (_fake_platform(cuda_alike=True), "CUDA"),
        (_fake_platform(rocm=True), "ROCM"),
        (_fake_platform(cpu=True), "CPU"),
        (_fake_platform(), "CUDA"),
    ],
)
def test_quant_linear_resolve_source_platform(monkeypatch, platform, expected_name):
    from vllm.platforms import PlatformEnum

    monkeypatch.setattr(quant_linear, "current_platform", platform)
    assert quant_linear._resolve_source_platform() == getattr(
        PlatformEnum, expected_name
    )


def test_quant_linear_add_oot_kernel_skips_present_entries(monkeypatch):
    from vllm.model_executor.kernels import linear as linear_kernels
    from vllm.platforms import PlatformEnum

    from vllm_fl.quantization.w8a8 import linear as linear_adapter
    from vllm_fl.quantization.w8a8 import moe as moe_adapter

    sentinel = object()
    for registry in (
        linear_kernels._POSSIBLE_KERNELS,
        linear_kernels._POSSIBLE_INT8_KERNELS,
        linear_kernels._POSSIBLE_FP8_KERNELS,
        linear_kernels._POSSIBLE_FP8_BLOCK_KERNELS,
        linear_kernels._POSSIBLE_MXFP8_KERNELS,
    ):
        monkeypatch.setitem(registry, PlatformEnum.OOT, [sentinel])

    monkeypatch.setattr(
        quant_linear, "_resolve_source_platform", lambda: PlatformEnum.CUDA
    )
    monkeypatch.setattr(
        linear_adapter, "register_fl_w8a8_linear_kernel", lambda registry: True
    )
    monkeypatch.setattr(moe_adapter, "install_fl_w8a8_moe_selector", lambda: None)
    monkeypatch.setattr(packed, "install_packed_w8a8_scheme", lambda: True)

    quant_linear.add_oot_quant_kernel()

    for registry in (
        linear_kernels._POSSIBLE_KERNELS,
        linear_kernels._POSSIBLE_INT8_KERNELS,
        linear_kernels._POSSIBLE_FP8_KERNELS,
        linear_kernels._POSSIBLE_FP8_BLOCK_KERNELS,
        linear_kernels._POSSIBLE_MXFP8_KERNELS,
    ):
        assert registry[PlatformEnum.OOT] == [sentinel]


# ---------------------------------------------------------------------------
# moe_experts.py helper functions and validation branches.
# ---------------------------------------------------------------------------


def _moe_quant_config(**overrides):
    config = SimpleNamespace(
        use_int8_w8a8=True,
        per_act_token_quant=True,
        block_shape=None,
        w1_scale=torch.ones((2, 8, 1), dtype=torch.float32),
        w2_scale=torch.ones((2, 4, 1), dtype=torch.float32),
        w1_bias=None,
        w2_bias=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _moe_instance(quant_config=None, *, clamp_limit=None, lora=None):
    return SimpleNamespace(
        quant_config=_moe_quant_config() if quant_config is None else quant_config,
        activation_config=SimpleNamespace(clamp_limit=clamp_limit),
        _lora_context=lora,
    )


def _moe_apply_arguments():
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    return {
        "output": torch.empty_like(hidden),
        "hidden_states": hidden,
        "w1": torch.ones((2, 8, 4), dtype=torch.int8),
        "w2": torch.ones((2, 4, 4), dtype=torch.int8),
        "topk_weights": torch.ones((2, 1), dtype=torch.float32),
        "topk_ids": torch.zeros((2, 1), dtype=torch.int64),
        "activation": MoEActivation.SILU,
        "global_num_experts": 2,
        "expert_map": None,
        "a1q_scale": None,
        "a2_scale": None,
        "workspace13": torch.empty(0),
        "workspace2": torch.empty(0),
        "expert_tokens_meta": None,
        "apply_router_weight_on_input": False,
    }


def test_is_ascend_npu_tensor_detects_device_type():
    assert moe_experts._is_ascend_npu_tensor(torch.ones(1)) is False
    fake = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert moe_experts._is_ascend_npu_tensor(fake) is True


def test_dynamic_int8_qdq_zero_rows_are_finite():
    value = torch.zeros((2, 3), dtype=torch.float32)
    assert torch.count_nonzero(moe_experts._dynamic_int8_qdq(value)) == 0
    value[0, 0] = 3.0
    quantized = moe_experts._dynamic_int8_qdq(value)
    assert quantized.dtype == torch.float32
    assert torch.isfinite(quantized).all()


@pytest.mark.parametrize("clamp_limit", [None, 0.0, 10.0])
def test_w8a8_swiglu_clamp_branches(clamp_limit):
    gate_up = torch.randn((2, 6), dtype=torch.bfloat16)
    output = moe_experts._w8a8_swiglu(gate_up, clamp_limit)
    assert output.shape == (2, 3)
    assert output.dtype == torch.bfloat16


def test_native_w8a8_fused_experts_covers_bias_and_skip_routes():
    hidden = torch.ones((2, 4), dtype=torch.float32)
    w1 = torch.ones((2, 6, 4), dtype=torch.int8)
    w2 = torch.ones((2, 4, 3), dtype=torch.int8)
    w1_scale = torch.full((2, 6, 1), 0.01, dtype=torch.float32)
    w2_scale = torch.full((2, 4, 1), 0.02, dtype=torch.float32)
    w1_bias = torch.zeros((2, 6), dtype=torch.float32)
    w2_bias = torch.zeros((2, 4), dtype=torch.float32)
    topk_weights = torch.ones((2, 2), dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int64)

    output = moe_experts._native_w8a8_fused_experts(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        None,
        True,
        clamp_limit=None,
    )
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


def test_native_w8a8_fused_experts_applies_router_and_expert_map():
    hidden = torch.ones((2, 4), dtype=torch.float32)
    w1 = torch.ones((2, 6, 4), dtype=torch.int8)
    w2 = torch.ones((2, 4, 3), dtype=torch.int8)
    w1_scale = torch.full((2, 6, 1), 0.01, dtype=torch.float32)
    w2_scale = torch.full((2, 4, 1), 0.02, dtype=torch.float32)
    topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    expert_map = torch.tensor([0, 1], dtype=torch.int64)

    output = moe_experts._native_w8a8_fused_experts(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        None,
        None,
        expert_map,
        False,
        clamp_limit=None,
    )
    assert output.shape == hidden.shape


def _install_torch_npu_stub(monkeypatch):
    torch_npu = ModuleType("torch_npu")

    def npu_dynamic_quant(value):
        scale = value.float().abs().amax(dim=-1, keepdim=True) / 127.0
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        quantized = (
            torch.round(value.float() / safe).clamp(-127, 127).to(torch.int8)
        )
        return quantized, scale

    def npu_grouped_matmul(
        *,
        x,
        weight,
        bias,
        scale,
        per_token_scale,
        group_list,
        split_item,
        group_type,
        group_list_type,
        output_dtype,
    ):
        codes = x[0]
        cols = weight[0].shape[-1]
        return [torch.zeros((codes.shape[0], cols), dtype=output_dtype)]

    def npu_swiglu(value, dim=-1):
        gate, up = value.chunk(2, dim=dim)
        return torch.nn.functional.silu(gate) * up

    torch_npu.npu_dynamic_quant = npu_dynamic_quant
    torch_npu.npu_grouped_matmul = npu_grouped_matmul
    torch_npu.npu_swiglu = npu_swiglu
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)


def test_ascend_grouped_experts_rejects_non_float_dtype(monkeypatch):
    _install_torch_npu_stub(monkeypatch)
    with pytest.raises(TypeError, match="bfloat16 or float16"):
        moe_experts._ascend_w8a8_grouped_experts(
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones((2, 6, 4), dtype=torch.int8),
            torch.ones((2, 4, 3), dtype=torch.int8),
            torch.ones((2, 2)),
            torch.zeros((2, 2), dtype=torch.int64),
            torch.ones((2, 6, 1)),
            torch.ones((2, 4, 1)),
            None,
            None,
            None,
            False,
        )


def test_ascend_grouped_experts_float16_and_swiglu_path(monkeypatch):
    _install_torch_npu_stub(monkeypatch)
    hidden = torch.ones((2, 4), dtype=torch.float16)
    output = moe_experts._ascend_w8a8_grouped_experts(
        hidden,
        torch.ones((2, 6, 4), dtype=torch.int8),
        torch.ones((2, 4, 3), dtype=torch.int8),
        torch.ones((2, 2), dtype=torch.float32),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        torch.ones((2, 6, 1), dtype=torch.float32),
        torch.ones((2, 4, 1), dtype=torch.float32),
        None,
        None,
        None,
        False,
        clamp_limit=None,
    )
    assert output.shape == (2, 4)
    assert output.dtype == torch.float16


def test_ascend_grouped_experts_empty_tokens(monkeypatch):
    _install_torch_npu_stub(monkeypatch)
    output = moe_experts._ascend_w8a8_grouped_experts(
        torch.ones((0, 4), dtype=torch.bfloat16),
        torch.ones((2, 6, 4), dtype=torch.int8),
        torch.ones((2, 4, 3), dtype=torch.int8),
        torch.ones((0, 2), dtype=torch.float32),
        torch.zeros((0, 2), dtype=torch.int64),
        torch.ones((2, 6, 1)),
        torch.ones((2, 4, 1)),
        None,
        None,
        None,
        False,
    )
    assert output.shape == (0, 4)


def test_ascend_grouped_experts_all_routes_nonlocal(monkeypatch):
    _install_torch_npu_stub(monkeypatch)
    output = moe_experts._ascend_w8a8_grouped_experts(
        torch.ones((2, 4), dtype=torch.bfloat16),
        torch.ones((2, 6, 4), dtype=torch.int8),
        torch.ones((2, 4, 3), dtype=torch.int8),
        torch.ones((2, 2), dtype=torch.float32),
        torch.full((2, 2), -1, dtype=torch.int64),
        torch.ones((2, 6, 1)),
        torch.ones((2, 4, 1)),
        None,
        None,
        None,
        False,
    )
    assert output.shape == (2, 4)
    assert torch.count_nonzero(output) == 0


def _contract_baseline():
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    w1 = torch.ones((2, 8, 4), dtype=torch.int8)
    w2 = torch.ones((2, 4, 4), dtype=torch.int8)
    w1_scale = torch.ones((2, 8, 1), dtype=torch.float32)
    w2_scale = torch.ones((2, 4, 1), dtype=torch.float32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.zeros((2, 1), dtype=torch.int64)
    return (
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        w1_scale,
        w2_scale,
        None,
        None,
    )


def test_validate_contract_accepts_baseline():
    moe_experts._validate_w8a8_contract(*_contract_baseline())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda args: args.__setitem__(0, torch.ones(2, 8)[:, ::2]),
            "contiguous",
        ),
        (
            lambda args: args.__setitem__(1, torch.ones((8, 4), dtype=torch.int8)),
            "3D expert weights",
        ),
        (
            lambda args: args.__setitem__(1, torch.ones((2, 8, 4))),
            "must be int8",
        ),
        (
            lambda args: args.__setitem__(2, torch.ones((1, 4, 4), dtype=torch.int8)),
            "shapes are incompatible",
        ),
        (
            lambda args: args.__setitem__(1, torch.ones((2, 6, 4), dtype=torch.int8)),
            "down-projection shapes are incompatible",
        ),
    ],
)
def test_validate_contract_rejects_weight_problems(mutation, match):
    args = list(_contract_baseline())
    mutation(args)
    with pytest.raises((ValueError, TypeError), match=match):
        moe_experts._validate_w8a8_contract(*args)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda args: args.__setitem__(3, torch.ones((2, 2))),
            "matching",
        ),
        (
            lambda args: args.__setitem__(4, torch.zeros((4,), dtype=torch.int64)),
            "matching",
        ),
        (
            lambda args: (
                args.__setitem__(3, torch.zeros((3, 1), dtype=torch.float32)),
                args.__setitem__(4, torch.zeros((3, 1), dtype=torch.int64)),
            ),
            "rows must match",
        ),
        (
            lambda args: args.__setitem__(
                4, torch.zeros((2, 2), dtype=torch.int64)[:, ::2]
            ),
            "contiguous in top_k",
        ),
    ],
)
def test_validate_contract_rejects_routing_problems(mutation, match):
    args = list(_contract_baseline())
    mutation(args)
    with pytest.raises(ValueError, match=match):
        moe_experts._validate_w8a8_contract(*args)


@pytest.mark.parametrize(
    ("mutation", "match", "error"),
    [
        (lambda args: args.__setitem__(6, None), "w1_scale must have shape", ValueError),
        (
            lambda args: args.__setitem__(6, torch.ones((2, 8), dtype=torch.float32)),
            "w1_scale must have shape",
            ValueError,
        ),
        (
            lambda args: args.__setitem__(7, torch.ones((2, 4, 1), dtype=torch.bfloat16)),
            "must be float32",
            TypeError,
        ),
        (
            lambda args: args.__setitem__(
                7, torch.ones((2, 4, 2), dtype=torch.float32)[:, :, ::2]
            ),
            "must be contiguous",
            ValueError,
        ),
        (
            lambda args: args.__setitem__(8, torch.ones((2, 7), dtype=torch.float32)),
            "w1_bias must have shape",
            ValueError,
        ),
        (
            lambda args: args.__setitem__(9, torch.ones((2, 5), dtype=torch.float32)),
            "w2_bias must have shape",
            ValueError,
        ),
    ],
)
def test_validate_contract_rejects_scale_and_bias_problems(mutation, match, error):
    args = list(_contract_baseline())
    mutation(args)
    with pytest.raises(error, match=match):
        moe_experts._validate_w8a8_contract(*args)


def test_validate_contract_rejects_unsupported_activation():
    args = list(_contract_baseline())
    args[5] = MoEActivation.GELU
    with pytest.raises(NotImplementedError, match="only silu"):
        moe_experts._validate_w8a8_contract(*args)


@pytest.mark.parametrize(
    "experts_cls",
    [moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts],
)
def test_apply_guards_reject_bad_quant_config(experts_cls):
    bad = _moe_quant_config(use_int8_w8a8=False)
    with pytest.raises(ValueError, match="INT8 W8A8 quant config"):
        experts_cls.apply(_moe_instance(bad), **_moe_apply_arguments())


@pytest.mark.parametrize(
    "experts_cls",
    [moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts],
)
def test_apply_guards_reject_static_or_blockwise(experts_cls):
    bad = _moe_quant_config(per_act_token_quant=False)
    with pytest.raises(ValueError, match="dynamic per-token/channel-wise INT8"):
        experts_cls.apply(_moe_instance(bad), **_moe_apply_arguments())


@pytest.mark.parametrize(
    "experts_cls",
    [moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts],
)
def test_apply_guards_reject_lora(experts_cls):
    with pytest.raises(NotImplementedError, match="LoRA"):
        experts_cls.apply(
            _moe_instance(lora=object()), **_moe_apply_arguments()
        )


@pytest.mark.parametrize(
    "experts_cls",
    [moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts],
)
def test_apply_guards_reject_prequantized_activation(experts_cls):
    arguments = _moe_apply_arguments()
    arguments["a2_scale"] = torch.ones((2, 1))
    with pytest.raises(ValueError, match="quantized before"):
        experts_cls.apply(_moe_instance(), **arguments)


@pytest.mark.parametrize(
    "experts_cls",
    [moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts],
)
def test_apply_guards_reject_integer_hidden_states(experts_cls):
    arguments = _moe_apply_arguments()
    arguments["hidden_states"] = torch.ones((2, 4), dtype=torch.int8)
    with pytest.raises(TypeError, match="floating-point hidden states"):
        experts_cls.apply(_moe_instance(), **arguments)


# ---------------------------------------------------------------------------
# modelslim_w8a8.py config/parsing/MoE branches.
# ---------------------------------------------------------------------------


def _modelslim_config(description):
    return modelslim_w8a8.ModelSlimW8A8Config(description)


def test_modelslim_class_metadata_and_from_config():
    config = modelslim_w8a8.ModelSlimW8A8Config.from_config({"a.weight": "FLOAT"})
    assert isinstance(config, modelslim_w8a8.ModelSlimW8A8Config)
    assert modelslim_w8a8.ModelSlimW8A8Config.get_name() == "fl_modelslim_w8a8"
    assert torch.bfloat16 in modelslim_w8a8.ModelSlimW8A8Config.get_supported_act_dtypes()
    assert modelslim_w8a8.ModelSlimW8A8Config.get_min_capability() == 0
    assert modelslim_w8a8.ModelSlimW8A8Config.get_config_filenames() == [
        "quant_model_description.json"
    ]


def test_modelslim_root_prefixes_handles_both_roots():
    config = _modelslim_config({})
    assert modelslim_w8a8.ModelSlimW8A8Config._root_prefixes("model.a") == [
        "model.a",
        "a",
    ]
    assert modelslim_w8a8.ModelSlimW8A8Config._root_prefixes("a") == [
        "a",
        "model.a",
    ]
    assert config._is_static_w8a8("model.a") is False
    assert config._is_dynamic_w8a8("model.a") is False


def test_modelslim_quant_kind_rejects_unsupported_scheme():
    config = _modelslim_config({"a.weight": "W4A8"})
    with pytest.raises(NotImplementedError, match="not supported"):
        config._quant_kind("a")


def test_modelslim_fused_qkv_grouping_and_incomplete_entries():
    config = _modelslim_config(
        {
            "layers.0.attn.q_a_proj.weight": "W8A8",
            "layers.0.attn.kv_a_proj_with_mqa.weight": "W8A8",
        }
    )
    assert config._quant_kind("model.layers.0.attn.fused_qkv_a_proj") == "W8A8"
    del config.description["layers.0.attn.kv_a_proj_with_mqa.weight"]
    with pytest.raises(ValueError, match="missing quantization entries"):
        config._quant_kind("model.layers.0.attn.fused_qkv_a_proj")


def test_modelslim_down_proj_and_gate_up_grouping():
    config = _modelslim_config(
        {
            "layers.0.mlp.down_proj.weight": "FLOAT",
            "layers.0.mlp.w2.weight": "FLOAT",
            "layers.0.mlp.gate_up_proj.weight": "FLOAT",
            "layers.0.mlp.w1.weight": "FLOAT",
            "layers.0.mlp.w3.weight": "FLOAT",
        }
    )
    assert config._quant_kind("model.layers.0.mlp.down_proj") == "FLOAT"
    assert config._quant_kind("model.layers.0.mlp.gate_up_proj") == "FLOAT"


def test_modelslim_get_quant_method_linear_branches(monkeypatch):
    class FakeLinear(torch.nn.Module):
        pass

    class FakeMethod:
        def __init__(self, *a, **k):
            self.args = a

    monkeypatch.setattr(modelslim_w8a8, "LinearBase", FakeLinear)
    monkeypatch.setattr(modelslim_w8a8, "RoutedExperts", type("R", (), {}))
    monkeypatch.setattr(modelslim_w8a8, "UnquantizedLinearMethod", FakeMethod)
    monkeypatch.setattr(modelslim_w8a8, "CompressedTensorsLinearMethod", FakeMethod)
    monkeypatch.setattr(
        modelslim_w8a8, "_ModelSlimW8A8StaticLinearScheme", lambda: "static-scheme"
    )
    monkeypatch.setattr(
        modelslim_w8a8,
        "_ModelSlimW8A8LinearScheme",
        lambda **k: SimpleNamespace(kind="dynamic", kwargs=k),
    )

    unquant = _modelslim_config({"a.weight": "FLOAT"})
    layer = FakeLinear()
    assert isinstance(unquant.get_quant_method(layer, "a"), FakeMethod)

    missing = _modelslim_config({})
    assert isinstance(missing.get_quant_method(FakeLinear(), "b"), FakeMethod)

    static = _modelslim_config({"a.weight": "W8A8"})
    layer = FakeLinear()
    method = static.get_quant_method(layer, "a")
    assert isinstance(method, FakeMethod)
    assert layer.scheme == "static-scheme"

    dynamic = _modelslim_config({"a.weight": "W8A8_DYNAMIC"})
    layer = FakeLinear()
    dynamic.get_quant_method(layer, "a")
    assert layer.scheme.kind == "dynamic"


def test_modelslim_get_quant_method_rejects_non_layer():
    config = _modelslim_config({})
    assert config.get_quant_method(object(), "a") is None


def test_modelslim_get_quant_method_routed_experts(monkeypatch):
    class FakeExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = SimpleNamespace(name="moe")

    class FakeMethod:
        def __init__(self, *a, **k):
            self.args = a
            self.kwargs = k

    monkeypatch.setattr(modelslim_w8a8, "LinearBase", type("L", (), {}))
    monkeypatch.setattr(modelslim_w8a8, "RoutedExperts", FakeExperts)
    monkeypatch.setattr(modelslim_w8a8, "UnquantizedFusedMoEMethod", FakeMethod)
    monkeypatch.setattr(modelslim_w8a8, "_ModelSlimW8A8MoEMethod", FakeMethod)

    float_config = _modelslim_config({"a.weight": "FLOAT"})
    assert isinstance(
        float_config.get_quant_method(FakeExperts(), "a"), FakeMethod
    )

    static_config = _modelslim_config({"a.weight": "W8A8"})
    with pytest.raises(NotImplementedError, match="static W8A8 routed experts"):
        static_config.get_quant_method(FakeExperts(), "a")

    dynamic_config = _modelslim_config({"a.weight": "W8A8_DYNAMIC"})
    method = dynamic_config.get_quant_method(FakeExperts(), "a")
    assert isinstance(method, FakeMethod)
    assert method.kwargs.get("layer_name") == "a"


def test_modelslim_dynamic_linear_scheme_registers_offset(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Int8,
    )

    calls = []

    def fake_create_weights(self, layer, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(CompressedTensorsW8A8Int8, "create_weights", fake_create_weights)

    scheme = object.__new__(modelslim_w8a8._ModelSlimW8A8LinearScheme)
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=[4, 6],
        input_size_per_partition=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert calls
    assert layer.weight_offset.shape == (10, 1)
    assert layer.weight_offset.dtype == torch.float32


def test_modelslim_static_process_rejects_scale_count_mismatch():
    scheme = modelslim_w8a8._ModelSlimW8A8StaticLinearScheme()
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=[2, 3],
        input_size_per_partition=4,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    layer.input_scale.data = torch.zeros(1, dtype=torch.float32)
    layer.input_offset.data = torch.zeros(1, dtype=torch.int8)
    with pytest.raises(ValueError, match="one input scale per logical"):
        scheme.process_weights_after_loading(layer)


def test_modelslim_static_apply_rejects_prequantized_multi_partition():
    scheme = modelslim_w8a8._ModelSlimW8A8StaticLinearScheme()
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=[2, 3],
        input_size_per_partition=4,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    with pytest.raises(ValueError, match="single logical matrix"):
        scheme.apply_weights(
            layer, torch.ones((2, 4), dtype=torch.int8), bias=None
        )


class _FakeRowParallel(torch.nn.Module):
    pass


def test_modelslim_static_apply_row_parallel_rank_lookup(monkeypatch):
    from types import ModuleType as _ModuleType

    torch_npu = _ModuleType("torch_npu")
    quantized_calls = []
    matmul_calls = []

    def fake_quantize(x, scale, offset, dtype, axis, sqrt_mode):
        quantized_calls.append((x.shape, dtype, axis, sqrt_mode))
        return torch.zeros(x.shape, dtype=torch.int8)

    def fake_matmul(x, weight, deq_scale, *, bias, output_dtype):
        matmul_calls.append((x, weight, deq_scale, bias, output_dtype))
        return torch.zeros((x.shape[0], weight.shape[1]), dtype=output_dtype)

    torch_npu.npu_quantize = fake_quantize
    torch_npu.npu_quant_matmul = fake_matmul
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setattr(modelslim_w8a8, "RowParallelLinear", _FakeRowParallel)
    monkeypatch.setattr(
        modelslim_w8a8, "get_tensor_model_parallel_rank", lambda: 1
    )

    layer = _FakeRowParallel()
    layer.tp_rank = None
    layer.logical_widths = [3]
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.zeros((4, 3), dtype=torch.int8)
    layer.deq_scale = torch.ones(3, dtype=torch.float32)
    layer.quant_bias = torch.zeros(3, dtype=torch.int32)
    layer.aclnn_input_scale_reciprocal = torch.ones((1, 4), dtype=torch.bfloat16)
    layer.aclnn_input_offset = torch.zeros((1, 4), dtype=torch.bfloat16)

    scheme = modelslim_w8a8._ModelSlimW8A8StaticLinearScheme()
    output = scheme.apply_weights(
        layer, torch.ones((2, 4), dtype=torch.bfloat16), bias=torch.ones(3)
    )

    assert output.shape == (2, 3)
    assert len(quantized_calls) == 1
    assert matmul_calls[0][3] is None  # tp_rank != 0 -> no int32 bias


def test_modelslim_moe_method_registers_offsets(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (
        CompressedTensorsW8A8Int8MoEMethod,
    )

    monkeypatch.setattr(
        CompressedTensorsW8A8Int8MoEMethod,
        "create_weights",
        lambda self, *a, **k: None,
    )
    method = object.__new__(modelslim_w8a8._ModelSlimW8A8MoEMethod)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert layer.w13_weight_offset.shape == (2, 6, 1)
    assert layer.w2_weight_offset.shape == (2, 4, 1)
    assert hasattr(layer.w13_weight_offset, "quant_method")


# ---------------------------------------------------------------------------
# int8_mode.py full selector policy.
# ---------------------------------------------------------------------------


def test_int8_mode_selector_policy_paths():
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm_fl.quantization.w8a8 import int8_mode

    def _args(strategy, *, dynamic=False, group_size=None):
        return QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=strategy,
            symmetric=True,
            dynamic=dynamic,
            group_size=group_size,
        )

    channel = _args(QuantizationStrategy.CHANNEL)
    group = _args(QuantizationStrategy.GROUP, group_size=128)
    dynamic_token = _args(QuantizationStrategy.TOKEN, dynamic=True)
    static_token = _args(QuantizationStrategy.TOKEN, dynamic=False)

    assert int8_mode.is_dynamic_token_int8(dynamic_token) is True
    assert int8_mode.is_dynamic_token_int8(static_token) is False
    assert int8_mode.is_packed_int8_weight(group, "pack-quantized") is True
    assert int8_mode.should_use_packed_w8a8(
        channel, dynamic_token, "pack-quantized"
    ) is True
    assert int8_mode.should_use_packed_w8a8(
        channel, dynamic_token, "dense"
    ) is False
    assert int8_mode.should_use_packed_w8a8(
        channel, None, "pack-quantized"
    ) is False
    assert int8_mode.should_use_packed_w8a8(
        channel, static_token, "pack-quantized"
    ) is False
    with pytest.raises(ValueError, match="per-channel"):
        int8_mode.should_use_packed_w8a8(group, dynamic_token, "pack-quantized")


# ---------------------------------------------------------------------------
# linear.py FlagGems scaled-mm adapter.
# ---------------------------------------------------------------------------


def test_w8a8_linear_supported_and_config_guards(monkeypatch):
    from vllm.model_executor.kernels.linear import Int8ScaledMMLinearLayerConfig

    from vllm_fl.quantization.w8a8 import linear as linear_adapter

    kernel_cls = linear_adapter.FLW8A8DynamicLinearKernel
    monkeypatch.setattr(linear_adapter, "is_oot_enabled", lambda: False)
    assert kernel_cls.is_supported() == (False, "FL W8A8 kernels are disabled")

    monkeypatch.setattr(linear_adapter, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(linear_adapter, "use_flaggems_op", lambda op: False)
    supported, reason = kernel_cls.is_supported()
    assert supported is False
    assert "disabled by policy" in reason

    monkeypatch.setattr(linear_adapter, "use_flaggems_op", lambda op: True)
    assert kernel_cls.is_supported() == (True, None)

    good = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    assert kernel_cls.can_implement(good) == (True, None)
    for kwargs, needle in (
        (dict(is_channelwise=False), "per-channel"),
        (dict(is_static_input_scheme=True), "dynamic"),
        (dict(input_symmetric=False), "symmetric"),
    ):
        base = dict(
            is_channelwise=True, is_static_input_scheme=False, input_symmetric=True
        )
        base.update(kwargs)
        ok, why = kernel_cls.can_implement(
            Int8ScaledMMLinearLayerConfig(**base)
        )
        assert ok is False
        assert needle in why


def test_w8a8_linear_registry_prepends_once(monkeypatch):
    from vllm.platforms import PlatformEnum

    from vllm_fl.quantization.w8a8 import linear as linear_adapter

    native = object()
    registry = {PlatformEnum.OOT: [native]}
    assert linear_adapter.register_fl_w8a8_linear_kernel(registry) is True
    assert linear_adapter.register_fl_w8a8_linear_kernel(registry) is True
    assert registry[PlatformEnum.OOT] == [
        linear_adapter.FLW8A8DynamicLinearKernel,
        native,
    ]


def test_w8a8_linear_process_and_apply(monkeypatch):
    from vllm_fl.quantization.w8a8 import linear as linear_adapter

    weight = torch.randint(-8, 8, (3, 4), dtype=torch.int8)
    scale = torch.rand(3, 1, dtype=torch.float32) * 0.1
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(weight.clone(), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(scale.clone(), requires_grad=False)
    )
    layer.register_parameter("input_scale", None)
    layer.register_parameter("input_zero_point", None)
    layer.register_parameter("azp_adj", None)

    kernel = object.__new__(linear_adapter.FLW8A8DynamicLinearKernel)
    kernel.layer_param_names = [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]
    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 3)
    assert layer.weight.stride() == (3, 1)
    assert layer.weight_scale.shape == (3,)
    assert layer.input_scale is None
    assert layer.input_zero_point is None
    assert layer.azp_adj is None

    calls = {}

    def fake_quant(value):
        return (
            torch.zeros(value.shape, dtype=torch.int8),
            torch.ones((value.shape[0], 1), dtype=torch.float32),
        )

    monkeypatch.setattr(linear_adapter, "_dynamic_per_token_quant_int8", fake_quant)
    flag_gems = ModuleType("flag_gems")

    def fake_scaled_mm(x_q, w, x_scale, w_scale, *, bias, out_dtype):
        calls["shape"] = tuple(x_q.shape)
        calls["out_dtype"] = out_dtype
        return torch.zeros((x_q.shape[0], w.shape[1]), dtype=out_dtype)

    flag_gems.scaled_mm = fake_scaled_mm
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    output = kernel.apply_weights(layer, x, bias=None)
    assert output.shape == (2, 3)
    assert output.dtype == torch.bfloat16
    assert calls["shape"] == (2, 4)


# ---------------------------------------------------------------------------
# moe.py W8A8 MoE backend selector installation.
# ---------------------------------------------------------------------------


def _install_moe_with_fake_modules(monkeypatch, upstream_selector):
    from vllm_fl.quantization.w8a8 import moe as moe_adapter

    oracle = SimpleNamespace(
        Int8MoeBackend=SimpleNamespace(TRITON="triton"),
        select_int8_moe_backend=upstream_selector,
    )
    scheme = SimpleNamespace(
        select_int8_moe_backend=upstream_selector,
        make_int8_moe_quant_config=lambda **kwargs: kwargs,
    )
    modules = {
        moe_adapter._ORACLE_MODULE: oracle,
        moe_adapter._SCHEME_MODULE: scheme,
    }
    monkeypatch.setattr(moe_adapter, "import_module", lambda name: modules[name])
    return moe_adapter, oracle, scheme


def _patch_platform(monkeypatch, *, cuda, forest):
    import vllm.platforms as platforms

    monkeypatch.setattr(
        type(platforms.current_platform), "is_cuda", lambda self: cuda
    )
    monkeypatch.setattr(
        type(platforms.current_platform),
        "is_out_of_tree",
        lambda self: forest,
    )


def _patch_fl_policy(monkeypatch, *, enabled, allow):
    import vllm_fl.utils as fl_utils

    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: enabled)
    monkeypatch.setattr(fl_utils, "use_flaggems_op", lambda op: allow)


def _moe_config():
    return SimpleNamespace(
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )


def test_moe_installer_preserves_upstream_builder(monkeypatch):
    moe_adapter, oracle, scheme = _install_moe_with_fake_modules(
        monkeypatch, lambda *a, **k: "upstream"
    )
    assert moe_adapter.install_fl_w8a8_moe_selector() is True
    assert oracle.select_int8_moe_backend is scheme.select_int8_moe_backend
    assert callable(scheme.make_int8_moe_quant_config)


def test_moe_selector_flaggems_path(monkeypatch):
    calls = []

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return "upstream"

    moe_adapter, oracle, _ = _install_moe_with_fake_modules(monkeypatch, upstream)
    _patch_platform(monkeypatch, cuda=False, forest=True)
    _patch_fl_policy(monkeypatch, enabled=True, allow=True)
    moe_adapter.install_fl_w8a8_moe_selector()

    backend, experts_cls = oracle.select_int8_moe_backend(
        _moe_config(), weight_key=None, activation_key=None
    )
    from vllm_fl.quantization.w8a8.moe_experts import FlagGemsW8A8Experts

    assert backend == "triton"
    assert experts_cls is FlagGemsW8A8Experts
    assert calls == []


def test_moe_selector_nvidia_native_fallback(monkeypatch):
    moe_adapter, oracle, _ = _install_moe_with_fake_modules(
        monkeypatch, lambda *a, **k: "upstream"
    )
    _patch_platform(monkeypatch, cuda=True, forest=False)
    _patch_fl_policy(monkeypatch, enabled=True, allow=False)
    moe_adapter.install_fl_w8a8_moe_selector()

    backend, experts_cls = oracle.select_int8_moe_backend(
        _moe_config(), weight_key=None, activation_key=None
    )
    from vllm_fl.quantization.w8a8.moe_experts import VllmFunctionalW8A8Experts

    assert backend == "triton"
    assert experts_cls is VllmFunctionalW8A8Experts


def test_moe_selector_noncanonical_uses_upstream(monkeypatch):
    moe_adapter, oracle, _ = _install_moe_with_fake_modules(
        monkeypatch, lambda *a, **k: "upstream"
    )
    _patch_platform(monkeypatch, cuda=True, forest=True)
    _patch_fl_policy(monkeypatch, enabled=True, allow=True)
    moe_adapter.install_fl_w8a8_moe_selector()

    result = oracle.select_int8_moe_backend(
        _moe_config(), weight_key=object(), activation_key=None
    )
    assert result == "upstream"
