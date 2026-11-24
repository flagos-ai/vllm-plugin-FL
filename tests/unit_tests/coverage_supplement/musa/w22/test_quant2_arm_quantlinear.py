# Coverage tests: remaining branches in
# vllm_fl/quantization/arm_cpu_w4a8.py and vllm_fl/quantization/quant_linear.py.
# Uses fake vLLM collaborators so the patch body runs on a plain CPU host.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.platforms import PlatformEnum

from vllm_fl.quantization import arm_cpu_w4a8
from vllm_fl.quantization import quant_linear
from vllm_fl.quantization.w8a8 import linear as w8a8_linear
from vllm_fl.quantization.w8a8 import moe as w8a8_moe
from vllm_fl.quantization.w8a8 import packed as w8a8_packed


PACK_FORMAT = "pack-quantized"


# ---------------------------------------------------------------------------
# quant_linear.py
# ---------------------------------------------------------------------------


def test_add_oot_quant_kernel_fills_only_missing_entries(monkeypatch):
    from vllm.model_executor.kernels import linear as linear_kernels

    monkeypatch.setattr(
        quant_linear, "_resolve_source_platform", lambda: PlatformEnum.CUDA
    )
    names = [
        "_POSSIBLE_KERNELS",
        "_POSSIBLE_INT8_KERNELS",
        "_POSSIBLE_FP8_KERNELS",
        "_POSSIBLE_FP8_BLOCK_KERNELS",
        "_POSSIBLE_MXFP8_KERNELS",
    ]
    for index, name in enumerate(names):
        table = getattr(linear_kernels, name)
        monkeypatch.setitem(table, PlatformEnum.CUDA, ["candidate"])
        if index < 2:
            monkeypatch.setitem(table, PlatformEnum.OOT, ["kept"])
        else:
            monkeypatch.delitem(table, PlatformEnum.OOT, raising=False)

    monkeypatch.setattr(
        w8a8_linear, "register_fl_w8a8_linear_kernel", lambda registry: None
    )
    monkeypatch.setattr(w8a8_packed, "install_packed_w8a8_scheme", lambda: None)
    monkeypatch.setattr(w8a8_moe, "install_fl_w8a8_moe_selector", lambda: None)

    quant_linear.add_oot_quant_kernel()

    for index, name in enumerate(names):
        table = getattr(linear_kernels, name)
        expected = ["kept"] if index < 2 else ["candidate"]
        assert table[PlatformEnum.OOT] == expected


# ---------------------------------------------------------------------------
# arm_cpu_w4a8.py
# ---------------------------------------------------------------------------


def _parameter(data=None, **kwargs):
    del kwargs
    return torch.nn.Parameter(data, requires_grad=False)


def _module(name, **attrs):
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


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
            self.quant_format = PACK_FORMAT
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


def _wire(monkeypatch):
    FakeKernel, FakeScheme, FakeConfig = _build_classes()
    calls = {"pack": [], "linear": []}

    def unpack_from_int32(quantized, bits, shape):
        return torch.zeros(tuple(shape), dtype=torch.int8)

    def pack_rhs_qsi4c128p(weight, scales):
        calls["pack"].append((weight.clone(), scales.clone()))
        return torch.zeros(
            (weight.shape[0], weight.shape[1] // 2), dtype=torch.uint8
        )

    def w4a8_g128_linear(inp, weight, n, k):
        calls["linear"].append((inp, weight, n, k))
        return torch.zeros((inp.shape[0], n), dtype=torch.bfloat16)

    modules = {
        "compressed_tensors": _module("compressed_tensors"),
        "compressed_tensors.compressors": _module("compressed_tensors.compressors"),
        "compressed_tensors.compressors.pack_quantized": _module(
            "compressed_tensors.compressors.pack_quantized"
        ),
        "compressed_tensors.compressors.pack_quantized.helpers": _module(
            "compressed_tensors.compressors.pack_quantized.helpers",
            unpack_from_int32=unpack_from_int32,
        ),
        "compressed_tensors.config": _module(
            "compressed_tensors.config",
            CompressionFormat=SimpleNamespace(
                pack_quantized=SimpleNamespace(value=PACK_FORMAT)
            ),
        ),
        "vllm.model_executor.kernels.linear": _module(
            "vllm.model_executor.kernels.linear",
            MPLinearLayerConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            choose_mp_linear_kernel=lambda config: FakeKernel,
        ),
        "vllm.model_executor.kernels.linear.mixed_precision": _module(
            "vllm.model_executor.kernels.linear.mixed_precision"
        ),
        "vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit": _module(
            "vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit",
            Dynamic4bitLinearKernel=FakeKernel,
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors": _module(
            "vllm.model_executor.layers.quantization.compressed_tensors"
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors": _module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors",
            CompressedTensorsConfig=FakeConfig,
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes": _module(
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes"
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a8_int": _module(
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w4a8_int",
            CompressedTensorsW4A8Int=FakeScheme,
        ),
        "vllm.model_executor.layers.quantization.utils": _module(
            "vllm.model_executor.layers.quantization.utils",
            replace_parameter=lambda layer, name, value: setattr(layer, name, value),
        ),
        "vllm.model_executor.parameter": _module(
            "vllm.model_executor.parameter",
            BasevLLMParameter=_parameter,
            ChannelQuantScaleParameter=_parameter,
            GroupQuantScaleParameter=_parameter,
            PackedvLLMParameter=_parameter,
        ),
        "flag_gems": _module("flag_gems"),
        "flag_gems.quantized_linear": _module(
            "flag_gems.quantized_linear",
            pack_rhs_qsi4c128p=pack_rhs_qsi4c128p,
            w4a8_g128_linear=w4a8_g128_linear,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return FakeKernel, FakeScheme, FakeConfig, calls


def _install(monkeypatch):
    FakeKernel, FakeScheme, FakeConfig, calls = _wire(monkeypatch)
    assert arm_cpu_w4a8.install_arm_cpu_packed_w4a8() is True
    return FakeKernel, FakeScheme, FakeConfig, calls


def test_install_rejects_partial_patch(monkeypatch):
    FakeKernel, FakeScheme, FakeConfig, _ = _wire(monkeypatch)
    FakeScheme._vllm_fl_arm_packed_w4a8 = True

    with pytest.raises(RuntimeError, match="incomplete"):
        arm_cpu_w4a8.install_arm_cpu_packed_w4a8()


def test_create_weights_group_minus_one_square_partition(monkeypatch):
    FakeKernel, FakeScheme, FakeConfig, _ = _install(monkeypatch)
    scheme = FakeScheme(strategy="group", num_bits=4, group_size=-1, packed=True)
    layer = torch.nn.Module()

    FakeScheme.create_weights(
        scheme,
        layer,
        output_size=8,
        input_size=8,
        output_partition_sizes=[8],
        input_size_per_partition=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.weight_scale.shape == (8, 1)
    assert scheme.kernel.config.group_size == 8


def test_get_scheme_from_parts_resolves_explicit_and_default_format(monkeypatch):
    FakeKernel, FakeScheme, FakeConfig, _ = _install(monkeypatch)
    weight_quant = SimpleNamespace(num_bits=4, strategy="group", group_size=128)
    input_quant = SimpleNamespace(symmetric=True)

    dense = FakeConfig()
    dense.quant_format = "dense"
    dense._dynamic = True
    assert dense._get_scheme_from_parts(weight_quant, input_quant)[0] == "original"

    explicit_dense = FakeConfig()
    explicit_dense.quant_format = PACK_FORMAT
    explicit_dense._dynamic = True
    assert (
        explicit_dense._get_scheme_from_parts(
            weight_quant, input_quant, format="dense"
        )[0]
        == "original"
    )

    explicit_packed = FakeConfig()
    explicit_packed.quant_format = "dense"
    explicit_packed._dynamic = True
    packed = explicit_packed._get_scheme_from_parts(
        weight_quant, input_quant, format=PACK_FORMAT
    )
    assert packed._vllm_fl_checkpoint_packed is True


def test_process_weights_zero_points_falls_back(monkeypatch):
    FakeKernel, FakeScheme, FakeConfig, _ = _install(monkeypatch)
    kernel = FakeKernel(
        SimpleNamespace(
            group_size=128, zero_points=True, partition_weight_shape=(128, 8)
        ),
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
    )
    kernel._vllm_fl_arm_packed_checkpoint = True
    layer = torch.nn.Module()

    FakeKernel.process_weights_after_loading(kernel, layer)

    assert kernel.original_processed is layer


def test_process_weights_keeps_distinct_weight_alias(monkeypatch):
    FakeKernel, FakeScheme, FakeConfig, calls = _install(monkeypatch)
    kernel = FakeKernel(
        SimpleNamespace(
            group_size=128, zero_points=False, partition_weight_shape=(128, 8)
        ),
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
        "weight",
        torch.nn.Parameter(torch.zeros((8, 8), dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((8, 1)), requires_grad=False),
    )

    FakeKernel.process_weights_after_loading(kernel, layer)

    # No weight_shape attribute -> validation is skipped; alias has separate
    # storage so it must be left untouched.
    assert layer.weight.numel() == 64
    assert layer.weight_packed.dtype == torch.uint8
    assert layer.weight_scale is None
    assert kernel._vllm_fl_arm_w4a8_shape == (8, 128)
    assert calls["pack"][0][0].shape == (8, 128)
