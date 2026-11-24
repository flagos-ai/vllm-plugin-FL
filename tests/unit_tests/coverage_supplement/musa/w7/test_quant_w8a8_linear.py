# Coverage tests: vllm_fl/quantization/w8a8/linear.py (CPU only).

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.kernels.linear import Int8ScaledMMLinearLayerConfig
from vllm.platforms import PlatformEnum

from vllm_fl.quantization.w8a8 import linear


PARAM_NAMES = [
    "weight",
    "weight_scale",
    "input_scale",
    "input_zero_point",
    "azp_adj",
]


def _kernel():
    kernel = object.__new__(linear.FLW8A8DynamicLinearKernel)
    kernel.layer_param_names = list(PARAM_NAMES)
    return kernel


def _layer():
    layer = SimpleNamespace(
        weight=None,
        weight_scale=None,
        input_scale=None,
        input_zero_point=None,
        azp_adj=None,
    )
    return layer


@pytest.mark.parametrize(
    ("oot", "flag_gems", "expected", "fragment"),
    [
        (False, True, False, "disabled"),
        (True, False, False, "policy"),
        (True, True, True, None),
    ],
)
def test_is_supported_policy(monkeypatch, oot, flag_gems, expected, fragment):
    monkeypatch.setattr(linear, "is_oot_enabled", lambda: oot)
    monkeypatch.setattr(linear, "use_flaggems_op", lambda op: flag_gems)

    supported, reason = linear.FLW8A8DynamicLinearKernel.is_supported()

    assert supported is expected
    if fragment is None:
        assert reason is None
    else:
        assert fragment in reason


@pytest.mark.parametrize(
    ("channelwise", "static_input", "symmetric", "expected", "fragment"),
    [
        (True, False, True, True, None),
        (False, False, True, False, "per-channel"),
        (True, True, True, False, "dynamic"),
        (True, False, False, False, "symmetric"),
    ],
)
def test_can_implement(channelwise, static_input, symmetric, expected, fragment):
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=channelwise,
        is_static_input_scheme=static_input,
        input_symmetric=symmetric,
    )
    supported, reason = linear.FLW8A8DynamicLinearKernel.can_implement(config)
    assert supported is expected
    if fragment is None:
        assert reason is None
    else:
        assert fragment in reason


def test_process_weights_transposes_and_clears_quantizers():
    checkpoint = torch.tensor(
        [[12, -7, 3, 9], [-5, 11, -13, 2], [4, 6, -8, 10]], dtype=torch.int8
    )
    scale = torch.tensor([[0.01], [0.025], [0.04]], dtype=torch.float32)
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(checkpoint.clone(), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(scale.clone(), requires_grad=False)
    )
    layer.register_parameter("input_scale", None)
    layer.register_parameter("input_zero_point", None)
    layer.register_parameter("azp_adj", None)

    _kernel().process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 3)
    assert layer.weight.is_contiguous()
    assert torch.equal(layer.weight, checkpoint.t().contiguous())
    assert layer.weight_scale.shape == (3,)
    assert layer.weight_scale.dtype == torch.float32
    assert torch.equal(layer.weight_scale, scale.reshape(-1))
    assert layer.input_scale is None
    assert layer.input_zero_point is None
    assert layer.azp_adj is None


def test_process_weights_rejects_bad_checkpoint():
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.zeros(3, dtype=torch.int8), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones(3), requires_grad=False),
    )
    layer.register_parameter("input_scale", None)
    layer.register_parameter("input_zero_point", None)
    layer.register_parameter("azp_adj", None)

    with pytest.raises(ValueError, match="2D int8"):
        _kernel().process_weights_after_loading(layer)

    bad_dtype = torch.nn.Module()
    bad_dtype.register_parameter(
        "weight",
        torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.float32), requires_grad=False),
    )
    bad_dtype.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.ones(3), requires_grad=False)
    )
    bad_dtype.register_parameter("input_scale", None)
    bad_dtype.register_parameter("input_zero_point", None)
    bad_dtype.register_parameter("azp_adj", None)
    with pytest.raises(ValueError, match="2D int8"):
        _kernel().process_weights_after_loading(bad_dtype)

    bad_scale = torch.nn.Module()
    bad_scale.register_parameter(
        "weight",
        torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.int8), requires_grad=False),
    )
    bad_scale.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.ones(2), requires_grad=False)
    )
    bad_scale.register_parameter("input_scale", None)
    bad_scale.register_parameter("input_zero_point", None)
    bad_scale.register_parameter("azp_adj", None)
    with pytest.raises(ValueError, match="one scale per output channel"):
        _kernel().process_weights_after_loading(bad_scale)


def _flag_gems(monkeypatch):
    calls = []
    module = ModuleType("flag_gems")

    def scaled_mm(x_q, weight, x_scale, weight_scale, *, bias, out_dtype):
        calls.append((x_q, weight, x_scale, weight_scale, bias, out_dtype))
        rows = x_q.shape[0]
        return torch.full((rows, weight.shape[1]), 1.0, dtype=out_dtype)

    module.scaled_mm = scaled_mm
    monkeypatch.setitem(sys.modules, "flag_gems", module)
    return calls


def test_apply_weights_reshape_and_bias(monkeypatch):
    x_q = torch.ones((6, 4), dtype=torch.int8)
    x_scale = torch.full((6, 1), 0.5, dtype=torch.float32)
    weight = torch.ones((4, 3), dtype=torch.int8)
    weight_scale = torch.ones(3, dtype=torch.float32)
    monkeypatch.setattr(
        linear,
        "_dynamic_per_token_quant_int8",
        lambda value: (x_q, x_scale),
    )
    calls = _flag_gems(monkeypatch)

    layer = _layer()
    layer.weight = weight
    layer.weight_scale = weight_scale
    x = torch.zeros((2, 3, 4), dtype=torch.bfloat16)
    bias = torch.zeros(3, dtype=torch.bfloat16)

    output = _kernel().apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 3)
    assert output.dtype == torch.bfloat16
    assert len(calls) == 1
    assert calls[0][0] is x_q
    assert calls[0][1] is weight
    assert calls[0][4] is bias
    assert calls[0][5] == torch.bfloat16


def test_apply_weights_2d_without_bias(monkeypatch):
    x_q = torch.ones((2, 4), dtype=torch.int8)
    x_scale = torch.ones((2, 1), dtype=torch.float32)
    weight = torch.ones((4, 5), dtype=torch.int8)
    weight_scale = torch.ones(5, dtype=torch.float32)
    monkeypatch.setattr(
        linear, "_dynamic_per_token_quant_int8", lambda value: (x_q, x_scale)
    )
    calls = _flag_gems(monkeypatch)

    layer = _layer()
    layer.weight = weight
    layer.weight_scale = weight_scale
    output = _kernel().apply_weights(layer, torch.zeros((2, 4), dtype=torch.float32))

    assert output.shape == (2, 5)
    assert calls[0][4] is None
    assert calls[0][5] == torch.float32


def test_register_prepends_once_and_keeps_fallback():
    native = object()

    registry = {}
    assert linear.register_fl_w8a8_linear_kernel(registry) is True
    assert registry[PlatformEnum.OOT] == [linear.FLW8A8DynamicLinearKernel]
    assert linear.register_fl_w8a8_linear_kernel(registry) is True
    assert registry[PlatformEnum.OOT] == [linear.FLW8A8DynamicLinearKernel]

    with_native = {PlatformEnum.OOT: [native]}
    assert linear.register_fl_w8a8_linear_kernel(with_native) is True
    assert with_native[PlatformEnum.OOT] == [
        linear.FLW8A8DynamicLinearKernel,
        native,
    ]
    assert linear.register_fl_w8a8_linear_kernel(with_native) is True
    assert with_native[PlatformEnum.OOT] == [
        linear.FLW8A8DynamicLinearKernel,
        native,
    ]
