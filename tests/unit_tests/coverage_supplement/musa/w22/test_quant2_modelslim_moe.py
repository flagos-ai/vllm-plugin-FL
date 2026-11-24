# Coverage tests: remaining branches in
# vllm_fl/quantization/modelslim_w8a8.py and
# vllm_fl/quantization/w8a8/moe_experts.py (CPU only).

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.quantization import modelslim_w8a8 as modelslim
from vllm_fl.quantization.w8a8 import moe_experts


@pytest.fixture(autouse=True)
def _single_rank_tensor_parallel(monkeypatch):
    # Real vLLM parameter constructors consult the TP group; keep them local
    # to a single CPU rank instead of an uninitialized distributed group.
    import vllm.model_executor.parameter as vllm_parameter

    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(modelslim, "get_tensor_model_parallel_rank", lambda: 0)


# ---------------------------------------------------------------------------
# modelslim_w8a8.py
# ---------------------------------------------------------------------------


def test_quant_kind_skips_absent_group_for_experts():
    config = modelslim.ModelSlimW8A8Config(
        {
            "x.experts.0.w1.weight": "W8A8",
            "x.experts.0.w2.weight": "W8A8",
            "x.experts.0.w3.weight": "W8A8",
        }
    )
    assert config._quant_kind("x.experts") == "W8A8"


def test_quant_kind_mixes_kinds_across_groups():
    config = modelslim.ModelSlimW8A8Config(
        {
            "x.experts.0.w1.weight": "W8A8",
            "x.experts.0.w2.weight": "W8A8",
            "x.experts.0.w3.weight": "W8A8",
            "x.experts.0.gate_proj.weight": "FLOAT",
            "x.experts.0.up_proj.weight": "FLOAT",
            "x.experts.0.down_proj.weight": "FLOAT",
        }
    )
    with pytest.raises(ValueError, match="mixes"):
        config._quant_kind("x.experts")


def test_quant_kind_down_proj_uses_secondary_group():
    config = modelslim.ModelSlimW8A8Config({"x.w2.weight": "W8A8"})
    assert config._quant_kind("x.down_proj") == "W8A8"
    assert config._quant_kind("model.x.down_proj") == "W8A8"


def _static_layer(widths, input_size):
    scheme = modelslim._ModelSlimW8A8StaticLinearScheme()
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=list(widths),
        input_size_per_partition=input_size,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.weight.data.zero_()
    layer.input_scale.data.fill_(0.5)
    layer.input_offset.data.zero_()
    layer.deq_scale.data.fill_(0.125)
    layer.quant_bias.data.zero_()
    scheme.process_weights_after_loading(layer)
    return scheme, layer


def test_static_apply_accepts_prequantized_single_matrix(monkeypatch):
    calls = []

    def quantize(*args, **kwargs):
        calls.append("quant")
        return torch.zeros((1, 1), dtype=torch.int8)

    def quant_matmul(x, weight, deq_scale, *, bias, output_dtype):
        calls.append("matmul")
        return torch.zeros((x.shape[0], weight.shape[1]), dtype=output_dtype)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_quantize=quantize, npu_quant_matmul=quant_matmul),
    )
    scheme, layer = _static_layer([4], 4)

    output = scheme.apply_weights(
        layer, torch.ones((2, 4), dtype=torch.int8), None
    )

    assert calls == ["matmul"]
    assert output.shape == (2, 4)
    assert output.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# w8a8/moe_experts.py
# ---------------------------------------------------------------------------


def test_native_experts_multi_slot_with_biases_and_clamp():
    hidden = torch.tensor(
        [[1.0, -1.0, 0.5, 2.0], [0.5, 0.25, -0.75, 1.0]], dtype=torch.float32
    )
    w1 = torch.ones((2, 8, 4), dtype=torch.int8)
    w2 = torch.ones((2, 4, 4), dtype=torch.int8)
    w1_scale = torch.full((2, 8, 1), 0.01, dtype=torch.float32)
    w2_scale = torch.full((2, 4, 1), 0.02, dtype=torch.float32)
    w1_bias = torch.full((2, 8), 0.1, dtype=torch.float32)
    w2_bias = torch.full((2, 4), 0.2, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32)

    output = moe_experts._native_w8a8_fused_experts(
        hidden,
        w1,
        w2,
        weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        None,
        False,
        clamp_limit=2.0,
    )

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) > 0


def test_native_experts_router_weights_on_input_skips_scaling():
    hidden = torch.ones((1, 4), dtype=torch.float32)
    w1 = torch.ones((1, 8, 4), dtype=torch.int8)
    w2 = torch.ones((1, 4, 4), dtype=torch.int8)
    w1_scale = torch.full((1, 8, 1), 0.01)
    w2_scale = torch.full((1, 4, 1), 0.02)
    topk_ids = torch.zeros((1, 1), dtype=torch.int64)

    unweighted = moe_experts._native_w8a8_fused_experts(
        hidden, w1, w2, torch.tensor([[0.5]]), topk_ids, w1_scale, w2_scale,
        None, None, None, True,
    )
    weighted = moe_experts._native_w8a8_fused_experts(
        hidden, w1, w2, torch.tensor([[0.5]]), topk_ids, w1_scale, w2_scale,
        None, None, None, False,
    )
    assert torch.allclose(weighted, unweighted * 0.5, rtol=0.1, atol=0.1)


def _fake_torch_npu(monkeypatch, **functions):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(**functions))


def test_ascend_grouped_float16_with_clamp_and_router_scaling(monkeypatch):
    def dynamic_quant(value):
        return value.to(torch.int8), value.float().abs().amax(dim=-1) / 127.0

    def grouped_matmul(*, x, weight, scale, per_token_scale, group_list,
                       output_dtype, **kwargs):
        # weight list holds the [E, K, N] expert tensor, so the output width
        # is its last dimension N (not the expert count E).
        return [torch.ones((x[0].shape[0], weight[0].shape[-1]), dtype=output_dtype)]

    def swiglu(value, dim=-1):
        gate, up = value.chunk(2, dim=dim)
        return torch.nn.functional.silu(gate) * up

    _fake_torch_npu(
        monkeypatch,
        npu_dynamic_quant=dynamic_quant,
        npu_grouped_matmul=grouped_matmul,
        npu_swiglu=swiglu,
    )

    hidden = torch.ones((2, 4), dtype=torch.float16)
    result = moe_experts._ascend_w8a8_grouped_experts(
        hidden,
        torch.ones((2, 4, 4), dtype=torch.int8),
        torch.ones((2, 4, 4), dtype=torch.int8),
        torch.ones((2, 1), dtype=torch.float32),
        torch.tensor([[0], [1]], dtype=torch.int64),
        torch.ones((2, 4, 1), dtype=torch.float32),
        torch.ones((2, 4, 1), dtype=torch.float32),
        None,
        None,
        None,
        False,
        clamp_limit=2.0,
    )

    assert result.shape == (2, 4)
    assert result.dtype == torch.float16
    assert torch.isfinite(result).all()
