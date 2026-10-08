# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.quantization.w8a8 import moe_experts


def _apply_arguments():
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    return {
        "output": torch.empty_like(hidden_states),
        "hidden_states": hidden_states,
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


def _quant_config():
    return SimpleNamespace(
        use_int8_w8a8=True,
        per_act_token_quant=True,
        block_shape=None,
        w1_scale=torch.ones((2, 8, 1), dtype=torch.float32),
        w2_scale=torch.ones((2, 4, 1), dtype=torch.float32),
        w1_bias=None,
        w2_bias=None,
    )


def _experts_instance(quant_config=None, *, clamp_limit=None):
    return SimpleNamespace(
        quant_config=_quant_config() if quant_config is None else quant_config,
        activation_config=SimpleNamespace(clamp_limit=clamp_limit),
        _lora_context=None,
    )


def _independent_w8a8_reference(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    expert_map,
    apply_router_weight_on_input,
    clamp_limit,
):
    """Scalar-route INT8 reference, independent of both production experts paths."""

    def quantize_rows(value):
        absmax = value.float().abs().amax(dim=-1, keepdim=True)
        scale = absmax / 127
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round(value.float() / safe_scale).clamp(-127, 127).to(torch.int8)
        return codes, scale

    def linear(value, weight, weight_scale, bias):
        codes, token_scale = quantize_rows(value)
        if bias is not None and bias.dtype == torch.int32:
            accumulator = codes.float() @ weight.float().T + bias.float()
            return (accumulator * weight_scale.squeeze(-1).float() * token_scale).to(
                value.dtype
            )
        dequantized = (codes.float() * token_scale).to(value.dtype)
        scaled_weight = weight.to(value.dtype) * weight_scale.squeeze(-1).to(
            value.dtype
        ).unsqueeze(-1)
        result = dequantized @ scaled_weight.T
        if bias is not None:
            result = result + bias.to(value.dtype)
        return result

    output = torch.zeros(
        (hidden_states.shape[0], w2.shape[1]), dtype=hidden_states.dtype
    )
    for token in range(topk_ids.shape[0]):
        for slot in range(topk_ids.shape[1]):
            global_id = int(topk_ids[token, slot])
            if global_id < 0:
                continue
            local_id = global_id if expert_map is None else int(expert_map[global_id])
            if local_id < 0:
                continue
            gate_up = linear(
                hidden_states[token : token + 1],
                w1[local_id],
                w1_scale[local_id],
                None if w1_bias is None else w1_bias[local_id],
            )
            gate, up = gate_up.float().chunk(2, dim=-1)
            if clamp_limit is not None and clamp_limit > 0:
                gate = gate.clamp(max=clamp_limit)
                up = up.clamp(-clamp_limit, clamp_limit)
            activated = (torch.nn.functional.silu(gate) * up).to(hidden_states.dtype)
            down = linear(
                activated,
                w2[local_id],
                w2_scale[local_id],
                None if w2_bias is None else w2_bias[local_id],
            )
            if not apply_router_weight_on_input:
                down = (down.float() * topk_weights[token, slot].float()).to(
                    hidden_states.dtype
                )
            output[token] += down[0]
    return output


@pytest.mark.parametrize(
    "experts_cls",
    [
        moe_experts.FlagGemsW8A8Experts,
        moe_experts.VllmFunctionalW8A8Experts,
    ],
)
def test_functional_experts_defer_activation_quantization(experts_cls):
    instance = SimpleNamespace()
    assert experts_cls.expects_unquantized_inputs.fget(instance) is True


def test_vllm_functional_experts_changes_the_triton_input_contract():
    instance = SimpleNamespace(_lora_context=None)

    assert moe_experts.TritonExperts.expects_unquantized_inputs.fget(instance) is False
    assert (
        moe_experts.VllmFunctionalW8A8Experts.expects_unquantized_inputs.fget(instance)
        is True
    )


def test_vllm_functional_experts_call_only_native_vllm(monkeypatch):
    calls = []

    def fake_vllm_fused_experts(**kwargs):
        calls.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 2)

    monkeypatch.setattr(
        moe_experts,
        "_vllm_fused_experts",
        fake_vllm_fused_experts,
    )
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("NVIDIA must not dispatch through FlagGems"),
    )
    quant_config = _quant_config()
    arguments = _apply_arguments()

    moe_experts.VllmFunctionalW8A8Experts.apply(
        SimpleNamespace(quant_config=quant_config),
        **arguments,
    )

    assert calls[0]["hidden_states"].dtype == torch.bfloat16
    assert calls[0]["quant_config"] is quant_config
    assert calls[0]["activation"] is MoEActivation.SILU
    assert "inplace" not in calls[0]
    assert torch.equal(
        arguments["output"],
        torch.full_like(arguments["output"], 2),
    )


def test_non_npu_small_batch_calls_flaggems_with_exact_w8a8_contract(monkeypatch):
    calls = []

    def fake_fused_experts_impl(**kwargs):
        calls.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 3)

    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        fake_fused_experts_impl,
    )
    quant_config = _quant_config()
    instance = _experts_instance(quant_config)
    arguments = _apply_arguments()

    moe_experts.FlagGemsW8A8Experts.apply(instance, **arguments)

    assert calls[0]["hidden_states"].dtype == torch.bfloat16
    assert calls[0]["hidden_states"].shape == (2, 4)
    assert calls[0]["w1"].shape == (2, 8, 4)
    assert calls[0]["w2"].shape == (2, 4, 4)
    assert calls[0]["w1_scale"] is quant_config.w1_scale
    assert calls[0]["w2_scale"] is quant_config.w2_scale
    assert calls[0]["w1_scale"].shape == (2, 8, 1)
    assert calls[0]["w2_scale"].shape == (2, 4, 1)
    assert calls[0]["a1_scale"] is None
    assert calls[0]["a2_scale"] is None
    assert calls[0]["use_int8_w8a8"] is True
    assert calls[0]["per_channel_quant"] is True
    assert calls[0]["activation"] == MoEActivation.SILU.value
    assert calls[0]["inplace"] is False
    assert torch.equal(
        arguments["output"],
        torch.full_like(arguments["output"], 3),
    )


def test_ascend_with_unsupported_bias_uses_native_w8a8_fallback(monkeypatch):
    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.backends.vendor.ascend import register_ops as ascend_ops

    def call_ascend_provider(op_name, **kwargs):
        assert op_name == "w8a8_moe_experts"
        return ascend_ops._w8a8_moe_experts(**kwargs)

    monkeypatch.setattr(moe_experts, "_is_ascend_npu_tensor", lambda value: True)
    monkeypatch.setattr(dispatch, "call_op", call_ascend_provider)
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("Ascend must bypass FlagGems Triton"),
    )
    arguments = _apply_arguments()
    quant_config = _quant_config()
    quant_config.w1_bias = torch.zeros((2, 8), dtype=torch.float32)
    moe_experts.FlagGemsW8A8Experts.apply(
        _experts_instance(quant_config),
        **arguments,
    )

    assert arguments["output"].shape == arguments["hidden_states"].shape
    assert torch.isfinite(arguments["output"]).all()
    assert torch.count_nonzero(arguments["output"]) > 0


def test_local_expert_ids_flattens_lookup_and_preserves_nonlocal_routes():
    topk_ids = torch.tensor([[2, -1], [0, 1]], dtype=torch.int32)
    expert_map = torch.tensor([1, -1, 0], dtype=torch.int64)

    local_ids = moe_experts._local_expert_ids(topk_ids, expert_map)

    assert local_ids.shape == topk_ids.shape
    assert local_ids.dtype == torch.int64
    assert torch.equal(local_ids, torch.tensor([[0, -1], [1, -1]]))


@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
@pytest.mark.parametrize("clamp_limit", [None, 10.0])
def test_ascend_grouped_w8a8_matches_independent_reference(
    monkeypatch,
    apply_router_weight_on_input,
    clamp_limit,
):
    calls = []
    quantized_inputs = []

    def fake_dynamic_quant(value):
        quantized_inputs.append(value.clone())
        scale = value.float().abs().amax(dim=-1) / 127.0
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quantized = (
            torch.round(value.float() / safe_scale[:, None])
            .clamp(-127, 127)
            .to(torch.int8)
        )
        return quantized, scale

    def fake_grouped_matmul(
        *,
        x,
        weight,
        bias,
        scale,
        per_token_scale,
        group_list,
        output_dtype,
        split_item,
        group_type,
        group_list_type,
        **kwargs,
    ):
        del kwargs
        counts = group_list.cpu()
        expert_ids = torch.repeat_interleave(torch.arange(len(counts)), counts)
        rows = []
        for row, expert_id in enumerate(expert_ids.tolist()):
            accumulator = x[0][row].float() @ weight[0][expert_id].float()
            if bias is not None:
                accumulator = accumulator + bias[0][expert_id].float()
            rows.append(
                accumulator
                * scale[0][expert_id].float()
                * per_token_scale[0][row].float()
            )
        calls.append(
            {
                "weight_shape": tuple(weight[0].shape),
                "scale_shape": tuple(scale[0].shape),
                "scale_dtype": scale[0].dtype,
                "counts": counts.tolist(),
                "split_item": split_item,
                "group_type": group_type,
                "group_list_type": group_list_type,
                "per_token_scale_shape": tuple(per_token_scale[0].shape),
            }
        )
        return [torch.stack(rows).to(output_dtype)]

    fake_torch_npu = SimpleNamespace(
        npu_dynamic_quant=fake_dynamic_quant,
        npu_grouped_matmul=fake_grouped_matmul,
        npu_swiglu=lambda value, dim=-1: (
            torch.nn.functional.silu(value.chunk(2, dim=dim)[0])
            * value.chunk(2, dim=dim)[1]
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    hidden_states = torch.tensor(
        [[2.0, 0.5, -0.25, 0.125], [3.0, -0.5, 0.25, 0.5]],
        dtype=torch.bfloat16,
    )
    w1 = torch.tensor(
        [
            [
                [10, 1, -1, 2],
                [8, -1, 2, 0],
                [-4, 0, 1, 1],
                [6, 1, 0, -1],
                [10, 0, 1, -1],
                [8, 1, -1, 0],
                [4, -1, 0, 1],
                [-6, 0, 1, 1],
            ],
            [
                [9, -1, 1, 0],
                [11, 0, -1, 1],
                [5, 1, 0, -1],
                [7, 0, 1, 1],
                [9, 1, 0, 1],
                [11, 0, 1, -1],
                [5, -1, 1, 0],
                [7, 1, -1, 0],
            ],
        ],
        dtype=torch.int8,
    )
    w2 = torch.tensor(
        [
            [[2, -1, 1, 0], [0, 2, 1, -1], [1, 0, -1, 2], [-1, 1, 2, 0]],
            [[1, 1, 0, -1], [2, -1, 1, 0], [0, 1, -2, 1], [1, 0, 1, 2]],
        ],
        dtype=torch.int8,
    )
    if apply_router_weight_on_input:
        topk_ids = torch.tensor([[2], [0]], dtype=torch.int64)
        topk_weights = torch.tensor([[0.75], [0.6]])
        # vLLM's modular prepare stage applies input-side router weights.
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
    else:
        topk_ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
        topk_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]])
    expert_map = torch.tensor([1, -1, 0], dtype=torch.int64)
    w1_scale = torch.tensor(
        [[1, 0.5, 2, 0.25, 1, 2, 0.5, 1], [1, 0.5, 2, 0.25, 1, 2, 0.5, 1]],
        dtype=torch.float32,
    ).unsqueeze(-1)
    w2_scale = torch.tensor(
        [[0.5, 1, 0.25, 2], [1, 0.5, 2, 0.25]], dtype=torch.float32
    ).unsqueeze(-1)
    w1_bias = torch.tensor(
        [[1, -2, 0, 2, 0, 1, -1, 0], [2, 0, -1, 1, 1, -1, 0, 2]],
        dtype=torch.int32,
    )
    w2_bias = torch.tensor([[1, 0, -2, 1], [0, 2, 1, -1]], dtype=torch.int32)

    actual = moe_experts._ascend_w8a8_grouped_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        expert_map,
        apply_router_weight_on_input,
        clamp_limit=clamp_limit,
    )
    expected = _independent_w8a8_reference(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        expert_map,
        apply_router_weight_on_input,
        clamp_limit,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=0.05)
    assert quantized_inputs[1].dtype == hidden_states.dtype
    if clamp_limit is not None:
        unclamped = _independent_w8a8_reference(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            expert_map,
            apply_router_weight_on_input,
            None,
        )
        assert not torch.allclose(expected.float(), unclamped.float(), rtol=0.1)
    expected_counts = [1, 1] if apply_router_weight_on_input else [2, 1]
    routed_rows = 2 if apply_router_weight_on_input else 3
    assert calls == [
        {
            "weight_shape": (2, 4, 8),
            "scale_shape": (2, 8),
            "scale_dtype": torch.bfloat16,
            "counts": expected_counts,
            "split_item": 2,
            "group_type": 0,
            "group_list_type": 1,
            "per_token_scale_shape": (routed_rows,),
        },
        {
            "weight_shape": (2, 4, 4),
            "scale_shape": (2, 4),
            "scale_dtype": torch.bfloat16,
            "counts": expected_counts,
            "split_item": 2,
            "group_type": 0,
            "group_list_type": 1,
            "per_token_scale_shape": (routed_rows,),
        },
    ]


def test_ascend_grouped_w8a8_skips_nonlocal_routes(monkeypatch):
    fail = lambda *args, **kwargs: pytest.fail("empty local routes must skip NPU ops")
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_dynamic_quant=fail,
            npu_grouped_matmul=fail,
            npu_swiglu=fail,
        ),
    )
    arguments = _apply_arguments()

    result = moe_experts._ascend_w8a8_grouped_experts(
        arguments["hidden_states"],
        arguments["w1"],
        arguments["w2"],
        arguments["topk_weights"],
        arguments["topk_ids"],
        _quant_config().w1_scale,
        _quant_config().w2_scale,
        None,
        None,
        torch.full((2,), -1, dtype=torch.int64),
        False,
    )

    assert torch.equal(result, torch.zeros_like(arguments["hidden_states"]))


@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_native_w8a8_clamp_matches_independent_reference(
    apply_router_weight_on_input,
):
    hidden_states = torch.tensor(
        [[2.0, 1.0, -0.5, 0.25], [3.0, -1.0, 0.5, 0.125]],
        dtype=torch.bfloat16,
    )
    w1 = torch.tensor(
        [
            [
                [10, 1, 0, 0],
                [8, 0, 1, 0],
                [5, -1, 0, 1],
                [4, 0, -1, 0],
                [10, 0, 1, 0],
                [8, 1, 0, 0],
                [5, 0, -1, 1],
                [4, 1, 0, -1],
            ]
        ],
        dtype=torch.int8,
    )
    w2 = torch.tensor(
        [
            [
                [2, -1, 1, 0],
                [0, 2, 1, -1],
                [1, 0, -1, 2],
                [-1, 1, 2, 0],
            ]
        ],
        dtype=torch.int8,
    )
    topk_ids = torch.zeros((2, 1), dtype=torch.int64)
    topk_weights = torch.tensor([[0.75], [0.6]], dtype=torch.float32)
    if apply_router_weight_on_input:
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
    w1_scale = torch.tensor(
        [[[1], [0.5], [2], [0.25], [1], [2], [0.5], [1]]],
        dtype=torch.float32,
    )
    w2_scale = torch.tensor([[[0.5], [1], [0.25], [2]]], dtype=torch.float32)
    w1_bias = torch.tensor([[0.5, -1, 0, 1, -0.5, 1, 0, -1]], dtype=torch.float32)
    w2_bias = torch.tensor([[0.5, -1, 0, 1]], dtype=torch.float32)

    actual = moe_experts._native_w8a8_fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        None,
        apply_router_weight_on_input,
        clamp_limit=10.0,
    )
    expected = _independent_w8a8_reference(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        None,
        apply_router_weight_on_input,
        10.0,
    )
    unclamped = _independent_w8a8_reference(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        None,
        apply_router_weight_on_input,
        None,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=0.05)
    assert not torch.allclose(expected.float(), unclamped.float(), rtol=0.1)


@pytest.mark.gpu
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_real_ascend_grouped_w8a8_clamp_matches_cpu_reference(
    apply_router_weight_on_input,
):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")

    # Grouped INT8 matmul requires aligned dimensions on real Ascend hardware.
    device = torch.device("npu:0")
    hidden_states = torch.zeros((2, 128), dtype=torch.bfloat16, device=device)
    hidden_states[:, 0] = torch.tensor([2.0, 3.0], dtype=torch.bfloat16)
    w1 = torch.zeros((2, 256, 128), dtype=torch.int8, device=device)
    w1[0, :, 0] = 10
    w1[1, :, 0] = 11
    w2 = torch.eye(128, dtype=torch.int8, device=device).repeat(2, 1, 1)
    w1_scale = torch.ones((2, 256, 1), dtype=torch.float32, device=device)
    w2_scale = torch.ones((2, 128, 1), dtype=torch.float32, device=device)
    w2_scale[1] = 0.5
    w1_bias = torch.ones((2, 256), dtype=torch.int32, device=device)
    w2_bias = torch.ones((2, 128), dtype=torch.int32, device=device)
    expert_map = torch.tensor([1, 0], dtype=torch.int64, device=device)
    if apply_router_weight_on_input:
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int64, device=device)
        topk_weights = torch.tensor([[0.75], [0.6]], dtype=torch.float32, device=device)
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
    else:
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device)
        topk_weights = torch.tensor(
            [[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32, device=device
        )

    actual = moe_experts._ascend_w8a8_grouped_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        expert_map,
        apply_router_weight_on_input,
        clamp_limit=10.0,
    )
    expected = _independent_w8a8_reference(
        hidden_states.cpu(),
        w1.cpu(),
        w2.cpu(),
        topk_weights.cpu(),
        topk_ids.cpu(),
        w1_scale.cpu(),
        w2_scale.cpu(),
        w1_bias.cpu(),
        w2_bias.cpu(),
        expert_map.cpu(),
        apply_router_weight_on_input,
        10.0,
    )

    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=0.05, atol=0.5
    )


def test_npu_w8a8_experts_bypass_flaggems_triton(monkeypatch):
    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.backends.vendor.ascend import register_ops as ascend_ops

    grouped_calls = []

    def fake_grouped_experts(**kwargs):
        grouped_calls.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 4)

    def call_ascend_provider(op_name, **kwargs):
        assert op_name == "w8a8_moe_experts"
        return ascend_ops._w8a8_moe_experts(**kwargs)

    monkeypatch.setattr(moe_experts, "_is_ascend_npu_tensor", lambda value: True)
    monkeypatch.setattr(dispatch, "call_op", call_ascend_provider)
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("Ascend must bypass FlagGems Triton MoE"),
    )
    monkeypatch.setattr(
        moe_experts,
        "_native_w8a8_fused_experts",
        lambda **kwargs: pytest.fail("Ascend must use native grouped INT8 MoE"),
    )
    monkeypatch.setattr(
        moe_experts,
        "_ascend_w8a8_grouped_experts",
        fake_grouped_experts,
    )
    arguments = _apply_arguments()

    moe_experts.FlagGemsW8A8Experts.apply(
        _experts_instance(clamp_limit=10.0),
        **arguments,
    )

    assert torch.equal(arguments["output"], torch.full_like(arguments["output"], 4))
    assert grouped_calls[0]["clamp_limit"] == 10.0


def test_npu_w8a8_experts_keep_float_bias_on_reference_path(monkeypatch):
    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch.backends.vendor.ascend import register_ops as ascend_ops

    native_calls = []

    def fake_native_experts(**kwargs):
        native_calls.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 5)

    def call_ascend_provider(op_name, **kwargs):
        assert op_name == "w8a8_moe_experts"
        return ascend_ops._w8a8_moe_experts(**kwargs)

    monkeypatch.setattr(moe_experts, "_is_ascend_npu_tensor", lambda value: True)
    monkeypatch.setattr(dispatch, "call_op", call_ascend_provider)
    monkeypatch.setattr(
        moe_experts,
        "_ascend_w8a8_grouped_experts",
        lambda **kwargs: pytest.fail("quantized GMM requires int32 bias"),
    )
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("Ascend must bypass FlagGems Triton MoE"),
    )
    monkeypatch.setattr(
        moe_experts,
        "_native_w8a8_fused_experts",
        fake_native_experts,
    )
    quant_config = _quant_config()
    quant_config.w1_bias = torch.zeros((2, 8), dtype=torch.float32)
    arguments = _apply_arguments()

    moe_experts.FlagGemsW8A8Experts.apply(
        _experts_instance(quant_config, clamp_limit=10.0),
        **arguments,
    )

    assert torch.equal(arguments["output"], torch.full_like(arguments["output"], 5))
    assert native_calls[0]["clamp_limit"] == 10.0


@pytest.mark.parametrize(
    ("hidden_dtype", "w1_bias_dtype", "w2_bias_dtype", "expected_path"),
    [
        (torch.bfloat16, None, None, "grouped"),
        (torch.float16, torch.int32, torch.int32, "grouped"),
        (torch.bfloat16, torch.float32, None, "native"),
        (torch.bfloat16, torch.int32, torch.float32, "native"),
        (torch.float32, None, None, "native"),
    ],
)
def test_ascend_w8a8_provider_selects_supported_kernel(
    monkeypatch, hidden_dtype, w1_bias_dtype, w2_bias_dtype, expected_path
):
    from vllm_fl.dispatch.backends.vendor.ascend import register_ops as ascend_ops

    calls = []

    def record(path):
        def implementation(**kwargs):
            calls.append((path, kwargs))
            return kwargs["hidden_states"]

        return implementation

    monkeypatch.setattr(moe_experts, "_ascend_w8a8_grouped_experts", record("grouped"))
    monkeypatch.setattr(moe_experts, "_native_w8a8_fused_experts", record("native"))
    hidden_states = torch.ones((2, 4), dtype=hidden_dtype)
    w1_bias = (
        None if w1_bias_dtype is None else torch.zeros((2, 8), dtype=w1_bias_dtype)
    )
    w2_bias = (
        None if w2_bias_dtype is None else torch.zeros((2, 4), dtype=w2_bias_dtype)
    )

    actual = ascend_ops._w8a8_moe_experts(
        hidden_states=hidden_states,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        activation=MoEActivation.SILU.value,
        global_num_experts=2,
        clamp_limit=10.0,
    )

    assert actual is hidden_states
    assert len(calls) == 1
    path, kwargs = calls[0]
    assert path == expected_path
    assert kwargs["w1_bias"] is w1_bias
    assert kwargs["w2_bias"] is w2_bias
    assert kwargs["clamp_limit"] == 10.0
    assert "activation" not in kwargs
    assert "global_num_experts" not in kwargs


def test_ascend_w8a8_provider_rejects_unsupported_activation(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.ascend import register_ops as ascend_ops

    monkeypatch.setattr(
        moe_experts,
        "_ascend_w8a8_grouped_experts",
        lambda **kwargs: pytest.fail("unsupported activation reached grouped kernel"),
    )
    monkeypatch.setattr(
        moe_experts,
        "_native_w8a8_fused_experts",
        lambda **kwargs: pytest.fail("unsupported activation reached native kernel"),
    )
    with pytest.raises(NotImplementedError, match="SwiGLU only"):
        ascend_ops._w8a8_moe_experts(
            hidden_states=torch.ones((1, 4), dtype=torch.bfloat16),
            w1_bias=None,
            w2_bias=None,
            activation="gelu",
            global_num_experts=2,
        )


def test_functional_experts_rejects_prequantized_input(monkeypatch):
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("FlagGems fused_experts must not run"),
    )
    instance = _experts_instance()
    arguments = _apply_arguments()
    arguments["hidden_states"] = torch.ones((2, 4), dtype=torch.int8)
    arguments["a1q_scale"] = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="quantized before"):
        moe_experts.FlagGemsW8A8Experts.apply(instance, **arguments)


def test_functional_experts_rejects_activation_not_supported_by_flaggems(
    monkeypatch,
):
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("FlagGems fused_experts must not run"),
    )
    arguments = _apply_arguments()
    arguments["activation"] = MoEActivation.GELU

    with pytest.raises(NotImplementedError, match="only silu"):
        moe_experts.FlagGemsW8A8Experts.apply(
            _experts_instance(),
            **arguments,
        )


@pytest.mark.parametrize(
    ("scale_name", "bad_scale"),
    [
        ("w1_scale", torch.ones((2, 1, 8), dtype=torch.float32)),
        ("w2_scale", torch.ones((2, 1, 4), dtype=torch.float32)),
        ("w1_scale", torch.ones((2, 8, 1), dtype=torch.bfloat16)),
    ],
)
def test_functional_experts_rejects_incompatible_weight_scale(
    monkeypatch,
    scale_name,
    bad_scale,
):
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("FlagGems fused_experts must not run"),
    )
    quant_config = _quant_config()
    setattr(quant_config, scale_name, bad_scale)
    instance = _experts_instance(quant_config)

    with pytest.raises((TypeError, ValueError), match="weight scales|scale must"):
        moe_experts.FlagGemsW8A8Experts.apply(instance, **_apply_arguments())
