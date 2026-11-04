# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
flag_gems = pytest.importorskip("flag_gems")

from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)

from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL
from vllm_fl.ops.fused_moe.router import fused_topk
from vllm_fl.utils import get_flag_gems_whitelist_blacklist


@pytest.mark.gpu
@pytest.mark.parametrize("tokens", [48, 49, 65])
def test_ascend_topk_softmax_renormalization(tokens):
    generator = torch.Generator().manual_seed(91)
    experts, top_k = 256, 8
    logits = torch.randn(tokens, experts, generator=generator, dtype=torch.bfloat16)
    hidden_states = torch.empty(tokens, 1, dtype=torch.bfloat16, device="npu")
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()

    with (
        torch.inference_mode(),
        flag_gems.use_gems(include=whitelist, exclude=blacklist),
    ):
        weights, ids, _ = fused_topk(
            hidden_states,
            logits.npu(),
            top_k,
            renormalize=True,
        )

    ids_cpu = ids.cpu().long()
    scores = torch.softmax(logits.float(), dim=-1)
    selected_scores = scores.gather(1, ids_cpu)
    expected_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    expected_top_scores = scores.topk(top_k, dim=-1).values

    assert torch.isfinite(weights).all()
    assert (ids_cpu >= 0).all() and (ids_cpu < experts).all()
    torch.testing.assert_close(
        selected_scores.sort(dim=-1).values,
        expected_top_scores.sort(dim=-1).values,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "tokens,local_map,on_input,bias,activation,clamp",
    [
        (1, False, False, False, "silu", None),
        (17, False, False, False, "silu", None),
        (17, True, False, False, "silu", None),
        (17, True, True, False, "silu", None),
        (17, False, False, True, "silu", None),
        (17, False, False, False, "silu", 0.2),
        (17, False, False, False, "gelu", None),
        (17, False, False, False, "gelu_tanh", None),
        (17, False, False, False, "silu_no_mul", None),
        (17, False, False, False, "gelu_no_mul", None),
        (17, False, False, False, "gelu_tanh_no_mul", None),
        (0, False, False, False, "silu", None),
    ],
)
def test_modular_ascend_experts_match_cpu(
    tokens,
    local_map,
    on_input,
    bias,
    activation,
    clamp,
    monkeypatch,
):
    generator = torch.Generator().manual_seed(73)
    experts, hidden, intermediate = 8, 128, 64
    top_k = 1 if on_input else 3
    gated = not activation.endswith("_no_mul")

    def randn(*shape):
        return (torch.randn(*shape, generator=generator) * 0.1).bfloat16()

    x = randn(tokens, hidden)
    w1 = randn(experts, intermediate * (2 if gated else 1), hidden)
    w2 = randn(experts, hidden, intermediate)
    b1 = randn(experts, w1.shape[1]) if bias else None
    b2 = randn(experts, hidden) if bias else None
    # Leave expert 7 empty; duplicate expert choices exercise row restoration.
    ids = torch.randint(7, (tokens, top_k), generator=generator, dtype=torch.int32)
    weights = torch.rand(tokens, top_k, generator=generator)
    mapping = torch.tensor([2, -1, 0, -1, 4, 1, 3, 5]) if local_map else None
    expected_routes = torch.zeros(tokens, top_k, hidden, dtype=torch.bfloat16)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(ids[token, slot])
            if mapping is not None:
                expert = int(mapping[expert])
            if expert < 0:
                continue
            row = x[token]
            if on_input:
                row = row * weights[token, slot].to(row.dtype)
            gate_up = F.linear(row.float(), w1[expert].float())
            if b1 is not None:
                gate_up += b1[expert].float()
            gate_up = gate_up.bfloat16().float()
            if gated:
                gate, up = gate_up.chunk(2)
                if clamp is not None:
                    gate = gate.clamp(max=clamp)
                    up = up.clamp(-clamp, clamp)
            else:
                gate = gate_up
            act = (
                F.silu(gate)
                if activation.startswith("silu")
                else F.gelu(
                    gate, approximate="tanh" if "tanh" in activation else "none"
                )
            )
            if gated:
                act *= up
            result = F.linear(act.bfloat16().float(), w2[expert].float())
            if b2 is not None:
                result += b2[expert].float()
            if not on_input:
                result *= weights[token, slot]
            expected_routes[token, slot] = result

    def triton_must_not_run(*args, **kwargs):
        raise AssertionError("Ascend modular experts entered CUDA Triton MoE")

    monkeypatch.setattr(
        "vllm_fl.ops.fused_moe.fused_moe_utils.try_get_optimal_moe_config",
        triton_must_not_run,
    )
    quant = FusedMoEQuantConfig.make()
    stub = SimpleNamespace(
        _lora_context=None,
        quant_config=quant,
        w1_bias=None if b1 is None else b1.npu(),
        w2_bias=None if b2 is None else b2.npu(),
        activation_config=ApplyMoEActivationConfig(clamp_limit=clamp),
    )
    npu_weights, npu_ids = weights.npu(), ids.npu()
    npu_map = None if mapping is None else mapping.npu()
    output = torch.full(x.shape, 123, dtype=x.dtype, device="npu")
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    with (
        torch.inference_mode(),
        flag_gems.use_gems(include=whitelist, exclude=blacklist),
    ):
        prepared, _, _, _, _ = MoEPrepareAndFinalizeNoDPEPModular().prepare(
            x.npu(),
            npu_weights,
            npu_ids,
            experts,
            npu_map,
            on_input,
            quant,
        )
        TritonExpertsFL.apply(
            stub,
            output,
            prepared,
            w1.npu(),
            w2.npu(),
            npu_weights,
            npu_ids,
            MoEActivation(activation),
            experts,
            npu_map,
            None,
            None,
            torch.empty(0, device="npu"),
            torch.empty(0, device="npu"),
            None,
            on_input,
        )
    torch.testing.assert_close(
        output.cpu().float(),
        expected_routes.sum(dim=1).float(),
        rtol=0.03,
        atol=0.006,
    )


@pytest.mark.gpu
def test_grouped_experts_with_no_local_routes():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.grouped_moe import (
        grouped_experts,
    )

    x = torch.ones(3, 128, device="npu", dtype=torch.bfloat16)
    output = grouped_experts(
        x,
        torch.ones(2, 128, 128, device="npu", dtype=x.dtype),
        torch.ones(2, 128, 64, device="npu", dtype=x.dtype),
        torch.ones(3, 2, device="npu"),
        torch.tensor([[0, 1], [1, 0], [-1, 1]], device="npu"),
        activation="silu",
        expert_map=torch.full((2,), -1, device="npu"),
        apply_router_weight_on_input=False,
    )
    torch.testing.assert_close(output.cpu(), torch.zeros(3, 128, dtype=x.dtype))


@pytest.mark.gpu
def test_native_grouped_experts_drop_padded_routes():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.grouped_moe import grouped_experts

    generator = torch.Generator().manual_seed(32)
    x = (torch.randn(4, 128, generator=generator) * 0.1).bfloat16()
    w1 = (torch.randn(8, 128, 128, generator=generator) * 0.1).bfloat16()
    w2 = (torch.randn(8, 128, 64, generator=generator) * 0.1).bfloat16()
    ids = torch.tensor([[1, 2], [-1, -1], [3, 5], [-1, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.0, 0.0], [0.4, 0.6], [0.0, 0.0]])
    valid_rows = [0, 2]
    expected_routes = torch.zeros(2, 2, 128, dtype=torch.float32)
    for compact_row, row in enumerate(valid_rows):
        for slot in range(2):
            expert = int(ids[row, slot])
            gate, up = (
                F.linear(x[row].float(), w1[expert].float()).bfloat16().float().chunk(2)
            )
            act = (F.silu(gate) * up).bfloat16().float()
            # The second GMM also produces BF16 before unpermute applies
            # FP32 router weights and accumulates the selected experts.
            down = F.linear(act, w2[expert].float()).bfloat16().float()
            expected_routes[compact_row, slot] = down * weights[row, slot]
    expected = expected_routes.sum(dim=1).bfloat16()
    x_npu, w1_npu, w2_npu = x.npu(), w1.npu(), w2.npu()
    weights_npu, ids_npu = weights.npu(), ids.npu()
    output = grouped_experts(
        x_npu,
        w1_npu,
        w2_npu,
        weights_npu,
        ids_npu,
        activation="silu",
        expert_map=None,
        apply_router_weight_on_input=False,
    ).cpu()
    # Assert the padding contract before any numerical-reference assertion,
    # so a valid-row precision failure cannot hide a padding corruption.
    assert torch.count_nonzero(output[[1, 3]]) == 0
    compact = grouped_experts(
        x_npu[valid_rows],
        w1_npu,
        w2_npu,
        weights_npu[valid_rows],
        ids_npu[valid_rows],
        activation="silu",
        expert_map=None,
        apply_router_weight_on_input=False,
    ).cpu()
    # Both native calls evaluate exactly the same valid expert rows. This
    # comparison isolates invalid-route handling from CPU/NPU BF16 rounding.
    torch.testing.assert_close(output[valid_rows], compact, rtol=0, atol=0)
    torch.testing.assert_close(compact, expected, rtol=0.03, atol=0.006)


@pytest.mark.gpu
def test_native_grouped_experts_drop_partial_invalid_routes():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.grouped_moe import grouped_experts

    generator = torch.Generator().manual_seed(132)
    x = (torch.randn(4, 128, generator=generator) * 0.1).bfloat16()
    x[[1, 3]] = float("nan")
    w1 = (torch.randn(8, 128, 128, generator=generator) * 0.1).bfloat16().npu()
    w2 = (torch.randn(8, 128, 64, generator=generator) * 0.1).bfloat16().npu()
    ids = torch.tensor([[1, -1], [-1, -1], [99, 3], [-1, -1]], dtype=torch.int32).npu()
    # Invalid slots must be ignored independently of their unused probabilities.
    weights = torch.tensor(
        [[0.25, float("nan")], [4.0, 5.0], [7.0, 0.6], [9.0, 11.0]]
    ).npu()
    x = x.npu()
    kwargs = dict(
        activation="silu", expert_map=None, apply_router_weight_on_input=False
    )
    output = grouped_experts(x, w1, w2, weights, ids, **kwargs).cpu()
    assert torch.count_nonzero(output[[1, 3]]) == 0
    assert torch.isfinite(output).all()
    compact = grouped_experts(
        x[[0, 2]],
        w1,
        w2,
        torch.tensor([[0.25], [0.6]], device="npu"),
        torch.tensor([[1], [3]], dtype=torch.int32, device="npu"),
        **kwargs,
    ).cpu()
    torch.testing.assert_close(output[[0, 2]], compact, rtol=0, atol=0)
