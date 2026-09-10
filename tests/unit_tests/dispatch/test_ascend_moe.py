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
from vllm_fl.utils import get_flag_gems_whitelist_blacklist


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
    _, blacklist = get_flag_gems_whitelist_blacklist()
    with torch.inference_mode(), flag_gems.use_gems(exclude=blacklist):
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
