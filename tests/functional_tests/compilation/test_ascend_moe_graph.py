# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.mark.parametrize("tokens", [1, 8])
@torch.inference_mode()
def test_grouped_moe_capture_reads_live_expert_routes(tokens):
    from vllm_fl.dispatch.backends.vendor.ascend.impl.grouped_moe import grouped_experts

    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(37)
    experts, hidden, intermediate, topk = 8, 128, 64, 2
    w1_cpu = (
        torch.randn(experts, 2 * intermediate, hidden, generator=generator) * 0.1
    ).bfloat16()
    w2_cpu = (
        torch.randn(experts, hidden, intermediate, generator=generator) * 0.1
    ).bfloat16()
    w1, w2 = w1_cpu.npu(), w2_cpu.npu()
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16, device="npu")
    ids = torch.zeros(tokens, topk, dtype=torch.int32, device="npu")
    weights = torch.full((tokens, topk), 0.5, device="npu")

    def forward():
        return grouped_experts(
            x,
            w1,
            w2,
            weights,
            ids,
            activation="silu",
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = forward()

    for active_experts in (1, 8, 3):
        x_cpu = torch.randn(tokens, hidden, generator=generator).bfloat16()
        ids_cpu = torch.randint(active_experts, (tokens, topk), generator=generator)
        weights_cpu = torch.rand(tokens, topk, generator=generator)
        weights_cpu /= weights_cpu.sum(-1, keepdim=True)
        x.copy_(x_cpu)
        ids.copy_(ids_cpu)
        weights.copy_(weights_cpu)

        expected = torch.zeros_like(x_cpu, dtype=torch.float32)
        for token in range(tokens):
            for route in range(topk):
                expert = ids_cpu[token, route]
                gate, up = (
                    F.linear(x_cpu[token].float(), w1_cpu[expert].float())
                    .bfloat16()
                    .float()
                    .chunk(2)
                )
                activated = (F.silu(gate) * up).bfloat16().float()
                expected[token] += (
                    F.linear(activated, w2_cpu[expert].float())
                    * weights_cpu[token, route]
                )

        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu().float(), expected, rtol=0.03, atol=0.03)
