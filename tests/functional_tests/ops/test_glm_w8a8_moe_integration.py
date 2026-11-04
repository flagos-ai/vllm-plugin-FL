"""Two-stage W8A8 experts: GLM TP16 dimensions and changed-input Graph."""
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("tokens", [1, 64])
def test_glm_w8a8_experts_and_graph(tokens):
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType,
        int8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
    from vllm_fl.dispatch.backends.vendor.hygon.impl.moe.triton_experts import HygonTritonExpertsFL

    torch.manual_seed(23)
    e, h, i, topk = 256, 6144, 128, 8
    w1 = torch.randint(-8, 9, (e, 2*i, h), device="cuda", dtype=torch.int8)
    w2 = torch.randint(-8, 9, (e, h, i), device="cuda", dtype=torch.int8)
    s1 = torch.full((e, 2*i, 1), .002, device="cuda")
    s2 = torch.full((e, h, 1), .02, device="cuda")
    x = (torch.randn(tokens, h, device="cuda") * .25).bfloat16()
    ids = (torch.arange(tokens*topk, device="cuda").reshape(tokens, topk) % e).int()
    tw = torch.full((tokens, topk), 1/topk, device="cuda")
    quant = int8_w8a8_moe_quant_config(
        w1_scale=s1, w2_scale=s2, a1_scale=None, a2_scale=None,
        per_act_token_quant=True)
    config = FusedMoEConfig(
        num_experts=e, experts_per_token=topk, hidden_dim=h, intermediate_size=i,
        num_local_experts=e, num_logical_experts=e, activation=MoEActivation.SILU,
        device="cuda", routing_method=RoutingMethodType.Unspecified,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )
    experts = HygonTritonExpertsFL(config, quant)
    ws1, ws2, out_shape = experts.workspace_shapes(tokens, 2*i, h, topk, e, e, None, MoEActivation.SILU)
    workspace1 = torch.empty(ws1, device="cuda", dtype=torch.bfloat16)
    workspace2 = torch.empty(ws2, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(out_shape, device="cuda", dtype=torch.bfloat16)

    def run():
        qx, sx = moe_kernel_quantize_input(x, None, torch.int8, True, None)
        experts.apply(out, qx, w1, w2, tw, ids, MoEActivation.SILU,
                      e, None, sx, None, workspace1, workspace2, None, False)
        return out

    def qdq(t):
        scale = t.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        return (t.float() / scale).round().clamp(-127, 127) * scale

    def reference():
        rows = []
        for row in range(min(tokens, 3)):
            acc = torch.zeros(h, device="cuda")
            for slot in range(topk):
                expert = int(ids[row, slot])
                first = (qdq(x[row:row+1]) @ (w1[expert].float() * s1[expert]).T).bfloat16()
                gate, up = first.float().chunk(2, dim=-1)
                activated = (F.silu(gate) * up).bfloat16()
                second = qdq(activated) @ (w2[expert].float() * s2[expert]).T
                acc += (second * tw[row, slot]).bfloat16().float().squeeze(0)
            rows.append(acc.bfloat16())
        return torch.stack(rows)

    torch.testing.assert_close(run()[:min(tokens, 3)], reference(), rtol=.03, atol=.003)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.neg_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:min(tokens, 3)], reference(), rtol=.03, atol=.003)
