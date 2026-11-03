# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch

from vllm_fl.ops.fused_moe import router

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("indices_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("padding", ["none", "mixed", "all"])
def test_gcu_topk_softmax_preserves_padding_semantics(
    device, monkeypatch, indices_type, renormalize, padding, use_graph, record_property
):
    if device.type != "gcu":
        pytest.skip("Enflame routing contract")
    logits_cpu = torch.arange(17 * 8, dtype=torch.float32).reshape(17, 8) / 19
    logits = logits_cpu.to(device)
    hidden = torch.zeros(17, 4, device=device, dtype=torch.bfloat16)
    mask_cpu = torch.zeros(17, dtype=torch.bool)
    if padding == "mixed":
        mask_cpu[1::3] = True
    elif padding == "all":
        mask_cpu[:] = True
    mask = mask_cpu.to(device)
    monkeypatch.setattr(router, "_get_padding_mask", lambda count: mask[:count])

    def cuda_fallback(**kwargs):
        pytest.fail("GCU padding must not enter a CUDA-only routing operator")

    monkeypatch.setattr(router, "upstream_fused_topk", cuda_fallback)

    def apply():
        return router.fused_topk(
            hidden, logits, 2, renormalize, indices_type=indices_type
        )

    def check(weights, ids, values, row_mask):
        torch.gcu.synchronize()
        expected_weights, expected_ids = values.softmax(-1).topk(2, dim=-1)
        if renormalize:
            expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True)
        expected_ids[row_mask] = -1
        torch.testing.assert_close(weights.cpu(), expected_weights)
        assert ids.dtype == indices_type
        torch.testing.assert_close(ids.cpu(), expected_ids.to(indices_type))

    weights, ids, _ = apply()
    check(weights, ids, logits_cpu, mask_cpu)
    if use_graph:
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            weights, ids, _ = apply()
        for revision in (1, 2):
            values = logits_cpu.flip(-1) if revision == 1 else logits_cpu
            row_mask = mask_cpu.roll(revision)
            logits.copy_(values)
            mask.copy_(row_mask)
            graph.replay()
            check(weights, ids, values, row_mask)
        record_property("gcu_padded_router_graph_replays", 2)
