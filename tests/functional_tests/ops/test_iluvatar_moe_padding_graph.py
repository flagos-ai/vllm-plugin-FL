# Copyright (c) 2026 BAAI. All rights reserved.
"""Exercise the Iluvatar biased router and provider together during replay."""

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_biased_softmax_padding_align_graph(monkeypatch, dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires Iluvatar CUDA Graph support")
    if "Iluvatar" not in torch.cuda.get_device_name():
        pytest.skip("Iluvatar adapter regression")

    from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle

    from vllm.forward_context import ForwardContext, override_forward_context

    from vllm_fl.dispatch.backends.vendor.iluvatar.moe_router import topk_softmax

    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    rows, experts, topk, block = 32, 10, 4, 16
    logits = torch.empty((rows, experts), device="cuda")
    bias = torch.linspace(-0.2, 0.2, experts, device="cuda")
    padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
    weights = torch.empty((rows, topk), device="cuda")
    ids = torch.empty((rows, topk), device="cuda", dtype=dtype)
    source = torch.empty_like(ids, dtype=torch.int32)
    context = ForwardContext({}, {}, {}, is_padding=padding)

    def run():
        topk_softmax(weights, ids, source, logits, True, bias)
        return moe_align_block_size_no_tle(ids, block, experts)

    logits.normal_()
    with override_forward_context(context):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sorted_ids, expert_ids, count = run()
    for step, valid_rows in enumerate((32, 17, 1, 0, 23)):
        values = torch.sin(torch.arange(rows * experts).reshape(rows, experts) + step)
        logits.copy_(values)
        padding.copy_(torch.arange(rows) >= valid_rows)
        graph.replay()
        torch.cuda.synchronize()
        scores = values.softmax(-1)
        expected_ids = (scores + bias.cpu()).topk(topk, dim=-1).indices
        expected_weights = scores.gather(1, expected_ids)
        expected_weights /= expected_weights.sum(-1, keepdim=True)
        torch.testing.assert_close(weights.cpu(), expected_weights)
        expected_ids[valid_rows:] = -1
        assert torch.equal(ids.cpu().long(), expected_ids)
        expected_count = sum(
            ((int((expected_ids == e).sum()) + block - 1) // block) * block
            for e in range(experts)
        )
        assert count.item() == expected_count
        actual = sorted_ids[:expected_count].cpu().tolist()
        blocks = expert_ids[: expected_count // block].cpu().tolist()
        flat = expected_ids.flatten().tolist()
        seen = []
        for offset, token in enumerate(actual):
            if token < rows * topk:
                assert 0 <= token < valid_rows * topk
                assert flat[token] == blocks[offset // block]
                seen.append(token)
        assert sorted(seen) == list(range(valid_rows * topk))
