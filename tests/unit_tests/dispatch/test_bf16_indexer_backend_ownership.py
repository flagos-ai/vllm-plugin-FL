# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the composite BF16 Indexer decode backend."""

from __future__ import annotations

import importlib

import pytest
import torch


@pytest.mark.parametrize(
    ("candidate_indices", "expected"),
    [
        ([0], [0]),
        ([0, 2, 1], [1, 2, 0]),
        ([3, 0, 2, 1], [1, 2, 0, 3]),
    ],
)
def test_flaggems_decode_owns_candidate_ordering(
    monkeypatch, candidate_indices, expected
):
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.bf16_indexer"
    )
    logits = torch.tensor([[0.5, 4.0, 2.0, -1.0]], dtype=torch.float32)

    monkeypatch.setattr(
        module, "_bf16_paged_mqa_logits_flaggems", lambda *args, **kwargs: logits
    )

    def fake_topk(_logits, _seq_lens, indices, *, next_n):
        assert next_n == 1
        indices.copy_(torch.tensor([candidate_indices], dtype=torch.int32))

    monkeypatch.setattr(module, "_bf16_indexer_topk_flaggems", fake_topk)

    def reject_global_topk(*args, **kwargs):
        raise AssertionError("candidate ordering must not re-enter global torch.topk")

    monkeypatch.setattr(torch, "topk", reject_global_topk)
    indices = torch.empty((1, len(candidate_indices)), dtype=torch.int32)
    module.bf16_indexer_decode_flaggems(
        None,
        None,
        None,
        torch.tensor([4], dtype=torch.int32),
        None,
        None,
        indices,
        next_n=1,
        max_context_len=4,
    )
    assert indices.tolist() == [expected]


def test_reference_decode_honors_each_valid_length(monkeypatch):
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.reference.impl.bf16_indexer"
    )
    logits = torch.tensor(
        [[1.0, 5.0, 3.0, 99.0], [7.0, 80.0, 60.0, 40.0]],
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        module, "_bf16_paged_mqa_logits_torch", lambda *args, **kwargs: logits
    )
    q = torch.empty((1, 2, 1, 1), dtype=torch.bfloat16)
    indices = torch.empty((2, 3), dtype=torch.int32)
    module.bf16_indexer_decode_torch(
        q,
        None,
        None,
        torch.tensor([[3, 1]], dtype=torch.int32),
        None,
        None,
        indices,
        next_n=2,
        max_context_len=4,
    )
    assert indices.tolist() == [[1, 2, 0], [0, -1, -1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires accelerator")
def test_flaggems_decode_graph_replay_uses_changed_logits(monkeypatch):
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.bf16_indexer"
    )
    device = torch.device("cuda")
    logits = torch.tensor(
        [[1.0, 5.0, 3.0, 2.0, 4.0, -1.0], [6.0, 2.0, 5.0, 1.0, -1.0, -2.0]],
        device=device,
        dtype=torch.float32,
    )
    seq_lens = torch.tensor([5], device=device, dtype=torch.int32)
    indices = torch.empty((2, 3), device=device, dtype=torch.int32)
    monkeypatch.setattr(
        module, "_bf16_paged_mqa_logits_flaggems", lambda *args, **kwargs: logits
    )

    def run():
        module.bf16_indexer_decode_flaggems(
            None,
            None,
            None,
            seq_lens,
            None,
            None,
            indices,
            next_n=2,
            max_context_len=6,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    first = indices.cpu().clone()

    logits.copy_(
        torch.tensor(
            [[9.0, 1.0, 8.0, 2.0, 7.0, -1.0], [1.0, 9.0, 2.0, 8.0, -1.0, -2.0]],
            device=device,
        )
    )
    graph.replay()
    torch.cuda.synchronize()
    second = indices.cpu()
    assert not torch.equal(first, second)
    for row, valid_len in enumerate((4, 5)):
        chosen = second[row].to(torch.long)
        assert bool(((chosen >= 0) & (chosen < valid_len)).all())
        scores = logits[row].cpu().index_select(0, chosen)
        assert bool((scores[:-1] >= scores[1:]).all())
