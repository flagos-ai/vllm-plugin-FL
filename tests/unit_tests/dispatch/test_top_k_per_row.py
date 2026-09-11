# SPDX-License-Identifier: Apache-2.0
"""Shared row-wise top-k capability tests."""

from __future__ import annotations

import importlib

import pytest
import torch

from vllm_fl.dispatch.registry import OpRegistry


def test_shared_topk_capabilities_are_registered(monkeypatch):
    flaggems_registration = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.register_ops"
    )
    reference_registration = importlib.import_module(
        "vllm_fl.dispatch.backends.reference.register_ops"
    )
    monkeypatch.setattr(flaggems_registration, "use_flaggems_op", lambda _: True)
    registry = OpRegistry()

    flaggems_registration.register_builtins(registry)
    reference_registration.register_builtins(registry)

    entries = registry.snapshot().impls_by_op
    for op_name in ("top_k_per_row_prefill", "top_k_per_row_decode"):
        assert {impl.impl_id for impl in entries[op_name]} == {
            "default.flagos",
            "reference.torch",
        }


def test_flaggems_decode_uses_explicit_2d_lengths(monkeypatch):
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.top_k_per_row"
    )
    fused = importlib.import_module("flag_gems.fused")
    captured = {}

    def fake_topk(logits, next_n, seq_lens, indices, *shape_args):
        captured["next_n"] = next_n
        captured["seq_lens"] = seq_lens.clone()
        captured["shape_args"] = shape_args

    monkeypatch.setattr(fused, "top_k_per_row_decode", fake_topk)
    logits = torch.empty((2, 8), dtype=torch.float32)
    seq_lens = torch.tensor([[3, 1]], dtype=torch.int32)
    indices = torch.empty((2, 2), dtype=torch.int32)

    module.top_k_per_row_decode(logits, seq_lens, indices, next_n=2)

    assert captured["next_n"] == 1
    assert captured["seq_lens"].tolist() == [3, 1]
    assert captured["shape_args"] == (2, 8, 1, 2)


def test_flaggems_decode_preserves_final_length_abi(monkeypatch):
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.top_k_per_row"
    )
    fused = importlib.import_module("flag_gems.fused")
    captured = {}

    def fake_topk(logits, next_n, seq_lens, indices, *shape_args):
        captured["next_n"] = next_n
        captured["seq_lens"] = seq_lens.clone()

    monkeypatch.setattr(fused, "top_k_per_row_decode", fake_topk)
    logits = torch.empty((4, 8), dtype=torch.float32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    indices = torch.empty((4, 2), dtype=torch.int32)

    module.top_k_per_row_decode(logits, seq_lens, indices, next_n=2)

    assert captured["next_n"] == 2
    assert captured["seq_lens"].tolist() == [5, 7]


def test_flaggems_prefill_derives_provider_abi(monkeypatch):
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.top_k_per_row"
    )
    fused = importlib.import_module("flag_gems.fused")
    captured = {}

    def fake_topk(logits, starts, ends, indices, *shape_args):
        captured["starts"] = starts.clone()
        captured["ends"] = ends.clone()
        captured["shape_args"] = shape_args

    monkeypatch.setattr(fused, "top_k_per_row_prefill", fake_topk)
    logits = torch.empty((2, 8), dtype=torch.float32)
    starts = torch.tensor([1, 3], dtype=torch.int32)
    ends = torch.tensor([6, 8], dtype=torch.int32)
    indices = torch.empty((2, 3), dtype=torch.int32)

    module.top_k_per_row_prefill(logits, starts, ends, indices)

    assert captured["starts"].tolist() == [1, 3]
    assert captured["ends"].tolist() == [6, 8]
    assert captured["shape_args"] == (2, 8, 1, 3)


def test_reference_decode_honors_explicit_and_derived_lengths():
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.reference.impl.top_k_per_row"
    )
    logits = torch.tensor(
        [[1.0, 5.0, 3.0, 99.0], [7.0, 80.0, 60.0, 40.0]],
        dtype=torch.float32,
    )
    indices = torch.empty((2, 3), dtype=torch.int32)

    module.top_k_per_row_decode(
        logits, torch.tensor([[3, 1]], dtype=torch.int32), indices, next_n=2
    )
    assert indices.tolist() == [[1, 2, 0], [0, -1, -1]]

    module.top_k_per_row_decode(
        logits, torch.tensor([3], dtype=torch.int32), indices, next_n=2
    )
    assert indices.tolist() == [[1, 0, -1], [1, 2, 0]]


def test_reference_prefill_returns_request_relative_indices():
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.reference.impl.top_k_per_row"
    )
    logits = torch.tensor(
        [[1.0, 5.0, 3.0, 4.0], [9.0, 8.0, 7.0, 6.0]], dtype=torch.float32
    )
    indices = torch.empty((2, 2), dtype=torch.int32)
    module.top_k_per_row_prefill(
        logits,
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([4, 2], dtype=torch.int32),
        indices,
    )
    assert indices.tolist() == [[0, 2], [0, 1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires accelerator")
def test_flaggems_prefill_matches_request_relative_topk_set():
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.top_k_per_row"
    )
    device = torch.device("cuda")
    torch.manual_seed(31)
    logits = torch.randn((4, 128), device=device, dtype=torch.float32)
    starts = torch.tensor([0, 3, 11, 40], device=device, dtype=torch.int32)
    ends = torch.tensor([3, 19, 76, 128], device=device, dtype=torch.int32)
    indices = torch.empty((4, 8), device=device, dtype=torch.int32)

    module.top_k_per_row_prefill(logits, starts, ends, indices)
    torch.cuda.synchronize()

    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        count = min(indices.shape[1], end - start)
        chosen = indices[row, :count].to(torch.long)
        assert bool(((chosen >= 0) & (chosen < end - start)).all())
        actual = logits[row, start:end].index_select(0, chosen).sort().values
        expected = torch.topk(logits[row, start:end], count).values.sort().values
        torch.testing.assert_close(actual, expected)
        assert bool((indices[row, count:] == -1).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires accelerator")
def test_flaggems_decode_2d_lengths_graph_replay():
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.top_k_per_row"
    )
    device = torch.device("cuda")
    torch.manual_seed(7)
    logits = torch.randn((4, 128), device=device, dtype=torch.float32)
    seq_lens = torch.tensor([[11, 7], [13, 9]], device=device, dtype=torch.int32)
    indices = torch.empty((4, 4), device=device, dtype=torch.int32)

    def run():
        module.top_k_per_row_decode(logits, seq_lens, indices, next_n=2)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    first = indices.clone()

    logits.copy_(torch.randn_like(logits))
    seq_lens.copy_(torch.tensor([[8, 12], [6, 10]], device=device, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    second = indices.clone()
    assert not torch.equal(first, second)

    for row, valid_len in enumerate(seq_lens.reshape(-1).tolist()):
        chosen = second[row].to(torch.long)
        assert bool(((chosen >= 0) & (chosen < valid_len)).all())
        actual = logits[row].index_select(0, chosen).sort(descending=True).values
        expected = torch.topk(logits[row, :valid_len], 4).values
        torch.testing.assert_close(actual, expected)
