# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU regression tests for the portable sigmoid MoE top-k fallback."""

import torch

from vllm_fl.dispatch.backends.reference.impl.moe import topk_sigmoid_torch


def _run_topk(logits, topk=2, bias=None, renormalize=False):
    token_count = logits.size(0)
    weights = torch.empty(token_count, topk, dtype=torch.float32)
    indices = torch.empty(token_count, topk, dtype=torch.int32)
    source_rows = torch.empty(token_count, topk, dtype=torch.int32)
    ret_weights, ret_indices = topk_sigmoid_torch(
        weights, indices, source_rows, logits, renormalize, bias
    )
    assert ret_weights is weights and ret_indices is indices
    return weights, indices, source_rows


def test_bias_affects_ranking_but_not_weights():
    logits = torch.tensor([[0.0, 1.0, -1.0, 2.0]])
    bias = torch.tensor([0.0, -1.0, 2.0, 0.0])
    weights, indices, _ = _run_topk(logits, bias=bias)
    assert indices.tolist() == [[2, 3]]
    torch.testing.assert_close(weights[0], torch.sigmoid(logits.float())[0, [2, 3]])


def test_ties_choose_smallest_expert_and_no_duplicates():
    logits = torch.zeros(2, 4)
    weights, indices, source = _run_topk(logits, topk=3)
    assert indices.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert source.tolist() == [[0, 2, 4], [1, 3, 5]]
    torch.testing.assert_close(weights, torch.full((2, 3), 0.5))


def test_renormalization_uses_unbiased_sigmoid():
    logits = torch.tensor([[-2.0, 0.0, 2.0]])
    bias = torch.tensor([2.0, 0.0, 0.0])
    weights, ids, _ = _run_topk(logits, 2, bias, renormalize=True)
    assert ids.tolist() == [[0, 2]]
    raw = torch.sigmoid(logits[0, [0, 2]])
    torch.testing.assert_close(weights[0], raw / raw.sum())


def test_nan_and_zero_denominator_do_not_produce_nan():
    weights, indices, _ = _run_topk(
        torch.full((1, 3), float("nan")), topk=2, renormalize=True
    )
    assert indices.tolist() == [[0, 1]]
    assert weights.tolist() == [[0.0, 0.0]]


def test_float16_input_is_computed_in_float32():
    logits = torch.tensor([[1.0, -0.5, -2.0]], dtype=torch.float16)
    weights, indices, _ = _run_topk(logits, topk=2)
    assert indices.tolist() == [[0, 1]]
    torch.testing.assert_close(weights[0], torch.sigmoid(logits.float())[0, [0, 1]])


def test_cuda_native_sigmoid_availability_tracks_compiled_op(monkeypatch):
    from types import SimpleNamespace

    from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend

    backend = CudaBackend()
    monkeypatch.setattr(CudaBackend, "is_available", lambda self: True)
    monkeypatch.setattr(torch.ops, "_moe_C", SimpleNamespace(), raising=False)
    assert not backend.has_native_topk_sigmoid()
    monkeypatch.setattr(
        torch.ops, "_moe_C", SimpleNamespace(topk_sigmoid=object()), raising=False
    )
    assert backend.has_native_topk_sigmoid()
    monkeypatch.setattr(CudaBackend, "is_available", lambda self: False)
    assert not backend.has_native_topk_sigmoid()


def test_cuda_sigmoid_preserves_native_out_buffer_contract(monkeypatch):
    from vllm import _custom_ops

    from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend

    calls = []
    monkeypatch.setattr(_custom_ops, "topk_sigmoid", lambda *args: calls.append(args))
    weights = torch.empty(2, 3, dtype=torch.float32)
    indices = torch.empty(2, 3, dtype=torch.int32)
    rows = torch.empty(2, 3, dtype=torch.int32)
    logits = torch.randn(2, 8)
    bias = torch.randn(8)
    returned = CudaBackend().topk_sigmoid(weights, indices, rows, logits, True, bias)
    assert returned[0] is weights and returned[1] is indices
    assert len(calls) == 1 and len(calls[0]) == 6
    assert all(
        actual is expected
        for actual, expected in zip(calls[0][:4], (weights, indices, rows, logits))
    )
    assert calls[0][4] is True and calls[0][5] is bias


def test_topk_outside_expert_count_fails():
    """An invalid k must not silently reuse experts after all are exhausted."""
    import pytest

    logits = torch.tensor([[1.0, 2.0]])
    for invalid_k in (0, 3):
        with pytest.raises(ValueError, match="topk"):
            _run_topk(logits, topk=invalid_k)
