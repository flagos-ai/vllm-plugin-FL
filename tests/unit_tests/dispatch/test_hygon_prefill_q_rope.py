# Copyright (c) 2026 BAAI. All rights reserved.
"""Prefill Q-RoPE: production dispatch, exact replay, and decode isolation."""

import pytest
import torch


@pytest.fixture(scope="module")
def impl():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Hygon/HIP required")
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import bf16_indexer
    return bf16_indexer


@pytest.mark.parametrize("m", [257, 1024, 8131, 8192])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
def test_prefill_production_graph_exact(impl, monkeypatch, m, strided, table_dtype):
    torch.manual_seed(916)
    q = torch.randn(m, 128 if strided else 64, 128, device="cuda", dtype=torch.bfloat16)
    if strided:
        q = q[:, ::2]
    weights = torch.randn(m, 64, device="cuda", dtype=torch.float32 if strided else torch.bfloat16)
    positions = torch.arange(57344, 57344 + m, device="cuda", dtype=torch.int64)
    angles = torch.randn(131072, 32, device="cuda")
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(table_dtype)
    args = (positions, q, table, weights, 128**-0.5, 64**-0.5)
    # Exercise the actual wrapper, including the disable switch and allocation.
    monkeypatch.setenv("VLLM_FL_HYGON_PREFILL_Q_ROPE", "0")
    impl.fused_indexer_q_rope_bf16(*args)
    old_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph):
        expected = impl.fused_indexer_q_rope_bf16(*args)
    monkeypatch.setenv("VLLM_FL_HYGON_PREFILL_Q_ROPE", "1")
    impl.fused_indexer_q_rope_bf16(*args)
    new_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(new_graph):
        actual = impl.fused_indexer_q_rope_bf16(*args)
    for offset in [0, 16384, 57344]:
        positions.copy_(torch.arange(offset, offset + m, device="cuda"))
        q.normal_()
        weights.normal_()
        old_graph.replay()
        new_graph.replay()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert all(torch.isfinite(a).all() for a in actual)


@pytest.mark.parametrize("m", [16, 24, 32, 48, 64])
def test_decode_keeps_original_leaf(impl, monkeypatch, m):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import prefill_q_rope
    def forbidden(*args, **kwargs):
        pytest.fail("decode entered prefill-only leaf")
    monkeypatch.setattr(prefill_q_rope, "prefill_indexer_q_rope_bf16_out", forbidden)
    monkeypatch.setenv("VLLM_FL_HYGON_PREFILL_Q_ROPE", "1")
    q = torch.randn(m, 64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
    p = torch.zeros(m, device="cuda", dtype=torch.int64)
    table = torch.cat((torch.ones(1, 32, device="cuda"), torch.zeros(1, 32, device="cuda")), dim=-1)
    out, scale = impl.fused_indexer_q_rope_bf16(p, q, table, w, 0.5, 0.25)
    assert torch.equal(out, q)
    assert torch.equal(scale, w.float() * 0.125)


@pytest.mark.parametrize("heads,dtype", [(8, torch.bfloat16), (64, torch.float32)])
def test_other_shapes_keep_original_leaf(impl, monkeypatch, heads, dtype):
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import prefill_q_rope
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported shape entered specialized leaf")
    monkeypatch.setattr(prefill_q_rope, "prefill_indexer_q_rope_bf16_out", forbidden)
    q = torch.randn(512, heads, 128, device="cuda", dtype=dtype)
    w = torch.randn(512, heads, device="cuda", dtype=torch.bfloat16)
    p = torch.zeros(512, device="cuda", dtype=torch.int64)
    table = torch.cat((torch.ones(1, 32, device="cuda"), torch.zeros(1, 32, device="cuda")), dim=-1)
    out, scale = impl.fused_indexer_q_rope_bf16(p, q, table, w, 0.5, 0.25)
    assert torch.equal(out, q.to(torch.bfloat16))
    assert torch.equal(scale, w.float() * 0.125)
