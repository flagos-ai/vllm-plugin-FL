# SPDX-License-Identifier: Apache-2.0
"""Focused tests for HY4 model integration contracts."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vllm_fl.models import hy_v4


class _CountingHostCopy:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def tolist(self):
        self.calls += 1
        return list(self.values)


def test_hy4_prefill_host_metadata_is_copied_once_per_chunk():
    cu_seq_lens = _CountingHostCopy([0, 4, 9])
    starts = _CountingHostCopy([0, 1, 4, 7])
    ends = _CountingHostCopy([1, 4, 7, 9])
    chunk = SimpleNamespace(
        cu_seq_lens=cu_seq_lens,
        cu_seqlen_ks=starts,
        cu_seqlen_ke=ends,
    )

    first = hy_v4._get_hyv4_prefill_host_metadata(chunk)
    second = hy_v4._get_hyv4_prefill_host_metadata(chunk)

    assert first is second
    assert first == (
        [0, 4, 9],
        [0, 1, 4, 7],
        [1, 4, 7, 9],
        [0, 4],
        [0, 0, 1, 1],
    )
    assert (cu_seq_lens.calls, starts.calls, ends.calls) == (1, 1, 1)


def test_hy4_prefill_host_metadata_is_derived_from_cpu_builder_inputs():
    chunk = SimpleNamespace(
        token_start=0,
        token_end=5,
        num_reqs=2,
    )
    common_attn_metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([6, 9], dtype=torch.int32),
    )

    hy_v4._attach_hyv4_prefill_host_metadata(
        chunk,
        common_attn_metadata,
        compress_ratio=2,
    )

    assert hy_v4._get_hyv4_prefill_host_metadata(chunk) == (
        [0, 3, 7],
        [0, 0, 3, 3, 3],
        [2, 3, 6, 7, 7],
        [0, 3],
        [0, 0, 1, 1, 1],
    )


def test_hy4_prefill_host_metadata_honors_query_slice():
    chunk = SimpleNamespace(token_start=1, token_end=2, num_reqs=1)
    common_attn_metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([6], dtype=torch.int32),
    )

    hy_v4._attach_hyv4_prefill_host_metadata(chunk, common_attn_metadata, 2)

    assert hy_v4._get_hyv4_prefill_host_metadata(chunk) == ([0, 3], [0], [3], [0], [0])


def test_hy4_sparse_index_block_n_is_thead_shape_gated(monkeypatch):
    token_indices = SimpleNamespace(
        device=SimpleNamespace(type="cuda"), ndim=2, shape=(64, 2048)
    )
    monkeypatch.setattr(hy_v4.current_platform, "vendor_name", "thead")

    assert (
        hy_v4._select_hyv4_sparse_index_block_n(token_indices, 2048, 128, False, True)
        == 2048
    )
    assert (
        hy_v4._select_hyv4_sparse_index_block_n(token_indices, 2048, 128, True, True)
        == 128
    )
    assert (
        hy_v4._select_hyv4_sparse_index_block_n(token_indices, 2048, 128, False, False)
        == 128
    )

    token_indices.shape = (64, 1024)
    assert (
        hy_v4._select_hyv4_sparse_index_block_n(token_indices, 1024, 128, False, True)
        == 128
    )
    monkeypatch.setattr(hy_v4.current_platform, "vendor_name", "nvidia")
    token_indices.shape = (64, 2048)
    assert (
        hy_v4._select_hyv4_sparse_index_block_n(token_indices, 2048, 128, False, True)
        == 128
    )


def test_hy4_prefill_topk_uses_shared_capability(monkeypatch):
    captured = {}

    def fake_topk(logits, row_starts, row_ends, indices):
        captured["logits"] = logits.clone()
        captured["row_starts"] = row_starts.clone()
        captured["row_ends"] = row_ends.clone()
        indices.fill_(-1)
        indices[0, :2].copy_(torch.tensor([2, 0], dtype=torch.int32))

    monkeypatch.setattr(hy_v4, "_top_k_per_row_prefill", fake_topk)
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
    weights = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    keys = torch.tensor([[1.0, 1.0], [2.0, -1.0], [3.0, 4.0]], dtype=torch.bfloat16)
    output = torch.empty(4, dtype=torch.int32)

    hy_v4._select_topk(q, weights, keys, topk=4, output=output)

    assert captured["logits"].shape == (1, 3)
    assert captured["logits"].dtype == torch.float32
    assert captured["row_starts"].tolist() == [0]
    assert captured["row_ends"].tolist() == [3]
    assert output.tolist() == [2, 0, -1, -1]


def test_hy4_selected_prefix_translation_preserves_invalid_tail():
    indices = torch.tensor([3, 0, 7, -1, -1], dtype=torch.int32)

    hy_v4._translate_hyv4_selected_prefix(
        indices,
        count=3,
        request_offset=11,
    )

    assert indices.tolist() == [14, 11, 18, -1, -1]


def test_hy4_blocked_prefill_topk_matches_row_reference():
    torch.manual_seed(20260914)
    q = torch.randn((5, 2, 4), dtype=torch.float32)
    weights = torch.randn((5, 2), dtype=torch.float32)
    keys = torch.randn((12, 4), dtype=torch.float32)
    starts = torch.tensor([2, 2, 4, 6, 6], dtype=torch.int32)
    ends = torch.tensor([2, 5, 8, 8, 12], dtype=torch.int32)
    topk = 4
    request_start = 2
    output = torch.empty((5, topk), dtype=torch.int32)

    hy_v4._select_topk_block(
        q,
        weights,
        keys,
        starts,
        ends,
        key_start=2,
        key_end=12,
        request_start=request_start,
        topk=topk,
        output=output,
    )

    expected = torch.full_like(output, -1)
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        count = min(topk, end - start)
        if count == 0:
            continue
        scores = torch.matmul(keys[start:end], q[row].transpose(0, 1))
        logits = (
            (torch.relu(scores) * weights[row].to(scores.dtype).unsqueeze(0))
            .sum(dim=-1)
            .float()
        )
        selected = torch.topk(logits, count, sorted=True).indices.to(torch.int32)
        expected[row, :count].copy_(selected + start - request_start)

    assert torch.equal(output.sort(dim=1).values, expected.sort(dim=1).values)
    assert output[0].tolist() == [-1, -1, -1, -1]


def test_hy4_blocked_prefill_topk_uses_row_ranges(monkeypatch):
    captured = {}

    def fake_topk(logits, row_starts, row_ends, indices):
        captured["shape"] = logits.shape
        captured["starts"] = row_starts.tolist()
        captured["ends"] = row_ends.tolist()
        indices.fill_(-1)
        indices[0, :2].copy_(torch.tensor([0, 1], dtype=torch.int32))
        indices[1, :2].copy_(torch.tensor([1, 0], dtype=torch.int32))

    def forbidden_topk(*args, **kwargs):
        raise AssertionError("blocked Indexer must use row-wise Top-K")

    monkeypatch.setattr(hy_v4, "_top_k_per_row_prefill", fake_topk)
    monkeypatch.setattr(torch, "topk", forbidden_topk)
    q = torch.randn((2, 2, 4), dtype=torch.float32)
    weights = torch.randn((2, 2), dtype=torch.float32)
    keys = torch.randn((6, 4), dtype=torch.float32)
    starts = torch.tensor([2, 3], dtype=torch.int32)
    ends = torch.tensor([4, 6], dtype=torch.int32)
    output = torch.empty((2, 4), dtype=torch.int32)

    hy_v4._select_topk_block(
        q,
        weights,
        keys,
        starts,
        ends,
        key_start=2,
        key_end=6,
        request_start=1,
        topk=4,
        output=output,
    )

    assert captured == {"shape": (2, 4), "starts": [0, 1], "ends": [2, 4]}
    assert output.tolist() == [[1, 2, -1, -1], [3, 2, -1, -1]]


def test_hy4_full_range_indices_are_request_relative():
    starts = torch.tensor([2, 4, 7], dtype=torch.int32)
    ends = torch.tensor([2, 7, 9], dtype=torch.int32)
    output = torch.empty((3, 4), dtype=torch.int32)

    hy_v4._fill_hyv4_full_range_indices(starts, ends, request_start=2, output=output)

    assert output.tolist() == [
        [-1, -1, -1, -1],
        [2, 3, 4, -1],
        [5, 6, -1, -1],
    ]


def test_hy4_blocked_prefill_topk_empty_block():
    output = torch.empty((0, 4), dtype=torch.int32)
    hy_v4._select_topk_block(
        torch.empty((0, 2, 4)),
        torch.empty((0, 2)),
        torch.empty((0, 4)),
        torch.empty((0,), dtype=torch.int32),
        torch.empty((0,), dtype=torch.int32),
        key_start=0,
        key_end=0,
        request_start=0,
        topk=4,
        output=output,
    )
    assert output.shape == (0, 4)


class _FakeGate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeSharedExperts(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeRoutedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_num_experts = 8


class _FakeMoERunner(nn.Module):
    def __init__(self):
        super().__init__()
        self.routed_experts = _FakeRoutedExperts()


def test_hyv4_native_router_mm_is_shape_precision_and_backend_gated(monkeypatch):
    monkeypatch.setattr(hy_v4.current_platform, "vendor_name", "thead")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(hy_v4, "use_flaggems_op", lambda name: False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    args = {
        "input_size": 6144,
        "output_size": 256,
        "bias": False,
        "params_dtype": torch.float32,
        "out_dtype": torch.float32,
    }
    assert hy_v4._can_use_hyv4_native_router_mm(**args)

    for key, value in (
        ("input_size", 4096),
        ("output_size", 128),
        ("bias", True),
        ("params_dtype", torch.bfloat16),
        ("out_dtype", torch.bfloat16),
    ):
        changed = dict(args)
        changed[key] = value
        assert not hy_v4._can_use_hyv4_native_router_mm(**changed)

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    assert not hy_v4._can_use_hyv4_native_router_mm(**args)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")
    assert not hy_v4._can_use_hyv4_native_router_mm(**args)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(hy_v4, "use_flaggems_op", lambda name: True)
    assert not hy_v4._can_use_hyv4_native_router_mm(**args)
    monkeypatch.setattr(hy_v4.current_platform, "vendor_name", "nvidia")
    assert not hy_v4._can_use_hyv4_native_router_mm(**args)


def test_hyv4_router_linear_native_path_calls_mm(monkeypatch):
    layer = object.__new__(hy_v4.HYV4RouterLinear)
    nn.Module.__init__(layer)
    layer.weight = nn.Parameter(torch.randn(256, 6144, dtype=torch.float32))
    layer._use_thead_native_router_mm = True
    x = torch.randn(2, 6144, dtype=torch.bfloat16)
    expected = torch.mm(x.float(), layer.weight.T)
    original_mm = torch.mm
    calls = []

    def counted_mm(a, b):
        calls.append((a, b))
        return original_mm(a, b)

    monkeypatch.setattr(torch, "mm", counted_mm)
    output, output_bias = layer(x)

    assert output_bias is None
    assert len(calls) == 1
    assert calls[0][0].dtype == torch.float32
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_hyv4_router_linear_non_target_dtype_falls_back(monkeypatch):
    layer = object.__new__(hy_v4.HYV4RouterLinear)
    nn.Module.__init__(layer)
    layer.weight = nn.Parameter(torch.randn(256, 6144, dtype=torch.float32))
    layer._use_thead_native_router_mm = True
    sentinel = torch.empty(1)
    monkeypatch.setattr(
        hy_v4.GateLinear,
        "forward",
        lambda self, x: (sentinel, None),
    )

    output, output_bias = layer(torch.empty(2, 6144, dtype=torch.float16))

    assert output is sentinel
    assert output_bias is None


def test_hy4_moe_resolves_patched_factory_at_construction(monkeypatch):
    calls = []

    def patched_factory(*args, **kwargs):
        runner = _FakeMoERunner()
        calls.append((runner, args, kwargs))
        return runner

    monkeypatch.setattr(hy_v4, "HYV4RouterLinear", _FakeGate)
    monkeypatch.setattr(hy_v4, "HYV4DenseMLP", _FakeSharedExperts)
    monkeypatch.setattr(hy_v4.fused_moe, "FusedMoE", patched_factory)

    config = SimpleNamespace(
        hidden_act="silu",
        hidden_size=16,
        n_routed_experts=8,
        moe_intermediate_size=32,
        n_shared_experts=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
    )
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        enable_eplb=False,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        quant_config=None,
    )

    layer = hy_v4.HYV4MoE(config, vllm_config, prefix="model.layers.0.mlp")

    assert len(calls) == 1
    assert layer.experts is calls[0][0]
    assert calls[0][2]["num_experts"] == config.n_routed_experts
    assert layer.n_local_physical_experts == 8
