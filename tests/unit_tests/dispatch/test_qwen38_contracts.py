# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 integration and contract regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl import qwen3_8_flash_next as dispatch_ops
from vllm_fl.dispatch.backends.vendor.metax.impl import (
    qwen3_8_flash_next as dispatch_metax,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.qwen3_8_flash_next import (
    qsa_policy as dispatch_qsa_policy,
)


def dispatch_expansion_oracle(blocks, positions, lengths, requests, ratio, budget):
    result = []
    for row, position, request in zip(
        blocks.tolist(), positions.tolist(), requests.tolist()
    ):
        values = []
        length = lengths[request].item() if 0 <= request < lengths.numel() else 0
        complete = max(
            0, min((position + 1) // ratio, length // ratio, budget // ratio)
        )
        for block in row[:complete]:
            values.extend(
                token if 0 <= token < length and token <= position else -1
                for token in range(block * ratio, (block + 1) * ratio)
            )
        tail_start = max(0, (position + 1) // ratio) * ratio
        for token in range(tail_start, position + 1):
            values.append(token if 0 <= token < length else -1)
        width = budget + ratio - 1
        result.append((values + [-1] * width)[:width])
    return torch.tensor(result, dtype=torch.int32).reshape(
        blocks.shape[0], budget + ratio - 1
    )


@pytest.mark.parametrize("ratio,budget", [(1, 3), (2, 6), (4, 8), (4, 2048)])
def test_expand_matches_independent_causal_tail_oracle(ratio, budget):
    count = budget // ratio
    blocks = torch.arange(count - 1, -1, -1, dtype=torch.int32).repeat(8, 1)
    positions = torch.tensor([0, ratio - 1, ratio, budget + 2, -1, 7, 5, 3])
    lengths = torch.tensor([1, ratio, ratio + 1, budget + 7, 20, 8])
    requests = torch.tensor([0, 1, 2, 3, 4, -1, 99, 5])
    expected = dispatch_expansion_oracle(
        blocks, positions, lengths, requests, ratio, budget
    )
    out = torch.full_like(expected, -123)
    actual = dispatch_ops.expand_qsa_block_indices(
        blocks, positions, lengths, requests, ratio, budget, out
    )
    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_expand_handles_invalid_blocks_empty_requests_and_strides():
    blocks = torch.tensor([[-1, 30], [0, 1]], dtype=torch.int64)
    positions = torch.tensor([7, 5], dtype=torch.int32)
    requests = torch.tensor([0, 1], dtype=torch.int32)
    lengths = torch.tensor([8, 0, 6, 0], dtype=torch.int32)[::2]
    expected = dispatch_expansion_oracle(blocks, positions, lengths, requests, 4, 8)
    actual = dispatch_ops.expand_qsa_block_indices(
        blocks, positions, lengths, requests, 4, 8
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    empty_lengths = lengths[:0]
    actual = dispatch_ops.expand_qsa_block_indices(
        blocks, positions, empty_lengths, requests, 4, 8
    )
    assert torch.equal(actual, torch.full_like(actual, -1))
    empty = dispatch_ops.expand_qsa_block_indices(
        blocks[:0], positions[:0], lengths, requests[:0], 4, 8
    )
    assert empty.shape == (0, 11)


@pytest.mark.parametrize("ratio,budget", [(0, 8), (4, 0), (4, 7)])
def test_expand_rejects_invalid_geometry(ratio, budget):
    with pytest.raises(ValueError, match="divisible"):
        dispatch_ops.expand_qsa_block_indices(
            torch.zeros(1, 2, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            ratio,
            budget,
        )


def test_selection_uses_fgv_scores_and_topk_with_static_width(monkeypatch):
    scores = torch.tensor(
        [
            [1.0, 5.0, 3.0, 8.0, 2.0, 6.0],
            [2.0, 1.0, -99.0, -99.0, -99.0, -99.0],
        ]
    )
    visible = torch.tensor([6, 2], dtype=torch.int32)
    calls = []

    def score(*args):
        calls.append(("score", args[7]))
        return scores, visible

    def topk(logits, starts, ends, indices, rows, stride0, stride1, k):
        calls.append(("topk", rows, stride0, stride1, k))
        assert torch.equal(starts, torch.zeros(2, dtype=torch.int32))
        assert torch.equal(ends, visible)
        # Independent result for these inputs, including the short row.
        indices.copy_(torch.tensor([[3, 5], [0, 1]], dtype=torch.int32))

    functions = {"qsa_mqa_paged": score, "top_k_per_row_prefill": topk}
    monkeypatch.setattr(dispatch_ops, "get_flaggems_vllm_operator", functions.get)
    q = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
    cache = torch.zeros(3, 2, 1, 128, dtype=torch.bfloat16)
    table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    requests = torch.tensor([0, 0], dtype=torch.int32)
    positions = torch.tensor([25, 9], dtype=torch.int64)
    lengths = torch.tensor([26], dtype=torch.int64)
    expected = dispatch_expansion_oracle(
        torch.tensor([[3, 5], [0, 1]], dtype=torch.int32),
        positions,
        lengths,
        requests,
        4,
        8,
    )
    out = torch.empty_like(expected)

    def forbid_host_read(*args, **kwargs):
        raise AssertionError("Device tensor readback is not graph-safe")

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "item", forbid_host_read)
        scoped.setattr(torch.Tensor, "tolist", forbid_host_read)
        actual = dispatch_ops.qsa_select_paged_tokens(
            q, cache, table, requests, positions, lengths, 8, 4, out
        )
    assert actual is out
    assert calls == [("score", 6), ("topk", 2, 6, 1, 2)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_selection_empty_batch_does_not_resolve_kernels(monkeypatch):
    monkeypatch.setattr(
        dispatch_ops,
        "get_flaggems_vllm_operator",
        lambda name: pytest.fail(f"Unexpected kernel lookup: {name}"),
    )
    actual = dispatch_ops.qsa_select_paged_tokens(
        torch.empty(0, 4, 128),
        torch.empty(1, 4, 1, 128),
        torch.empty(1, 1, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        2048,
        4,
    )
    assert actual.shape == (0, 2051)


def test_mutable_operation_does_not_retry_after_failure(monkeypatch):
    calls = []

    def failed(*args, **kwargs):
        calls.append("called")
        raise RuntimeError("failure after a partial state write")

    monkeypatch.setattr(dispatch_ops, "get_flaggems_vllm_operator", lambda name: failed)
    with pytest.raises(RuntimeError, match="partial state"):
        dispatch_ops.qsa_store_cache_rows(
            torch.empty(1), torch.empty(1), torch.empty(1)
        )
    assert calls == ["called"]


def test_packed_gdn_rejects_spec_state_index_matrix_before_kernel(monkeypatch):
    monkeypatch.setattr(
        dispatch_ops,
        "get_flaggems_vllm_operator",
        lambda name: pytest.fail("Kernel resolved"),
    )
    with pytest.raises(ValueError, match="non-spec"):
        dispatch_ops.gdn_packed_decode(
            torch.empty(2, 16),
            None,
            None,
            None,
            None,
            1.0,
            None,
            None,
            torch.zeros(2, 2, dtype=torch.int32),
        )


def test_availability_is_per_symbol_and_honors_fgv_switch(monkeypatch):
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    monkeypatch.setattr(
        dispatch_ops,
        "get_flaggems_vllm_operator",
        lambda name: (lambda *a: None) if name == "qsa_mqa_paged" else None,
    )
    assert dispatch_ops.is_available("qsa_mqa_paged")
    assert not dispatch_ops.is_available("qsa_select_paged_tokens")
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    assert not dispatch_ops.is_available("qsa_mqa_paged")


def test_registration_preserves_default_and_scopes_vendor_difference():
    entries = []

    class Registry:
        def register_many(self, values):
            entries.extend(values)

    dispatch_ops.register_qwen38_operators(
        Registry(),
        allow=lambda name: (
            name
            in {
                "qsa_mqa_paged",
                "qsa_select_paged_tokens",
                "qsa_sparse_split_count",
                "qwen38_gdn_mixed_qkv",
            }
        ),
    )
    dispatch_metax.register_qwen38_operators(
        Registry(), is_platform_available=lambda: False
    )
    assert [(entry.op_name, entry.impl_id) for entry in entries] == [
        ("qsa_mqa_paged", "default.flagos"),
        ("qsa_select_paged_tokens", "default.flagos"),
        ("qsa_sparse_split_count", "default.flagos"),
        ("qwen38_gdn_mixed_qkv", "default.flagos"),
        ("gdn_prefill", "vendor.metax.qwen38"),
        ("qsa_select_paged_tokens", "vendor.metax.qwen38"),
        ("qsa_sparse_split_count", "vendor.metax.qwen38"),
        ("qsa_sparse_paged_attention", "vendor.metax.qwen38"),
        ("qwen38_small_projection", "vendor.metax.qwen38"),
        ("qwen38_grdown_projection", "vendor.metax.qwen38"),
        ("qwen38_grup_projection", "vendor.metax.qwen38"),
        ("qwen38_ple_gate", "vendor.metax.qwen38"),
        ("qwen38_gdn_mixed_qkv", "vendor.metax.qwen38"),
        ("qwen38_hc_grouped_norm", "vendor.metax.qwen38"),
        ("qwen38_hc_gate_reduce", "vendor.metax.qwen38"),
        ("qwen38_hc_inject_combine", "vendor.metax.qwen38"),
        ("qwen38_hc_down_activation", "vendor.metax.qwen38"),
        ("qwen38_qsa_rope", "vendor.metax.qwen38"),
        ("qwen38_qsa_compress_norm_rope", "vendor.metax.qwen38"),
    ]
    for entry in entries[4:]:
        assert entry.vendor == "metax"
        assert not entry.is_available()


def test_metax_selection_uses_all_short_row_blocks_without_topk_order(monkeypatch):
    import sys

    scores = torch.tensor(
        [[5.0, 4.0, -float("inf"), -float("inf")], [1.0, 9.0, 2.0, 10.0]]
    )
    visible = torch.tensor([2, 4], dtype=torch.int32)
    monkeypatch.setattr(
        dispatch_metax.portable,
        "qsa_mqa_paged",
        lambda *args, **kwargs: (scores, visible),
    )

    def topk(value, k, **kwargs):
        assert value is scores and k == 2
        assert kwargs == {"dim": -1, "largest": True, "sorted": False}
        return torch.empty(2, 2), torch.tensor([[3, 2], [3, 1]])

    monkeypatch.setitem(sys.modules, "flag_gems", SimpleNamespace(topk=topk))
    positions, lengths = torch.tensor([9, 17]), torch.tensor([20])
    requests = torch.tensor([0, 0], dtype=torch.int32)
    expected = dispatch_expansion_oracle(
        torch.tensor([[0, 1], [3, 1]], dtype=torch.int32),
        positions,
        lengths,
        requests,
        4,
        8,
    )
    actual = dispatch_metax.qsa_select_paged_tokens(
        torch.empty(2, 4, 8),
        torch.empty(1, 4, 1, 8),
        torch.zeros(1, 1, dtype=torch.int32),
        requests,
        positions,
        lengths,
        8,
        4,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_ple_gate_cast_keeps_bf16_rounding_boundaries():
    gate = torch.tensor(
        [
            -float("inf"),
            -65536.0,
            -4.0,
            -1e-6,
            -0.0,
            0.0,
            1e-6,
            4.0,
            65536.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.bfloat16,
    ).reshape(-1, 1, 1)
    expected = dispatch_ops.qwen38_ple_gate(gate)
    actual = dispatch_metax.qwen38_ple_gate(gate)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


def test_metax_short_bucket_never_scores_or_selects(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("short bucket must not score or select")

    monkeypatch.setattr(dispatch_metax.portable, "qsa_mqa_paged", unexpected)
    monkeypatch.setattr(dispatch_metax, "import_module", unexpected)
    requests = torch.tensor([0, 0, -1], dtype=torch.int32)
    positions = torch.tensor([0, 2046, 2000])
    actual = dispatch_metax.qsa_select_paged_tokens(
        torch.empty(3, 4, 128),
        torch.empty(11, 1600, 1, 128),
        torch.zeros(1, 11, dtype=torch.int32),
        requests,
        positions,
        torch.tensor([2047]),
        2048,
        4,
        num_columns=512,
    )
    expected = torch.full((3, 2051), -1, dtype=torch.int32)
    expected[0, 0] = 0
    expected[1, :2047] = torch.arange(2047, dtype=torch.int32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 128, 129])
def test_metax_selection_uses_native_only_for_small_batches(monkeypatch, rows):
    scores = torch.empty(rows, 2048)
    visible = torch.full((rows,), 2048, dtype=torch.int32)
    calls = []
    monkeypatch.setattr(
        dispatch_metax.portable, "qsa_mqa_paged", lambda *a, **kw: (scores, visible)
    )
    monkeypatch.setattr(
        dispatch_metax.portable, "expand_qsa_block_indices", lambda ids, *a: ids
    )

    def native(x, next_n, lens, out, count, stride0, stride1, k):
        assert x is scores and lens is visible and next_n == 1
        assert (count, stride0, stride1, k) == (rows, 2048, 1, 512)
        calls.append("native")
        out.fill_(3)

    def radix(x, lens, k):
        assert x is scores and lens is visible and k == 512
        calls.append("radix")
        return torch.full((rows, k), 4, dtype=torch.int32)

    def module(name):
        if name == "flaggems_vllm.ops.top_k_per_row_decode":
            return SimpleNamespace(HAS_TLE=False, top_k_per_row_decode=native)
        assert name == "flaggems_vllm.runtime.backend._metax.ops.radix_topk"
        return SimpleNamespace(radix_topk=radix)

    monkeypatch.setattr(dispatch_qsa_policy, "import_module", module)
    out = dispatch_metax.qsa_select_paged_tokens(
        torch.empty(rows, 4, 128),
        torch.empty(2, 1600, 1, 128),
        torch.zeros(1, 2, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int64),
        torch.tensor([8192]),
        2048,
        4,
        num_columns=2048,
    )
    assert calls == ["native" if rows <= 128 else "radix"]
    assert bool((out == (3 if rows <= 128 else 4)).all())
