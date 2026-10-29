# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Qwen selection composition and FGV dispatch boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl import qwen3_8_flash_next as ops
from vllm_fl.dispatch.backends.vendor.metax.impl import qwen3_8_flash_next as metax


def expansion_oracle(blocks, positions, lengths, requests, ratio, budget):
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
    expected = expansion_oracle(blocks, positions, lengths, requests, ratio, budget)
    out = torch.full_like(expected, -123)
    actual = ops.expand_qsa_block_indices(
        blocks, positions, lengths, requests, ratio, budget, out
    )
    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_expand_handles_invalid_blocks_empty_requests_and_strides():
    blocks = torch.tensor([[-1, 30], [0, 1]], dtype=torch.int64)
    positions = torch.tensor([7, 5], dtype=torch.int32)
    requests = torch.tensor([0, 1], dtype=torch.int32)
    lengths = torch.tensor([8, 0, 6, 0], dtype=torch.int32)[::2]
    expected = expansion_oracle(blocks, positions, lengths, requests, 4, 8)
    actual = ops.expand_qsa_block_indices(blocks, positions, lengths, requests, 4, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    empty_lengths = lengths[:0]
    actual = ops.expand_qsa_block_indices(
        blocks, positions, empty_lengths, requests, 4, 8
    )
    assert torch.equal(actual, torch.full_like(actual, -1))
    empty = ops.expand_qsa_block_indices(
        blocks[:0], positions[:0], lengths, requests[:0], 4, 8
    )
    assert empty.shape == (0, 11)


@pytest.mark.parametrize("ratio,budget", [(0, 8), (4, 0), (4, 7)])
def test_expand_rejects_invalid_geometry(ratio, budget):
    with pytest.raises(ValueError, match="divisible"):
        ops.expand_qsa_block_indices(
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
    monkeypatch.setattr(ops, "get_flaggems_vllm_operator", functions.get)
    q = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
    cache = torch.zeros(3, 2, 1, 128, dtype=torch.bfloat16)
    table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    requests = torch.tensor([0, 0], dtype=torch.int32)
    positions = torch.tensor([25, 9], dtype=torch.int64)
    lengths = torch.tensor([26], dtype=torch.int64)
    expected = expansion_oracle(
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
        actual = ops.qsa_select_paged_tokens(
            q, cache, table, requests, positions, lengths, 8, 4, out
        )
    assert actual is out
    assert calls == [("score", 6), ("topk", 2, 6, 1, 2)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_selection_empty_batch_does_not_resolve_kernels(monkeypatch):
    monkeypatch.setattr(
        ops,
        "get_flaggems_vllm_operator",
        lambda name: pytest.fail(f"Unexpected kernel lookup: {name}"),
    )
    actual = ops.qsa_select_paged_tokens(
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

    monkeypatch.setattr(ops, "get_flaggems_vllm_operator", lambda name: failed)
    with pytest.raises(RuntimeError, match="partial state"):
        ops.qsa_store_cache_rows(torch.empty(1), torch.empty(1), torch.empty(1))
    assert calls == ["called"]


def test_packed_gdn_rejects_spec_state_index_matrix_before_kernel(monkeypatch):
    monkeypatch.setattr(
        ops, "get_flaggems_vllm_operator", lambda name: pytest.fail("Kernel resolved")
    )
    with pytest.raises(ValueError, match="non-spec"):
        ops.gdn_packed_decode(
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
        ops,
        "get_flaggems_vllm_operator",
        lambda name: (lambda *a: None) if name == "qsa_mqa_paged" else None,
    )
    assert ops.is_available("qsa_mqa_paged")
    assert not ops.is_available("qsa_select_paged_tokens")
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    assert not ops.is_available("qsa_mqa_paged")


def test_registration_preserves_default_and_scopes_vendor_difference():
    entries = []

    class Registry:
        def register_many(self, values):
            entries.extend(values)

    ops.register_qwen38_operators(
        Registry(),
        allow=lambda name: (
            name in {"qsa_mqa_paged", "qsa_select_paged_tokens", "qwen38_gdn_mixed_qkv"}
        ),
    )
    metax.register_qwen38_operators(Registry(), is_platform_available=lambda: False)
    assert [(entry.op_name, entry.impl_id) for entry in entries] == [
        ("qsa_mqa_paged", "default.flagos"),
        ("qsa_select_paged_tokens", "default.flagos"),
        ("qwen38_gdn_mixed_qkv", "default.flagos"),
        ("gdn_prefill", "vendor.metax.qwen38"),
        ("qsa_select_paged_tokens", "vendor.metax.qwen38"),
        ("qwen38_small_projection", "vendor.metax.qwen38"),
        ("qwen38_ple_gate", "vendor.metax.qwen38"),
        ("qwen38_gdn_mixed_qkv", "vendor.metax.qwen38"),
    ]
    assert entries[-1].vendor == "metax"
    assert not entries[-1].is_available()


def test_packed_gdn_passes_fp32_beta_input_without_changing_values(monkeypatch):
    sent = {}

    def packed(*args):
        sent["b"] = args[2]
        return args[7], args[6]

    monkeypatch.setattr(ops, "get_flaggems_vllm_operator", lambda name: packed)
    b = torch.tensor([[0.3125, -0.421875]], dtype=torch.bfloat16)
    out, state = torch.empty(1, 1, 2, 4), torch.empty(3, 2, 4, 4)
    ops.gdn_packed_decode(
        torch.empty(1, 16),
        b,
        b,
        torch.empty(2),
        torch.empty(2),
        0.5,
        state,
        out,
        torch.tensor([1], dtype=torch.int32),
    )
    assert sent["b"].dtype == torch.float32
    torch.testing.assert_close(sent["b"], b.float(), atol=0, rtol=0)


def test_metax_prefill_preserves_fp32_state_and_metadata(monkeypatch):
    q = torch.randn(1, 8, 2, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 8, 6, 4, dtype=torch.bfloat16)
    g, beta = torch.randn(1, 8, 6), torch.rand(1, 8, 6)
    state = torch.randn(1, 6, 4, 4)
    offsets = torch.tensor([0, 8], dtype=torch.int32)
    chunks = torch.tensor([[0, 0]], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 1], dtype=torch.int32)
    expected_out, expected_state = torch.randn_like(v), torch.randn_like(state)
    sent = {}

    def chunk(**kwargs):
        sent.update(kwargs)
        return None, expected_out, None, expected_state, None, None, None

    monkeypatch.setattr(
        metax,
        "import_module",
        lambda name: SimpleNamespace(chunk_gated_delta_rule_fwd=chunk),
    )
    out, final = metax.gdn_prefill(
        q,
        q,
        v,
        g,
        beta,
        state,
        True,
        offsets,
        chunks,
        chunk_offsets,
        use_qk_l2norm_in_kernel=False,
    )
    assert out is expected_out and final is expected_state
    assert sent["initial_state"] is state
    assert sent["g"] is g and sent["beta"] is beta
    assert sent["cu_seqlens"] is offsets
    assert sent["chunk_indices"] is chunks
    assert sent["chunk_offsets"] is chunk_offsets
    assert sent["scale"] == 0.5
    with pytest.raises(NotImplementedError, match="pre-normalized"):
        metax.gdn_prefill(q, q, v, g, beta, state, True)
    with pytest.raises(ValueError, match="FP32"):
        metax.gdn_prefill(
            q,
            q,
            v,
            g,
            beta,
            state.bfloat16(),
            True,
            use_qk_l2norm_in_kernel=False,
        )


def test_metax_selection_uses_all_short_row_blocks_without_topk_order(monkeypatch):
    import sys

    scores = torch.tensor(
        [[5.0, 4.0, -float("inf"), -float("inf")], [1.0, 9.0, 2.0, 10.0]]
    )
    visible = torch.tensor([2, 4], dtype=torch.int32)
    monkeypatch.setattr(
        metax.portable, "qsa_mqa_paged", lambda *args, **kwargs: (scores, visible)
    )

    def topk(value, k, **kwargs):
        assert value is scores and k == 2
        assert kwargs == {"dim": -1, "largest": True, "sorted": False}
        return torch.empty(2, 2), torch.tensor([[3, 2], [3, 1]])

    monkeypatch.setitem(sys.modules, "flag_gems", SimpleNamespace(topk=topk))
    positions, lengths = torch.tensor([9, 17]), torch.tensor([20])
    requests = torch.tensor([0, 0], dtype=torch.int32)
    expected = expansion_oracle(
        torch.tensor([[0, 1], [3, 1]], dtype=torch.int32),
        positions,
        lengths,
        requests,
        4,
        8,
    )
    actual = metax.qsa_select_paged_tokens(
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
    expected = ops.qwen38_ple_gate(gate)
    actual = metax.qwen38_ple_gate(gate)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


def test_expand_strided_selection_never_materializes_python_mul_scalar(monkeypatch):
    blocks = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    positions = torch.tensor([7, 0, 9, 0], dtype=torch.int64)[::2]
    lengths = torch.tensor([8, 0, 10, 0], dtype=torch.int64)[::2]
    requests = torch.tensor([0, 0, 1, 0], dtype=torch.int64)[::2]
    expected = expansion_oracle(blocks, positions, lengths, requests, 4, 8)
    original_getitem = torch.Tensor.__getitem__
    original_mul = torch.Tensor.__mul__
    strided_products = []

    def column_major_selection(value, key):
        selected = original_getitem(value, key)
        if value is blocks:
            # Reproduce FG indexing's actual M>1 output stride (1, M).
            return selected.t().contiguous().t()
        return selected

    def tensor_only_mul(value, other):
        assert isinstance(other, torch.Tensor), "capture-unsafe scalar mul"
        if not value.is_contiguous():
            strided_products.append((value.stride(), other.shape, other.dtype))
        return original_mul(value, other)

    def forbid_tensor_constructor(*args, **kwargs):
        raise AssertionError("capture cannot materialize a host scalar tensor")

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "__getitem__", column_major_selection)
        scoped.setattr(torch.Tensor, "__mul__", tensor_only_mul)
        scoped.setattr(torch, "tensor", forbid_tensor_constructor)
        actual = ops.expand_qsa_block_indices(
            blocks, positions, lengths, requests, 4, 8
        )
    assert strided_products
    assert all(
        shape == torch.Size([]) and dtype == torch.int64
        for _, shape, dtype in strided_products
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mixed_gdn_default_keeps_view_and_metax_preserves_values(monkeypatch):
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    qkv = torch.arange(42 * 2048, dtype=torch.float32).reshape(42, 2048)[:, :1280]
    assert qkv.stride() == (2048, 1)
    assert ops.is_available("qwen38_gdn_mixed_qkv")
    assert ops.qwen38_gdn_mixed_qkv(qkv) is qkv
    actual = metax.qwen38_gdn_mixed_qkv(qkv)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, qkv, atol=0, rtol=0)
    assert metax.qwen38_gdn_mixed_qkv(actual) is actual
