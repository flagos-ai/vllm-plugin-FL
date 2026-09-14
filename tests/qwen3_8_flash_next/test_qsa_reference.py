"""CPU QSA metadata/cache/attention references and optional CUDA comparisons."""

from __future__ import annotations

import importlib

import pytest
import torch

from .reference import (
    compressed_slot_mapping_reference,
    expand_qsa_indices_reference,
    logical_to_physical_slots_reference,
    qsa_compress_groups_reference,
    qsa_mqa_paged_reference,
    qsa_sparse_paged_attention_reference,
    qsa_store_cache_rows_reference,
)


def _qsa_geometry(device: torch.device = torch.device("cpu")):
    # Two logical pages, deliberately mapped to non-contiguous physical pages.
    table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device=device)
    return table


def test_qsa_slot_mapping_and_compression_boundaries():
    table = _qsa_geometry()
    req = torch.tensor([0, 0, 1, 1, 1, -1], dtype=torch.int32)
    logical = torch.tensor([0, 3, 4, 7, 8, 2], dtype=torch.int64)
    physical = logical_to_physical_slots_reference(table, req, logical, 4)
    assert physical.tolist() == [8, 11, 12, 15, -1, -1]

    compressed = compressed_slot_mapping_reference(table, req, logical, 4, 4)
    # Only the last row of every complete group is stored in the compressed
    # cache; incomplete positions must remain -1.
    assert compressed.tolist() == [-1, 8, -1, 5, -1, -1]


def test_qsa_store_rows_and_compress_groups_reference():
    torch.manual_seed(10)
    table = _qsa_geometry()
    cache = torch.full((4, 4, 1, 3), -1.0, dtype=torch.bfloat16)
    rows = torch.arange(12, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16)
    slots = torch.tensor([8, 11, -1, 99], dtype=torch.int64)
    stored = qsa_store_cache_rows_reference(cache, slots, rows)
    torch.testing.assert_close(stored[2, 0, 0], rows[0])
    torch.testing.assert_close(stored[2, 3, 0], rows[1])
    assert bool((stored[0] == -1).all())

    raw = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 1, 3).to(torch.bfloat16)
    token_to_req = torch.tensor([0, 1], dtype=torch.int32)
    logical_positions = torch.tensor([3, 7], dtype=torch.int64)
    compressed_slots = torch.tensor([0, 1], dtype=torch.int64)
    pooled, first_positions = qsa_compress_groups_reference(
        raw,
        table,
        token_to_req,
        logical_positions,
        compressed_slots,
        4,
    )
    expected0 = torch.stack([raw[2, i, 0].float() for i in range(4)]).mean(0)
    expected1 = torch.stack([raw[3, i, 0].float() for i in range(4)]).mean(0)
    torch.testing.assert_close(pooled[0, 0].float(), expected0)
    torch.testing.assert_close(pooled[1, 0].float(), expected1)
    assert first_positions.tolist() == [[0, 0, 0], [4, 4, 4]]


def test_qsa_mqa_and_sparse_references_cover_gqa_and_invalid_entries():
    torch.manual_seed(11)
    table = _qsa_geometry()
    # MQA indexer: one compressed key head, two query heads.
    key = torch.randn(4, 4, 1, 3, dtype=torch.bfloat16)
    q = torch.randn(2, 2, 3, dtype=torch.bfloat16)
    token_to_req = torch.tensor([0, 1], dtype=torch.int32)
    positions = torch.tensor([7, 7], dtype=torch.int64)
    lengths = torch.tensor([8, 8], dtype=torch.int64)
    logits, visible = qsa_mqa_paged_reference(
        q, key, table, token_to_req, positions, lengths, 4
    )
    assert logits.shape == (2, 8)
    assert visible.tolist() == [2, 2]
    assert torch.isneginf(logits[:, 2:]).all()

    # Sparse GQA: four Q heads share two KV heads, and -1 is ignored.
    k = torch.randn(4, 4, 2, 3, dtype=torch.bfloat16)
    v = torch.randn(4, 4, 2, 3, dtype=torch.bfloat16)
    q_gqa = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1, 5, -1], [4, 7, -1, -1]], dtype=torch.int32)
    out = qsa_sparse_paged_attention_reference(
        q_gqa, k, v, indices, table, token_to_req
    )
    assert out.shape == q_gqa.shape
    assert torch.isfinite(out.float()).all()


def test_qsa_expand_reference_handles_complete_and_tail_groups():
    block_indices = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    query_positions = torch.tensor([8, 5], dtype=torch.int64)
    sequence_lengths = torch.tensor([9, 6], dtype=torch.int64)
    token_to_req = torch.tensor([0, 1], dtype=torch.int32)
    expanded = expand_qsa_indices_reference(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio=4,
        token_topk=8,
    )
    assert expanded.shape == (2, 11)
    # Complete blocks are expanded four tokens each. The incomplete tail has
    # at most ratio - 1 entries, per the Triton contract.
    assert expanded[0, :8].tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
    assert expanded[0, 8:].tolist() == [8, -1, -1]
    assert expanded[1, :4].tolist() == [0, 1, 2, 3]
    assert expanded[1, 4:].tolist() == [4, 5, -1, -1, -1, -1, -1]


def _load_qsa_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; Triton comparison is optional")
    try:
        return importlib.import_module(
            "vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa"
        )
    except Exception as exc:  # target-GPU jobs must not hide import failures
        pytest.fail(f"vLLM QSA plugin import failed: {type(exc).__name__}: {exc}")


@pytest.mark.gpu
def test_qsa_cuda_store_and_compress_match_reference():
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    torch.manual_seed(20)
    table = _qsa_geometry(device)
    cache = torch.full((4, 4, 1, 8), -1.0, dtype=torch.bfloat16, device=device)
    rows = torch.randn(4, 8, dtype=torch.bfloat16, device=device)
    slots = torch.tensor([8, 11, -1, 99], dtype=torch.int64, device=device)
    expected = qsa_store_cache_rows_reference(cache.cpu(), slots.cpu(), rows.cpu())
    ops.qsa_store_cache_rows(cache, slots, rows)
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)

    raw = torch.randn(4, 4, 1, 8, dtype=torch.bfloat16, device=device)
    req = torch.tensor([0, 1], dtype=torch.int32, device=device)
    logical = torch.tensor([3, 7], dtype=torch.int64, device=device)
    compressed_slots = torch.tensor([0, 1], dtype=torch.int64, device=device)
    expected_pool, expected_pos = qsa_compress_groups_reference(
        raw.cpu(), table.cpu(), req.cpu(), logical.cpu(), compressed_slots.cpu(), 4
    )
    actual_pool, actual_pos = ops.qsa_compress_groups_with_ratio(
        raw, table, req, logical, compressed_slots, 4
    )
    torch.testing.assert_close(actual_pool.cpu().float(), expected_pool.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_pos.cpu(), expected_pos)


@pytest.mark.gpu
def test_qsa_cuda_indexer_and_sparse_match_reference():
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    torch.manual_seed(21)
    table = _qsa_geometry(device)
    key = torch.randn(4, 4, 1, 8, dtype=torch.bfloat16, device=device)
    q = torch.randn(2, 2, 8, dtype=torch.bfloat16, device=device)
    req = torch.tensor([0, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([7, 7], dtype=torch.int64, device=device)
    lengths = torch.tensor([8, 8], dtype=torch.int64, device=device)
    expected_logits, expected_visible = qsa_mqa_paged_reference(
        q.cpu(), key.cpu(), table.cpu(), req.cpu(), positions.cpu(), lengths.cpu(), 4
    )
    actual_logits, actual_visible = ops.qsa_mqa_paged(
        q, key, table, req, positions, lengths, 4
    )
    torch.testing.assert_close(actual_logits.cpu(), expected_logits, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_visible.cpu(), expected_visible)

    block_indices = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32, device=device)
    expected_indices = expand_qsa_indices_reference(
        block_indices.cpu(), positions.cpu(), lengths.cpu(), req.cpu(), 4, 8
    )
    actual_indices = ops.expand_qsa_block_indices(
        block_indices, positions, lengths, req, 4, 8
    )
    torch.testing.assert_close(actual_indices.cpu(), expected_indices)

    k = torch.randn(4, 4, 2, 8, dtype=torch.bfloat16, device=device)
    v = torch.randn(4, 4, 2, 8, dtype=torch.bfloat16, device=device)
    q_gqa = torch.randn(2, 4, 8, dtype=torch.bfloat16, device=device)
    logical_indices = torch.tensor(
        [[0, 1, 5, -1], [4, 7, -1, -1]], dtype=torch.int32, device=device
    )
    expected_out = qsa_sparse_paged_attention_reference(
        q_gqa.cpu(), k.cpu(), v.cpu(), logical_indices.cpu(), table.cpu(), req.cpu()
    )
    actual_out = ops.qsa_sparse_paged_attention(
        q_gqa, k, v, logical_indices, table, req
    )
    torch.testing.assert_close(actual_out.cpu().float(), expected_out.float(), rtol=5e-2, atol=5e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 8, 64])
def test_qsa_cuda_full_select_matches_reference_and_graph(rows):
    """Exercise cooperative/persistent or cross-vendor TopK end to end."""

    ops = _load_qsa_ops()
    device = torch.device("cuda")
    torch.manual_seed(100 + rows)
    # Match the checkpoint contract exactly: token_topk=2048 and ratio=4
    # select k=512 compressed blocks, one of the private NVIDIA op's supported
    # K values. The same case stays valid through the generic vendor path.
    page_size, num_pages, compress_ratio, token_topk = 16, 32, 4, 2048
    table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(1, num_pages)
    key = torch.randn(
        num_pages, page_size, 1, 8, dtype=torch.bfloat16, device=device
    )
    q = torch.randn(rows, 2, 8, dtype=torch.bfloat16, device=device)
    req = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), 2047, dtype=torch.int64, device=device)
    lengths = torch.tensor([2048], dtype=torch.int64, device=device)

    expected_logits, _ = qsa_mqa_paged_reference(
        q.cpu(),
        key.cpu(),
        table.cpu(),
        req.cpu(),
        positions.cpu(),
        lengths.cpu(),
        compress_ratio,
    )
    expected_blocks = torch.topk(
        expected_logits, token_topk // compress_ratio, dim=-1
    ).indices.to(torch.int32)
    expected = expand_qsa_indices_reference(
        expected_blocks,
        positions.cpu(),
        lengths.cpu(),
        req.cpu(),
        compress_ratio,
        token_topk,
    )
    output = torch.empty(
        rows, token_topk + compress_ratio - 1, dtype=torch.int32, device=device
    )

    def select() -> torch.Tensor:
        return ops.qsa_select_paged_tokens(
            q,
            key,
            table,
            req,
            positions,
            lengths,
            token_topk,
            compress_ratio,
            output,
        )

    def assert_same_selected_tokens(actual: torch.Tensor) -> None:
        actual_cpu = actual.cpu()
        for row in range(rows):
            expected_valid = expected[row][expected[row] >= 0].sort().values
            actual_valid = actual_cpu[row][actual_cpu[row] >= 0].sort().values
            torch.testing.assert_close(actual_valid, expected_valid, rtol=0, atol=0)
            assert int((actual_cpu[row] < 0).sum()) == int(
                (expected[row] < 0).sum()
            )

    assert_same_selected_tokens(select())
    for _ in range(5):
        select()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        select()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert_same_selected_tokens(output)
