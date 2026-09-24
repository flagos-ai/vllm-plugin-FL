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

    kv_backing = torch.full(
        (4, 2, 4, 2, 8), -1.0, dtype=torch.bfloat16, device=device
    )
    k_cache, v_cache = kv_backing.unbind(1)
    key = torch.randn(4, 2, 8, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    ops.qsa_store_kv_cache_rows(k_cache, v_cache, slots, key, value)
    torch.testing.assert_close(k_cache[2, 0].cpu(), key[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(k_cache[2, 3].cpu(), key[1].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(v_cache[2, 0].cpu(), value[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(v_cache[2, 3].cpu(), value[1].cpu(), rtol=0, atol=0)
    assert bool((k_cache[0] == -1).all())
    assert bool((v_cache[0] == -1).all())

    for _ in range(3):
        ops.qsa_store_kv_cache_rows(k_cache, v_cache, slots, key, value)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        ops.qsa_store_kv_cache_rows(k_cache, v_cache, slots, key, value)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(k_cache[2, 0].cpu(), key[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(v_cache[2, 3].cpu(), value[1].cpu(), rtol=0, atol=0)

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
def test_qsa_fused_compress_norm_mrope_store_matches_reference_and_graph():
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    torch.manual_seed(25)
    rows, page_size, head_dim, rotary_dim = 2, 4, 128, 64
    table = _qsa_geometry(device)
    raw = torch.randn(
        4, page_size, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    req = torch.tensor([0, 1], dtype=torch.int32, device=device)
    logical = torch.tensor([3, 7], dtype=torch.int64, device=device)
    compressed_slots = torch.tensor([0, 5], dtype=torch.int64, device=device)
    rope_cache = torch.zeros(
        4, page_size, 1, 3, dtype=torch.int64, device=device
    )
    table_cpu = table.cpu()
    for request in range(rows):
        for position in range(8):
            physical = int(table_cpu[request, position // page_size])
            rope_cache[physical, position % page_size, 0] = torch.tensor(
                [position, position // 2, position // 3], device=device
            )
    norm_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    positions = torch.arange(16, dtype=torch.float32)
    # Real Qwen4 checkpoint config: mrope_section=[11, 11, 10].
    mrope_section = (11, 11, 10)
    frequencies = 1.0 / (
        1_000_000
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    angles = positions[:, None] * frequencies[None, :]
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(
        device=device, dtype=torch.bfloat16
    )
    compressed = torch.full(
        (4, page_size, 1, head_dim),
        -1.0,
        dtype=torch.bfloat16,
        device=device,
    )

    expected_pool, _ = qsa_compress_groups_reference(
        raw.cpu(),
        table_cpu,
        req.cpu(),
        logical.cpu(),
        compressed_slots.cpu(),
        4,
    )
    pooled_fp32 = expected_pool[:, 0].float()
    normalized = (
        pooled_fp32
        * torch.rsqrt(pooled_fp32.square().mean(dim=-1, keepdim=True) + 1e-6)
        * (norm_weight.cpu().float() + 1.0)
    ).to(torch.bfloat16)
    first_positions = torch.tensor([[0, 0, 0], [4, 2, 1]], dtype=torch.int64)
    freq = torch.arange(rotary_dim // 2)
    use_height = (freq % 3 == 1) & (freq < 3 * mrope_section[1])
    use_width = (freq % 3 == 2) & (freq < 3 * mrope_section[2])
    axis = torch.where(use_height, 1, torch.where(use_width, 2, 0))
    rope_positions = first_positions[:, axis]
    cos = cos_sin.cpu()[rope_positions, freq]
    sin = cos_sin.cpu()[rope_positions, rotary_dim // 2 + freq]
    first = normalized[:, : rotary_dim // 2]
    second = normalized[:, rotary_dim // 2 : rotary_dim]
    expected = torch.cat(
        (
            (first * cos - second * sin).to(torch.bfloat16),
            (second * cos + first * sin).to(torch.bfloat16),
            normalized[:, rotary_dim:],
        ),
        dim=-1,
    )

    def run() -> None:
        ops.qsa_compress_norm_mrope_store_groups(
            raw,
            table,
            req,
            logical,
            compressed_slots,
            compressed,
            norm_weight,
            cos_sin,
            4,
            1e-6,
            rotary_dim,
            mrope_section,
            True,
            rope_cache,
        )

    run()
    torch.cuda.synchronize()
    actual = torch.stack((compressed[0, 0, 0], compressed[1, 1, 0])).cpu()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        run()
    compressed.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    replay = torch.stack((compressed[0, 0, 0], compressed[1, 1, 0])).cpu()
    torch.testing.assert_close(replay.float(), expected.float(), rtol=2e-2, atol=2e-2)

    compressed.fill_(-1)
    ops.qsa_compress_norm_mrope_store_groups(
        raw,
        table,
        req,
        logical,
        compressed_slots,
        compressed,
        norm_weight,
        cos_sin,
        4,
        1e-6,
        rotary_dim,
        mrope_section,
        True,
    )
    text_positions = torch.tensor([0, 4], dtype=torch.int64)[:, None]
    text_cos = cos_sin.cpu()[text_positions, freq]
    text_sin = cos_sin.cpu()[text_positions, rotary_dim // 2 + freq]
    expected_text = torch.cat(
        (
            (first * text_cos - second * text_sin).to(torch.bfloat16),
            (second * text_cos + first * text_sin).to(torch.bfloat16),
            normalized[:, rotary_dim:],
        ),
        dim=-1,
    )
    actual_text = torch.stack(
        (compressed[0, 0, 0], compressed[1, 1, 0])
    ).cpu()
    torch.testing.assert_close(
        actual_text.float(), expected_text.float(), rtol=2e-2, atol=2e-2
    )


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

    gate = torch.randn_like(q_gqa)
    gated_out = torch.empty_like(q_gqa)
    ops.qsa_sparse_paged_attention(
        q_gqa,
        k,
        v,
        logical_indices,
        table,
        req,
        out=gated_out,
        gate=gate,
    )
    expected_gated = expected_out * torch.sigmoid(gate.cpu())
    torch.testing.assert_close(
        gated_out.cpu().float(), expected_gated.float(), rtol=5e-2, atol=5e-2
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        ops.qsa_sparse_paged_attention(
            q_gqa,
            k,
            v,
            logical_indices,
            table,
            req,
            out=gated_out,
            gate=gate,
        )
    replay_gate = torch.randn_like(gate)
    gate.copy_(replay_gate)
    graph.replay()
    torch.cuda.synchronize()
    expected_replay = expected_out * torch.sigmoid(replay_gate.cpu())
    torch.testing.assert_close(
        gated_out.cpu().float(), expected_replay.float(), rtol=5e-2, atol=5e-2
    )


@pytest.mark.gpu
def test_qsa_mqa_dot_production_shape_matches_reference_and_graph():
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    torch.manual_seed(22)
    rows, page_size, pages_per_request = 8, 16, 64
    total_pages = rows * pages_per_request
    table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        rows, pages_per_request
    )
    key = torch.randn(
        total_pages, page_size, 1, 128, dtype=torch.bfloat16, device=device
    )
    q = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device=device)
    req = torch.arange(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), 4095, dtype=torch.int64, device=device)
    lengths = torch.full((rows,), 4096, dtype=torch.int64, device=device)
    assert ops._use_qsa_mqa_dot(q, key)
    assert not ops._use_qsa_mqa_dot(q[:1], key)

    expected_logits, expected_visible = qsa_mqa_paged_reference(
        q.cpu(), key.cpu(), table.cpu(), req.cpu(), positions.cpu(), lengths.cpu(), 4
    )

    def run():
        return ops.qsa_mqa_paged(q, key, table, req, positions, lengths, 4)

    actual_logits, actual_visible = run()
    torch.testing.assert_close(actual_visible.cpu(), expected_visible, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_logits.cpu(), expected_logits, rtol=2e-4, atol=5e-6
    )
    expected_topk = torch.topk(expected_logits, 512, dim=-1).indices.sort().values
    actual_topk = torch.topk(actual_logits.cpu(), 512, dim=-1).indices.sort().values
    torch.testing.assert_close(actual_topk, expected_topk, rtol=0, atol=0)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        graph_logits, graph_visible = run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_visible.cpu(), expected_visible, rtol=0, atol=0)
    torch.testing.assert_close(
        graph_logits.cpu(), expected_logits, rtol=2e-4, atol=5e-6
    )


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
