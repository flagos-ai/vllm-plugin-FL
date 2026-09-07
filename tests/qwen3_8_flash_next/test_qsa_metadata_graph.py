"""Exactness and graph-replay tests for fused QSA forward metadata."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from .reference import compressed_slot_mapping_reference


def _load_qsa_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; Triton comparison is optional")
    try:
        return importlib.import_module("vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa")
    except Exception as exc:
        pytest.fail(f"vLLM QSA plugin import failed: {type(exc).__name__}: {exc}")


def _reference(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    common_slot_mapping: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_lens = torch.diff(query_start_loc)
    num_mapped_tokens = int(query_start_loc[-1])
    token_to_req = torch.zeros(rows, dtype=torch.int32)
    token_to_req[:num_mapped_tokens] = torch.repeat_interleave(
        torch.arange(query_lens.numel(), dtype=torch.int32),
        query_lens,
        output_size=num_mapped_tokens,
    )
    row_ids = torch.arange(rows, dtype=torch.int64)
    requests = token_to_req.to(torch.int64)
    request_valid = (requests >= 0) & (requests < seq_lens.numel())
    safe_requests = requests.clamp(0, max(seq_lens.numel() - 1, 0))
    query_starts = query_start_loc.index_select(0, safe_requests)
    query_ends = query_start_loc.index_select(0, safe_requests + 1)
    logical = (
        seq_lens.index_select(0, safe_requests).to(torch.int64)
        - (query_ends - query_starts).to(torch.int64)
        + row_ids
        - query_starts.to(torch.int64)
    )
    mapped = request_valid & (row_ids < int(query_start_loc[-1]))
    logical = torch.where(mapped, logical, torch.full_like(logical, -1))
    slots = compressed_slot_mapping_reference(
        block_table,
        token_to_req,
        logical,
        storage_block_size,
        compress_ratio,
    )
    slots.masked_fill_(common_slot_mapping < 0, -1)
    return token_to_req, logical, slots


def test_forward_metadata_holds_common_metadata_by_reference():
    from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import (
        QSAForwardMetadata,
    )

    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    common_slot_mapping = torch.tensor([7, 8], dtype=torch.int64)
    common = SimpleNamespace(
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=common_slot_mapping,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        num_actual_tokens=2,
    )
    logical = torch.full((2,), -1, dtype=torch.int64)
    metadata = QSAForwardMetadata(
        common=common,
        token_to_req=torch.zeros(2, dtype=torch.int32),
        logical_positions=logical,
        qsa_slot_mapping=common_slot_mapping,
        storage_block_size=4,
        compress_ratio=1,
    )
    assert metadata.block_table is common.block_table_tensor
    assert metadata.slot_mapping is common.slot_mapping
    assert metadata.seq_lens is common.seq_lens
    assert metadata.query_start_loc is common.query_start_loc
    assert metadata.num_actual_tokens == 2
    metadata.prepare()
    assert logical.tolist() == [0, 1]


def test_metadata_defer_follows_global_full_graph_modes():
    from vllm.config.compilation import CUDAGraphMode

    from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import (
        _global_full_cudagraph_enabled,
    )

    expected = {
        CUDAGraphMode.NONE: False,
        CUDAGraphMode.PIECEWISE: False,
        CUDAGraphMode.FULL: True,
        CUDAGraphMode.FULL_DECODE_ONLY: True,
        CUDAGraphMode.FULL_AND_PIECEWISE: True,
    }
    for mode, enabled in expected.items():
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=mode)
        )
        assert _global_full_cudagraph_enabled(config) is enabled
    unresolved = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=None)
    )
    assert _global_full_cudagraph_enabled(unresolved) is False


@pytest.mark.parametrize("with_new_token_to_req_api", [False, True])
def test_builder_fallback_does_not_require_common_token_to_request_api(
    with_new_token_to_req_api,
):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import QSAMetadataBuilder

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    builder = QSAMetadataBuilder(spec, ["qsa"], config, torch.device("cpu"))
    assert torch.count_nonzero(builder.token_to_req_buffer) == 0
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    common = SimpleNamespace(
        block_table_tensor=torch.tensor([[2, 0, 4, 6], [1, 3, 5, 7]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2, 3, -1, -1], dtype=torch.int64),
        seq_lens=torch.tensor([2, 5], dtype=torch.int32),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        num_actual_tokens=6,
    )
    if with_new_token_to_req_api:

        def token_to_req_indices(out):
            del out
            raise AssertionError("QSA must not build token_to_req before prepare()")

        common.token_to_req_indices = token_to_req_indices

    metadata = builder.build(0, common)
    assert metadata.common is common
    assert metadata.token_to_req.tolist() == [0, 0, 1, 1, 0, 0]
    assert metadata.logical_positions.tolist() == [0, 1, 3, 4, -1, -1]
    expected_token_to_req, expected_logical, expected_slots = _reference(
        common.query_start_loc,
        common.seq_lens,
        common.block_table_tensor,
        common.slot_mapping,
        4,
        4,
        6,
    )
    torch.testing.assert_close(metadata.token_to_req, expected_token_to_req)
    torch.testing.assert_close(metadata.logical_positions, expected_logical)
    torch.testing.assert_close(metadata.qsa_slot_mapping, expected_slots)


def test_common_token_to_req_cache_reuses_same_common_object(monkeypatch):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from vllm_fl.patches.qwen3_8_flash_next import (
        _patch_common_attention_token_to_req_cache,
    )

    _patch_common_attention_token_to_req_cache()
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([2, 5], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=3,
        max_seq_len=5,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int64),
    )
    first = torch.full((8,), -7, dtype=torch.int32)
    second = torch.full((8,), -9, dtype=torch.int32)
    original_repeat_interleave = torch.repeat_interleave
    calls = 0

    def counted_repeat_interleave(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_repeat_interleave(*args, **kwargs)

    monkeypatch.setattr(torch, "repeat_interleave", counted_repeat_interleave)
    first_result = common.token_to_req_indices(first)
    second_result = common.token_to_req_indices(second)

    assert calls == 1
    assert first_result.data_ptr() == first.data_ptr()
    assert second_result.data_ptr() == first.data_ptr()
    assert first_result.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert second.tolist() == [-9] * 8

    fresh_common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([2, 5], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=3,
        max_seq_len=5,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int64),
    )
    fresh_result = fresh_common.token_to_req_indices(second)
    assert calls == 2
    assert fresh_result.data_ptr() == second.data_ptr()
    assert fresh_result.tolist() == first_result.tolist()


@pytest.mark.gpu
@pytest.mark.parametrize("num_reqs,rows", [(1, 8), (64, 64)])
def test_kernel_matches_torch_for_boundaries_and_invalid_rows(num_reqs, rows):
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    if num_reqs == 1:
        query_start_loc = torch.tensor([0, rows - 2], dtype=torch.int32)
        seq_lens = torch.tensor([rows - 3], dtype=torch.int32)
    else:
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32)
        query_start_loc[-4:] = num_reqs - 3
        seq_lens = torch.arange(num_reqs, dtype=torch.int32) + 1
        seq_lens[0] = 0
    block_table = torch.arange(num_reqs * 8, dtype=torch.int32).reshape(num_reqs, 8)
    block_table[0, -1] = -1
    common_slots = torch.arange(rows, dtype=torch.int64)
    common_slots[1::17] = -1
    expected_token_to_req, expected_logical, expected_slots = _reference(
        query_start_loc,
        seq_lens,
        block_table,
        common_slots,
        4,
        4,
        rows,
    )
    token_to_req = torch.full((rows,), -99, dtype=torch.int32, device=device)
    logical = torch.empty(rows, dtype=torch.int64, device=device)
    slots = torch.empty_like(logical)
    assert ops.build_qsa_forward_metadata(
        token_to_req,
        query_start_loc.to(device),
        seq_lens.to(device),
        block_table.to(device),
        common_slots.to(device),
        logical,
        slots,
        4,
        4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        token_to_req.cpu(), expected_token_to_req, rtol=0, atol=0
    )
    torch.testing.assert_close(logical.cpu(), expected_logical, rtol=0, atol=0)
    torch.testing.assert_close(slots.cpu(), expected_slots, rtol=0, atol=0)


@pytest.mark.gpu
def test_kernel_graph_replay_updates_contents_at_fixed_addresses():
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    rows = num_reqs = 64
    token_to_req = torch.full((rows,), -99, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
    seq_lens = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    block_table = torch.arange(
        num_reqs * 8, dtype=torch.int32, device=device
    ).reshape(num_reqs, 8)
    common_slots = torch.arange(rows, dtype=torch.int64, device=device)
    logical = torch.empty(rows, dtype=torch.int64, device=device)
    slots = torch.empty_like(logical)
    output_addresses = (
        token_to_req.data_ptr(),
        logical.data_ptr(),
        slots.data_ptr(),
    )

    def run() -> None:
        assert ops.build_qsa_forward_metadata(
            token_to_req,
            query_start_loc,
            seq_lens,
            block_table,
            common_slots,
            logical,
            slots,
            4,
            4,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        run()
    graph.replay()
    torch.cuda.synchronize()
    first_token_to_req = token_to_req.cpu().clone()
    first_logical = logical.cpu().clone()
    first_slots = slots.cpu().clone()

    second_query_start_loc = (
        torch.arange(num_reqs + 1, dtype=torch.int32) // 2 * 2
    )
    second_seq_lens = torch.arange(5, num_reqs + 5, dtype=torch.int32)
    second_table = torch.flip(block_table.cpu(), dims=(1,))
    second_common_slots = common_slots.cpu().clone()
    second_common_slots[::9] = -1
    query_start_loc.copy_(second_query_start_loc)
    seq_lens.copy_(second_seq_lens)
    block_table.copy_(second_table)
    common_slots.copy_(second_common_slots)
    graph.replay()
    torch.cuda.synchronize()
    expected_token_to_req, expected_logical, expected_slots = _reference(
        second_query_start_loc,
        second_seq_lens,
        second_table,
        second_common_slots,
        4,
        4,
        rows,
    )
    torch.testing.assert_close(
        token_to_req.cpu(), expected_token_to_req, rtol=0, atol=0
    )
    torch.testing.assert_close(logical.cpu(), expected_logical, rtol=0, atol=0)
    torch.testing.assert_close(slots.cpu(), expected_slots, rtol=0, atol=0)
    assert not torch.equal(token_to_req.cpu(), first_token_to_req)
    assert not torch.equal(logical.cpu(), first_logical)
    assert not torch.equal(slots.cpu(), first_slots)
    assert (
        token_to_req.data_ptr(),
        logical.data_ptr(),
        slots.data_ptr(),
    ) == output_addresses


@pytest.mark.gpu
def test_ratio_one_reuses_common_slot_mapping_without_fused_kernel():
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    logical = torch.full((1,), 101, dtype=torch.int64, device=device)
    slots = torch.full((1,), 202, dtype=torch.int64, device=device)
    token_to_req = torch.full((1,), 303, dtype=torch.int32, device=device)
    used = ops.build_qsa_forward_metadata(
        token_to_req,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
        logical,
        slots,
        4,
        1,
    )
    assert used is False
    assert token_to_req.item() == 303
    assert logical.item() == 101
    assert slots.item() == 202
