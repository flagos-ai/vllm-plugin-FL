"""Core Qwen3.8 framework contracts; numerical tests live in FlagGems-vllm."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.compat import kv_cache as kv_bind
from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import QSAKeyStateCache


def compressed_slot_mapping_reference(table, requests, positions, block_size, ratio):
    compressed = positions // ratio
    valid = (positions >= 0) & (requests >= 0) & (requests < table.shape[0])
    blocks = compressed // block_size
    valid &= blocks < table.shape[1]
    pages = table[
        requests.long().clamp(0, table.shape[0] - 1),
        blocks.long().clamp(0, table.shape[1] - 1),
    ]
    valid &= (pages >= 0) & ((positions + 1) % ratio == 0)
    return torch.where(valid, pages.long() * block_size + compressed % block_size, -1)


def _load_qsa_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; Triton comparison is optional")
    try:
        return importlib.import_module("vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa")
    except Exception as exc:
        raise AssertionError(
            f"vLLM QSA plugin import failed: {type(exc).__name__}: {exc}"
        ) from exc


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


def test_sparse_attention_uses_compressed_metadata_request_map():
    from vllm_fl.models.qwen3_8_flash_next.gpu import model as _model  # noqa: F401
    from vllm_fl.models.qwen3_8_flash_next.gpu.qsa import _prepared_qsa_token_to_req

    raw = torch.zeros(6, dtype=torch.int32)
    compressed = torch.tensor([0, 0, 1, 1, 99, 99], dtype=torch.int32)
    metadata = {
        "raw": SimpleNamespace(num_actual_tokens=4, token_to_req=raw),
        "compressed": SimpleNamespace(num_actual_tokens=4, token_to_req=compressed),
    }
    indexer = SimpleNamespace(
        raw_key_cache=SimpleNamespace(prefix="raw"),
        compressed_key_cache=SimpleNamespace(prefix="compressed"),
    )
    actual = _prepared_qsa_token_to_req(metadata, indexer, 4)
    assert actual.tolist() == [0, 0, 1, 1]
    assert actual.data_ptr() == compressed.data_ptr()
    assert actual.data_ptr() != raw.data_ptr()
    # Fixed-address storage must expose updated mappings on later replays.
    compressed[:4].copy_(torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert actual.tolist() == [0, 1, 2, 3]


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
        block_table_tensor=torch.tensor(
            [[2, 0, 4, 6], [1, 3, 5, 7]], dtype=torch.int32
        ),
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


@pytest.mark.gpu
def test_kernel_graph_replay_updates_contents_at_fixed_addresses():
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    rows = num_reqs = 64
    token_to_req = torch.full((rows,), -99, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
    seq_lens = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    block_table = torch.arange(num_reqs * 8, dtype=torch.int32, device=device).reshape(
        num_reqs, 8
    )
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

    second_query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32) // 2 * 2
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


class _Owner:
    def __init__(self):
        self.bound = []

    def bind_kv_cache(self, kv_cache):
        self.bound.append(kv_cache)


def _noop_upstream(monkeypatch, *, mutates=True):
    calls = []

    def fake(kv_caches, forward_context, runner_kv_caches, num_attn_module=1):
        calls.append(list(kv_caches))
        if mutates:
            for layer_name, kv in kv_caches.items():
                forward_context[layer_name].kv_cache = kv
                runner_kv_caches.append(kv)

    from vllm_fl.worker import kv_cache_utils

    monkeypatch.setattr(kv_cache_utils, "bind_kv_cache", fake)
    monkeypatch.setattr(kv_bind, "upstream_binds_kv_cache_owners", lambda: False)
    return calls


def test_owner_hook_called_once_after_upstream(monkeypatch):
    upstream_calls = _noop_upstream(monkeypatch)
    owner = _Owner()
    kv_caches = {"qsa": torch.zeros(2, 4, 1, 8)}
    forward_context = {"qsa": owner}

    kv_bind.bind_kv_cache(kv_caches, forward_context, [], 1)
    assert len(upstream_calls) == 1
    assert len(owner.bound) == 1
    assert owner.bound[0] is kv_caches["qsa"]


def test_validation_failure_leaves_no_mutation(monkeypatch):
    _noop_upstream(monkeypatch)
    owner = _Owner()
    owner.validate_kv_cache = lambda kv: (_ for _ in ()).throw(ValueError("bad dtype"))
    runner: list = []
    kv_caches = {"qsa": torch.zeros(1)}
    with pytest.raises(ValueError, match="bad dtype"):
        kv_bind.bind_kv_cache(kv_caches, {"qsa": owner}, runner, 1)
    assert runner == []
    assert owner.bound == []


def test_owner_hooks_preserve_iluvatar_multi_cache_binding(monkeypatch):
    from vllm_fl.worker import kv_cache_utils

    monkeypatch.setattr(
        kv_cache_utils,
        "current_platform",
        SimpleNamespace(vendor_name="iluvatar", device_type="cuda"),
    )
    caches = {
        "model.layers.0.main_attn": torch.zeros(2),
        "model.layers.0.index_attn": torch.zeros(3),
    }
    owners = {name: _Owner() for name in caches}
    runner = []
    kv_bind.upstream_binds_kv_cache_owners.cache_clear()
    try:
        kv_bind.bind_kv_cache(caches, owners, runner)
    finally:
        kv_bind.upstream_binds_kv_cache_owners.cache_clear()
    assert len(runner) == 2
    for index, (name, tensor) in enumerate(caches.items()):
        assert runner[index] is tensor
        assert owners[name].kv_cache is tensor
        assert len(owners[name].bound) == 1
        assert owners[name].bound[0] is tensor


def test_rebind_replaces_typed_views():
    key_head_size = 512
    obj = object.__new__(QSAKeyStateCache)
    obj.head_size = ((key_head_size + 3) // 4) * 4 + 3 * 4
    obj.key_head_size = key_head_size
    obj.cache_rope_positions = True
    obj.rope_position_offset = ((key_head_size + 3) // 4) * 4
    obj.kv_cache = torch.tensor([])

    first = torch.zeros(2, 4, 1, obj.head_size, dtype=torch.bfloat16)
    obj.bind_kv_cache(first)
    assert (
        obj.key_cache.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    )

    second = torch.zeros(2, 4, 1, obj.head_size, dtype=torch.bfloat16)
    obj.bind_kv_cache(second)
    assert (
        obj.key_cache.untyped_storage().data_ptr()
        == second.untyped_storage().data_ptr()
    )
    assert (
        obj.rope_position_cache.untyped_storage().data_ptr()
        == second.untyped_storage().data_ptr()
    )
    assert obj.rope_position_cache.dtype == torch.int64
    assert obj.rope_position_cache.storage_offset() == obj.rope_position_offset // 4


kv_bind.register_kv_cache_owner(_Owner)
