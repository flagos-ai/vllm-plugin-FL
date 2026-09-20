# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the per-request KPool tail ring."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_fl.models.glm5_next_kpool import (
    KpoolTailMetadataBuilder,
    compute_kpool_tail_slot_mapping,
)

KPOOL = 4


def _make_tail_block_table(own_blocks: list[int], width: int = 64) -> torch.Tensor:
    block_table = torch.zeros(len(own_blocks), width, dtype=torch.int32)
    block_table[:, 0] = torch.tensor(own_blocks, dtype=torch.int32)
    return block_table


def _legacy_tail_slots(
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    slots = []
    for request in range(block_table.shape[0]):
        for token in range(query_start_loc[request], query_start_loc[request + 1]):
            position = int(positions[token])
            block = int(block_table[request, position // KPOOL])
            slots.append(block * KPOOL + position % KPOOL)
    return torch.tensor(slots, dtype=torch.int64)


def _make_batch(per_request_positions, padded_len=None):
    positions = torch.cat(
        [torch.tensor(values, dtype=torch.int64) for values in per_request_positions]
    )
    num_actual_tokens = positions.numel()
    num_reqs = len(per_request_positions)
    lengths = [len(values) for values in per_request_positions]
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int64), 0, out=query_start_loc[1:])
    if padded_len is None:
        padded_len = num_actual_tokens
    slot_mapping = torch.full((padded_len,), -1, dtype=torch.int64)
    return (
        positions,
        query_start_loc,
        slot_mapping,
        num_actual_tokens,
        num_reqs,
    )


def _circular_tail_slots(
    slot_mapping,
    block_table,
    query_start_loc,
    positions,
    num_actual_tokens,
    num_reqs,
):
    return compute_kpool_tail_slot_mapping(
        slot_mapping,
        block_table,
        query_start_loc,
        positions,
        num_actual_tokens,
        num_reqs,
        KPOOL,
    )


def test_legacy_mapping_collapses_long_requests_onto_block_zero() -> None:
    own_blocks = [5, 9]
    per_request = [list(range(10)), list(range(12))]
    positions, query_start_loc, _, _, _ = _make_batch(per_request)
    legacy = _legacy_tail_slots(
        _make_tail_block_table(own_blocks), query_start_loc, positions
    )

    offset = 0
    request_slots = []
    for request, prompt in enumerate(per_request):
        slots = set()
        for position in range(len(prompt)):
            slot = int(legacy[offset + position])
            if position >= KPOOL:
                assert slot // KPOOL == 0
                assert slot // KPOOL != own_blocks[request]
            slots.add(slot)
        request_slots.append(slots)
        offset += len(prompt)
    assert request_slots[0] & request_slots[1]


def test_circular_mapping_isolates_concurrent_requests() -> None:
    own_blocks = [5, 9]
    per_request = [list(range(10)), list(range(12))]
    args = _make_batch(per_request)
    out = _circular_tail_slots(
        args[2],
        _make_tail_block_table(own_blocks),
        args[1],
        args[0],
        args[3],
        args[4],
    )

    offset = 0
    request_slots = []
    for request, prompt in enumerate(per_request):
        slots = set()
        for position in range(len(prompt)):
            slot = int(out[offset + position])
            assert slot // KPOOL == own_blocks[request]
            assert slot % KPOOL == position % KPOOL
            slots.add(slot)
        request_slots.append(slots)
        offset += len(prompt)
    assert not request_slots[0] & request_slots[1]


@pytest.mark.parametrize("prompt_len", [1, 2, 3, 4])
def test_circular_mapping_matches_generic_for_first_pool(prompt_len: int) -> None:
    per_request = [list(range(prompt_len))]
    positions, query_start_loc, slot_mapping, num_actual_tokens, num_reqs = _make_batch(
        per_request
    )
    block_table = _make_tail_block_table([7])
    legacy = _legacy_tail_slots(block_table, query_start_loc, positions)
    circular = _circular_tail_slots(
        slot_mapping,
        block_table,
        query_start_loc,
        positions,
        num_actual_tokens,
        num_reqs,
    )
    assert torch.equal(circular, legacy)


def test_circular_mapping_preserves_padding_and_empty_batch() -> None:
    per_request = [list(range(10)), list(range(12))]
    args = _make_batch(per_request, padded_len=30)
    block_table = _make_tail_block_table([5, 9])
    out = _circular_tail_slots(args[2], block_table, args[1], args[0], args[3], args[4])
    assert torch.equal(out[args[3] :], torch.full_like(out[args[3] :], -1))

    empty = _circular_tail_slots(args[2], block_table, args[1], args[0][:0], 0, args[4])
    assert torch.equal(empty, args[2])


def _make_common_metadata(
    per_request_positions, own_blocks, with_positions: bool = True
):
    positions, query_start_loc, slot_mapping, num_actual_tokens, num_reqs = _make_batch(
        per_request_positions,
        padded_len=sum(len(values) for values in per_request_positions) + 4,
    )
    seq_lens = torch.tensor(
        [max(values) + 1 if values else 1 for values in per_request_positions],
        dtype=torch.int64,
    )
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max(map(len, per_request_positions), default=1),
        max_seq_len=int(seq_lens.max()) if num_reqs else 1,
        block_table_tensor=_make_tail_block_table(own_blocks),
        slot_mapping=slot_mapping,
        positions=positions if with_positions else None,
    )


def _make_tail_builder():
    builder = object.__new__(KpoolTailMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(block_size=KPOOL)
    return builder


def test_builder_uses_circular_mapping() -> None:
    per_request = [list(range(10)), list(range(12))]
    own_blocks = [5, 9]
    metadata = _make_common_metadata(per_request, own_blocks)
    built = KpoolTailMetadataBuilder.build(_make_tail_builder(), 0, metadata)

    offset = 0
    for request, prompt in enumerate(per_request):
        for position in range(len(prompt)):
            slot = int(built.slot_mapping[offset + position])
            assert slot // KPOOL == own_blocks[request]
            assert slot % KPOOL == position % KPOOL
        offset += len(prompt)


def test_builder_keeps_generic_mapping_without_positions() -> None:
    metadata = _make_common_metadata([list(range(10))], [5], with_positions=False)
    built = KpoolTailMetadataBuilder.build(_make_tail_builder(), 0, metadata)
    assert built.slot_mapping is metadata.slot_mapping
