# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields

import pytest
import torch

from vllm_fl.models.hy_v4_flash.nvidia.indexpool_metadata import (
    expand_groups,
    prepare_indexpool_metadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Hy4 Flash graph metadata tests require NVIDIA CUDA",
)


def make_inputs(mrope):
    tokens, ratio = 8, 4
    starts = torch.tensor([0, 2, 2, 7], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([7, 0, 11], device="cuda", dtype=torch.int32)
    pos = torch.arange(3 * tokens, device="cuda", dtype=torch.int64).reshape(3, tokens)
    tail_storage = torch.arange(
        3 * 8 * 262, device="cuda", dtype=torch.float32
    ).reshape(3, 8, 262)
    tail = tail_storage[..., ::2]
    cache = torch.empty((6, 2, 132), device="cuda", dtype=torch.uint8)
    pool_table = torch.tensor([[1, 4], [4, 5], [2, 3]], device="cuda")
    tail_table = torch.tensor([[2], [-1], [0]], device="cuda")
    return (
        pos if mrope else pos[0],
        tail,
        cache,
        pool_table,
        tail_table,
        starts,
        lengths,
        tokens,
        ratio,
    )


def reference(args):
    pos, tail, pool, pt, tt, starts, lengths, tokens, ratio = args
    pos, tail, pt, tt = (x.cpu() for x in (pos, tail, pt, tt))
    starts, lengths = starts.tolist(), lengths.tolist()
    batch, ring, page_size = len(lengths), tail.shape[1], pool.shape[1]
    spots = (tokens + batch * (ratio - 1)) // ratio
    group_keys = torch.full((spots, ratio), -1, dtype=torch.int64)
    group_tails = torch.full_like(group_keys, -1)
    group_positions = torch.zeros(
        (3, spots) if pos.ndim == 2 else (spots,), dtype=torch.int64
    )
    pool_slots = torch.full((spots,), -1, dtype=torch.int64)
    tail_keys = torch.full((batch * ring,), -1, dtype=torch.int64)
    tail_slots = torch.full_like(tail_keys, -1)
    owner = torch.full((tokens,), batch - 1, dtype=torch.int64)
    logical = (
        torch.arange(tokens)
        - starts[-2]
        + max(lengths[-1] - (starts[-1] - starts[-2]), 0)
    )
    upper = torch.zeros(tokens, dtype=torch.int32)
    live = torch.zeros(tokens, dtype=torch.bool)
    slot = 0

    for req, (start, end, length) in enumerate(zip(starts, starts[1:], lengths)):
        context = length - (end - start)
        if not (0 <= start < end <= tokens and context >= 0):
            continue
        owner[start:end] = req
        logical[start:end] = torch.arange(context, length)
        upper[start:end] = torch.arange(context, length) // ratio
        live[start:end] = True
        block = int(tt[req, 0])
        for group in range(context // ratio, length // ratio):
            column, offset = divmod(group, page_size)
            if column < pt.shape[1] and 0 <= pt[req, column] < len(pool):
                pool_slots[slot] = pt[req, column] * page_size + offset
            for member in range(ratio):
                time = group * ratio + member
                if time < context:
                    if 0 <= block < len(tail):
                        group_tails[slot, member] = block * ring + time % ring
                        if member == 0:
                            coord = tail[block, time % ring, 128:131].long()
                            group_positions[..., slot] = (
                                coord if pos.ndim == 2 else coord[0]
                            )
                else:
                    row = start + time - context
                    group_keys[slot, member] = row
                    if member == 0:
                        group_positions[..., slot] = pos[..., row]
            slot += 1
        if 0 <= block < len(tail):
            for delta, time in enumerate(range(length - ring, length)):
                if time >= context:
                    tail_keys[req * ring + delta] = start + time - context
                    tail_slots[req * ring + delta] = block * ring + time % ring
    return dict(
        group_positions=group_positions,
        gather_key_rows=group_keys,
        gather_tail_slots=group_tails,
        pool_slots=pool_slots,
        tail_key_rows=tail_keys,
        tail_slots=tail_slots,
        owner=owner,
        logical=logical,
        upper=upper,
        live=live,
    )


def assert_plan(plan, args):
    expected = reference(args)
    for field in fields(plan):
        torch.testing.assert_close(
            getattr(plan, field.name).cpu(), expected[field.name], msg=field.name
        )


@pytest.mark.parametrize("mrope", [True, False])
def test_indexpool_metadata_plans_carry_ring_wrap_and_padding(mrope):
    args = make_inputs(mrope)
    assert_plan(prepare_indexpool_metadata(*args), args)


def test_graph_replay_remaps_reordered_requests_and_reused_pages():
    args = make_inputs(True)
    pos, tail, _, pt, tt, starts, lengths, _, ratio = args
    group_ids = torch.zeros((8, 2), device="cuda", dtype=torch.int32)
    prepare_indexpool_metadata(*args)
    expand_groups(group_ids, torch.arange(8, device="cuda"), ratio)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan = prepare_indexpool_metadata(*args)
        expanded = expand_groups(group_ids, plan.logical, ratio, active=plan.live)
    starts.copy_(torch.tensor([0, 5, 7, 7], device="cuda"))
    lengths.copy_(torch.tensor([13, 9, 0], device="cuda"))
    pt.copy_(torch.tensor([[2, 3], [1, 4], [-1, -1]], device="cuda"))
    tt.copy_(torch.tensor([[0], [2], [-1]], device="cuda"))
    pos.add_(37)
    tail[0, :, 128:].add_(196)
    group_ids[:, 1].fill_(1)
    graph.replay()
    assert_plan(plan, args)
    expected = []
    ref = reference(args)
    for groups, logical, live in zip(
        group_ids.tolist(), ref["logical"].tolist(), ref["live"].tolist()
    ):
        history = []
        if live:
            current, offset = divmod(logical, ratio)
            for group in groups:
                if 0 <= group < current:
                    history.extend(range(group * ratio, group * ratio + ratio))
            history.extend(range(current * ratio, current * ratio + offset + 1))
        expected.append(history + [-1] * (expanded.shape[1] - len(history)))
    torch.testing.assert_close(
        expanded.cpu(), torch.tensor(expected, dtype=torch.int32)
    )
