# Copyright (c) 2026 BAAI. All rights reserved.
"""Invalid-route alignment contracts for the MetaX Triton MoE adapter.

The CPU oracle tests its own contract only. The GPU tests run the real
launcher; atomic ordering of token indices within an expert is not assumed.
"""

import math

import pytest
import torch


def _require_metax():
    if not torch.cuda.is_available():
        pytest.skip("Requires a MetaX GPU")
    from tests.utils.device_utils import get_backend

    if get_backend() != "metax":
        pytest.skip("MetaX-only MoE alignment regression")


def _capacity(numel, block_size, num_experts, pad_sorted_ids):
    capacity = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        capacity = math.ceil(capacity / block_size) * block_size
    if numel < num_experts:
        capacity = min(numel * block_size, capacity)
    return capacity


def _oracle(topk_ids, block_size, num_experts, expert_map, ignore_invalid_experts):
    """Group flattened route indices by global expert, preserving membership."""
    ids = topk_ids.cpu().reshape(-1).tolist()
    mapping = expert_map.cpu().tolist() if expert_map is not None else None
    groups = []
    for expert in range(num_experts):
        mapped_expert = mapping[expert] if mapping is not None else expert
        if ignore_invalid_experts and mapped_expert < 0:
            continue
        members = [index for index, expert_id in enumerate(ids) if expert_id == expert]
        if members:
            groups.append((mapped_expert, members))
    return groups


def _assert_alignment(
    outputs,
    topk_ids,
    block_size,
    num_experts,
    expert_map=None,
    pad_sorted_ids=False,
    ignore_invalid_experts=False,
):
    sorted_ids, expert_ids, postpad = outputs
    assert all(tensor.dtype == torch.int32 for tensor in outputs)
    capacity = _capacity(topk_ids.numel(), block_size, num_experts, pad_sorted_ids)
    assert sorted_ids.shape == (capacity,)
    assert expert_ids.shape == (math.ceil(capacity / block_size),)
    assert postpad.shape == (1,)
    groups = _oracle(
        topk_ids, block_size, num_experts, expert_map, ignore_invalid_experts
    )
    sorted_cpu, experts_cpu, postpad_cpu = (tensor.cpu() for tensor in outputs)
    expected_postpad = sum(
        math.ceil(len(members) / block_size) * block_size for _, members in groups
    )
    assert postpad_cpu.item() == expected_postpad
    assert expected_postpad % block_size == 0
    cursor = 0
    for expert, members in groups:
        count = math.ceil(len(members) / block_size) * block_size
        region = sorted_cpu[cursor : cursor + count]
        assert ((region >= 0) & (region <= topk_ids.numel())).all().item()
        real = region[region < topk_ids.numel()].sort().values.tolist()
        assert real == members
        assert (region[len(members) :] == topk_ids.numel()).all().item()
        assert (
            (
                experts_cpu[cursor // block_size : (cursor + count) // block_size]
                == expert
            )
            .all()
            .item()
        )
        cursor += count
    assert (sorted_cpu[cursor:] == topk_ids.numel()).all().item()
    assert (experts_cpu[cursor // block_size :] == -1).all().item()


def _ids(tokens, experts, pattern, dtype):
    ids = ((torch.arange(tokens * 8) * 37 + 11) % experts).reshape(tokens, 8)
    if pattern == "invalid":
        bad_values = [-1, -7, experts, experts + 13]
        if dtype == torch.int64:
            bad_values += [2**32 + 1, -(2**32)]
        bad = torch.tensor(bad_values)
        ids = bad[torch.arange(tokens * 8) % bad.numel()].reshape(tokens, 8)
    elif pattern == "mixed":
        flat = ids.reshape(-1)
        flat[::5] = -1
        flat[::11] = experts + 3
        flat[::17] = -7
        if dtype == torch.int64:
            # Bounds must be checked before narrowing to int32: these values
            # otherwise wrap to apparently valid expert IDs 1 and 0.
            flat[1::19] = 2**32 + 1
            flat[2::23] = -(2**32)
        if tokens:
            ids[-1].fill_(-1)
    return ids.to(dtype)


def _mapping(experts):
    # Preserve the production global expert order while mapping to dense local
    # expert indices. Even global experts are non-local on this synthetic rank.
    mapping = torch.full((experts,), -1, dtype=torch.int32)
    mapping[1::2] = torch.arange(experts // 2, dtype=torch.int32)
    return mapping


@pytest.mark.parametrize("ignore_invalid_experts", [False, True])
def test_cpu_oracle_excludes_raw_invalid_routes_and_handles_nonlocal_experts(
    ignore_invalid_experts,
):
    ids = torch.tensor([[2, -1, 7, 0], [2, 1, -2, 6]])
    mapping = torch.tensor([-1, 2, 0, -1, -1, -1, 1], dtype=torch.int32)
    # The result order follows global IDs 0, 1, 2, 6, not mapped local IDs.
    expected = [(2, [5]), (0, [0, 4]), (1, [7])]
    if not ignore_invalid_experts:
        expected.insert(0, (-1, [3]))
    assert _oracle(ids, 4, 7, mapping, ignore_invalid_experts) == expected


def test_cpu_oracle_empty_and_all_invalid_have_zero_padded_tokens():
    for ids in (
        torch.empty((0, 8), dtype=torch.int64),
        _ids(256, 7, "invalid", torch.int32),
    ):
        assert _oracle(ids, 16, 7, None, False) == []
        capacity = _capacity(ids.numel(), 16, 7, True)
        outputs = (
            torch.full((capacity,), ids.numel(), dtype=torch.int32),
            torch.full((math.ceil(capacity / 16),), -1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )
        _assert_alignment(outputs, ids, 16, 7, pad_sorted_ids=True)


# Curated combinations avoid a full Cartesian product while covering the
# sampler's 256-row all-padding batch, >8-row alignment, E=7 tails, and E=256.
_CASES = [
    pytest.param(0, 7, "mixed", torch.int32, 16, False, False, False, id="empty"),
    pytest.param(0, 256, "invalid", torch.int64, 16, True, True, True, id="empty-map"),
    pytest.param(
        9, 7, "valid", torch.int32, 4, False, False, False, id="valid9-e7-i32"
    ),
    pytest.param(
        256, 256, "valid", torch.int64, 32, False, False, False, id="valid256-e256-i64"
    ),
    pytest.param(
        9, 7, "mixed", torch.int64, 16, True, False, False, id="mixed9-e7-i64"
    ),
    pytest.param(
        256, 256, "mixed", torch.int32, 16, True, False, False, id="mixed256-e256-i32"
    ),
    pytest.param(
        9, 7, "invalid", torch.int32, 16, False, False, False, id="invalid9-e7"
    ),
    pytest.param(
        256, 256, "invalid", torch.int64, 16, False, False, False, id="invalid256-e256"
    ),
    pytest.param(
        8, 256, "valid", torch.int32, 64, False, False, False, id="valid8-e256"
    ),
    pytest.param(
        8, 256, "mixed", torch.int64, 32, True, False, False, id="mixed8-e256"
    ),
    pytest.param(
        9, 7, "mixed", torch.int32, 8, False, True, False, id="mixed9-map-keep"
    ),
    pytest.param(
        9, 7, "mixed", torch.int64, 8, True, True, True, id="mixed9-map-ignore"
    ),
    pytest.param(
        256, 7, "invalid", torch.int32, 16, False, True, False, id="invalid256-map-keep"
    ),
    pytest.param(
        256, 7, "invalid", torch.int64, 16, True, True, True, id="invalid256-map-ignore"
    ),
    pytest.param(
        256, 256, "valid", torch.int32, 32, True, True, False, id="valid256-map-keep"
    ),
    pytest.param(
        256, 256, "mixed", torch.int64, 32, False, True, True, id="mixed256-map-ignore"
    ),
]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "tokens,experts,pattern,dtype,block_size,pad,mapped,ignore", _CASES
)
@torch.inference_mode()
def test_metax_moe_align_invalid_routes(
    tokens, experts, pattern, dtype, block_size, pad, mapped, ignore
):
    _require_metax()
    from vllm_fl.dispatch.backends.vendor.metax.impl.moe_align_block_size import (
        moe_align_block_size,
    )

    ids_cpu = _ids(tokens, experts, pattern, dtype)
    mapping_cpu = _mapping(experts) if mapped else None
    ids = ids_cpu.cuda()
    mapping = mapping_cpu.cuda() if mapped else None
    outputs = moe_align_block_size(ids, block_size, experts, mapping, pad, ignore)
    assert all(tensor.device == ids.device for tensor in outputs)
    torch.cuda.synchronize()
    _assert_alignment(outputs, ids_cpu, block_size, experts, mapping_cpu, pad, ignore)
    torch.testing.assert_close(ids.cpu(), ids_cpu, rtol=0, atol=0)
    if mapped:
        torch.testing.assert_close(mapping.cpu(), mapping_cpu, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("experts,mapped", [(7, False), (256, True)])
@torch.inference_mode()
def test_metax_moe_align_graph_replay_refreshes_counts_and_sentinels(experts, mapped):
    _require_metax()
    from vllm_fl.dispatch.backends.vendor.metax.impl.moe_align_block_size import (
        moe_align_block_size,
    )

    tokens, block_size = 256, 16
    ids = _ids(tokens, experts, "valid", torch.int64).cuda()
    mapping_cpu = _mapping(experts) if mapped else None
    mapping = mapping_cpu.cuda() if mapped else None
    for _ in range(2):
        moe_align_block_size(ids, block_size, experts, mapping, True, mapped)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = moe_align_block_size(ids, block_size, experts, mapping, True, mapped)
    assert all(tensor.device == ids.device for tensor in outputs)
    # Data-dependent counts and all output tails must be reset every replay.
    # No tensor reads, CPU copies or synchronization occur inside capture.
    for pattern in ("valid", "invalid", "mixed"):
        ids_cpu = _ids(tokens, experts, pattern, torch.int64)
        ids.copy_(ids_cpu)
        graph.replay()
        torch.cuda.synchronize()
        _assert_alignment(
            outputs, ids_cpu, block_size, experts, mapping_cpu, True, mapped
        )


def _poisoned_expert_output(ids, experts, mapping, width):
    """CPU values for cleanup tests; no provider or alignment code is reused."""
    valid = (ids >= 0) & (ids < experts)
    if mapping is not None:
        local = mapping[ids.clamp(0, experts - 1).long()]
        valid &= local >= 0
    rng = torch.Generator().manual_seed(320)
    finite = torch.randn((*ids.shape, width), generator=rng).bfloat16()
    route_index = torch.arange(ids.numel()).reshape(ids.shape)
    poison = torch.where(route_index % 2 == 0, float("nan"), -101.0).bfloat16()
    original = torch.where(valid[..., None], finite, poison[..., None])
    expected = torch.where(valid[..., None], original, 0.0)
    return original, expected, valid


def _assert_cleaned_expert_output(actual, original, expected, valid):
    actual_cpu = actual.cpu()
    torch.testing.assert_close(actual_cpu, expected, rtol=0, atol=0)
    assert torch.equal(
        actual_cpu[valid].view(torch.int16), original[valid].view(torch.int16)
    )
    assert (actual_cpu[~valid] == 0).all().item()
    assert torch.isfinite(actual_cpu).all().item()
    # This checks the following reduction too: dropped NaN/old workspace rows
    # contribute zero and valid expert contributions remain unchanged.
    torch.testing.assert_close(
        actual.float().sum(1).cpu(), expected.float().sum(1), rtol=1e-5, atol=1e-6
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "tokens,experts,dtype,mapped,width",
    [
        pytest.param(9, 7, torch.int32, False, 17, id="mixed-i32-tail"),
        pytest.param(256, 256, torch.int64, True, 128, id="mixed-i64-nonlocal"),
        pytest.param(0, 7, torch.int32, True, 17, id="empty"),
    ],
)
@torch.inference_mode()
def test_metax_moe_cleanup_zeros_only_invalid_route_slots(
    tokens, experts, dtype, mapped, width
):
    _require_metax()
    from vllm_fl.dispatch.backends.vendor.metax.impl.moe_align_block_size import (
        zero_invalid_expert_outputs,
    )

    ids_cpu = _ids(tokens, experts, "mixed", dtype)
    mapping_cpu = _mapping(experts) if mapped else None
    original, expected, valid = _poisoned_expert_output(
        ids_cpu, experts, mapping_cpu, width
    )
    ids = ids_cpu.cuda()
    output = original.cuda()
    mapping = mapping_cpu.cuda() if mapped else None
    result = zero_invalid_expert_outputs(output, ids, experts, mapping)
    assert result is None
    torch.cuda.synchronize()
    _assert_cleaned_expert_output(output, original, expected, valid)
    torch.testing.assert_close(ids.cpu(), ids_cpu, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("mapped", [False, True])
@torch.inference_mode()
def test_metax_moe_cleanup_graph_replay_with_changed_ids_and_poison(mapped):
    _require_metax()
    from vllm_fl.dispatch.backends.vendor.metax.impl.moe_align_block_size import (
        zero_invalid_expert_outputs,
    )

    tokens, experts, width = 256, 7, 17
    dtype = torch.int64 if mapped else torch.int32
    ids_cpu = _ids(tokens, experts, "valid", dtype)
    mapping_cpu = _mapping(experts) if mapped else None
    original, _, _ = _poisoned_expert_output(ids_cpu, experts, mapping_cpu, width)
    ids = ids_cpu.cuda()
    output = original.cuda()
    mapping = mapping_cpu.cuda() if mapped else None
    for _ in range(2):
        zero_invalid_expert_outputs(output, ids, experts, mapping)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = zero_invalid_expert_outputs(output, ids, experts, mapping)
    assert result is None
    for pattern in ("valid", "invalid", "mixed"):
        ids_cpu = _ids(tokens, experts, pattern, dtype)
        original, expected, valid = _poisoned_expert_output(
            ids_cpu, experts, mapping_cpu, width
        )
        ids.copy_(ids_cpu)
        output.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        _assert_cleaned_expert_output(output, original, expected, valid)
