# Copyright (c) 2025 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators import all2all
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from vllm_fl.distributed.device_communicators import npu_communicator as npu_module
from vllm_fl.distributed.device_communicators.npu_communicator import (
    NPUCommunicator,
)


def _bare_communicator(
    world_size: int, rank: int, *, global_rank: int | None = None
) -> NPUCommunicator:
    communicator = object.__new__(NPUCommunicator)
    communicator.world_size = world_size
    communicator.rank = rank if global_rank is None else global_rank
    communicator.rank_in_group = rank
    communicator.device_group = object()
    return communicator


def _sizes_for(world_size: int, pattern: str) -> list[int]:
    if pattern == "equal":
        return [2] * world_size
    if pattern == "single-real-multi-idle":
        return [3] + [0] * (world_size - 1)
    if pattern == "uneven":
        return [(rank * 2 + 1) % 4 for rank in range(world_size)]
    raise AssertionError(f"unknown pattern: {pattern}")


def _rank_tensor(size: int, rank: int, width: int = 3) -> torch.Tensor:
    if size == 0:
        return torch.empty((0, width), dtype=torch.float32)
    return torch.arange(size * width, dtype=torch.float32).reshape(size, width) + (
        rank * 100
    )


def _pad_dim_zero(tensor: torch.Tensor, padded_size: int) -> torch.Tensor:
    padding = padded_size - tensor.shape[0]
    if padding == 0:
        return tensor.contiguous()
    return torch.cat(
        [tensor, torch.zeros((padding,) + tensor.shape[1:], dtype=tensor.dtype)],
        dim=0,
    ).contiguous()


@pytest.mark.parametrize("backend", ["naive", "allgather_reducescatter"])
@pytest.mark.parametrize("world_size,rank", [(1, 0), (3, 2), (8, 7)])
def test_initializes_agrs_manager_for_supported_backend(
    monkeypatch, backend, world_size, rank
):
    created_groups = []

    class FakeAgRsAll2AllManager:
        def __init__(self, cpu_group):
            created_groups.append(cpu_group)

    def fake_base_init(self, cpu_group, device, device_group, unique_name):
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.world_size = world_size
        self.rank = rank
        self.rank_in_group = rank
        self.use_all2all = True
        self.all2all_backend = backend
        self.all2all_manager = None

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base_init)
    monkeypatch.setattr(all2all, "AgRsAll2AllManager", FakeAgRsAll2AllManager)
    monkeypatch.setattr(
        npu_module.torch,
        "npu",
        SimpleNamespace(current_device=lambda: 7),
        raising=False,
    )

    cpu_group = object()
    communicator = NPUCommunicator(
        cpu_group, device_group=object(), unique_name="ep:any-topology"
    )

    assert isinstance(communicator.all2all_manager, FakeAgRsAll2AllManager)
    assert created_groups == [cpu_group]
    assert communicator.world_size == world_size
    assert communicator.rank_in_group == rank


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("pattern", ["equal", "uneven", "single-real-multi-idle"])
def test_all_gatherv_padding_and_unpacking_matrix(
    monkeypatch, world_size: int, pattern: str
):
    sizes = _sizes_for(world_size, pattern)
    tensors = [_rank_tensor(size, rank) for rank, size in enumerate(sizes)]
    max_size = max(sizes)
    expected = torch.cat(tensors, dim=0)

    for rank, local_tensor in enumerate(tensors):
        communicator = _bare_communicator(world_size, rank)
        calls = []

        def fake_all_gather(
            output,
            padded_input,
            group,
            calls=calls,
            local_tensor=local_tensor,
        ):
            calls.append((output.shape, padded_input.clone(), group))
            assert padded_input.shape == (max_size, 3)
            torch.testing.assert_close(
                padded_input, _pad_dim_zero(local_tensor, max_size)
            )
            output.copy_(torch.cat([_pad_dim_zero(t, max_size) for t in tensors]))

        monkeypatch.setattr(npu_module.dist, "all_gather_into_tensor", fake_all_gather)
        actual = communicator.all_gatherv(local_tensor, dim=0, sizes=sizes)

        torch.testing.assert_close(actual, expected)
        assert actual.shape == (sum(sizes), 3)
        assert actual.dtype == local_tensor.dtype
        assert actual.device == local_tensor.device
        assert len(calls) == (0 if world_size == 1 else 1)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("pattern", ["equal", "uneven", "single-real-multi-idle"])
def test_reduce_scatterv_padding_and_unpacking_matrix(
    monkeypatch, world_size: int, pattern: str
):
    sizes = _sizes_for(world_size, pattern)
    max_size = max(sizes)
    contributions = []
    for source_rank in range(world_size):
        values = torch.arange(sum(sizes) * 2, dtype=torch.float32).reshape(-1, 2)
        contributions.append(values + source_rank * 1000)

    def padded_contribution(tensor):
        return torch.cat(
            [_pad_dim_zero(chunk, max_size) for chunk in tensor.split(sizes, dim=0)],
            dim=0,
        )

    padded_contributions = [padded_contribution(t) for t in contributions]
    reduced_padded = torch.stack(padded_contributions).sum(dim=0)

    for rank, local_input in enumerate(contributions):
        communicator = _bare_communicator(world_size, rank)
        calls = []

        def fake_reduce_scatter(
            output,
            padded_input,
            group,
            calls=calls,
            rank=rank,
        ):
            calls.append((output.shape, padded_input.clone(), group))
            assert padded_input.shape == (world_size * max_size, 2)
            torch.testing.assert_close(padded_input, padded_contributions[rank])
            output.copy_(reduced_padded[rank * max_size : (rank + 1) * max_size])

        monkeypatch.setattr(
            npu_module.dist, "reduce_scatter_tensor", fake_reduce_scatter
        )
        actual = communicator.reduce_scatterv(local_input, dim=0, sizes=sizes)
        expected = reduced_padded[rank * max_size : rank * max_size + sizes[rank]]

        torch.testing.assert_close(actual, expected)
        assert actual.shape == (sizes[rank], 2)
        assert actual.dtype == local_input.dtype
        assert actual.device == local_input.device
        assert len(calls) == (0 if world_size == 1 else 1)


def test_all_gatherv_supports_tensor_list_and_nonzero_dim(monkeypatch):
    communicator = _bare_communicator(world_size=3, rank=1)
    sizes = [1, 2, 3]
    rank_payloads = [
        [
            torch.arange(8, dtype=torch.float32).reshape(2, 4) + rank * 100,
            torch.arange(6, dtype=torch.int64).reshape(2, 3) + rank * 10,
        ]
        for rank in range(3)
    ]
    # Slice each rank's payload along dim 1 according to its public size.
    rank_payloads = [
        [payload[:, : sizes[rank]] for payload in payload_list]
        for rank, payload_list in enumerate(rank_payloads)
    ]
    call_index = 0

    def fake_all_gather(output, padded_input, group):
        nonlocal call_index
        tensor_index = call_index
        call_index += 1
        moved = [payload[tensor_index].movedim(1, 0) for payload in rank_payloads]
        output.copy_(torch.cat([_pad_dim_zero(t, 3) for t in moved]))

    monkeypatch.setattr(npu_module.dist, "all_gather_into_tensor", fake_all_gather)
    actual = communicator.all_gatherv(rank_payloads[1], dim=1, sizes=sizes)

    assert isinstance(actual, list)
    assert len(actual) == 2
    assert call_index == 2
    for tensor_index, output in enumerate(actual):
        expected = torch.cat(
            [payload[tensor_index] for payload in rank_payloads], dim=1
        )
        torch.testing.assert_close(output, expected)
        assert output.dtype == rank_payloads[1][tensor_index].dtype


def test_reduce_scatterv_supports_nonzero_dim(monkeypatch):
    world_size = 3
    sizes = [1, 2, 3]
    max_size = max(sizes)
    contributions = [
        torch.arange(12, dtype=torch.float32).reshape(2, 6) + source_rank * 100
        for source_rank in range(world_size)
    ]

    def padded_contribution(tensor):
        moved = tensor.movedim(1, 0).contiguous()
        return torch.cat(
            [_pad_dim_zero(chunk, max_size) for chunk in moved.split(sizes, dim=0)],
            dim=0,
        )

    padded_contributions = [padded_contribution(t) for t in contributions]
    reduced_padded = torch.stack(padded_contributions).sum(dim=0)

    for rank_in_group, local_input in enumerate(contributions):
        communicator = _bare_communicator(
            world_size,
            rank_in_group,
            global_rank=71 + rank_in_group * 4,
        )
        calls = []

        def fake_reduce_scatter(
            output,
            padded_input,
            group,
            calls=calls,
            rank_in_group=rank_in_group,
        ):
            calls.append((output, padded_input, group))
            assert padded_input.shape == (world_size * max_size, 2)
            torch.testing.assert_close(
                padded_input, padded_contributions[rank_in_group]
            )
            output.copy_(
                reduced_padded[
                    rank_in_group * max_size : (rank_in_group + 1) * max_size
                ]
            )

        monkeypatch.setattr(
            npu_module.dist, "reduce_scatter_tensor", fake_reduce_scatter
        )
        actual = communicator.reduce_scatterv(local_input, dim=1, sizes=sizes)
        expected = reduced_padded[
            rank_in_group * max_size : rank_in_group * max_size + sizes[rank_in_group]
        ].movedim(0, 1)

        torch.testing.assert_close(actual, expected)
        assert actual.shape == (2, sizes[rank_in_group])
        assert actual.dtype == local_input.dtype
        assert actual.device == local_input.device
        assert communicator.rank != communicator.rank_in_group
        assert len(calls) == 1


def test_world_size_one_does_not_call_collectives(monkeypatch):
    communicator = _bare_communicator(world_size=1, rank=0)
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def unexpected(*args, **kwargs):
        raise AssertionError("world_size=1 must not invoke a distributed collective")

    monkeypatch.setattr(npu_module.dist, "all_gather_into_tensor", unexpected)
    monkeypatch.setattr(npu_module.dist, "reduce_scatter_tensor", unexpected)

    torch.testing.assert_close(communicator.all_gatherv(tensor, sizes=[2]), tensor)
    torch.testing.assert_close(
        communicator.all_gatherv([tensor, tensor.to(torch.int64)], sizes=[2])[1],
        tensor.to(torch.int64),
    )
    torch.testing.assert_close(
        communicator.reduce_scatterv(tensor, dim=0, sizes=[2]), tensor
    )


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("sizes_mode", ["none", "equal"])
def test_equal_all_gatherv_uses_direct_collective_for_every_subgroup_rank(
    monkeypatch, world_size, sizes_mode
):
    sizes = None if sizes_mode == "none" else [2] * world_size
    rank_tensors = [_rank_tensor(2, rank) for rank in range(world_size)]

    for rank_in_group, local_tensor in enumerate(rank_tensors):
        communicator = _bare_communicator(
            world_size,
            rank_in_group,
            global_rank=101 + rank_in_group * 3,
        )
        calls = []

        def fake_all_gather(
            output,
            collective_input,
            group,
            calls=calls,
            local_tensor=local_tensor,
        ):
            calls.append((output, collective_input, group))
            assert collective_input.data_ptr() == local_tensor.data_ptr()
            output.copy_(torch.cat(rank_tensors, dim=0))

        monkeypatch.setattr(npu_module.dist, "all_gather_into_tensor", fake_all_gather)
        actual = communicator.all_gatherv(local_tensor, dim=-2, sizes=sizes)

        torch.testing.assert_close(actual, torch.cat(rank_tensors, dim=0))
        assert communicator.rank != communicator.rank_in_group
        assert len(calls) == (0 if world_size == 1 else 1)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("sizes_mode", ["none", "equal"])
def test_equal_reduce_scatterv_uses_direct_collective_for_every_subgroup_rank(
    monkeypatch, world_size, sizes_mode
):
    sizes = None if sizes_mode == "none" else [2] * world_size
    local_input = torch.arange(world_size * 4, dtype=torch.float32).reshape(-1, 2)

    for rank_in_group in range(world_size):
        communicator = _bare_communicator(
            world_size,
            rank_in_group,
            global_rank=211 + rank_in_group * 5,
        )
        calls = []

        def fake_reduce_scatter(
            output,
            collective_input,
            group,
            calls=calls,
            rank_in_group=rank_in_group,
        ):
            calls.append((output, collective_input, group))
            assert collective_input.data_ptr() == local_input.data_ptr()
            start = rank_in_group * 2
            output.copy_(local_input[start : start + 2] * world_size)

        monkeypatch.setattr(
            npu_module.dist, "reduce_scatter_tensor", fake_reduce_scatter
        )
        actual = communicator.reduce_scatterv(local_input, dim=-2, sizes=sizes)
        expected = local_input[rank_in_group * 2 : (rank_in_group + 1) * 2] * world_size

        torch.testing.assert_close(actual, expected)
        assert communicator.rank != communicator.rank_in_group
        assert len(calls) == (0 if world_size == 1 else 1)


def test_equal_all_gatherv_fast_path_preserves_negative_nonzero_dim(monkeypatch):
    world_size = 3
    rank_in_group = 1
    communicator = _bare_communicator(world_size, rank_in_group, global_rank=41)
    rank_tensors = [
        (torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank * 100).T
        for rank in range(world_size)
    ]
    local_tensor = rank_tensors[rank_in_group]

    def fake_all_gather(output, collective_input, group):
        assert collective_input.shape == (2, 3)
        # movedim(-1, 0) makes this transposed input contiguous without a copy.
        assert collective_input.data_ptr() == local_tensor.data_ptr()
        output.copy_(
            torch.cat([tensor.movedim(-1, 0) for tensor in rank_tensors], dim=0)
        )

    monkeypatch.setattr(npu_module.dist, "all_gather_into_tensor", fake_all_gather)
    actual = communicator.all_gatherv(local_tensor, dim=-1, sizes=[2] * world_size)

    torch.testing.assert_close(actual, torch.cat(rank_tensors, dim=-1))
    assert actual.shape == (3, 6)


def test_equal_reduce_scatterv_fast_path_preserves_negative_nonzero_dim(monkeypatch):
    world_size = 3
    rank_in_group = 1
    communicator = _bare_communicator(world_size, rank_in_group, global_rank=43)
    base = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    local_input = base.T

    def fake_reduce_scatter(output, collective_input, group):
        assert collective_input.shape == (6, 3)
        assert collective_input.data_ptr() == local_input.data_ptr()
        output.copy_(
            collective_input[rank_in_group * 2 : (rank_in_group + 1) * 2] * world_size
        )

    monkeypatch.setattr(npu_module.dist, "reduce_scatter_tensor", fake_reduce_scatter)
    actual = communicator.reduce_scatterv(local_input, dim=-1, sizes=[2] * world_size)

    expected = (base[rank_in_group * 2 : (rank_in_group + 1) * 2] * world_size).T
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3, 2)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
def test_all_zero_sizes_for_every_noncontiguous_subgroup_rank(monkeypatch, world_size):
    def unexpected(*args, **kwargs):
        raise AssertionError("all-zero sizes must not invoke a collective")

    monkeypatch.setattr(npu_module.dist, "all_gather_into_tensor", unexpected)
    monkeypatch.setattr(npu_module.dist, "reduce_scatter_tensor", unexpected)
    sizes = [0] * world_size
    tensor = torch.empty((0, 3), dtype=torch.float32)

    for rank_in_group in range(world_size):
        communicator = _bare_communicator(
            world_size,
            rank_in_group,
            global_rank=307 + rank_in_group * 7,
        )
        gathered = communicator.all_gatherv(tensor, dim=-2, sizes=sizes)
        scattered = communicator.reduce_scatterv(tensor, dim=-2, sizes=sizes)

        assert gathered.shape == (0, 3)
        assert scattered.shape == (0, 3)
        assert communicator.rank != communicator.rank_in_group


def test_manager_methods_forward_all_arguments_and_results():
    communicator = _bare_communicator(world_size=3, rank=1, global_rank=17)
    communicator.all2all_manager = Mock()
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 8)
    topk_weights = torch.randn(2, 2)
    topk_ids = torch.ones(2, 2, dtype=torch.int64)
    extra_tensors = [torch.randn(2, 1), torch.ones(2, dtype=torch.int64)]
    router_result = (hidden_states + 1, router_logits + 1, extra_tensors)
    dispatch_result = (hidden_states + 2, topk_weights + 2, topk_ids, extra_tensors)
    combine_result = hidden_states + 3
    communicator.all2all_manager.dispatch_router_logits.return_value = router_result
    communicator.all2all_manager.dispatch.return_value = dispatch_result
    communicator.all2all_manager.combine.return_value = combine_result

    assert (
        communicator.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel=True,
            extra_tensors=extra_tensors,
        )
        is router_result
    )
    assert (
        communicator.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel=True,
            extra_tensors=extra_tensors,
        )
        is dispatch_result
    )
    assert (
        communicator.combine(hidden_states, is_sequence_parallel=True) is combine_result
    )
    communicator.all2all_manager.dispatch_router_logits.assert_called_once_with(
        hidden_states, router_logits, True, extra_tensors
    )
    communicator.all2all_manager.dispatch.assert_called_once_with(
        hidden_states,
        topk_weights,
        topk_ids,
        True,
        extra_tensors=extra_tensors,
    )
    communicator.all2all_manager.combine.assert_called_once_with(hidden_states, True)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("dispatch_router_logits", (torch.ones(1, 2), torch.ones(1, 3))),
        (
            "dispatch",
            (torch.ones(1, 2), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.int64)),
        ),
        ("combine", (torch.ones(1, 2),)),
    ],
)
def test_manager_methods_require_initialized_manager(method, args):
    communicator = _bare_communicator(world_size=2, rank=0)
    communicator.all2all_manager = None
    with pytest.raises(AssertionError):
        getattr(communicator, method)(*args)


def test_destroy_releases_manager_and_communicator_surface():
    communicator = _bare_communicator(world_size=2, rank=0)
    manager = Mock()
    communicator.all2all_manager = manager
    communicator.ca_comm = object()

    communicator.destroy()

    manager.destroy.assert_called_once_with()
    assert communicator.all2all_manager is None
    assert communicator.ca_comm is None


def test_all_to_all_variable_sizes_use_subgroup_rank(monkeypatch):
    communicator = _bare_communicator(
        world_size=3,
        rank=1,
        global_rank=11,
    )
    scatter_sizes = [1, 2, 3]
    gather_sizes = [2, 1, 3]
    input_tensor = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    calls = []

    def fake_all_to_all(output_list, input_list, group):
        calls.append((output_list, input_list, group))
        assert [tensor.shape for tensor in input_list] == [
            (1, 4),
            (2, 4),
            (3, 4),
        ]
        assert [tensor.shape for tensor in output_list] == [
            (2, 2),
            (2, 1),
            (2, 3),
        ]
        for source_rank, output in enumerate(output_list):
            output.fill_(source_rank + 1)

    monkeypatch.setattr(npu_module.dist, "all_to_all", fake_all_to_all)
    actual = communicator.all_to_all(
        input_tensor,
        scatter_dim=0,
        gather_dim=1,
        scatter_sizes=scatter_sizes,
        gather_sizes=gather_sizes,
    )

    expected = torch.cat(
        [
            torch.full((2, size), source_rank + 1, dtype=torch.float32)
            for source_rank, size in enumerate(gather_sizes)
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected)
    assert communicator.rank == 11
    assert communicator.rank_in_group == 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("method", "tensor", "dim", "sizes", "message"),
    [
        ("all_gatherv", torch.tensor(1.0), 0, [1, 1], "at least one dimension"),
        ("all_gatherv", torch.ones(1, 2), 2, [1, 1], "Invalid dim"),
        ("all_gatherv", torch.ones(1, 2), 0, [1], "sizes length"),
        ("all_gatherv", torch.ones(1, 2), 0, [1, -1], "non-negative"),
        ("all_gatherv", torch.ones(1, 2), 0, [2, 0], "local input size"),
        ("reduce_scatterv", torch.tensor(1.0), 0, [1, 1], "at least one dimension"),
        ("reduce_scatterv", torch.ones(2, 2), 2, [1, 1], "Invalid dim"),
        ("reduce_scatterv", torch.ones(2, 2), 0, [1], "sizes length"),
        ("reduce_scatterv", torch.ones(2, 2), 0, [1, -1], "non-negative"),
        ("reduce_scatterv", torch.ones(3, 2), 0, [1, 1], "sum of sizes"),
    ],
)
def test_variable_collectives_validate_contract(method, tensor, dim, sizes, message):
    communicator = _bare_communicator(world_size=2, rank=0)
    with pytest.raises(AssertionError, match=message):
        getattr(communicator, method)(tensor, dim=dim, sizes=sizes)


def test_variable_collectives_validate_rank_and_list_type():
    communicator = _bare_communicator(world_size=2, rank=2)
    with pytest.raises(AssertionError, match="rank_in_group"):
        communicator.all_gatherv(torch.ones(1, 2), sizes=[1, 1])

    communicator.rank_in_group = 0
    with pytest.raises(AssertionError, match="Tensor"):
        communicator.all_gatherv([torch.ones(1, 2), "not-a-tensor"], sizes=[1, 1])
