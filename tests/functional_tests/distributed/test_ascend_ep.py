# Copyright (c) 2026 BAAI. All rights reserved.
"""Run with torchrun --nproc-per-node=2 -m pytest -q <this file> on Ascend."""

import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="Requires two NPU processes launched with torchrun",
)


@pytest.fixture(scope="module")
def groups():
    from vllm.config import (
        CompilationConfig,
        CUDAGraphMode,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    config = VllmConfig(
        # These component tests capture NPUGraph directly, not a model graph.
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2, enable_expert_parallel=True
        ),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=2,
            rank=int(os.environ["RANK"]),
            local_rank=local_rank,
            backend="hccl",
        )
        initialize_model_parallel(2, backend="hccl")
        try:
            yield get_tp_group()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_all_reduce_preserves_input(groups, dtype):
    from vllm_fl.distributed.device_communicators.npu_communicator import (
        NPUCommunicator,
    )

    comm = groups.device_communicator
    assert isinstance(comm, NPUCommunicator)
    rank = groups.rank_in_group
    backing = torch.full((6, 8), rank + 1, dtype=dtype, device="npu")
    tensor = backing[1:5]
    alias = tensor.view(-1)
    expected_sum = groups.world_size * (groups.world_size + 1) // 2

    # Exercise both the communicator and the public group wrapper: callers
    # may retain the input or a view after the reduction returns.
    for reduce in (comm.all_reduce, groups.all_reduce):
        backing.fill_(rank + 1)
        before = backing.clone()
        for _ in range(3):
            result = reduce(tensor)
        torch.npu.synchronize()
        torch.testing.assert_close(backing, before, rtol=0, atol=0)
        torch.testing.assert_close(alias, before[1:5].reshape(-1), rtol=0, atol=0)
        assert result.data_ptr() != tensor.data_ptr()
        torch.testing.assert_close(
            result, torch.full_like(tensor, expected_sum), rtol=0, atol=0
        )

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = reduce(tensor)
        for step in range(1, 4):
            tensor.add_(1)
            before = backing.clone()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(backing, before, rtol=0, atol=0)
            torch.testing.assert_close(alias, before[1:5].reshape(-1), rtol=0, atol=0)
            assert result.data_ptr() != tensor.data_ptr()
            torch.testing.assert_close(
                result,
                torch.full_like(tensor, expected_sum + groups.world_size * step),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("sizes", [[4, 4], [2, 4], [0, 4], [0, 0]])
def test_padded_collectives(groups, sizes):
    from vllm_fl.distributed.device_communicators.npu_communicator import (
        NPUCommunicator,
    )

    comm = groups.device_communicator
    assert isinstance(comm, NPUCommunicator)
    rank = groups.rank_in_group
    tensor = torch.full((sizes[rank], 8), rank + 1, dtype=torch.bfloat16, device="npu")
    expected = torch.cat(
        [tensor.new_full((size, 8), other + 1) for other, size in enumerate(sizes)]
    )

    def run():
        gathered = comm.all_gatherv([tensor, tensor], dim=0, sizes=sizes)
        return gathered[0], comm.reduce_scatterv(gathered[1], dim=0, sizes=sizes)

    for _ in range(3):
        gathered, reduced = run()
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    torch.testing.assert_close(reduced, tensor * 2, rtol=0, atol=0)
    if sum(sizes):
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            gathered, reduced = run()
        tensor.add_(1)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(gathered, expected + 1, rtol=0, atol=0)
        torch.testing.assert_close(reduced, tensor * 2, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize(
    "shape,dim,sizes",
    [
        ((2, 3, 4), -1, [2, 2]),
        ((2, 3, 4), 2, [2, 2]),
        ((2, 3, 4), -1, None),
        ((2, 3, 4), 2, [1, 3]),
        ((2, 3, 4), -1, [0, 4]),
        ((2, 4, 3), 1, None),
        ((4, 2, 3), 0, [2, 2]),
        ((2, 3, 0), -1, None),
        ((4,), -1, None),
    ],
)
def test_reduce_scatterv_dimensions(groups, shape, dim, sizes, noncontiguous, dtype):
    """Equal/uneven/implicit splits share the same axis and reduction contract."""
    from math import prod

    comm = groups.device_communicator
    rank = groups.rank_in_group
    world = groups.world_size
    normalized_dim = dim % len(shape)
    values = torch.arange(prod(shape), dtype=torch.float32, device="npu").reshape(shape)
    tensor = (values + rank).to(dtype)
    if noncontiguous:
        backing = tensor.new_full((*shape[:-1], shape[-1] * 2), -31)
        tensor = backing[..., ::2]
        tensor.copy_((values + rank).to(dtype))
    else:
        backing = tensor
    before = backing.clone()
    split_sizes = sizes or [shape[normalized_dim] // world] * world
    start = sum(split_sizes[:rank])
    expected = (
        (values * world + world * (world - 1) // 2)
        .narrow(normalized_dim, start, split_sizes[rank])
        .to(dtype)
    )

    for _ in range(2):
        result = comm.reduce_scatterv(tensor, dim=dim, sizes=sizes)
    torch.npu.synchronize()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
    assert result.is_contiguous()
    if tensor.numel() == 0:
        return

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = comm.reduce_scatterv(tensor, dim=dim, sizes=sizes)
    for step in range(1, 3):
        tensor.add_(1)
        before = backing.clone()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result, expected + world * step, rtol=0, atol=0)
        torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,dim,sizes",
    [
        ((2, 3, 4), -4, None),
        ((2, 3, 4), 3, [2, 2]),
        ((2, 3, 3), -1, None),
        ((2, 3, 4), -1, [1, 1]),
        ((2, 3, 4), -1, [4]),
    ],
)
def test_reduce_scatterv_rejects_invalid_dimensions_or_sizes(groups, shape, dim, sizes):
    tensor = torch.zeros(shape, dtype=torch.bfloat16, device="npu")
    with pytest.raises(ValueError):
        groups.device_communicator.reduce_scatterv(tensor, dim=dim, sizes=sizes)


@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_native_expert_partition(groups, topk, num_tokens, monkeypatch):
    import torch_npu

    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe import (
        _torch_fused_experts_impl,
    )

    rank = groups.rank_in_group
    torch.manual_seed(41)
    hidden = (torch.randn(num_tokens, 128, device="npu") * 0.2).bfloat16()
    w1 = (torch.randn(4, 128, 128, device="npu") * 0.1).bfloat16()
    w2 = (torch.randn(4, 128, 64, device="npu") * 0.1).bfloat16()
    expert_map = torch.full((4,), -1, dtype=torch.int32, device="npu")
    expert_map[rank * 2 : rank * 2 + 2] = torch.arange(2, device="npu")
    ids = torch.tensor(
        [[0, 2], [1, 3], [2, 0], [3, 1]] * 2, device="npu", dtype=torch.int32
    )[:num_tokens, :topk].contiguous()
    probs = torch.full((num_tokens, topk), 1 / topk, device="npu")

    grouped_matmul = torch_npu.npu_grouped_matmul

    def poison_unused_capacity(**kwargs):
        result = grouped_matmul(**kwargs)
        # The API only defines the local-route prefix. Deliberately poison
        # its unused capacity to exercise the adapter independently of the
        # allocator's previous contents, including the N=1, topk=1 boundary.
        unused = (
            torch.arange(result[0].size(0), device=hidden.device)
            >= kwargs["group_list"].sum()
        )
        result[0].masked_fill_(unused[:, None], float("nan"))
        return result

    monkeypatch.setattr(torch_npu, "npu_grouped_matmul", poison_unused_capacity)

    def run():
        local = _torch_fused_experts_impl(
            hidden,
            w1[rank * 2 : rank * 2 + 2],
            w2[rank * 2 : rank * 2 + 2],
            probs,
            ids,
            global_num_experts=4,
            expert_map=expert_map,
        )
        return groups.all_reduce(local)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = run()
    for expert_ids in (ids.clone(), torch.zeros_like(ids), torch.full_like(ids, 3)):
        ids.copy_(expert_ids)
        graph.replay()
        torch.npu.synchronize()
        expected = torch.zeros_like(hidden, dtype=torch.float32)
        for expert in range(4):
            gate, up = F.linear(hidden, w1[expert]).chunk(2, dim=-1)
            down = F.linear((F.silu(gate) * up), w2[expert]).float()
            weight = ((ids == expert) * probs).sum(dim=-1, keepdim=True)
            expected += down * weight
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.004)
