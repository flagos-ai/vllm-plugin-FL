# Copyright (c) 2026 BAAI. All rights reserved.
"""Real MetaX FlagCX C-API collectives, independent of a Torch FlagCX plugin.

Run inside an isolated MACA environment with a built FlagCX source tree:

    FLAGCX_PATH=/path/to/FlagCX PYTHONPATH=/path/to/FlagCX:$PYTHONPATH \
      torchrun --standalone --nproc-per-node=2 --module pytest --noconftest \
      tests/functional_tests/distributed/test_metax_flagcx_collectives.py -q -s

Gloo exchanges the bootstrap ID only. Every tested device operation calls
PyFlagcxCommunicator directly and rejects a disabled communicator.
"""

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.gpu, pytest.mark.multi_gpu, pytest.mark.flagcx]


@pytest.fixture(scope="module")
def flagcx_runtime():
    if int(os.getenv("WORLD_SIZE", "1")) < 2:
        pytest.skip("Launch with torchrun and at least two processes")
    if not os.getenv("FLAGCX_PATH"):
        pytest.skip("Set FLAGCX_PATH to a built MetaX FlagCX source tree")
    if not torch.cuda.is_available():
        pytest.skip("Requires MACA CUDA devices")
    from tests.utils.device_utils import get_backend

    if get_backend() != "metax":
        pytest.skip("MetaX-only FlagCX integration test")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Resolve the service's real platform before low-level communication, so
    # missing FL dependencies fail here instead of appearing as a stream error.
    from vllm.platforms import current_platform

    assert getattr(current_platform, "vendor_name", None) == "metax", (
        f"The real FL platform must be selected, got {type(current_platform)}"
    )
    assert current_platform.device_type == "cuda"
    assert not dist.is_initialized(), "Use an isolated torchrun process group"
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

    library = Path(os.environ["FLAGCX_PATH"]) / "build/lib/libflagcx.so"
    assert library.is_file(), f"Missing FlagCX native library: {library}"
    comm = PyFlagcxCommunicator(dist.group.WORLD, device, str(library))
    assert comm.available and not comm.disabled, (
        f"FlagCX must execute the device collectives: {getattr(comm, 'init_error', None)}"
    )
    runtime = SimpleNamespace(
        comm=comm,
        device=device,
        rank=dist.get_rank(),
        world=dist.get_world_size(),
    )
    assert runtime.world in (2, 4), "This probe supports two or four ranks"
    print(
        f"FLAGCX_READY rank={runtime.rank} world={runtime.world} device={device} "
        f"library={library.resolve()} wrapper={type(comm.flagcx).__module__} "
        f"available={comm.available} disabled={comm.disabled}",
        flush=True,
    )
    try:
        yield runtime
    finally:
        torch.cuda.synchronize(device)
        dist.barrier()
        comm.destroy()
        dist.destroy_process_group()


def _check(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.to(actual.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_flagcx_eager_collectives_against_cpu_oracles(flagcx_runtime, dtype):
    ctx = flagcx_runtime
    comm, device, rank, world = ctx.comm, ctx.device, ctx.rank, ctx.world
    dist.barrier()
    values = torch.full((2, 3), rank + 1, dtype=dtype, device=device)
    original = values.clone()
    reduced = comm.all_reduce(values)
    assert reduced is not None and reduced.data_ptr() != values.data_ptr()
    gathered = torch.empty((2 * world, 3), dtype=dtype, device=device)
    comm.all_gather(gathered, values)

    rs_cpu = torch.arange(2 * world * 3).reshape(2 * world, 3).float()
    rs_input = (rs_cpu + rank).to(device=device, dtype=dtype)
    scattered = torch.empty((2, 3), dtype=dtype, device=device)
    comm.reduce_scatter(scattered, rs_input)
    broadcasted = values.clone()
    comm.broadcast(broadcasted, src=0)
    torch.cuda.synchronize(device)
    _check(values, original.cpu())
    _check(reduced, torch.full((2, 3), world * (world + 1) // 2))
    _check(gathered, torch.cat([torch.full((2, 3), i + 1) for i in range(world)]))
    rs_expected = sum(rs_cpu + i for i in range(world))
    _check(scattered, rs_expected.chunk(world, dim=0)[rank])
    _check(broadcasted, torch.ones(2, 3))

    sizes = list(range(1, world + 1))
    variable_input = torch.full((sizes[rank], 3), rank + 1, dtype=dtype, device=device)
    variable_gathered = torch.empty((sum(sizes), 3), dtype=dtype, device=device)
    comm.all_gatherv(variable_gathered, variable_input, sizes)
    variable_cpu = torch.arange(sum(sizes) * 3).reshape(sum(sizes), 3).float()
    variable_rs_input = (variable_cpu + rank).to(device=device, dtype=dtype)
    variable_scattered = torch.empty((sizes[rank], 3), dtype=dtype, device=device)
    comm.reduce_scatterv(variable_scattered, variable_rs_input, sizes)
    torch.cuda.synchronize(device)
    _check(
        variable_gathered,
        torch.cat([torch.full((size, 3), i + 1) for i, size in enumerate(sizes)]),
    )
    variable_expected = sum(variable_cpu + i for i in range(world))
    _check(variable_scattered, variable_expected.split(sizes, dim=0)[rank])

    partner = rank ^ 1
    sent = (torch.arange(6).reshape(2, 3) + 10 * rank).to(device=device, dtype=dtype)
    received = torch.empty_like(sent)
    if rank % 2 == 0:
        comm.send(sent, partner)
        comm.recv(received, partner)
    else:
        comm.recv(received, partner)
        comm.send(sent, partner)
    torch.cuda.synchronize(device)
    _check(received, torch.arange(6).reshape(2, 3) + 10 * partner)
    dist.barrier()
    print(f"FLAGCX_EAGER_PASS rank={rank} world={world} dtype={dtype}", flush=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_flagcx_graph_replay_reads_live_inputs(flagcx_runtime, dtype):
    ctx = flagcx_runtime
    comm, device, rank, world = ctx.comm, ctx.device, ctx.rank, ctx.world
    values = torch.full((2, 3), rank + 1, dtype=dtype, device=device)
    reduced = torch.empty_like(values)
    gathered = torch.empty((2 * world, 3), dtype=dtype, device=device)
    rs_cpu = torch.arange(2 * world * 3).reshape(2 * world, 3).float()
    rs_input = (rs_cpu + rank).to(device=device, dtype=dtype)
    scattered = torch.empty_like(values)
    broadcasted = values.clone()

    def communicate():
        comm.all_reduce(values, out_tensor=reduced)
        comm.all_gather(gathered, values)
        comm.reduce_scatter(scattered, rs_input)
        comm.broadcast(broadcasted, src=0)

    communicate()
    torch.cuda.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        communicate()
    for multiplier in (1, 2, 3):
        values.fill_((rank + 1) * multiplier)
        rs_input.copy_(((rs_cpu + rank) * multiplier).to(device=device, dtype=dtype))
        broadcasted.fill_((rank + 1) * multiplier)
        reduced.fill_(-7)
        gathered.fill_(-7)
        scattered.fill_(-7)
        graph.replay()
        torch.cuda.synchronize(device)
        _check(reduced, torch.full((2, 3), world * (world + 1) // 2 * multiplier))
        _check(
            gathered,
            torch.cat([torch.full((2, 3), (i + 1) * multiplier) for i in range(world)]),
        )
        expected_rs = sum(rs_cpu + i for i in range(world)) * multiplier
        _check(scattered, expected_rs.chunk(world, dim=0)[rank])
        _check(broadcasted, torch.full((2, 3), multiplier))
    dist.barrier()
    print(f"FLAGCX_GRAPH_PASS rank={rank} world={world} dtype={dtype}", flush=True)
