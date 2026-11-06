# Copyright (c) 2026 BAAI. All rights reserved.
"""Real MetaX FlagCX C-API collectives, independent of a Torch FlagCX plugin.

Run inside an isolated MACA environment with a built FlagCX source tree:

    FLAGCX_PATH=/path/to/FlagCX PYTHONPATH=/path/to/FlagCX:$PYTHONPATH \
      torchrun --standalone --nproc-per-node=2 --module pytest --noconftest \
      tests/functional_tests/distributed/test_metax_flagcx_collectives.py -q -s

Set FLAGCX_EXPECT_NODES=2 with a two-node torchrun launch to require two
physical host boot identities. Gloo exchanges bootstrap and host metadata;
every tested device operation calls PyFlagcxCommunicator directly and rejects
a disabled communicator.
"""

import hashlib
import ipaddress
import json
import os
import socket
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.gpu, pytest.mark.multi_gpu, pytest.mark.flagcx]


def _expected_nodes(value: str | None, world_size: int) -> int | None:
    if value is None:
        return None
    try:
        nodes = int(value)
    except ValueError as exc:
        raise ValueError("FLAGCX_EXPECT_NODES must be a positive integer") from exc
    if not 1 <= nodes <= world_size:
        raise ValueError("FLAGCX_EXPECT_NODES must be between 1 and WORLD_SIZE")
    return nodes


def _host_identity(rank: int, local_rank: int, world_size: int, device_count: int):
    # Containers on one physical host share its kernel boot ID. Hostnames and
    # container machine IDs alone cannot distinguish a two-host launch.
    try:
        boot_id = str(
            uuid.UUID(
                Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="ascii")
                .strip()
            )
        )
        boot_hash = hashlib.sha256(boot_id.encode("ascii")).hexdigest()
    except (OSError, ValueError):
        boot_hash = None
    ips = {}
    # Record only explicitly named IP settings, never the full environment or
    # interface inventory. DNS names are not copied into this IP-only field.
    for name in ("VLLM_HOST_IP", "MASTER_ADDR"):
        value = os.environ.get(name, "").strip()
        try:
            ips[name] = str(ipaddress.ip_address(value))
        except ValueError:
            continue
    return {
        "host_boot_id_sha256": boot_hash,
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        "device_count": device_count,
        "ips": ips,
    }


def _validate_host_topology(records: list[dict], expected_nodes: int):
    world_size = len(records)
    assert 1 <= expected_nodes <= world_size, "Invalid expected physical host count"
    hosts = {}
    for record in records:
        for field in (
            "rank",
            "local_rank",
            "world_size",
            "local_world_size",
            "device_count",
        ):
            assert type(record[field]) is int, (
                f"Host metadata {field} must be an integer"
            )
        assert record["world_size"] == world_size, "Inconsistent WORLD_SIZE metadata"
        boot_hash = record["host_boot_id_sha256"]
        assert (
            isinstance(boot_hash, str)
            and len(boot_hash) == 64
            and all(char in "0123456789abcdef" for char in boot_hash)
        ), "Every rank must supply a valid /proc host boot ID SHA256"
        hosts.setdefault(boot_hash, []).append(record)
    assert sorted(record["rank"] for record in records) == list(range(world_size)), (
        "Host metadata must contain every global rank exactly once"
    )
    assert len(hosts) == expected_nodes, (
        f"Expected {expected_nodes} physical hosts, observed {len(hosts)} boot identities"
    )
    for ranks in hosts.values():
        local_size = len(ranks)
        assert all(record["local_world_size"] == local_size for record in ranks), (
            "LOCAL_WORLD_SIZE must match the ranks on each physical host"
        )
        assert sorted(record["local_rank"] for record in ranks) == list(
            range(local_size)
        ), "Each physical host must have distinct, contiguous local ranks"
        device_counts = {record["device_count"] for record in ranks}
        assert len(device_counts) == 1 and next(iter(device_counts)) >= local_size, (
            "Each physical host must expose enough devices for its local ranks"
        )


@pytest.fixture(scope="module")
def flagcx_runtime():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    expected_nodes = _expected_nodes(os.getenv("FLAGCX_EXPECT_NODES"), world_size)
    if world_size < 2:
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
    if expected_nodes is not None:
        try:
            identity = _host_identity(
                dist.get_rank(),
                local_rank,
                dist.get_world_size(),
                torch.cuda.device_count(),
            )
            records = [None] * dist.get_world_size()
            dist.all_gather_object(records, identity, group=dist.group.WORLD)
            _validate_host_topology(records, expected_nodes)
            print(
                "FLAGCX_TOPOLOGY "
                + json.dumps(
                    {"expected_nodes": expected_nodes, "ranks": records}, sort_keys=True
                ),
                flush=True,
            )
        except Exception:
            dist.destroy_process_group()
            raise
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
