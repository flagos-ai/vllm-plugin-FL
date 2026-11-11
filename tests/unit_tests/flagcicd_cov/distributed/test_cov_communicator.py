# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-rank collective tests for CommunicatorFL through the flagcx backend.

Pilot P2 of the supplementary coverage suite (flagcicd_cov). Uses the only
sanctioned multi-GPU shape in this repo (see
tests/unit_tests/test_kunlunxin_tp_graph_replay.py): the test process spawns
two workers via torch.multiprocessing.spawn; each worker pins its own device,
forms a gloo world plus a flagcx device group, and checks collective results
against values derivable without communication.

The controller asserts the workers succeeded; per-rank failures propagate as
a nonzero worker exit.
"""

import datetime
import socket
import time

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.gpu,
    pytest.mark.skipif(
        getattr(current_platform, "vendor_name", None) != "nvidia",
        reason="P2 pilot targets the NVIDIA flagcx build",
    ),
]

WORLD_SIZE = 2
TIMEOUT_S = 180


def _collective_worker(rank: int, port: int, result_queue) -> None:
    """Worker: run each collective and push per-rank pass/fail results."""
    import torch.distributed as dist

    from vllm_fl.distributed.communicator import CommunicatorFL

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(seconds=TIMEOUT_S),
    )
    try:
        device_group = dist.new_group(backend="nccl")
        comm = CommunicatorFL(
            dist.group.WORLD, torch.device("cuda", rank), device_group
        )
        assert comm.world_size == WORLD_SIZE

        results = {}

        # all_reduce: rank r contributes (r+1) ones; sum is world_size * (r+1)
        # summed over ranks = 1+2 = 3 per element.
        x = torch.full((64, 16), rank + 1, device="cuda", dtype=torch.float32)
        out = comm.all_reduce(x)
        torch.cuda.synchronize()
        results["all_reduce"] = bool(
            torch.allclose(out.cpu(), torch.full((64, 16), 3.0))
        )

        # reduce_scatter on dim=-1: each rank keeps its half of the sum.
        y = torch.full((64, 32), rank + 1, device="cuda", dtype=torch.float32)
        out = comm.reduce_scatter(y, dim=-1)
        torch.cuda.synchronize()
        results["reduce_scatter"] = bool(
            out.shape == (64, 16) and torch.allclose(out.cpu(), torch.full((64, 16), 3.0))
        )

        # all_gatherv with uniform sizes: rank r contributes rows of value r+1;
        # the gathered tensor interleaves them in rank order.
        z = torch.full((8, 4), rank + 1, device="cuda", dtype=torch.float32)
        out = comm.all_gatherv(z, dim=0, sizes=[8, 8])
        torch.cuda.synchronize()
        expected = torch.cat(
            [torch.full((8, 4), r + 1) for r in range(WORLD_SIZE)], dim=0
        )
        results["all_gatherv"] = bool(
            out.shape == (16, 4) and torch.allclose(out.cpu(), expected)
        )

        # send/recv ring: rank 0 sends 7s forward, rank 1 sends 11s back.
        if rank == 0:
            comm.send(torch.full((4,), 7.0, device="cuda", dtype=torch.float32), dst=1)
            got = comm.recv((4,), torch.float32, src=1)
        else:
            got = comm.recv((4,), torch.float32, src=0)
            comm.send(torch.full((4,), 11.0, device="cuda", dtype=torch.float32), dst=0)
        torch.cuda.synchronize()
        want = 7.0 if rank == 1 else 11.0
        results["send_recv"] = bool(
            torch.allclose(got.cpu(), torch.full((4,), want))
        )

        comm.destroy()
        result_queue.put((rank, results))
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(TIMEOUT_S + 60)
def test_flagcx_collectives_two_ranks():
    """all_reduce/reduce_scatter/all_gatherv/send/recv match closed forms."""
    try:
        import flagcx  # noqa: F401  # the CommunicatorFL import needs it
    except ImportError:
        pytest.fail("flagcx package not available in the CI image")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    workers = torch.multiprocessing.spawn(
        _collective_worker, args=(port, result_queue), nprocs=WORLD_SIZE, join=False
    )
    deadline = time.monotonic() + TIMEOUT_S
    try:
        while not workers.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("collective workers did not finish within timeout")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
        for process in workers.processes:
            process.join(timeout=10)

    collected = {}
    while len(collected) < WORLD_SIZE:
        rank, results = result_queue.get(timeout=30)
        collected[rank] = results

    failures = {
        f"rank{rank}.{op}": detail
        for rank, results in sorted(collected.items())
        for op, detail in results.items()
        if not detail
    }
    assert not failures, f"collective mismatches: {sorted(failures)}"
