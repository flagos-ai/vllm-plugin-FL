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

"""Two-rank collective tests for CommunicatorFL on the NVIDIA CI image.

The v0.24.0-cuda-ci image ships torch.distributed NCCL but no flagcx
package, so ``PyFlagcxCommunicator`` disables itself at construction
(_flagcx_available=False) and every CommunicatorFL collective takes the
``torch.distributed`` fallback. These tests exercise exactly that path —
the one production takes on this platform — plus the CPU-testable
flagcx import ladder, using the sanctioned multi-GPU spawn pattern (see
tests/unit_tests/test_kunlunxin_tp_graph_replay.py).

P2 pilot of the supplementary coverage suite (flagcicd_cov).
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
        reason="P2 pilot targets the NVIDIA CI image",
    ),
]

WORLD_SIZE = 2
TIMEOUT_S = 180


def _collective_worker(rank: int, port: int, result_queue) -> None:
    """Worker: run collectives through CommunicatorFL, push per-rank results."""
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
        results = {}

        # On this image PyFlagcxCommunicator must disable itself and the
        # collectives fall back to torch.distributed (nccl device group).
        results["constructed"] = comm.world_size == WORLD_SIZE
        pyc = comm.pyflagcx_comm
        results["flagcx_disabled"] = bool(
            pyc is None
            or (getattr(pyc, "available", None) is False)
            or (getattr(pyc, "disabled", None) is True)
        )

        # all_reduce: rank r contributes (r+1) per element; sum is 3.
        x = torch.full((64, 16), rank + 1, device="cuda", dtype=torch.float32)
        out = comm.all_reduce(x)
        torch.cuda.synchronize()
        results["all_reduce"] = bool(
            torch.allclose(out.cpu(), torch.full((64, 16), 3.0))
        )

        # reduce_scatter: NOT asserted here — on flagcx-less images
        # PyFlagcxCommunicator.reduce_scatter returns early (disabled) and
        # CommunicatorFL.reduce_scatter has no torch.distributed fallback,
        # so the output stays uninitialized (vllm_fl/distributed/communicator.py:91
        # + device_communicators/flagcx.py:297-298; SOURCE_BUGS.md S3).

        # all_gatherv: NOT exercised here — vllm_fl/distributed/communicator.py:176
        # asserts pyflagcx_comm is not None and not disabled, i.e. the op has
        # no torch.distributed fallback and is unreachable on images without
        # flagcx (handed to the plugin owners; would need an nccl fallback
        # like all_reduce/send/recv have).

        # send/recv ring: rank 0 sends 7s forward, rank 1 sends 11s back.
        if rank == 0:
            comm.send(torch.full((4,), 7.0, device="cuda", dtype=torch.float32), dst=1)
            got = comm.recv((4,), torch.float32, src=1)
        else:
            got = comm.recv((4,), torch.float32, src=0)
            comm.send(torch.full((4,), 11.0, device="cuda", dtype=torch.float32), dst=0)
        torch.cuda.synchronize()
        want = 7.0 if rank == 1 else 11.0
        results["send_recv"] = bool(torch.allclose(got.cpu(), torch.full((4,), want)))

        comm.destroy()
        result_queue.put((rank, results))
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(TIMEOUT_S + 60)
def test_collectives_two_ranks_nccl_fallback():
    """Collectives via CommunicatorFL match closed forms on the NCCL path."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    workers = torch.multiprocessing.spawn(
        _collective_worker,
        args=(port, result_queue),
        nprocs=WORLD_SIZE,
        join=False,
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


def test_flagcx_import_ladder_disabled_without_package(monkeypatch):
    """Without flagcx installed and FLAGCX_PATH unset the module stays disabled.

    The NVIDIA CI image ships no flagcx package; importing the communicator
    module must yield _flagcx_available=False and FLAGCXLibrary=None so the
    torch.distributed fallback is used (regression: a half-bound ctypes
    facade would return uninitialized results).
    """
    import importlib
    import sys

    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.delitem(sys.modules, "flagcx", raising=False)
    monkeypatch.delitem(sys.modules, "flagcx.api", raising=False)
    module_name = "vllm_fl.distributed.device_communicators.flagcx"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    fc = importlib.import_module(module_name)

    assert fc._flagcx_available is False
    assert fc.FLAGCXLibrary is None
