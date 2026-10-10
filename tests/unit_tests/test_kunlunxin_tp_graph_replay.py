# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

import datetime
import socket
import time

import pytest
import torch
import torch.distributed as dist

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.skipif(
        getattr(current_platform, "vendor_name", None) != "kunlunxin",
        reason="Real Kunlunxin TP graph replay",
    ),
]


def _tp_replay_worker(rank, port, mode_name):
    import flagcx  # noqa: F401

    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
        VllmConfig,
    )
    from vllm.forward_context import BatchDescriptor, set_forward_context

    from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
        patch_breakable_full_only,
        patch_breakable_private_pools,
        patch_eager_all_gather,
        patch_graph_all_reduce,
    )
    from vllm_fl.distributed.communicator import CommunicatorFL

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=180),
    )
    group = dist.new_group(backend="flagcx")
    comm = CommunicatorFL(dist.group.WORLD, torch.device("cuda", rank), group)
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        )
    )
    patch_breakable_private_pools()
    patch_breakable_full_only()
    patch_graph_all_reduce()
    patch_eager_all_gather()
    stream = torch.cuda.Stream()
    outputs, inputs = {}, {}
    calls = []

    def run(value):
        calls.append(True)
        result = value
        for _ in range(17):
            result = comm.all_reduce(result)
        outputs[value.shape[0]].copy_(result)
        return outputs[value.shape[0]]

    wrapper = BreakableCUDAGraphWrapper(run, config)
    mode = CUDAGraphMode[mode_name]
    try:
        for size in (4, 8):
            inputs[size] = torch.ones((size, 16), device="cuda")
            outputs[size] = torch.empty_like(inputs[size])
            torch.cuda.synchronize()
            with (
                torch.cuda.stream(stream),
                set_forward_context(
                    None,
                    config,
                    cudagraph_runtime_mode=mode,
                    batch_descriptor=BatchDescriptor(num_tokens=size, uniform=True),
                ),
            ):
                wrapper(inputs[size])
        captures = len(calls)
        # Replay old and new descriptors after both pools have been populated.
        for size, step in ((4, 2), (8, 3), (4, 5), (8, 7)):
            inputs[size].fill_(step + rank)
            outputs[size].fill_(float("nan"))
            torch.cuda.synchronize()
            with (
                torch.cuda.stream(stream),
                set_forward_context(
                    None,
                    config,
                    cudagraph_runtime_mode=mode,
                    batch_descriptor=BatchDescriptor(num_tokens=size, uniform=True),
                ),
            ):
                actual = wrapper(inputs[size])
            torch.cuda.synchronize()
            assert len(calls) == captures
            expected = torch.full(
                (size, 16), (2 * step + 1) * 2**16, dtype=torch.float32
            )
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            gathered = comm.all_gather(actual, dim=-1)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                gathered.cpu(), expected.repeat(1, 2), rtol=0, atol=0
            )
        assert all(entry.capture.num_graphs >= 2 for entry in wrapper.entries.values())
        assert all(
            entry.capture.num_eager_breaks == 0 for entry in wrapper.entries.values()
        )
    finally:
        wrapper.clear_graphs()
        dist.destroy_process_group()


@pytest.mark.parametrize("mode_name", ["FULL", "PIECEWISE"])
def test_tp_graph_segments_replay_refreshed_collectives(mode_name):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    workers = torch.multiprocessing.spawn(
        _tp_replay_worker, args=(port, mode_name), nprocs=2, join=False
    )
    deadline = time.monotonic() + 180
    try:
        while not workers.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("TP graph replay workers did not finish within 180 seconds")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
        for process in workers.processes:
            process.join(timeout=10)
