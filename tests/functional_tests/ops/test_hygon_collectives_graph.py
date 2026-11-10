"""Minimal real two-rank collective compile / HIP Graph regression."""
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp


def _worker(rank, init_method):
    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
    config = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=2, disable_custom_all_reduce=True))
    with set_current_vllm_config(config):
        _run(rank, init_method)


def _run(rank, init_method):
    from vllm.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel, get_tp_group,
        cleanup_dist_env_and_memory,
    )
    from vllm.platforms import current_platform
    assert current_platform.vendor_name == "hygon"
    assert current_platform.use_custom_op_collectives()
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=2, rank=rank, local_rank=rank,
        distributed_init_method=init_method, backend="nccl")
    try:
        initialize_model_parallel(tensor_model_parallel_size=2)
        group = get_tp_group()
        assert group.use_custom_op_call
        def run(x):
            return group.all_reduce(x)
        x = torch.full((32, 16), rank + 1., device="cuda")
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), torch.full_like(x, 3.))
        with group.graph_capture() as context:
            for _ in range(3):
                run(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=context.stream):
                output = run(x)
        x.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(x, 5.))
    finally:
        cleanup_dist_env_and_memory()


def test_hygon_collective_compile_and_graph():
    with tempfile.TemporaryDirectory(prefix="glm52-collective-") as tmp:
        mp.spawn(_worker, args=(Path(tmp, "rendezvous").as_uri(),),
                 nprocs=2, join=True)
