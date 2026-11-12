# Copyright (c) 2026 BAAI. All rights reserved.

"""Process-local Gloo fixtures for supplementary tests."""

from contextlib import ExitStack, contextmanager
from datetime import timedelta
from unittest.mock import patch


@contextmanager
def single_rank_gloo(*groups):
    """Own a one-rank Gloo world and only the requested vLLM subgroups."""
    import torch.distributed as dist

    import vllm.distributed.parallel_state as ps

    assert not dist.is_initialized(), "test requires its own process group"
    timeout = timedelta(seconds=30)
    store = dist.TCPStore("127.0.0.1", 0, 1, True, timeout, use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1, timeout=timeout)
    with ExitStack() as stack:
        stack.callback(ps.destroy_distributed_environment)
        ps.init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="gloo"
        )
        for name in groups:
            group = ps.init_model_parallel_group([[0]], 0, "gloo", group_name=name)
            stack.callback(group.destroy)
            stack.enter_context(patch.object(ps, f"_{name.upper()}", group))
        yield
