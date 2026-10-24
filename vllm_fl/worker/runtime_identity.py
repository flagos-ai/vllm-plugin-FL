"""Optional CPU metadata for startup auditing, independent of profiling."""

import os
from vllm.distributed import (
    get_world_group,
    get_tp_group,
    get_dp_group,
    get_ep_group,
    get_pp_group,
)


def worker_profile_identity(worker):
    runner = getattr(worker, "model_runner", None)
    return {
        "node": os.environ.get("K3_NODE"),
        "worker_rank": int(worker.rank),
        "runner_type": None
        if runner is None
        else type(runner).__module__ + "." + type(runner).__name__,
        "local_rank": int(worker.local_rank),
        "world_rank": int(get_world_group().rank),
        "pp_rank": int(get_pp_group().rank_in_group),
        "pp_world_size": int(get_pp_group().world_size),
        "expert_parallel_enabled": bool(worker.parallel_config.enable_expert_parallel),
        "tp_rank": int(get_tp_group().rank_in_group),
        "dp_rank": int(get_dp_group().rank_in_group),
        "ep_rank": int(get_ep_group().rank_in_group),
        "tp_world_size": int(get_tp_group().world_size),
        "dp_world_size": int(get_dp_group().world_size),
        "ep_world_size": int(get_ep_group().world_size),
    }
