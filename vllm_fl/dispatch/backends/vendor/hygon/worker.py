"""Hygon DCU worker helpers."""

from __future__ import annotations

import os

from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

#: Default request, in GiB, kept out of the KV-cache budget for graph capture.
DEFAULT_HEADROOM_GB = 4.0
#: Upper bound as a fraction of one card, so small cards are not starved.
DEVICE_BUDGET_FRACTION = 0.3


def cudagraph_memory_headroom(vllm_config) -> int:
    """Bytes to reserve from the KV-cache budget for graph capture.

    Hygon DCUs report ``is_cuda_alike() == False``, so the upstream cudagraph
    memory estimator never runs and the KV cache takes the whole
    ``gpu_memory_utilization`` budget. The first capture then fails with
    ``HIP out of memory``, which also leaves the DCU memory unreclaimable.
    """
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if getattr(compilation_config, "cudagraph_mode", CUDAGraphMode.NONE) == (
        CUDAGraphMode.NONE
    ):
        return 0

    requested = int(
        float(os.environ.get("VLLM_FL_HYGON_GRAPH_HEADROOM_GB", DEFAULT_HEADROOM_GB))
        * 1024**3
    )
    device_budget_cap = int(
        current_platform.get_device_total_memory() * DEVICE_BUDGET_FRACTION
    )
    headroom = min(requested, device_budget_cap)
    logger.info(
        "Hygon: reserving %.1f GiB of the KV cache budget for graph capture "
        "(requested %.1f GiB, device budget cap %.1f GiB).",
        headroom / 1024**3,
        requested / 1024**3,
        device_budget_cap / 1024**3,
    )
    return headroom
