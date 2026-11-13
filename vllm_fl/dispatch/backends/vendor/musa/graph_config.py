# Copyright (c) 2026 BAAI. All rights reserved.

"""MUSA-specific cudagraph configuration.

Two torch_musa properties have to be reflected in ``CompilationConfig`` before
vLLM finalizes (and compiles/caches) the piecewise graph:

* ``ProcessGroupMCCL`` submits collectives on its own communication stream and
  its watchdog polls them with MUSA events; neither is permitted while a graph
  is being recorded. The TP collectives therefore have to be listed in
  ``splitting_ops`` so they run eagerly between the captured regions.
* The graph memory pool is not shared across captures, so every captured batch
  size costs a full activation footprint (~2.4 GiB for a 27B model at TP4 with
  a 512-token batch). vLLM's default 51-size ladder then needs >100 GiB and
  fails the memory check.

The logic lives here instead of the shared platform module so that no other
vendor sees MUSA-only policy.
"""

from __future__ import annotations

from typing import Any

# Collectives that torch_musa's MCCL cannot submit while a graph is recording.
MUSA_COLLECTIVE_SPLITTING_OPS = (
    "_c10d_functional::all_reduce_",
    "_c10d_functional::all_reduce",
    "_c10d_functional::reduce_scatter",
    "_c10d_functional::all_gather_into_tensor",
    "_c10d_functional::wait_tensor",
)

MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES = 10


def append_collective_splitting_ops(splitting_ops: list[str]) -> bool:
    """Append the c10d collectives that vLLM 0.32 lowers TP to.

    Returns True when ``splitting_ops`` was modified.
    """
    added = False
    for op in MUSA_COLLECTIVE_SPLITTING_OPS:
        if op not in splitting_ops:
            splitting_ops.append(op)
            added = True
    return added


def thin_cudagraph_capture_sizes(sizes: list[int]) -> list[int]:
    """Keep a power-of-two ladder (plus the largest size) for MUSA."""
    kept = sorted({s for s in sizes if s & (s - 1) == 0})
    if len(kept) < min(4, len(sizes)):
        # No useful power-of-two ladder: sample the range evenly instead.
        count = min(len(sizes), MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES)
        step = len(sizes) / count
        kept = sorted({sizes[min(int(i * step), len(sizes) - 1)] for i in range(count)})
    if sizes and sizes[-1] not in kept:
        kept.append(sizes[-1])
    kept = sorted(set(kept))
    if len(kept) > MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES:
        count = MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES
        step = len(kept) / (count - 1)
        kept = sorted(
            {kept[min(int(i * step), len(kept) - 1)] for i in range(count - 1)}
            | {kept[-1]}
        )
    return kept


def limit_cudagraph_capture_sizes(compilation_config: Any) -> list[int] | None:
    """Thin the capture ladder so the graphs fit in device memory.

    Returns the new ladder when it was changed, otherwise None.
    """
    sizes = list(getattr(compilation_config, "cudagraph_capture_sizes", None) or [])
    if len(sizes) <= MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES:
        return None
    sizes = thin_cudagraph_capture_sizes(sizes)
    compilation_config.cudagraph_capture_sizes = sizes
    return sizes
