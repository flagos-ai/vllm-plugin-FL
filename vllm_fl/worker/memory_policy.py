# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Host-side physical KV memory budgeting."""

from __future__ import annotations


def physical_kv_cache_budget(
    requested: int | float, non_kv_memory: int, initial_non_torch: int
) -> int:
    """Include pre-existing non-Torch occupancy in the physical memory budget.

    The profile's non-Torch field is an increment from the initial snapshot.
    Subtract the nonnegative initial occupancy once; never increase the old
    budget if another process releases memory during profiling.
    """
    return max(0, int(requested - non_kv_memory - max(0, initial_non_torch)))
