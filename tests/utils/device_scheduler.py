# Copyright (c) 2025 BAAI. All rights reserved.

"""
Slot-based device scheduler for parallel e2e test execution.

Each e2e test case consumes ``tensor_parallel_size`` device slots.  The
scheduler tracks which slots are free and hands out non-overlapping ranges
so that multiple test subprocesses can run concurrently without fighting
over the same accelerator cards.

Usage::

    scheduler = DeviceScheduler(total_devices=8)

    # Blocking acquire — returns e.g. [2, 3, 4, 5] for tp=4
    slots = scheduler.acquire(tp_size=4)

    # ... run subprocess with VISIBLE_DEVICES set to slots ...

    scheduler.release(slots)
"""

from __future__ import annotations

import threading
from collections.abc import Sequence


class DeviceScheduler:
    """Thread-safe slot allocator for GPU/accelerator devices.

    Slots are integer device indices (0-based within the container's visible
    device space).  ``acquire`` blocks until ``tp_size`` contiguous slots are
    free, then marks them as used and returns them.  ``release`` marks them
    free again and wakes any waiting threads.

    Contiguous allocation ensures that the subprocess's ``VISIBLE_DEVICES``
    maps to a coherent set of physical cards, which matters for NVLink / HCCL
    topologies that prefer adjacent indices.
    """

    def __init__(self, total_devices: int) -> None:
        self._total = total_devices
        # True = free
        self._free: list[bool] = [True] * total_devices
        self._lock = threading.Condition()

    @property
    def total(self) -> int:
        return self._total

    def acquire(self, tp_size: int) -> list[int]:
        """Block until ``tp_size`` contiguous free slots are available.

        Returns the list of allocated slot indices.  Raises ``ValueError`` if
        ``tp_size`` exceeds the total device count (can never be satisfied).
        """
        if tp_size > self._total:
            raise ValueError(f"tp_size={tp_size} exceeds total devices={self._total}")
        with self._lock:
            while True:
                slots = self._find_contiguous(tp_size)
                if slots is not None:
                    for s in slots:
                        self._free[s] = False
                    return slots
                self._lock.wait()

    def release(self, slots: Sequence[int]) -> None:
        """Return ``slots`` to the free pool and notify waiting threads."""
        with self._lock:
            for s in slots:
                self._free[s] = True
            self._lock.notify_all()

    def _find_contiguous(self, n: int) -> list[int] | None:
        """Return the first run of ``n`` free slots, or None."""
        start = 0
        while start <= self._total - n:
            run = []
            for i in range(start, start + n):
                if self._free[i]:
                    run.append(i)
                else:
                    start = i + 1
                    break
            else:
                return run
        return None
