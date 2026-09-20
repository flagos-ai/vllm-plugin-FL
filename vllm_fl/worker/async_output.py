# SPDX-License-Identifier: Apache-2.0
"""Optional, bounded native completion waits for asynchronous CPU output."""

from __future__ import annotations

import logging
import math
import os
import select
import threading
import time
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Waiting on a CUDA event from the async-output thread introduces a second
# stream of CUDA runtime calls into the output-rank process. Under a busy
# CUDA-graph serving loop, those calls contend with the model thread's launch
# path and turn rank-arrival skew into apparent first-collective time.
#
# A Python cudaLaunchHostFunc callback is not safe here: the callback needs the
# GIL, while a CUDA API running on the model thread may hold the GIL and wait
# for the callback's stream (cudaProfilerStop is one concrete example). The
# plugin C++ extension instead enqueues a native callback which only writes to
# a Linux eventfd. This optimization is opt-in: the default is the original
# accelerator Event synchronization. Polling is bounded because CUDA may skip
# a host callback after an asynchronous context error. A slot with an unconsumed
# notification is quarantined until process exit, never recycled or closed
# underneath a potentially late native callback.
_ASYNC_OUTPUT_COMPLETION_POOL_SIZE = 64
_ASYNC_OUTPUT_WAIT_TIMEOUT_S = 30.0
_ASYNC_OUTPUT_POLL_INTERVAL_S = 0.1


class _NativeEventfdCompletionPool:
    """Process-local pool of native stream-callback completion events."""

    def __init__(self, capacity: int = _ASYNC_OUTPUT_COMPLETION_POOL_SIZE):
        if capacity <= 0:
            raise ValueError("Native completion pool capacity must be positive.")
        if not hasattr(os, "eventfd"):
            raise RuntimeError("Linux eventfd is unavailable.")

        # Import lazily because non-NVIDIA platforms may intentionally ship no
        # plugin extension. The op itself dynamically resolves libcuda.
        import importlib

        import torch

        importlib.import_module("vllm_fl._C")
        namespace = getattr(torch.ops, "vllm_fl", None)
        enqueue_op = (
            getattr(namespace, "enqueue_cuda_eventfd_completion", None)
            if namespace is not None
            else None
        )
        if enqueue_op is None:
            raise RuntimeError("vllm_fl native eventfd completion op is unavailable.")

        event_fds: list[int] = []
        try:
            for _ in range(capacity):
                event_fds.append(os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK))
        except Exception:
            for event_fd in event_fds:
                os.close(event_fd)
            raise

        self._enqueue_op = enqueue_op
        self._event_fds = event_fds
        self._available = list(range(capacity - 1, -1, -1))
        self._lock = threading.Lock()
        self._supported = True
        self._closed = threading.Event()
        self._quarantined: set[int] = set()

    def acquire(self) -> int | None:
        with self._lock:
            if not self._supported or not self._available:
                return None
            return self._available.pop()

    def enqueue(self, stream: Any) -> "_NativeEventfdCompletion | None":
        slot = self.acquire()
        if slot is None:
            return None
        try:
            status = int(
                self._enqueue_op(int(stream.cuda_stream), self._event_fds[slot])
            )
            if status != 0:
                raise RuntimeError(
                    f"native CUDA host callback enqueue failed with status {status}"
                )
        except Exception as exc:
            self.retire(slot)
            logger.warning(
                "Native async-output completion is unavailable; using "
                "accelerator Event synchronization instead: %s",
                exc,
            )
            return None
        return _NativeEventfdCompletion(self, slot)

    def wait(self, slot: int, event: Any) -> bool:
        """Return whether the callback notification was consumed.

        Querying the recorded copy event periodically exposes CUDA errors even
        when the host callback is never invoked. If the event completes before
        the callback, CPU copies are safe but the descriptor must be retired.
        """
        deadline = time.monotonic() + _ASYNC_OUTPUT_WAIT_TIMEOUT_S
        poller = select.poll()
        poller.register(self._event_fds[slot], select.POLLIN)
        while True:
            if self._closed.is_set():
                raise RuntimeError("Native async-output completion pool is closed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Native async-output completion timed out")
            ready = poller.poll(
                max(1, math.ceil(min(remaining, _ASYNC_OUTPUT_POLL_INTERVAL_S) * 1000))
            )
            if ready:
                if ready[0][1] != select.POLLIN:
                    raise OSError("Native completion descriptor is unavailable")
                payload = os.read(self._event_fds[slot], 8)
                if len(payload) != 8 or int.from_bytes(payload, "little") != 1:
                    raise RuntimeError("Invalid native completion eventfd payload")
                return True
            # Do not replace this with synchronize(): a missed callback or a
            # stuck stream must not turn the bounded wait into another hang.
            if event.query():
                return False

    def release(self, slot: int) -> None:
        with self._lock:
            if slot not in self._quarantined and not self._closed.is_set():
                self._available.append(slot)

    def retire(self, slot: int) -> None:
        with self._lock:
            self._quarantined.add(slot)
            self._supported = False
        # Keep the fd open. The C++ callback carries only its integer value;
        # closing it would let a late write target an unrelated reused fd.
        # The fixed pool bounds this retention to 64 descriptors per process.

    def close(self) -> None:
        """Cancel pending waits without invalidating late callback descriptors."""
        with self._lock:
            self._supported = False
            self._closed.set()
            # Only free slots have no outstanding callback. Retain the rest.
            for slot in self._available:
                os.close(self._event_fds[slot])
            self._available.clear()


class _NativeEventfdCompletion(NamedTuple):
    pool: _NativeEventfdCompletionPool
    slot: int


_native_completion_pool: _NativeEventfdCompletionPool | bool | None = None
_native_completion_pool_lock = threading.Lock()


def _shutdown_native_completion_pool() -> None:
    """Cancel waits before worker teardown enters accelerator synchronization.

    The opt-in pool has a single worker-process lifetime. Do not recreate it
    after shutdown: callbacks from that lifetime may still reference its fds.
    """
    global _native_completion_pool
    with _native_completion_pool_lock:
        pool = _native_completion_pool
        _native_completion_pool = False
        if isinstance(pool, _NativeEventfdCompletionPool):
            pool.close()


def _get_native_completion_pool() -> _NativeEventfdCompletionPool | None:
    global _native_completion_pool
    enabled = os.environ.get("VLLM_FL_ASYNC_OUTPUT_NATIVE_COMPLETION", "0")
    if enabled not in {"0", "1"}:
        raise ValueError("VLLM_FL_ASYNC_OUTPUT_NATIVE_COMPLETION must be 0 or 1")
    if enabled == "0" or _native_completion_pool is False:
        return None
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or current_platform.is_rocm():
        return None
    if _native_completion_pool is None:
        with _native_completion_pool_lock:
            if _native_completion_pool is None:
                try:
                    _native_completion_pool = _NativeEventfdCompletionPool()
                except (
                    ImportError,
                    AttributeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    _native_completion_pool = False
                    logger.warning(
                        "Native async-output completion initialization "
                        "failed; using accelerator Event synchronization: %s",
                        exc,
                    )
    return (
        _native_completion_pool
        if isinstance(_native_completion_pool, _NativeEventfdCompletionPool)
        else None
    )


def _enqueue_native_completion(
    stream: Any,
) -> _NativeEventfdCompletion | None:
    pool = _get_native_completion_pool()
    if pool is None:
        return None
    completion = pool.enqueue(stream)
    if completion is None and pool._supported:
        logger.warning(
            "Native async-output completion pool exhausted; using "
            "accelerator Event synchronization for this output."
        )
    return completion


def _wait_for_async_output_event(
    event: Any, completion: _NativeEventfdCompletion | None
) -> None:
    """Wait until async D2H copies are safe to consume on the CPU."""
    if completion is None:
        event.synchronize()
        return
    try:
        notified = completion.pool.wait(completion.slot, event)
    except BaseException:
        # Includes cancellation. Never release a slot whose callback may still
        # arrive, and propagate accelerator errors rather than hiding them.
        completion.pool.retire(completion.slot)
        raise
    if notified:
        completion.pool.release(completion.slot)
    else:
        completion.pool.retire(completion.slot)
