# Copyright (c) 2026 BAAI. All rights reserved.
"""Bound MUSA executables during capture; never restore a cold runtime replay.

Capture and native profiler updates may rotate executables while retaining every
captured template. Ordinary replay uses only the resident set. A nonresident
piece returns False so GraphWrapper can execute its original runnable, instead
of synchronizing, destroying and instantiating one executable for every piece.
The 1900-executable budget is unchanged, and every configured bucket is captured.
"""

import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# MooreThreads documents a per-process limit near 2048 live executables.
# Leave room for graphs outside FL; this does not limit retained graph shapes.
# https://github.com/MooreThreads/torchada#cuda-graphs
_MUSA_GRAPH_EXEC_CAP = 1900


@dataclass
class _GraphState:
    reference: Any
    device: int


class _MusaGraphExecCache:
    def __init__(self, device_api, load_ops, *, cap=_MUSA_GRAPH_EXEC_CAP):
        if cap < 1:
            raise ValueError("MUSA graph executable cap must be positive")
        self._device_api = device_api
        self._load_ops = load_ops
        self._cap = cap
        self._ops = None
        self._states = {}
        self._live = OrderedDict()
        self._lock = threading.RLock()
        self._capture_reserved = False
        self._replay_reserved = 0

    def _forget(self, key, reference):
        with self._lock:
            state = self._states.get(key)
            if state is not None and state.reference is reference:
                self._states.pop(key)
                self._live.pop(key, None)

    def _get_ops(self):
        if self._ops is None:
            # Building is lazy, before capture starts, and failure is fatal.
            self._ops = self._load_ops()
        return self._ops

    def _reserve(self):
        while len(self._live) >= self._cap:
            key, state = next(iter(self._live.items()))
            victim = state.reference()
            if victim is None:
                self._forget(key, state.reference)
                continue
            # Never build, synchronize, or evict inside a native capture.
            if self._device_api.is_current_stream_capturing():
                raise RuntimeError("Cannot rotate MUSA graphs during capture")
            ops = self._get_ops()
            with self._device_api.device(state.device):
                if self._device_api.is_current_stream_capturing():
                    raise RuntimeError(
                        "Cannot evict MUSA graph from a capturing stream"
                    )
                self._device_api.synchronize()
                if not ops.has_exec(victim):
                    raise RuntimeError("Tracked MUSA graph executable is missing")
                ops.free_exec(victim)
                if ops.has_exec(victim):
                    raise RuntimeError("MUSA graph executable was not released")
            # Retain template ownership and only remove the live-exec record
            # after native destruction has succeeded.
            self._live.pop(key)

    @contextmanager
    def capture(self, graph):
        # Keep reservation, native capture and registration atomic with respect
        # to other FL captures/replays in this process.
        with self._lock:
            if self._device_api.is_current_stream_capturing():
                raise RuntimeError("Cannot start MUSA graph capture during capture")
            if self._capture_reserved:
                raise RuntimeError("MUSA graph capture reservation is already open")
            if self._replay_reserved:
                raise RuntimeError("Cannot start MUSA graph capture during replay")
            key = id(graph)
            if key in self._states:
                raise RuntimeError("MUSA graph is already registered")
            self._capture_reserved = True
            try:
                reference = weakref.ref(
                    graph, lambda reference: self._forget(key, reference)
                )
                device = self._device_api.current_device()
                self._reserve()
                yield
                state = _GraphState(reference, device)
                self._states[key] = state
                self._live[key] = state
            finally:
                self._capture_reserved = False

    def _state_for(self, graph):
        state = self._states.get(id(graph))
        if state is None or state.reference() is not graph:
            raise RuntimeError("MUSA graph was not captured by this cache")
        return state

    def _ensure_live(self, graph, state):
        key = id(graph)
        if key not in self._live:
            if self._replay_reserved:
                raise RuntimeError("Cannot restore MUSA graph during replay")
            if self._capture_reserved:
                raise RuntimeError(
                    "Cannot restore MUSA graph during capture reservation"
                )
            if self._device_api.is_current_stream_capturing():
                raise RuntimeError("Cannot instantiate MUSA graphs during capture")
            self._reserve()
            ops = self._get_ops()
            ops.inst_exec(graph)
            if not ops.has_exec(graph):
                raise RuntimeError("MUSA graph executable was not instantiated")
            self._live[key] = state
        self._live.move_to_end(key)

    def replay(self, graph, before_replay=None):
        """Replay a resident executable, or return False without native upkeep.

        The callback runs only for resident pieces, under the same reservation
        lock and device guard as replay. It can stage supported inputs, or return
        False to request the original runnable when staging is unsafe. Cold
        pieces never stage into their captured input storage.
        """
        with self._lock:
            state = self._state_for(graph)
            # Native MUSAGraph.replay accepts a different current device and
            # restores it afterwards. Keep that behavior for both decisions.
            with self._device_api.device(state.device):
                if id(graph) not in self._live:
                    if self._replay_reserved:
                        raise RuntimeError("Cannot restore MUSA graph during replay")
                    if self._capture_reserved:
                        raise RuntimeError(
                            "Cannot restore MUSA graph during capture reservation"
                        )
                    if self._device_api.is_current_stream_capturing():
                        raise RuntimeError(
                            "Cannot instantiate MUSA graphs during capture"
                        )
                    return False
                # RLock prevents other threads from rotating the live set. A
                # reservation also rejects same-thread capture/profiler/cold
                # restore from a staging callback, which could evict this graph.
                self._replay_reserved += 1
                try:
                    if before_replay is not None and before_replay() is False:
                        return False
                    self._live.move_to_end(id(graph))
                    # Keep native replay's RNG prologue and lock through enqueue.
                    graph.replay()
                    return True
                finally:
                    self._replay_reserved -= 1

    def reinstantiate(self, graph, native_reinstantiate):
        # torch_musa's profiler revisits every captured graph and requires a
        # live executable. Restore evicted FL graphs before its native update.
        with self._lock:
            if self._replay_reserved:
                raise RuntimeError("Cannot update MUSA graph profiler during replay")
            if self._capture_reserved:
                raise RuntimeError(
                    "Cannot update MUSA graph profiler during capture reservation"
                )
            if self._device_api.is_current_stream_capturing():
                raise RuntimeError("Cannot update MUSA graph profiler during capture")
            state = self._state_for(graph)
            with self._device_api.device(state.device):
                if self._device_api.is_current_stream_capturing():
                    raise RuntimeError(
                        "Cannot update MUSA graph profiler during capture"
                    )
                self._ensure_live(graph, state)
                return native_reinstantiate()


_cache = None
_cache_lock = threading.Lock()


def _get_cache():
    global _cache
    with _cache_lock:
        if _cache is None:
            import torch

            from vllm_fl.compilation.musa_graph_exec import load_musa_graph_exec_ops

            _cache = _MusaGraphExecCache(torch.musa, load_musa_graph_exec_ops)
        return _cache


def musa_graph_capture(graph):
    return _get_cache().capture(graph)


def musa_graph_replay(graph, before_replay=None):
    return _get_cache().replay(graph, before_replay)


def musa_graph_class(native_graph_class):
    """Preserve native profiler updates for FL-owned, possibly evicted graphs."""

    class MusaGraph(native_graph_class):
        def reinstantiate_graph(self, *args, **kwargs):
            native = super().reinstantiate_graph
            return _get_cache().reinstantiate(self, lambda: native(*args, **kwargs))

    return MusaGraph
