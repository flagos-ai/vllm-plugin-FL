# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU lifetime controls; no MUSA installation or device access is required."""

import gc
import importlib.util
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path

import pytest


def _load_cache_module():
    source = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/compilation/musa_graph_exec_cache.py"
    )
    name = "_musa_graph_exec_cache_cpu_test"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Device:
    def __init__(self):
        self.current = 0
        self.capturing = False
        self.capturing_by_device = {}
        self.calls = []
        self.graphs = weakref.WeakSet()
        self.limit = 2043
        self.fail_sync = False

    def current_device(self):
        return self.current

    def is_current_stream_capturing(self):
        return self.capturing or self.capturing_by_device.get(self.current, False)

    @contextmanager
    def device(self, device):
        previous = self.current
        self.current = device
        try:
            yield
        finally:
            self.current = previous

    def synchronize(self):
        assert not self.capturing
        self.calls.append(("sync", self.current))
        if self.fail_sync:
            raise RuntimeError("sync failed")

    @property
    def live(self):
        return sum(graph.executable for graph in self.graphs)


class _Graph:
    def __init__(self, device):
        self.device = device
        self.ordinal = device.current
        self.executable = False
        self.template = object()
        self.inputs = object()
        self.outputs = object()
        self.pool = object()
        self.rng = object()
        self.replays = 0
        device.graphs.add(self)

    def capture(self):
        assert self.device.live < self.device.limit, "simulated native exec cap"
        self.executable = True
        self.device.calls.append(("capture", self.ordinal))

    def replay(self):
        assert self.executable
        assert self.device.current == self.ordinal
        self.replays += 1
        self.device.calls.append(("native_replay", self.ordinal))


class _Ops:
    def __init__(self, device):
        self.device = device
        self.fail_free = False
        self.fail_inst = False

    def has_exec(self, graph):
        return graph.executable

    def free_exec(self, graph):
        assert self.device.current == graph.ordinal
        assert self.device.calls[-1] == ("sync", graph.ordinal)
        if self.fail_free:
            raise RuntimeError("free failed")
        graph.executable = False
        self.device.calls.append(("free", graph.ordinal))

    def inst_exec(self, graph):
        assert self.device.current == graph.ordinal
        assert self.device.live < self.device.limit
        if self.fail_inst:
            raise RuntimeError("instantiate failed")
        graph.executable = True
        self.device.calls.append(("instantiate", graph.ordinal))


def _fixture(cap=2):
    module = _load_cache_module()
    device = _Device()
    ops = _Ops(device)
    loads = []

    def load():
        assert not device.capturing
        loads.append("load")
        return ops

    return module._MusaGraphExecCache(device, load, cap=cap), device, ops, loads


def _capture(cache, device):
    graph = _Graph(device)
    with cache.capture(graph):
        device.capturing = True
        try:
            graph.capture()
        finally:
            device.capturing = False
    return graph


def test_old_unbounded_51_by_121_capture_hits_native_limit():
    device = _Device()
    graphs = []
    with pytest.raises(AssertionError, match="simulated native exec cap"):
        for _ in range(51 * 121):
            graph = _Graph(device)
            graph.capture()
            graphs.append(graph)
    assert len(graphs) == 2043


def test_51_by_121_templates_and_replay_survive_rotation():
    cache, device, _, loads = _fixture(cap=1900)
    graphs = [_capture(cache, device) for _ in range(51 * 121)]
    identities = [
        (graph.template, graph.inputs, graph.outputs, graph.pool, graph.rng)
        for graph in graphs
    ]
    assert device.live == len(cache._live) == 1900
    assert len(cache._states) == len(graphs) == 6171
    assert loads == ["load"]
    for graph in graphs:
        cache.replay(graph)
        assert device.live <= 1900
    assert all(graph.replays == 1 for graph in graphs)
    assert identities == [
        (graph.template, graph.inputs, graph.outputs, graph.pool, graph.rng)
        for graph in graphs
    ]
    assert len(cache._states) == 6171


def test_below_cap_never_builds_or_synchronizes():
    cache, device, _, loads = _fixture()
    graphs = [_capture(cache, device), _capture(cache, device)]
    for graph in graphs:
        cache.replay(graph)
    assert not loads
    assert not [call for call in device.calls if call[0] == "sync"]


def test_replay_updates_lru_and_preserves_original_native_replay():
    cache, device, _, _ = _fixture()
    first, second = _capture(cache, device), _capture(cache, device)
    cache.replay(first)
    third = _capture(cache, device)
    assert first.executable and third.executable and not second.executable
    cache.replay(second)
    assert second.executable and second.replays == 1
    assert not first.executable
    assert device.calls[-1] == ("native_replay", 0)


def test_gc_releases_manager_tracking_without_retaining_graphs():
    cache, device, _, _ = _fixture()
    graph = _capture(cache, device)
    reference = weakref.ref(graph)
    del graph
    gc.collect()
    assert reference() is None
    assert not cache._states and not cache._live


def test_stale_weakref_callback_cannot_forget_new_record():
    cache, device, _, _ = _fixture()
    graph = _capture(cache, device)
    unrelated = _Graph(device)
    cache._forget(id(graph), weakref.ref(unrelated))
    assert id(graph) in cache._states and id(graph) in cache._live


def test_capture_failure_does_not_register_or_hide_exception():
    cache, device, _, _ = _fixture()
    graph = _Graph(device)
    error = ValueError("original capture failed")

    def capture_then_fail():
        with cache.capture(graph):
            raise error

    with pytest.raises(ValueError) as caught:
        capture_then_fail()
    assert caught.value is error
    assert not cache._states and not cache._live


@pytest.mark.parametrize("failure", ["load", "sync", "free"])
def test_reservation_failure_blocks_next_capture_with_honest_accounting(failure):
    cache, device, ops, _ = _fixture(cap=1)
    first = _capture(cache, device)
    calls_before = list(device.calls)
    if failure == "load":

        def failed_load():
            raise RuntimeError("load failed")

        cache._load_ops = failed_load
    elif failure == "sync":
        device.fail_sync = True
    else:
        ops.fail_free = True
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        _capture(cache, device)
    assert first.executable and list(cache._live) == [id(first)]
    assert len(cache._states) == 1
    assert sum(call[0] == "capture" for call in device.calls) == sum(
        call[0] == "capture" for call in calls_before
    )


def test_instantiation_failure_blocks_replay_and_does_not_mark_live():
    cache, device, ops, _ = _fixture(cap=1)
    first, second = _capture(cache, device), _capture(cache, device)
    ops.fail_inst = True
    with pytest.raises(RuntimeError, match="instantiate failed"):
        cache.replay(first)
    assert not first.executable and not second.executable
    assert not cache._live and len(cache._states) == 2
    assert first.replays == 0


def test_eviction_syncs_victim_device_and_restores_current_device():
    cache, device, _, _ = _fixture(cap=1)
    first = _capture(cache, device)
    device.current = 1
    second = _capture(cache, device)
    assert not first.executable and second.executable
    assert ("sync", 0) in device.calls and device.current == 1
    cache.replay(first)
    assert ("sync", 1) in device.calls and device.current == 1
    assert first.replays == 1 and not second.executable


@pytest.mark.parametrize("cap", [1, 3])
def test_nested_capture_cannot_build_or_evict(cap):
    cache, device, _, loads = _fixture(cap=cap)
    first = _capture(cache, device)
    device.capturing = True
    with pytest.raises(RuntimeError, match="during capture"):
        _capture(cache, device)
    assert first.executable and not loads


def test_unknown_graph_replay_is_rejected():
    cache, device, _, _ = _fixture()
    with pytest.raises(RuntimeError, match="not captured"):
        cache.replay(_Graph(device))


def test_duplicate_capture_is_rejected():
    cache, device, _, _ = _fixture()
    graph = _capture(cache, device)
    with (
        pytest.raises(RuntimeError, match="already registered"),
        cache.capture(graph),
    ):
        pytest.fail("duplicate capture entered")


def test_nonpositive_cap_is_rejected():
    module = _load_cache_module()
    with pytest.raises(ValueError, match="positive"):
        module._MusaGraphExecCache(_Device(), lambda: None, cap=0)


def test_silent_native_free_failure_is_rejected():
    cache, device, ops, _ = _fixture(cap=1)
    first = _capture(cache, device)
    ops.free_exec = lambda graph: None
    with pytest.raises(RuntimeError, match="not released"):
        _capture(cache, device)
    assert first.executable and list(cache._live) == [id(first)]


def test_silent_native_instantiation_failure_is_rejected():
    cache, device, ops, _ = _fixture(cap=1)
    first, second = _capture(cache, device), _capture(cache, device)
    ops.inst_exec = lambda graph: None
    with pytest.raises(RuntimeError, match="not instantiated"):
        cache.replay(first)
    assert not first.executable and not second.executable
    assert not cache._live and first.replays == 0


def test_native_profiler_can_reinstantiate_evicted_fl_graphs():
    module = _load_cache_module()
    device = _Device()
    ops = _Ops(device)
    cache = module._MusaGraphExecCache(device, lambda: ops, cap=2)
    module._cache = cache

    class NativeGraph(_Graph):
        def reinstantiate_graph(self, token):
            assert self.executable
            assert self.device.current == self.ordinal
            self.device.calls.append(("native_profile", self.ordinal))
            return token

    Graph = module.musa_graph_class(NativeGraph)
    graphs = []
    for _ in range(7):
        graph = Graph(device)
        with cache.capture(graph):
            graph.capture()
        graphs.append(graph)
    assert sum(graph.executable for graph in graphs) == 2
    device.current = 1
    for graph in graphs:
        assert graph.reinstantiate_graph("profile result") == "profile result"
        assert device.live == 2
        assert device.current == 1
    assert len([c for c in device.calls if c[0] == "native_profile"]) == 7
    assert NativeGraph.reinstantiate_graph is not Graph.reinstantiate_graph


def test_native_profiler_failure_is_propagated_unchanged():
    cache, device, _, _ = _fixture(cap=1)
    first, second = _capture(cache, device), _capture(cache, device)
    error = ValueError("native profile failed")

    def update():
        raise error

    with pytest.raises(ValueError) as caught:
        cache.reinstantiate(first, update)
    assert caught.value is error
    assert first.executable and not second.executable
    assert device.current == 0


def test_open_capture_reservation_rejects_reentrant_capture_without_overflow():
    cache, device, _, _ = _fixture(cap=1)
    outer = _Graph(device)
    with cache.capture(outer):
        with pytest.raises(RuntimeError, match="reservation is already open"):
            _capture(cache, device)
        outer.capture()
    assert device.live == len(cache._live) == 1
    assert not cache._capture_reserved


def test_open_capture_reservation_rejects_cold_restore():
    cache, device, _, _ = _fixture(cap=1)
    old, latest = _capture(cache, device), _capture(cache, device)
    outer = _Graph(device)
    with cache.capture(outer):
        assert not old.executable and not latest.executable
        with pytest.raises(RuntimeError, match="during capture reservation"):
            cache.replay(old)
        outer.capture()
    assert device.live == len(cache._live) == 1
    assert old.replays == 0


def test_hot_replay_remains_available_with_open_capture_reservation():
    cache, device, _, _ = _fixture(cap=2)
    existing = _capture(cache, device)
    outer = _Graph(device)
    with cache.capture(outer):
        cache.replay(existing)
        outer.capture()
    assert existing.replays == 1
    assert device.live == 2


def test_capture_failure_closes_reservation_for_later_capture():
    cache, device, _, _ = _fixture(cap=1)

    def capture_then_fail():
        with cache.capture(_Graph(device)):
            raise ValueError("capture failed")

    with pytest.raises(ValueError):
        capture_then_fail()
    assert not cache._capture_reserved
    graph = _capture(cache, device)
    assert graph.executable


def test_capturing_victim_device_blocks_sync_and_eviction():
    cache, device, _, _ = _fixture(cap=1)
    first = _capture(cache, device)
    device.current = 1
    device.capturing_by_device[0] = True
    with pytest.raises(RuntimeError, match="capturing stream"):
        _capture(cache, device)
    assert first.executable and device.current == 1
    assert not [call for call in device.calls if call[0] in {"sync", "free"}]
    assert not cache._capture_reserved


def test_profiler_rejects_capture_on_recorded_device_even_if_graph_is_live():
    cache, device, _, _ = _fixture(cap=2)
    graph = _capture(cache, device)
    device.current = 1
    device.capturing_by_device[0] = True
    with pytest.raises(RuntimeError, match="during capture"):
        cache.reinstantiate(graph, lambda: pytest.fail("native update entered"))
    assert graph.executable and device.current == 1


def test_profiler_rejects_reentrant_update_with_open_reservation():
    cache, device, _, _ = _fixture(cap=2)
    graph = _capture(cache, device)
    outer = _Graph(device)
    with cache.capture(outer):
        with pytest.raises(RuntimeError, match="capture reservation"):
            cache.reinstantiate(graph, lambda: pytest.fail("native update entered"))
        outer.capture()
    assert graph.executable and device.live == 2
