# SPDX-License-Identifier: Apache-2.0
"""Real-import contracts for remaining WorkerFL edge branches.

Covers compilation match-table delegation, interpreter-shutdown guards, draft
weight-update absence, sleep/wake buffer edge paths, profiler reuse, and the XPU
sleep-mode memory-pool branch.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch


def _ensure_pkg(monkeypatch, name):
    import sys
    import types

    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _install(monkeypatch, name, **attrs):
    import sys
    import types

    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def make_worker(cls, **attrs):
    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _platform(**over):
    base = dict(
        device_type="cuda",
        vendor_name="",
        is_cuda_alike=lambda: False,
        is_cuda=lambda: False,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
        is_rocm=lambda: False,
        is_sleep_mode_available=lambda: True,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Allocator:
    def __init__(self, usage=0):
        self.usage = usage
        self.pools = []

    def get_current_usage(self):
        return self.usage

    def use_memory_pool(self, tag):
        self.pools.append(tag)
        return contextlib.nullcontext("pool:" + tag)


def test_get_compilation_match_table_delegates(monkeypatch):
    import vllm_fl.worker.worker as w

    _install(
        monkeypatch,
        "vllm.compilation.passes.vllm_inductor_pass",
        get_match_table=lambda: {"split": 3},
    )
    worker = make_worker(w.WorkerFL)
    assert worker.get_compilation_match_table() == {"split": 3}


def test_shutdown_tolerates_none_transfer_shutdowns(monkeypatch):
    import vllm_fl.worker.worker as w

    calls = []
    monkeypatch.setattr(w, "ensure_kv_transfer_shutdown", None)
    monkeypatch.setattr(w, "ensure_ec_transfer_shutdown", None)
    monkeypatch.setattr(w, "current_platform", _platform(is_cuda_alike=lambda: False))
    worker = make_worker(w.WorkerFL, profiler=None, _marker=calls)
    worker.shutdown()
    assert calls == []


def test_shutdown_skips_cumem_when_instance_absent(monkeypatch):
    import vllm_fl.worker.worker as w

    calls = []
    monkeypatch.setattr(w, "ensure_kv_transfer_shutdown", lambda: calls.append("kv"))
    monkeypatch.setattr(w, "ensure_ec_transfer_shutdown", lambda: calls.append("ec"))
    monkeypatch.setattr(w, "current_platform", _platform(is_cuda_alike=lambda: True))
    _install(
        monkeypatch,
        "vllm.device_allocator.cumem",
        CuMemAllocator=SimpleNamespace(instance=None),
    )
    worker = make_worker(w.WorkerFL, profiler=None)
    worker.shutdown()
    assert calls == ["kv", "ec"]


def test_supports_draft_weight_updates_false_without_engine():
    import vllm_fl.worker.worker as w

    worker = make_worker(
        w.WorkerFL,
        weight_transfer_engine=None,
        speculative_config=SimpleNamespace(draft_model_config=object()),
        model_runner=SimpleNamespace(get_draft_model=lambda: object()),
    )
    assert worker.supports_draft_weight_updates() is False


def test_wake_up_clears_saved_draft_buffers_when_model_missing():
    import vllm_fl.worker.worker as w

    resumed = []
    worker = make_worker(
        w.WorkerFL,
        _sleep_mode_backend=SimpleNamespace(resume=lambda tags: resumed.append(tags)),
        model_runner=SimpleNamespace(
            model=torch.nn.Module(), get_draft_model=lambda: None
        ),
        _sleep_saved_buffers={},
        _sleep_saved_draft_buffers={"d": torch.tensor([1.0])},
    )
    worker.wake_up(tags=["weights"])
    assert resumed == [["weights"]]
    assert worker._sleep_saved_draft_buffers == {}


def test_profile_start_reuses_existing_wrapper(monkeypatch):
    import vllm_fl.worker.worker as w

    started = []

    class _Profiler:
        def start(self):
            started.append("start")

        def stop(self):
            started.append("stop")

    _install(
        monkeypatch,
        "vllm.distributed.utils",
        get_worker_rank_suffix=lambda global_rank: "rank0",
    )
    profiler = _Profiler()
    worker = make_worker(
        w.WorkerFL,
        profiler=profiler,
        profiler_config=SimpleNamespace(profiler="torch"),
        rank=0,
        local_rank=0,
    )
    worker.profile(is_start=True)
    assert started == ["start"]
    assert worker.profiler is profiler


def test_memory_pool_context_xpu_sleep_enabled_uses_allocator(monkeypatch):
    import vllm_fl.worker.worker as w

    allocator = _Allocator()
    monkeypatch.setattr(w, "get_mem_allocator_instance", lambda: allocator)
    monkeypatch.setattr(
        w,
        "current_platform",
        _platform(is_cuda_alike=lambda: False, is_xpu=lambda: True),
    )
    worker = make_worker(
        w.WorkerFL,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=True)
        ),
    )
    with worker._maybe_get_memory_pool_context("weights") as value:
        assert value == "pool:weights"
    assert allocator.pools == ["weights"]


def test_memory_pool_context_unavailable_but_sleep_enabled_uses_allocator(monkeypatch):
    import vllm_fl.worker.worker as w

    allocator = _Allocator()
    monkeypatch.setattr(w, "get_mem_allocator_instance", lambda: allocator)
    monkeypatch.setattr(
        w,
        "current_platform",
        _platform(is_sleep_mode_available=lambda: False),
    )
    worker = make_worker(
        w.WorkerFL,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=True)
        ),
    )
    with worker._maybe_get_memory_pool_context("kv_cache") as value:
        assert value == "pool:kv_cache"


def test_update_max_model_len_without_runner(monkeypatch):
    import vllm_fl.worker.worker as w

    worker = make_worker(
        w.WorkerFL,
        model_runner=None,
        model_config=SimpleNamespace(max_model_len=8),
    )
    worker.update_max_model_len(64)
    assert worker.model_config.max_model_len == 64
