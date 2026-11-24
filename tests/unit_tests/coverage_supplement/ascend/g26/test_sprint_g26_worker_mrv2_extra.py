# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G26 coverage: remaining WorkerFL gaps plus the Ascend V2 runner adapter.

WorkerFL targets (814b):
  _apply_worker_vendor_patches 163-176 (iluvatar + failure branches),
  _maybe_get_memory_pool_context 398-408 (sleep-unavailable / weights guard),
  _get_sleep_mode_backend 299-308,
  determine_available_memory 636-877 (txda, kv_cache_memory_bytes, profile,
  cudagraph estimate on/off),
  get_supported_kv_cache_layouts 903-907, reset_mm/encoder_cache 1160-1164,
  get_model/get_draft_model 1166-1170, get_compilation_match_table 1208-1211,
  save_sharded_state 1487-1500.

model_runner_v2.py targets: _AscendBlockZeroer 75-84, __init__ rejection
branches 90-104, _allocate_and_bind_kv_cache longcat/context 114-147,
_init_kv_zero_meta 354-355, update_requests 357-382, shutdown 384-390.
"""

from __future__ import annotations

import contextlib
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.worker import worker as wmod

_ASCEND_REL = "vllm_fl/dispatch/backends/vendor/ascend"


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


class _Ctx:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


def _platform(**over):
    base = dict(
        device_type="cuda",
        vendor_name="nvidia",
        is_cuda_alike=lambda: False,
        is_cuda=lambda: False,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
        is_rocm=lambda: False,
        is_sleep_mode_available=lambda: True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _logger():
    return SimpleNamespace(
        info=lambda *a, **k: None,
        info_once=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )


# --------------------------------------------------------------------------
# _apply_worker_vendor_patches
# --------------------------------------------------------------------------
def _ilu_chain(monkeypatch, leaf):
    pkg = types.ModuleType("vllm_fl.dispatch.backends.vendor.iluvatar")
    pkg.__path__ = []
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.iluvatar", pkg
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar",
        leaf,
    )


def test_apply_vendor_patches_iluvatar_runs_both(monkeypatch):
    calls = []
    leaf = types.ModuleType("ilu_leaf")
    leaf.patch_torch_inductor_for_iluvatar = lambda: calls.append("inductor")
    leaf.patch_sampler_compile_for_iluvatar = lambda: calls.append("sampler")
    _ilu_chain(monkeypatch, leaf)
    with _stack((wmod, "current_platform", SimpleNamespace(vendor_name="iluvatar"))):
        wmod._apply_worker_vendor_patches()
    assert calls == ["inductor", "sampler"]


def test_apply_vendor_patches_iluvatar_failure_warns(monkeypatch):
    warned = []

    def _boom():
        raise RuntimeError("nope")

    leaf = types.ModuleType("ilu_leaf")
    leaf.patch_torch_inductor_for_iluvatar = _boom
    leaf.patch_sampler_compile_for_iluvatar = lambda: None
    _ilu_chain(monkeypatch, leaf)
    log = _logger()
    log.warning = lambda *a, **k: warned.append(a)
    with _stack(
        (wmod, "current_platform", SimpleNamespace(vendor_name="iluvatar")),
        (wmod, "logger", log),
    ):
        wmod._apply_worker_vendor_patches()
    assert len(warned) == 1


# --------------------------------------------------------------------------
# _maybe_get_memory_pool_context extra branches
# --------------------------------------------------------------------------
def test_maybe_memory_pool_sleep_unavailable_is_null():
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=False)
        )
    )
    platform = _platform(is_sleep_mode_available=lambda: False)
    with _stack((wmod, "current_platform", platform)):
        with wmod.WorkerFL._maybe_get_memory_pool_context(worker, "weights") as v:
            assert v is None


def test_maybe_memory_pool_weights_guard_asserts_when_in_use():
    alloc = SimpleNamespace(
        get_current_usage=lambda: 7,
        use_memory_pool=lambda tag: _Ctx("pool"),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=True)
        )
    )
    with _stack(
        (wmod, "current_platform", _platform()),
        (wmod, "get_mem_allocator_instance", lambda: alloc),
    ):
        with pytest.raises(AssertionError):
            wmod.WorkerFL._maybe_get_memory_pool_context(worker, "weights")


def test_maybe_memory_pool_non_weights_skips_guard():
    alloc = SimpleNamespace(
        get_current_usage=lambda: 7,
        use_memory_pool=lambda tag: _Ctx(tag),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=True)
        )
    )
    with _stack(
        (wmod, "current_platform", _platform()),
        (wmod, "get_mem_allocator_instance", lambda: alloc),
    ):
        with wmod.WorkerFL._maybe_get_memory_pool_context(worker, "kv_cache") as v:
            assert v == "kv_cache"


# --------------------------------------------------------------------------
# _get_sleep_mode_backend
# --------------------------------------------------------------------------
def test_get_sleep_mode_backend_creates_and_caches(monkeypatch):
    try:
        import vllm.device_allocator.sleep_mode_backend as smb
    except Exception:
        pytest.skip("sleep_mode_backend unavailable")

    made = []
    backend = object()

    class _Factory:
        @staticmethod
        def create_backend(model_config):
            made.append(model_config)
            return backend

    monkeypatch.setattr(smb, "SleepModeBackendFactory", _Factory, raising=False)
    worker = SimpleNamespace(
        _sleep_mode_backend=None,
        vllm_config=SimpleNamespace(model_config="mc"),
    )
    assert wmod.WorkerFL._get_sleep_mode_backend(worker) is backend
    assert worker._sleep_mode_backend is backend
    assert wmod.WorkerFL._get_sleep_mode_backend(worker) is backend
    assert made == ["mc"]


# --------------------------------------------------------------------------
# simple delegations
# --------------------------------------------------------------------------
def test_simple_model_delegations():
    calls = []
    runner = SimpleNamespace(
        get_model=lambda: "gm",
        get_draft_model=lambda: "dm",
        reset_mm_cache=lambda: calls.append("mm"),
        reset_encoder_cache=lambda: calls.append("enc"),
    )
    worker = SimpleNamespace(model_runner=runner)
    assert wmod.WorkerFL.get_model(worker) == "gm"
    assert wmod.WorkerFL.get_draft_model(worker) == "dm"
    wmod.WorkerFL.reset_mm_cache(worker)
    wmod.WorkerFL.reset_encoder_cache(worker)
    assert calls == ["mm", "enc"]




def test_get_compilation_match_table(monkeypatch):
    try:
        import vllm.compilation.passes.vllm_inductor_pass as vip
    except Exception:
        pytest.skip("vllm_inductor_pass unavailable")
    monkeypatch.setattr(vip, "get_match_table", lambda: {"op": 3})
    assert wmod.WorkerFL.get_compilation_match_table(SimpleNamespace()) == {"op": 3}


def test_save_sharded_state_delegates(monkeypatch):
    try:
        import vllm.model_executor.model_loader as ml
    except Exception:
        pytest.skip("model_loader unavailable")
    rec = []

    def _save(model, path, pattern=None, max_size=None):
        rec.append((model, path, pattern, max_size))

    monkeypatch.setattr(ml.ShardedStateLoader, "save_model", _save)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model="m"))
    wmod.WorkerFL.save_sharded_state(worker, "p", pattern="x", max_size=9)
    assert rec == [("m", "p", "x", 9)]


# --------------------------------------------------------------------------
# determine_available_memory
# --------------------------------------------------------------------------
def _log_patch():
    return (wmod, "logger", _logger())


def test_determine_available_memory_txda_env_default(monkeypatch):
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", "4096")
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=None)
    )
    with _stack(
        _log_patch(),
        (wmod, "current_platform", SimpleNamespace(device_type="txda")),
    ):
        assert wmod.WorkerFL.determine_available_memory(worker) == 4096


def test_determine_available_memory_txda_bad_env(monkeypatch):
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", "not-int")
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=None)
    )
    with _stack(
        _log_patch(),
        (wmod, "current_platform", SimpleNamespace(device_type="txda")),
    ):
        with pytest.raises(ValueError):
            wmod.WorkerFL.determine_available_memory(worker)


def test_determine_available_memory_txda_non_positive():
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=0)
    )
    with _stack(
        _log_patch(),
        (wmod, "current_platform", SimpleNamespace(device_type="txda")),
    ):
        with pytest.raises(ValueError):
            wmod.WorkerFL.determine_available_memory(worker)


def _profile_worker(llm_mode, estimate, env_flag, profiling=True):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        init_snapshot=SimpleNamespace(free_memory=100, total_memory=200),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=llm_mode)
        ),
        model_runner=SimpleNamespace(
            profile_run=lambda: None,
            profile_cudagraph_memory=lambda: estimate,
            model_memory_usage=123,
        ),
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(_api_process_count=1),
        requested_memory=50,
        use_v2_model_runner=False,
    )


def _run_profile(monkeypatch, worker, profile_result, env_flag, cuda_alike):
    platform = _platform(
        device_type="cuda",
        vendor_name="nvidia",
        is_cuda_alike=lambda: cuda_alike,
    )

    def fake_mem_profiling(snapshot, weights_memory=0, **k):
        return _Ctx(profile_result)

    reserved = []
    with _stack(
        _log_patch(),
        (wmod, "current_platform", platform),
        (wmod, "maybe_apply_startup_plan", lambda w: None),
        (wmod, "memory_profiling", fake_mem_profiling),
        (
            wmod,
            "envs",
            SimpleNamespace(
                VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=env_flag
            ),
        ),
        (
            wmod,
            "reserve_mm_ipc_gpu_memory",
            lambda *a: reserved.append(a) or 777,
        ),
    ):
        out = wmod.WorkerFL.determine_available_memory(worker)
    return out, reserved


def _profile_result():
    return SimpleNamespace(
        total_consumed=10,
        transient_peak_headroom=5,
        after_profile=SimpleNamespace(free_memory=80),
        non_kv_cache_memory=20,
    )


def test_determine_available_memory_profile_no_cudagraph(monkeypatch):
    worker = _profile_worker(object(), 0, False)
    out, reserved = _run_profile(
        monkeypatch, worker, _profile_result(), False, cuda_alike=False
    )
    assert out == 777
    assert worker.total_consumed == 10
    assert worker.peak_activation_memory == 5
    assert worker.cudagraph_memory_estimate == 0
    assert worker.available_kv_cache_memory_bytes == 30
    assert reserved[0][0] == 30


def test_determine_available_memory_profile_cudagraph_applied(monkeypatch):
    worker = _profile_worker(object(), 5, True)
    out, _ = _run_profile(
        monkeypatch, worker, _profile_result(), True, cuda_alike=True
    )
    assert worker.cudagraph_memory_estimate == 5
    assert worker.peak_activation_memory == 10
    assert worker.available_kv_cache_memory_bytes == 25
    assert out == 777


def test_determine_available_memory_profile_cudagraph_not_applied(monkeypatch):
    worker = _profile_worker(object(), 5, False)
    out, _ = _run_profile(
        monkeypatch, worker, _profile_result(), False, cuda_alike=True
    )
    assert worker.peak_activation_memory == 5
    assert worker.available_kv_cache_memory_bytes == 30
    assert out == 777


def test_determine_available_memory_kv_cache_memory_bytes_shortcut(monkeypatch):
    called = []
    worker = _profile_worker(object(), 0, False)
    worker.cache_config.kv_cache_memory_bytes = 1024
    worker.model_runner.profile_run = lambda: called.append(True)
    reserved = []
    with _stack(
        _log_patch(),
        (wmod, "current_platform", _platform()),
        (wmod, "maybe_apply_startup_plan", lambda w: None),
        (
            wmod,
            "reserve_mm_ipc_gpu_memory",
            lambda *a: reserved.append(a) or 999,
        ),
    ):
        out = wmod.WorkerFL.determine_available_memory(worker)
    assert out == 999
    assert called == [True]
    assert reserved[0][0] == 1024


# --------------------------------------------------------------------------
# Ascend V2 runner loader (mirrors g6 package shims)
# --------------------------------------------------------------------------
def _find_repo():
    import os

    env = os.environ.get("FL_REPO_ROOT") or os.environ.get("FL_WORKSPACE")
    candidates = []
    if env:
        candidates.append(pathlib.Path(env))
    here = pathlib.Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    candidates.append(pathlib.Path.cwd())
    for entry in sys.path:
        if entry:
            candidates.append(pathlib.Path(entry))
    for base in candidates:
        if (base / "vllm_fl" / "dispatch" / "backends").is_dir():
            return base
    return None


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_v2_runner():
    repo = _find_repo()
    if repo is None:
        return None, RuntimeError("cannot locate vllm_fl tree")
    ascend = repo / _ASCEND_REL
    worker = ascend / "worker"
    patches = ascend / "patches"

    pkg = types.ModuleType("g26mrv")
    pkg.__path__ = [str(ascend)]
    sys.modules["g26mrv"] = pkg

    worker_pkg = types.ModuleType("g26mrv.worker")
    worker_pkg.__path__ = [str(worker)]
    sys.modules["g26mrv.worker"] = worker_pkg

    patches_pkg = types.ModuleType("g26mrv.patches")
    patches_pkg.__path__ = [str(patches)]
    sys.modules["g26mrv.patches"] = patches_pkg

    device_shim = types.ModuleType("g26mrv.patches.ascend_v2_device")
    device_shim.install_ascend_v2_device_shims = lambda: False
    sys.modules["g26mrv.patches.ascend_v2_device"] = device_shim

    sampling_shim = types.ModuleType("g26mrv.patches.ascend_v2_sampling")
    sampling_shim.install_ascend_v2_sampling_shims = lambda: False
    sys.modules["g26mrv.patches.ascend_v2_sampling"] = sampling_shim

    block_table = types.ModuleType("g26mrv.worker.ascend_v2_block_table")

    class _AscendBlockTables:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    block_table.AscendBlockTables = _AscendBlockTables
    sys.modules["g26mrv.worker.ascend_v2_block_table"] = block_table

    graph = types.ModuleType("g26mrv.worker.ascend_v2_graph")

    class _AscendModelGraphManager:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def release_graphs(self):
            return None

    graph.AscendModelGraphManager = _AscendModelGraphManager
    graph.install_graph_metadata_bridge = lambda *a, **k: None
    sys.modules["g26mrv.worker.ascend_v2_graph"] = graph

    return _load_path(
        "g26mrv.worker.model_runner_v2", worker / "model_runner_v2.py"
    ), None


try:
    mrv2, _MRV2_ERROR = _load_v2_runner()
except Exception as exc:  # vllm V2 modules unavailable
    mrv2, _MRV2_ERROR = None, exc


@pytest.fixture()
def mr():
    if mrv2 is None:
        pytest.skip(f"model_runner_v2 unavailable: {_MRV2_ERROR}")
    return mrv2


# --------------------------------------------------------------------------
# _AscendBlockZeroer / _init_kv_zero_meta
# --------------------------------------------------------------------------
def test_ascend_block_zeroer_zero_and_warmup(mr, monkeypatch):
    calls = []
    monkeypatch.setattr(
        mr, "zero_block_planes", lambda planes, ids: calls.append((planes, ids))
    )
    zeroer = mr._AscendBlockZeroer(["p0"])
    zeroer.zero_block_ids([1, 2])
    zeroer.warmup(0)
    zeroer.warmup(3)
    assert calls == [(["p0"], [1, 2]), (["p0"], [0])]


def test_init_kv_zero_meta_builds_zeroer(mr):
    runner = object.__new__(mr.GPUModelRunnerFLV2)
    runner._ascend_block_planes = ["p0", "p1"]
    mr.GPUModelRunnerFLV2._init_kv_zero_meta(runner)
    assert isinstance(runner.kv_block_zeroer, mr._AscendBlockZeroer)
    assert runner.kv_block_zeroer.planes == ["p0", "p1"]


# --------------------------------------------------------------------------
# __init__ rejection branches
# --------------------------------------------------------------------------
def test_init_requires_v2_runner_flag(mr):
    fake = object.__new__(mr.GPUModelRunnerFLV2)
    cfg = SimpleNamespace(
        use_v2_model_runner=False,
        compilation_config=SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE),
    )
    with pytest.raises(AssertionError):
        mr.GPUModelRunnerFLV2.__init__(fake, cfg, torch.device("cpu"))


def test_init_rejects_unsupported_cudagraph_mode(mr):
    fake = object.__new__(mr.GPUModelRunnerFLV2)
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=object()),
    )
    with pytest.raises(NotImplementedError):
        mr.GPUModelRunnerFLV2.__init__(fake, cfg, torch.device("npu"))


def test_init_rejects_hybrid_cache_with_kv_transfer(mr, monkeypatch):
    monkeypatch.setattr(mr, "uses_dense_hybrid_cache", lambda mc: True)
    fake = object.__new__(mr.GPUModelRunnerFLV2)
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE),
        model_config=object(),
        kv_transfer_config=object(),
    )
    with pytest.raises(NotImplementedError):
        mr.GPUModelRunnerFLV2.__init__(fake, cfg, torch.device("npu"))


# --------------------------------------------------------------------------
# _allocate_and_bind_kv_cache extra branches
# --------------------------------------------------------------------------
def test_allocate_bind_longcat_and_allocation_context(mr, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        mr,
        "allocate_ascend_kv_cache",
        lambda *a, **k: ({"a": "va"}, {"a": "ma"}, ["plane"]),
    )
    monkeypatch.setattr(mr, "get_shared_kv_cache_layers", lambda cfg: {})
    monkeypatch.setattr(
        mr,
        "bind_kv_cache_to_layers",
        lambda views, ctx, **k: recorded.append((views, k)),
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=None)
        ),
        device=torch.device("cpu"),
        kv_cache_config=SimpleNamespace(kv_cache_groups=["g"]),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: "NHD"),
        kernel_block_sizes=[128],
        attn_groups=[["g0"]],
        use_ascend_hybrid_cache=False,
        compilation_config=SimpleNamespace(static_forward_context={"a": 1}),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="longcat_flash")),
    )
    managers = mr.GPUModelRunnerFLV2._allocate_and_bind_kv_cache(
        runner, kv_cache_allocation_context=contextlib.nullcontext()
    )
    assert managers == {"a": "ma"}
    assert runner._ascend_block_planes == ["plane"]
    assert recorded[0][1]["num_attn_module"] == 2


# --------------------------------------------------------------------------
# update_requests
# --------------------------------------------------------------------------
def _update_runner():
    return SimpleNamespace(
        req_states=SimpleNamespace(
            req_id_to_index={"r0": 0, "r1": 1},
            num_computed_tokens_np=np.array([5, 7], dtype=np.int64),
            prefill_len=SimpleNamespace(np=np.array([3, 8], dtype=np.int64)),
            num_computed_prefill_tokens=np.zeros(2, dtype=np.int64),
        ),
        block_tables=SimpleNamespace(append_block_ids=lambda *a, **k: _UPD.append(a)),
        _ascend_block_planes=["p0"],
    )


_UPD = []


def test_update_requests_appends_zeros_and_copies(mr, monkeypatch):
    _UPD.clear()
    zeroed, copied = [], []
    monkeypatch.setattr(
        mr, "zero_block_planes", lambda planes, ids: zeroed.append((planes, ids))
    )
    monkeypatch.setattr(
        mr, "copy_block_planes", lambda planes, copies: copied.append((planes, copies))
    )
    runner = _update_runner()
    sched = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r0", "r1"],
            num_computed_tokens=[5, 7],
            new_block_ids=[None, [10, 11]],
        ),
        new_block_ids_to_zero=[3],
        kv_cache_block_copies=[(1, 2)],
    )
    mr.GPUModelRunnerFLV2.update_requests(runner, sched)
    assert _UPD == [(1, [10, 11])]
    assert runner.req_states.num_computed_prefill_tokens.tolist() == [3, 7]
    assert zeroed == [(["p0"], [3])]
    assert copied == [(["p0"], [(1, 2)])]


def test_update_requests_skips_zero_and_copy_when_empty(mr, monkeypatch):
    _UPD.clear()
    zeroed, copied = [], []
    monkeypatch.setattr(
        mr, "zero_block_planes", lambda *a: zeroed.append(a)
    )
    monkeypatch.setattr(
        mr, "copy_block_planes", lambda *a: copied.append(a)
    )
    runner = _update_runner()
    sched = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r0"],
            num_computed_tokens=[5],
            new_block_ids=[None],
        ),
        new_block_ids_to_zero=[],
        kv_cache_block_copies=[],
    )
    mr.GPUModelRunnerFLV2.update_requests(runner, sched)
    assert zeroed == []
    assert copied == []
    assert _UPD == []


# --------------------------------------------------------------------------
# shutdown
# --------------------------------------------------------------------------
def test_shutdown_releases_graphs_and_clears_state(mr, monkeypatch):
    base = []
    monkeypatch.setattr(
        mr.GPUModelRunner, "shutdown", lambda self: base.append(True)
    )
    graph = mr.AscendModelGraphManager()
    released = []
    graph.release_graphs = lambda: released.append(True)
    runner = object.__new__(mr.GPUModelRunnerFLV2)
    runner.cudagraph_manager = graph
    runner._ascend_block_planes = ["p0"]
    runner._ascend_manager_views = {"a": 1}
    runner.kv_block_zeroer = "z"
    mr.GPUModelRunnerFLV2.shutdown(runner)
    assert released == [True]
    assert runner._ascend_block_planes == []
    assert runner._ascend_manager_views == {}
    assert runner.kv_block_zeroer is None
    assert base == [True]


def test_shutdown_without_graph_manager_skips_release(mr, monkeypatch):
    base = []
    monkeypatch.setattr(
        mr.GPUModelRunner, "shutdown", lambda self: base.append(True)
    )
    runner = object.__new__(mr.GPUModelRunnerFLV2)
    runner.cudagraph_manager = object()
    runner._ascend_block_planes = []
    runner._ascend_manager_views = {}
    runner.kv_block_zeroer = None
    mr.GPUModelRunnerFLV2.shutdown(runner)
    assert base == [True]
