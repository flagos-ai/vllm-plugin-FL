# SPDX-License-Identifier: Apache-2.0
"""Real-import contracts for WorkerFL.init_device vendor branches.

The real ``vllm_fl.worker.worker`` module is imported (not assembled from the
AST).  vLLM/platform internals that would need an accelerator are replaced with
explicit doubles via monkeypatch; instances are built with ``object.__new__`` so
no base ``__init__`` (and no device) is touched.
"""

from types import SimpleNamespace

import pytest


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


class _Dev:
    def __init__(self, spec):
        self.spec = spec
        self.type = spec.split(":")[0]
        self.index = int(spec.split(":")[1])

    def __repr__(self):
        return self.spec


def _platform(**over):
    base = dict(
        device_type="cuda",
        vendor_name="",
        dist_backend="nccl",
        logical_device_id_to_visible_device_id=lambda index: index + 1,
        set_device=lambda device: None,
        check_if_supports_dtype=lambda dtype: None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _parallel(**over):
    base = dict(
        assigned_physical_gpu_ids=None,
        local_world_size=1,
        distributed_executor_backend="mp",
        data_parallel_backend=None,
        nnodes_within_dp=1,
        data_parallel_rank_local=None,
        data_parallel_index=0,
        enable_dbo=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def make_worker(cls, **attrs):
    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


class _RunnerFactory:
    def __init__(self, rec, tag="runner"):
        self.rec = rec
        self.tag = tag

    def __call__(self, cfg, device):
        built = SimpleNamespace(cfg=cfg, device=device, tag=self.tag)
        self.rec.append((self.tag, cfg, device))
        return built


@pytest.fixture
def env(monkeypatch):
    import vllm_fl.worker.worker as w

    rec = []
    fake_torch = SimpleNamespace(
        device=lambda spec: _Dev(spec),
        accelerator=SimpleNamespace(
            device_count=lambda: 8, empty_cache=lambda: rec.append("empty")
        ),
    )
    monkeypatch.setattr(w, "torch", fake_torch)
    monkeypatch.setattr(w, "_apply_worker_vendor_patches", lambda: rec.append("vendor"))
    monkeypatch.setattr(
        w,
        "init_worker_distributed_environment",
        lambda *args: rec.append(("dist", args)),
    )
    monkeypatch.setattr(w, "set_random_seed", lambda seed: rec.append(("seed", seed)))
    monkeypatch.setattr(w, "request_memory", lambda snapshot, cache: 111)
    monkeypatch.setattr(w, "report_usage_stats", lambda cfg: rec.append("usage"))
    monkeypatch.setattr(w, "MemorySnapshot", lambda device: SimpleNamespace(device=device))
    monkeypatch.setattr(
        w,
        "init_workspace_manager",
        lambda device, ubatches, lanes: rec.append(("workspace", ubatches, lanes)),
    )
    monkeypatch.setattr(w, "_num_workspace_lanes", lambda cfg, v2: 3)
    monkeypatch.setattr(
        w, "gc", SimpleNamespace(collect=lambda: rec.append("collect"))
    )
    return w, rec


def _worker(w, parallel, **over):
    config_device_type = over.pop("config_device_type", "cuda")
    vllm_config = SimpleNamespace(parallel_config=parallel)
    return make_worker(
        w.WorkerFL,
        device_config=SimpleNamespace(device_type=config_device_type),
        parallel_config=parallel,
        local_rank=over.pop("local_rank", 0),
        rank=over.pop("rank", 0),
        distributed_init_method="tcp://127.0.0.1:1",
        model_config=SimpleNamespace(dtype="float16", seed=7),
        cache_config=SimpleNamespace(),
        vllm_config=vllm_config,
        use_v2_model_runner=over.pop("use_v2", False),
    )


def test_v2_amd_runs_vendor_compat_and_builds_runner(env, monkeypatch):
    w, rec = env
    called = []
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.amd.v2_runner_compat",
        install_v2_runner_compat=lambda: called.append("amd"),
    )
    runner_cls = _RunnerFactory(rec)
    _install(
        monkeypatch,
        "vllm.v1.worker.gpu.model_runner",
        GPUModelRunner=runner_cls,
    )
    monkeypatch.setattr(w, "current_platform", _platform(vendor_name="amd"))

    parallel = _parallel()
    worker = _worker(w, parallel, use_v2=True)
    worker.init_device()

    assert called == ["amd"]
    assert worker.device.type == "cuda" and worker.device.index == 1
    assert ("runner", worker.vllm_config, worker.device) in rec
    assert ("workspace", 1, 3) in rec
    assert ("seed", 7) in rec and "empty" in rec and "usage" in rec
    assert ("dist", (worker.vllm_config, 0, "tcp://127.0.0.1:1", 0, "nccl")) in rec
    assert worker.init_snapshot.device is worker.device
    assert worker.requested_memory == 111


def test_v2_thead_runs_vendor_compat(env, monkeypatch):
    w, rec = env
    called = []
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.thead.patches.v2_runner_compat",
        install_v2_runner_compat=lambda: called.append("thead"),
    )
    _install(
        monkeypatch,
        "vllm.v1.worker.gpu.model_runner",
        GPUModelRunner=_RunnerFactory(rec, "thead-runner"),
    )
    monkeypatch.setattr(w, "current_platform", _platform(vendor_name="thead"))

    worker = _worker(w, _parallel(), use_v2=True)
    worker.init_device()
    assert called == ["thead"]
    assert ("thead-runner", worker.vllm_config, worker.device) in rec


def test_v2_npu_installs_ascend_shims_and_runner(env, monkeypatch):
    w, rec = env
    calls = []
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils",
        init_device_properties_triton=lambda: calls.append("triton"),
    )
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_v2_device",
        install_ascend_v2_device_shims=lambda: calls.append("shims"),
    )
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.worker.model_runner_v2",
        GPUModelRunnerFLV2=_RunnerFactory(rec, "ascend-runner"),
    )
    monkeypatch.setattr(
        w,
        "current_platform",
        _platform(device_type="npu", vendor_name="ascend", dist_backend="hccl"),
    )

    worker = _worker(w, _parallel(), use_v2=True, config_device_type="npu")
    worker.init_device()
    assert calls == ["triton", "shims"]
    assert worker.device.type == "npu"
    assert ("ascend-runner", worker.vllm_config, worker.device) in rec


def test_v2_iluvatar_overrides_runner(env, monkeypatch):
    w, rec = env
    _install(
        monkeypatch,
        "vllm.v1.worker.gpu.model_runner",
        GPUModelRunner=_RunnerFactory(rec, "upstream-runner"),
    )
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.iluvatar.worker.model_runner_v2",
        IluvatarGPUModelRunnerV2=_RunnerFactory(rec, "iluvatar-runner"),
    )
    monkeypatch.setattr(w, "current_platform", _platform(vendor_name="iluvatar"))

    worker = _worker(w, _parallel(), use_v2=True)
    worker.init_device()
    assert worker.model_runner.tag == "iluvatar-runner"
    assert ("iluvatar-runner", worker.vllm_config, worker.device) in rec


def test_v1_builds_fl_model_runner(env, monkeypatch):
    w, rec = env
    _install(
        monkeypatch,
        "vllm_fl.worker.model_runner",
        ModelRunnerFL=_RunnerFactory(rec, "fl-v1"),
    )
    monkeypatch.setattr(w, "current_platform", _platform())

    worker = _worker(w, _parallel(), use_v2=False, rank=1)
    worker.init_device()
    assert worker.model_runner.tag == "fl-v1"
    assert "usage" not in rec  # only rank 0 reports usage stats


def test_assigned_physical_ids_publish_and_bound_check(env, monkeypatch):
    w, rec = env
    published = []
    _install(
        monkeypatch,
        "vllm.platforms.interface",
        set_assigned_physical_gpu_ids=lambda ids: published.append(list(ids)),
    )
    _install(
        monkeypatch,
        "vllm_fl.worker.model_runner",
        ModelRunnerFL=_RunnerFactory(rec),
    )
    monkeypatch.setattr(w, "current_platform", _platform())

    parallel = _parallel(
        assigned_physical_gpu_ids=[10, 11],
        local_world_size=2,
        distributed_executor_backend="mp",
    )
    worker = _worker(w, parallel, use_v2=False, local_rank=1)
    worker.init_device()
    assert published == [[10, 11]]
    assert worker.local_rank == 1  # a rank slice skips the DP offset


def test_dp_offset_applied_when_not_rank_slice(env, monkeypatch):
    w, rec = env
    _install(
        monkeypatch,
        "vllm_fl.worker.model_runner",
        ModelRunnerFL=_RunnerFactory(rec),
    )
    monkeypatch.setattr(w, "current_platform", _platform())

    parallel = _parallel(
        assigned_physical_gpu_ids=None,
        local_world_size=2,
        distributed_executor_backend="mp",
        data_parallel_rank_local=2,
        nnodes_within_dp=1,
    )
    worker = _worker(w, parallel, use_v2=False, local_rank=1)
    worker.init_device()
    assert worker.local_rank == 5  # 1 + 2 * 2, still below device_count 8


def test_unsupported_device_type_raises(env, monkeypatch):
    w, rec = env
    monkeypatch.setattr(w, "current_platform", _platform())
    worker = make_worker(
        w.WorkerFL,
        device_config=SimpleNamespace(device_type="xpu", device="xpu"),
        parallel_config=_parallel(),
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:1",
        model_config=SimpleNamespace(dtype="float16", seed=7),
        cache_config=SimpleNamespace(),
        vllm_config=SimpleNamespace(parallel_config=_parallel()),
        use_v2_model_runner=False,
    )
    with pytest.raises(RuntimeError, match="Unsupported device type"):
        worker.init_device()
    assert rec == ["vendor"]
