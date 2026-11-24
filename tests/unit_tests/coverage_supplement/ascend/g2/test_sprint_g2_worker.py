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

"""G2 sprint: coverage for vllm_fl/worker/worker.py lines 65-338.

Targets the 814b worktree source. Three clusters:

* module-level helpers ``_prepare_comm_buffer`` fallback (65),
  ``_normalize_process_group_backend`` (130), ``_is_hybrid_kv_model``
  (147-160), ``_apply_worker_vendor_patches`` exception path (175-176);
* the ``AsyncIntermediateTensors`` lazy-comm wrapper (188-208);
* ``WorkerFL.__init__`` (226-297) and ``WorkerFL.sleep`` (311-338).

Unbound-method / SimpleNamespace drives follow the existing blueprint
``coverage_tests/unit/worker`` pattern; ``WorkerBase.__init__`` is the only
patched method on the class under test's own hierarchy (it is the vLLM
input, not ``WorkerFL`` logic).
"""

import importlib
import os
import sys
import types
from types import SimpleNamespace

import torch

import vllm.envs as envs
from vllm.v1.worker.worker_base import WorkerBase

import vllm_fl.worker.worker as worker_module
from vllm_fl.worker.worker import AsyncIntermediateTensors, WorkerFL


class _Patch:
    """Deterministic set/restore of an attribute (R2 input boundary)."""

    def __init__(self, obj, name, value):
        self.obj = obj
        self.name = name
        self.value = value

    def __enter__(self):
        self.saved = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self.saved)
        return False


class _EnvUnset:
    """Ensure an environment variable is absent for the block, then restore."""

    def __init__(self, key):
        self.key = key

    def __enter__(self):
        self.saved = os.environ.pop(self.key, None)
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            os.environ[self.key] = self.saved
        return False


class _ModuleInject:
    """Inject a fake module object at ``name`` for the block, then restore."""

    def __init__(self, name, module):
        self.name = name
        self.module = module

    def __enter__(self):
        self.saved = sys.modules.get(self.name)
        sys.modules[self.name] = self.module
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            sys.modules.pop(self.name, None)
        else:
            sys.modules[self.name] = self.saved
        return False


# ---------------------------------------------------------------------------
# _prepare_comm_buffer fallback (line 65)
# ---------------------------------------------------------------------------


def test_prepare_comm_buffer_fallback_returns_none():
    """The ImportError fallback body returns None for any arguments."""
    import vllm.distributed.parallel_state as parallel_state

    symbol = "prepare_communication_buffer_for_model"
    if not hasattr(parallel_state, symbol):
        # Newer vLLM already dropped the symbol, so the module-level fallback
        # is already the live ``_prepare_comm_buffer``.
        assert worker_module._prepare_comm_buffer(3, y=4) is None
        return

    saved = getattr(parallel_state, symbol)
    delattr(parallel_state, symbol)
    try:
        reloaded = importlib.reload(worker_module)
        assert reloaded._prepare_comm_buffer(3, y=4) is None
    finally:
        setattr(parallel_state, symbol, saved)
        # Restore the module to its normal (symbol-present) state.
        importlib.reload(worker_module)


# ---------------------------------------------------------------------------
# _normalize_process_group_backend (line 130)
# ---------------------------------------------------------------------------


def test_normalize_backend_amd_flagcx_uses_nccl():
    """AMD vendors submit flagcx collectives but keep torch groups on RCCL."""
    platform = SimpleNamespace(device_type="cuda", vendor_name="amd")
    with _Patch(worker_module, "current_platform", platform):
        assert worker_module._normalize_process_group_backend("flagcx") == "nccl"


def test_normalize_backend_npu_flagcx_and_ptpu():
    """NPU flagcx -> hccl; PTPU flagcx/nccl -> pccl."""
    with _Patch(
        worker_module,
        "current_platform",
        SimpleNamespace(device_type="npu", vendor_name="npu"),
    ):
        assert worker_module._normalize_process_group_backend("flagcx") == "hccl"
    with _Patch(
        worker_module,
        "current_platform",
        SimpleNamespace(device_type="ptpu", vendor_name="ptpu"),
    ):
        assert worker_module._normalize_process_group_backend("nccl") == "pccl"


# ---------------------------------------------------------------------------
# _is_hybrid_kv_model (lines 147-160)
# ---------------------------------------------------------------------------


def _hybrid_cfg(*, cache=None, model=None):
    return SimpleNamespace(cache_config=cache, model_config=model)


def test_is_hybrid_kv_model_mamba_cache_modes():
    """mamba_cache_mode align/all short-circuit to True."""
    for mode in ("align", "all"):
        cfg = _hybrid_cfg(cache=SimpleNamespace(mamba_cache_mode=mode))
        assert worker_module._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_flag_on_model_config():
    """model_config.is_hybrid True resolves to True."""
    cfg = _hybrid_cfg(
        cache=SimpleNamespace(mamba_cache_mode="none"),
        model=SimpleNamespace(is_hybrid=True),
    )
    assert worker_module._is_hybrid_kv_model(cfg) is True


class _RaisesHybrid:
    @property
    def is_hybrid(self):
        raise RuntimeError("cannot resolve before init")


def test_is_hybrid_kv_model_is_hybrid_error_falls_back_to_layer_types():
    """An is_hybrid property failure falls back to hf layer_types."""
    model = _RaisesHybrid()
    model.hf_text_config = SimpleNamespace(
        layer_types=["full_attention", "linear_attention"]
    )
    cfg = _hybrid_cfg(cache=SimpleNamespace(mamba_cache_mode="none"), model=model)
    assert worker_module._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_all_full_attention_is_not_hybrid():
    """Only full-attention layers (or no layer_types) is not hybrid."""
    model = SimpleNamespace(
        is_hybrid=False,
        hf_text_config=SimpleNamespace(layer_types=["full_attention"]),
    )
    cfg = _hybrid_cfg(cache=SimpleNamespace(mamba_cache_mode="none"), model=model)
    assert worker_module._is_hybrid_kv_model(cfg) is False

    bare = SimpleNamespace(is_hybrid=False, hf_text_config=None)
    cfg = _hybrid_cfg(cache=None, model=bare)
    assert worker_module._is_hybrid_kv_model(cfg) is False


# ---------------------------------------------------------------------------
# _apply_worker_vendor_patches (lines 175-176)
# ---------------------------------------------------------------------------


def test_apply_worker_vendor_patches_swallows_patch_failure():
    """A failing iluvatar patch import is caught and warned, not raised."""
    warnings = []

    def _boom():
        raise RuntimeError("patch exploded")

    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar")
    fake.patch_torch_inductor_for_iluvatar = _boom
    fake.patch_sampler_compile_for_iluvatar = lambda: None

    with (
        _Patch(
            worker_module,
            "current_platform",
            SimpleNamespace(device_type="cuda", vendor_name="iluvatar"),
        ),
        _Patch(
            worker_module,
            "logger",
            SimpleNamespace(warning=lambda *a, **k: warnings.append(a)),
        ),
        _ModuleInject(
            "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar", fake
        ),
    ):
        worker_module._apply_worker_vendor_patches()

    assert warnings, "expected the failure to be logged"
    assert "Iluvatar worker patch failed" in warnings[0][0]


def test_apply_worker_vendor_patches_noop_for_other_vendors():
    """Non-iluvatar vendors return before importing anything."""
    with _Patch(
        worker_module,
        "current_platform",
        SimpleNamespace(device_type="cuda", vendor_name="nvidia"),
    ):
        assert worker_module._apply_worker_vendor_patches() is None


# ---------------------------------------------------------------------------
# AsyncIntermediateTensors (lines 188-208)
# ---------------------------------------------------------------------------


def test_async_intermediate_tensors_waits_once_then_tensors_is_ready():
    order = []
    handles = [SimpleNamespace(wait=lambda: order.append("wait"))]
    post = [lambda: order.append("post")]

    obj = AsyncIntermediateTensors({"h": torch.zeros(1)}, handles, post)
    assert obj._comm_waited is False

    obj.wait_for_comm()
    assert order == ["wait", "post"]

    # Second wait is a no-op (idempotent guard).
    obj.wait_for_comm()
    assert order == ["wait", "post"]

    # ``.tensors`` access after waiting must not re-run the hooks.
    assert "h" in obj.tensors
    assert order == ["wait", "post"]


def test_async_intermediate_tensors_lazily_waits_on_tensors_access():
    order = []
    handles = [SimpleNamespace(wait=lambda: order.append("wait"))]
    post = [lambda: order.append("post")]

    obj = AsyncIntermediateTensors({"h": torch.ones(1)}, handles, post)
    assert order == []

    tensors = obj.tensors
    assert "h" in tensors
    assert order == ["wait", "post"]


def test_async_intermediate_tensors_no_hooks_is_ready():
    """No handles / no postprocess still marks the object waited."""
    obj = AsyncIntermediateTensors({"h": torch.zeros(1)}, None, None)
    obj.wait_for_comm()
    assert obj._comm_waited is True
    assert "h" in obj.tensors


# ---------------------------------------------------------------------------
# WorkerFL.__init__ (lines 226-297)
# ---------------------------------------------------------------------------


class _FakeElasticExecutor:
    def __init__(self, worker):
        self.worker = worker


class _FakeSentinel:
    def __init__(self, worker=None):
        self.worker = worker


def _make_vllm_config(
    *, num_spec=2, async_sched=False, enable_ft=False, device_type="cuda"
):
    return SimpleNamespace(
        num_speculative_tokens=num_spec,
        scheduler_config=SimpleNamespace(
            async_scheduling=async_sched, scheduler_cls="orig.Scheduler"
        ),
        parallel_config=SimpleNamespace(enable_fault_tolerance=enable_ft),
        model_config=SimpleNamespace(),
        use_v2_model_runner=False,
        profiler_config=SimpleNamespace(profiler="torch"),
        device_type=device_type,
    )


def _fake_base_init(
    self,
    vllm_config=None,
    local_rank=0,
    rank=0,
    distributed_init_method="",
    is_driver_worker=False,
):
    self.vllm_config = vllm_config
    self.local_rank = local_rank
    self.rank = rank
    self.distributed_init_method = distributed_init_method
    self.is_driver_worker = is_driver_worker
    self.parallel_config = vllm_config.parallel_config
    self.scheduler_config = vllm_config.scheduler_config
    self.model_config = vllm_config.model_config


def _build_worker(platform, vllm_config, rank=0):
    """Create a real WorkerFL with the vLLM base __init__ replaced."""
    fake_exec = types.ModuleType("vllm.distributed.elastic_ep.elastic_execute")
    fake_exec.ElasticEPScalingExecutor = _FakeElasticExecutor
    runtime_calls = []
    precision_calls = []

    with (
        _Patch(worker_module, "current_platform", platform),
        _Patch(worker_module, "WorkerBase", WorkerBase),
        _Patch(WorkerBase, "__init__", _fake_base_init),
        _Patch(
            worker_module,
            "_initialize_fl_runtime",
            lambda r: runtime_calls.append(r),
        ),
        _Patch(
            worker_module, "_is_hybrid_kv_model", lambda cfg: cfg.device_type == "k"
        ),
        _Patch(worker_module, "WorkerSentinel", _FakeSentinel),
        _Patch(
            torch,
            "set_float32_matmul_precision",
            lambda p: precision_calls.append(p),
        ),
        _Patch(envs, "VLLM_FLOAT32_MATMUL_PRECISION", "high"),
        _ModuleInject(
            "vllm.distributed.elastic_ep.elastic_execute", fake_exec
        ),
    ):
        obj = WorkerFL.__new__(WorkerFL)
        WorkerFL.__init__(
            obj,
            vllm_config=vllm_config,
            local_rank=0,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:1",
        )
    return obj, runtime_calls, precision_calls


def test_worker_init_gcu_sets_triton_cache_and_async_scheduler():
    """GCU without an explicit cache dir gets a per-rank dir; async scheduler
    is installed for single-token speculative decoding; FT builds a sentinel."""
    cfg = _make_vllm_config(num_spec=1, async_sched=True, enable_ft=True)
    platform = SimpleNamespace(device_type="gcu", vendor_name="gcu")

    with _EnvUnset("TRITON_CACHE_DIR"), _EnvUnset("USE_RESHAPE_AND_CACHE_FLASH"):
        obj, runtime_calls, precision_calls = _build_worker(
            platform, cfg, rank=5
        )

        assert os.environ["TRITON_CACHE_DIR"] == "/tmp/triton-cache-fl-rank-5"
        assert (
            cfg.scheduler_config.scheduler_cls
            == "vllm_fl.worker.scheduler_fl.AsyncSchedulerFL"
        )
        assert runtime_calls == [5]
        assert precision_calls == ["high"]
        assert obj.use_v2_model_runner is False
        assert isinstance(obj.elastic_ep_executor, _FakeElasticExecutor)
        assert isinstance(obj.worker_sentinel, _FakeSentinel)
        assert obj._sleep_saved_buffers == {}
        assert obj._sleep_saved_draft_buffers == {}
        assert obj.weight_transfer_engine is None
        assert obj._weight_update_active is False
        assert obj._weight_update_is_draft is False
        assert obj.profiler is None
        assert obj.profiler_config is cfg.profiler_config
        assert obj._pp_send_work == []
        assert obj._sleep_mode_backend is None


def test_worker_init_kunlunxin_hybrid_enables_fused_kv_path():
    """Kunlunxin hybrid-KV models default the fused paged write path on."""
    cfg = _make_vllm_config(num_spec=2, async_sched=False)
    cfg.device_type = "k"
    platform = SimpleNamespace(device_type="cuda", vendor_name="kunlunxin")

    with _EnvUnset("TRITON_CACHE_DIR"), _EnvUnset(
        "USE_RESHAPE_AND_CACHE_FLASH"
    ):
        obj, runtime_calls, _ = _build_worker(platform, cfg, rank=1)

        assert os.environ["USE_RESHAPE_AND_CACHE_FLASH"] == "1"
        assert cfg.scheduler_config.scheduler_cls == "orig.Scheduler"
        assert obj.worker_sentinel is None
        assert runtime_calls == [1]


def test_get_sleep_mode_backend_resolves_lazily_once():
    """_get_sleep_mode_backend builds on first call and caches the backend."""
    import vllm.device_allocator.sleep_mode_backend as sback

    created = []
    sentinel = object()

    class _Factory:
        @staticmethod
        def create_backend(model_config):
            created.append(model_config)
            return sentinel

    worker = SimpleNamespace(
        _sleep_mode_backend=None,
        vllm_config=SimpleNamespace(model_config="MODELCFG"),
    )
    with _Patch(sback, "SleepModeBackendFactory", _Factory):
        first = WorkerFL._get_sleep_mode_backend(worker)
    second = WorkerFL._get_sleep_mode_backend(worker)

    assert first is sentinel and second is sentinel
    assert created == ["MODELCFG"]
    assert worker._sleep_mode_backend is sentinel


# ---------------------------------------------------------------------------
# WorkerFL.sleep (lines 311-338)
# ---------------------------------------------------------------------------


def test_sleep_level2_saves_buffers_waits_for_free_and_asserts():
    """Level-2 sleep snapshots buffers, suspends, and drains memory to free."""
    order = []
    mem = iter([(100, 1000), (40, 1000), (300, 1000)])
    clock = iter([0.0, 0.05, 0.1])

    class _Clock:
        @staticmethod
        def monotonic():
            return next(clock)

        @staticmethod
        def sleep(seconds):
            order.append(("sleep", seconds))

    model = SimpleNamespace(
        named_buffers=lambda: [("b", torch.zeros(2))],
    )
    draft = SimpleNamespace(
        named_buffers=lambda: [("d", torch.ones(2))],
    )
    backend = SimpleNamespace(suspend=lambda level: order.append(("suspend", level)))

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=model),
        get_draft_model=lambda: draft,
        _get_sleep_mode_backend=lambda: backend,
    )

    with (
        _Patch(
            torch.accelerator,
            "synchronize",
            lambda: order.append("sync"),
        ),
        _Patch(
            torch.accelerator,
            "get_memory_info",
            lambda: next(mem),
        ),
        _Patch(worker_module, "time", _Clock),
        _Patch(
            worker_module,
            "current_platform",
            SimpleNamespace(is_rocm=lambda: True),
        ),
    ):
        WorkerFL.sleep(worker, 2)

    assert ("suspend", 2) in order
    assert ("sleep", 0.1) in order
    assert order.count("sync") == 2
    assert set(worker._sleep_saved_buffers) == {"b"}
    assert set(worker._sleep_saved_draft_buffers) == {"d"}
    assert worker._sleep_saved_buffers["b"].device.type == "cpu"
