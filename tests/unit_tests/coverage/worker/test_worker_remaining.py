# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for the remaining ``vllm_fl/worker/worker.py`` lines.

``test_worker_fl.py`` (which must not be modified) covers the module-level
helpers, the WorkerFL constructor matrix, sleep/wake, checkpointing, weight
transfer, init_device's nvidia/npu paths, execute_model's main plumbing,
profiling and the compile/warmup flows on the default vendors.  Still
uncovered after that file plus the core suite runs:

* ``sleep``'s retry loop when freed memory is momentarily negative — the
  ``time.sleep(0.1)`` retry that only arms on ROCm's 5s deadline (line 335);
* the amd and thead V2-runner compat patches inside ``init_device``
  (567-571 and 575-579), loaded before the upstream V2 runner;
* ``determine_available_memory``'s amd warmup branch (679-681): the
  ``warmup_cached_ops(amd_compile=True)`` call plus the profile config
  context the branch arms;
* ``get_supported_kv_cache_layouts``' config context (906-907);
* ``compile_or_warm_up_model``'s cudagraph estimate log (1020-1022), the
  peak-activation KV suggestion message with its
  ``maybe_save_startup_plan`` handoff (1045-1087), the V1 pooling-model
  pooler warmup (1110) and the inductor lazy-init trigger (1122-1126);
* ``sample_tokens`` delegation (1338) and ``execute_model``'s V2 pooling
  output fixup plus pooling-after-send (1378, 1398-1399).

The tests import and reuse ``test_worker_fl``'s infrastructure (the
``worker_env`` fixture factory, ``WorkerVllmTree``, ``make_vllm_config``,
``_RunnerDouble``, ``_PPGroupStub`` and the sleep-mode double) exactly as
shipped — only new test code is added, no existing file changes.  The
GPU-only vendor compat modules are stubbed through ``sys.modules`` so the
real AMD/Thead patch chains never execute on a CPU test machine.
"""

import contextlib
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.unit_tests.coverage.worker.test_worker_fl import (
    _PPGroupStub,
    _RunnerDouble,
    _SleepModeBackendStub,
    make_vllm_config,
    worker_env as _fl_worker_env,
)

# Reuse the sibling file's environment fixture under its canonical name; the
# indirection avoids ruff's import-vs-parameter F811 collision with the
# ``worker_env`` test parameters below.
worker_env = _fl_worker_env


def _make_worker(worker, **cfg_overrides):
    """Construct a WorkerFL plus recording runner (sibling-file pattern)."""
    cfg = make_vllm_config(**cfg_overrides)
    w = worker.WorkerFL(
        vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
    )
    w.model_runner = _RunnerDouble()
    return w, cfg


# ---------------------------------------------------------------------------
# _prepare_comm_buffer ImportError fallback (lines 62-65)
# ---------------------------------------------------------------------------


class TestPrepareCommBufferFallback:
    """The vLLM >= 0.28 shim when parallel_state lacks the symbol."""

    def test_missing_symbol_becomes_noop(self, worker_env, monkeypatch):
        """A parallel_state without the symbol leaves a None-returning shim
        (the except-import fallback body at lines 62-65)."""
        worker, tree = worker_env()
        # Remove the symbol from the installed stub leaf, then reload the
        # module so the try/except at import time takes the fallback branch.
        installed = sys.modules["vllm.distributed.parallel_state"]
        monkeypatch.delattr(installed, "prepare_communication_buffer_for_model")
        import importlib

        reloaded = importlib.reload(worker)
        assert reloaded._prepare_comm_buffer(object(), args=1) is None


# ---------------------------------------------------------------------------
# sleep: the ROCm retry loop (line 335)
# ---------------------------------------------------------------------------


class TestSleepRetryLoop:
    """``sleep``'s while-loop when the first post-suspend reading is negative."""

    def test_negative_freed_memory_retries_then_recovers(self, worker_env, monkeypatch):
        """On ROCm a transient negative freed-bytes reading sleeps 0.1s and
        re-reads; the recovered reading ends the loop without an assert."""
        worker, tree = worker_env()
        monkeypatch.setattr(tree.platform, "is_rocm", lambda: True)
        w, _ = _make_worker(worker)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(w, "_get_sleep_mode_backend", lambda: backend)

        readings = iter([(100, 256), (90, 256), (100, 256), (100, 256)])
        sleeps = []
        monotonic = iter([0.0, 1.0, 2.0, 3.0, 4.0])

        monkeypatch.setattr(worker.torch.accelerator, "synchronize", lambda: None)
        monkeypatch.setattr(
            worker.torch.accelerator,
            "get_memory_info",
            lambda: next(readings),
            raising=False,
        )
        monkeypatch.setattr(worker.time, "monotonic", lambda: next(monotonic))
        monkeypatch.setattr(worker.time, "sleep", lambda s: sleeps.append(s))

        w.sleep(level=1)

        # One retry cycle: negative reading -> sleep(0.1) -> re-read -> exit.
        assert sleeps == [0.1]
        assert backend.calls == [("suspend", 1)]

    def test_deadline_expiry_stops_retrying(self, worker_env, monkeypatch):
        """When the deadline passes with still-negative readings the loop
        exits via the deadline arm and the post-loop assert fires."""
        worker, tree = worker_env()
        monkeypatch.setattr(tree.platform, "is_rocm", lambda: True)
        w, _ = _make_worker(worker)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(w, "_get_sleep_mode_backend", lambda: backend)

        # Before-sleep reading 100, every post-suspend reading 90: the loop
        # can only exit through the deadline arm.
        readings = iter([(100, 256)] + [(90, 256)] * 6)
        monkeypatch.setattr(worker.torch.accelerator, "synchronize", lambda: None)
        monkeypatch.setattr(
            worker.torch.accelerator,
            "get_memory_info",
            lambda: next(readings),
            raising=False,
        )
        # monotonic starts at 0 (deadline = 5.0), then advances past it.
        tick = {"now": 0.0}

        def advance():
            tick["now"] += 3.0
            return tick["now"]

        monkeypatch.setattr(worker.time, "monotonic", advance)
        sleeps = []
        monkeypatch.setattr(worker.time, "sleep", lambda s: sleeps.append(s))

        with pytest.raises(AssertionError, match="increased after sleeping"):
            w.sleep(level=1)
        # Deadline arm: exactly one sleep happened before expiry.
        assert sleeps == [0.1]


# ---------------------------------------------------------------------------
# init_device: amd / thead V2-runner compat branches (567-579)
# ---------------------------------------------------------------------------


class TestInitDeviceVendorCompatBranches:
    """The amd/thead patches loaded before the upstream V2 runner."""

    @staticmethod
    def _stub_vendor_compat(monkeypatch, leaf, calls):
        """Install a recording ``install_v2_runner_compat`` leaf plus its
        parent package so the method-body import resolves through it."""
        module = ModuleType(leaf)
        module.install_v2_runner_compat = lambda: calls.append(leaf)
        monkeypatch.setitem(sys.modules, leaf, module)
        parent_name = leaf.rsplit(".", 1)[0]
        parent = ModuleType(parent_name)
        setattr(parent, leaf.rsplit(".", 1)[1], module)
        monkeypatch.setitem(sys.modules, parent_name, parent)
        return module

    @pytest.mark.parametrize("vendor_name", ["amd", "thead"])
    def test_v2_runner_compat_installed_before_runner(
        self, worker_env, monkeypatch, vendor_name
    ):
        """vendor=amd/thead installs the compat patch, then builds the
        upstream V2 runner; the patch ran exactly once."""
        leaf = (
            "vllm_fl.dispatch.backends.vendor.amd.v2_runner_compat"
            if vendor_name == "amd"
            else "vllm_fl.dispatch.backends.vendor.thead.patches.v2_runner_compat"
        )
        worker, tree = worker_env(vendor_name=vendor_name, device_type="cuda")
        calls = []
        self._stub_vendor_compat(monkeypatch, leaf, calls)
        cfg = make_vllm_config(use_v2_model_runner=True)
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_init_device_fakes(worker, monkeypatch, tree)
        w.init_device()
        assert calls == [leaf]
        # The upstream V2 runner class was used (recorded by the stub tree).
        assert w.model_runner.__class__ is tree.gpu_model_runner_cls


# ---------------------------------------------------------------------------
# determine_available_memory: amd warmup (679-681)
# ---------------------------------------------------------------------------


class TestDetermineAvailableMemoryAmdWarmup:
    """The AMD pre-profile warmup and profile config context."""

    def test_amd_warms_cached_ops_with_compile(self, worker_env, monkeypatch):
        """vendor=amd runs warmup_cached_ops(amd_compile=True) before the
        profile run, arms the config context around the profile run, and
        returns the IPC-reserved KV budget (independent arithmetic oracle)."""
        worker, tree = worker_env(vendor_name="amd", device_type="cuda")
        w = _make_worker(worker)[0]
        w.cache_config.kv_cache_memory_bytes = None
        w.init_snapshot = SimpleNamespace(free_memory=100, total_memory=256)
        w.requested_memory = 90

        # The method imports warmup_cached_ops from vllm_fl.dispatch at call
        # time; steer that import at a recording double.
        dispatch_calls = []
        fake_dispatch = ModuleType("vllm_fl.dispatch")
        fake_dispatch.warmup_cached_ops = lambda **kw: dispatch_calls.append(dict(kw))
        monkeypatch.setitem(sys.modules, "vllm_fl.dispatch", fake_dispatch)

        # A recording stand-in for the real set_current_vllm_config context.
        config_entries = []

        @contextlib.contextmanager
        def fake_set_config(value):
            config_entries.append(value)
            yield

        monkeypatch.setattr(worker, "set_current_vllm_config", fake_set_config)

        # memory_profiling: real context yielding a full result double.
        result = SimpleNamespace(
            total_consumed=10,
            transient_peak_headroom=5,
            non_kv_cache_memory=20,
            after_profile=SimpleNamespace(free_memory=80),
        )

        @contextlib.contextmanager
        def fake_profiling(*args, **kwargs):
            yield result

        monkeypatch.setattr(worker, "memory_profiling", fake_profiling)

        returned = w.determine_available_memory()
        assert dispatch_calls == [{"amd_compile": True}]
        # The profile run happened inside the worker's own config context.
        assert config_entries == [w.vllm_config]
        assert ("profile_run", (), {}) in w.model_runner.calls
        # Oracle: requested 90 - non_kv 20 - graph estimate 0 = 70, and the
        # tree's reserve_mm_ipc_gpu_memory returns its first argument.
        assert returned == 70
        assert w.available_kv_cache_memory_bytes == 70


def _setup_init_device_fakes(worker, monkeypatch, tree, device_count=8):
    """Patch the torch/vllm surfaces init_device touches (sibling pattern)."""
    import gc

    acc = SimpleNamespace(
        device_count=lambda: device_count,
        synchronize=lambda: None,
        get_memory_info=lambda: (10, 20),
        empty_cache=lambda: None,
    )
    monkeypatch.setattr(worker.torch.accelerator, "device_count", acc.device_count)
    monkeypatch.setattr(worker.torch.accelerator, "empty_cache", acc.empty_cache)
    monkeypatch.setattr(gc, "collect", lambda: None)

    def _snapshot(device=None):
        return SimpleNamespace(free_memory=100, total_memory=256)

    monkeypatch.setattr(worker, "MemorySnapshot", _snapshot)
    monkeypatch.setattr(worker, "request_memory", lambda snap, cc: 90)


# ---------------------------------------------------------------------------
# get_supported_kv_cache_layouts (906-907)
# ---------------------------------------------------------------------------


class TestGetSupportedKvCacheLayouts:
    """The RPC wraps the base call in the current-config context."""

    def test_wraps_base_in_config_context(self, worker_env, monkeypatch):
        """super().get_supported_kv_cache_layouts() runs inside
        set_current_vllm_config(vllm_config) — the context captured the
        worker's own config before the base lookup ran."""
        worker, tree = worker_env()
        w = _make_worker(worker)[0]
        entered = []

        @contextlib.contextmanager
        def fake_set_config(value):
            entered.append(value)
            yield

        monkeypatch.setattr(worker, "set_current_vllm_config", fake_set_config)
        # WorkerBaseStub has no get_supported_kv_cache_layouts, so the real
        # super() lookup raises inside the context — proving both the wrap
        # and the ordering.
        with pytest.raises(AttributeError):
            w.get_supported_kv_cache_layouts()
        assert entered == [w.vllm_config]


# ---------------------------------------------------------------------------
# compile_or_warm_up_model: remaining branches
# ---------------------------------------------------------------------------


class TestCompileOrWarmUpRemainingBranches:
    """Cudagraph estimate log, KV suggestion, V1 pooling warmup, inductor."""

    def _make(self, worker_env, **cfg_overrides):
        worker, tree = worker_env()
        cfg = make_vllm_config(**cfg_overrides)
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.model_runner = _RunnerDouble()
        w.device = SimpleNamespace(type="cuda")
        w.scheduler_config = cfg.scheduler_config
        w.observability_config = SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        )
        return w, worker, tree

    @staticmethod
    def _patch_jit_boundaries(monkeypatch, activated):
        """Patch the jit monitor and inductor lazy-init leaves."""
        jit_mod = ModuleType("vllm.utils.jit_monitor")
        jit_mod.activate = lambda **kw: activated.append(kw)
        monkeypatch.setitem(sys.modules, "vllm.utils.jit_monitor", jit_mod)
        comp_iface = ModuleType("vllm.compilation.compiler_interface")
        comp_iface.trigger_inductor_lazy_init = lambda device: activated.append(
            ("inductor", device)
        )
        monkeypatch.setitem(
            sys.modules, "vllm.compilation.compiler_interface", comp_iface
        )

    def test_cudagraph_memory_estimate_logged(self, worker_env, monkeypatch, caplog):
        """A positive cudagraph_memory_estimate logs the actual-vs-estimated
        comparison message with the pool sizes and difference."""
        w, worker, tree = self._make(worker_env)
        activated = []
        self._patch_jit_boundaries(monkeypatch, activated)
        w.model_runner.capture_model = lambda: 1000 * (1024**3)
        w.cudagraph_memory_estimate = 900 * (1024**3)
        with caplog.at_level(logging.INFO):
            w.compile_or_warm_up_model()
        assert any("CUDA graph pool memory" in rec.message for rec in caplog.records), [
            r.message for r in caplog.records
        ]
        assert any("estimated" in rec.message for rec in caplog.records)

    def test_peak_activation_suggestion_saved_to_startup_plan(
        self, worker_env, monkeypatch
    ):
        """kv_cache_memory_bytes None + peak_activation_memory set computes
        the requested-limit suggestion and hands it to maybe_save_startup_plan."""
        w, worker, tree = self._make(worker_env)
        activated = []
        self._patch_jit_boundaries(monkeypatch, activated)
        w.model_runner.capture_model = lambda: 0
        w.cache_config.kv_cache_memory_bytes = None
        w.peak_activation_memory = 10
        w.total_consumed = 20
        w.init_snapshot = SimpleNamespace(free_memory=500, total_memory=1024)
        w.requested_memory = 100
        w.available_kv_cache_memory_bytes = 30
        w.compile_or_warm_up_model()
        # Independent oracle: non_kv = 20 + 10 + 0 = 30; 150MiB redundancy.
        redundancy = 150 * (1 << 20)
        expected = 100 - 30 - redundancy
        assert ("maybe_save_startup_plan", (w, expected), {}) in tree.calls

    def test_v1_pooling_model_warms_pooler(self, worker_env, monkeypatch):
        """A V1 pooling model's last-rank warmup runs _dummy_pooler_run
        instead of _dummy_sampler_run (line 1110)."""
        w, worker, tree = self._make(worker_env, use_v2_model_runner=False)
        pp = _PPGroupStub(is_first_rank=True, is_last_rank=True)
        monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
        activated = []
        self._patch_jit_boundaries(monkeypatch, activated)
        w.model_runner.is_pooling_model = True
        w.model_runner.capture_model = lambda: 0
        w.compile_or_warm_up_model()
        assert any(name == "_dummy_pooler_run" for name, _, _ in w.model_runner.calls)
        assert not any(
            name == "_dummy_sampler_run" for name, _, _ in w.model_runner.calls
        )

    def test_v1_non_last_rank_skips_sampler_and_triggers_inductor(
        self, worker_env, monkeypatch
    ):
        """A non-last-rank V1 worker skips the sampler/pooler priming, and
        the inductor lazy-init still triggers when the compilation backend
        is inductor (lines 1122-1126)."""
        w, worker, tree = self._make(worker_env, use_v2_model_runner=False)
        pp = _PPGroupStub(is_first_rank=True, is_last_rank=False)
        monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
        activated = []
        self._patch_jit_boundaries(monkeypatch, activated)
        w.model_runner.capture_model = lambda: 0
        # Arm the inductor branch: mode != NONE and backend == "inductor".
        w.vllm_config.compilation_config.mode = worker.CompilationMode.VLLM_COMPILE
        w.vllm_config.compilation_config.backend = "inductor"
        w.compile_or_warm_up_model()
        assert not any(
            name in ("_dummy_sampler_run", "_dummy_pooler_run")
            for name, _, _ in w.model_runner.calls
        )
        assert ("inductor", w.device) in activated


# ---------------------------------------------------------------------------
# sample_tokens delegation (1338)
# ---------------------------------------------------------------------------


class TestSampleTokens:
    """sample_tokens forwards the grammar output to the runner unchanged."""

    def test_delegates_to_runner(self, worker_env):
        """The grammar output reaches model_runner.sample_tokens verbatim."""
        worker, tree = worker_env()
        w = _make_worker(worker)[0]
        grammar = SimpleNamespace(grammar="out")
        result = w.sample_tokens(grammar)
        assert ("sample_tokens", (grammar,), {}) in w.model_runner.calls
        assert result is not None

    def test_none_grammar_forwarded(self, worker_env):
        """A None grammar output is forwarded unchanged."""
        worker, tree = worker_env()
        w = _make_worker(worker)[0]
        w.sample_tokens(None)
        assert ("sample_tokens", (None,), {}) in w.model_runner.calls


# ---------------------------------------------------------------------------
# execute_model: V2 pooling output fixup and pooling-after-send
# ---------------------------------------------------------------------------


class TestExecuteModelPoolingFixup:
    """execute_model's V2 pooling-model output fixup (lines 1378, 1398-1399)."""

    @staticmethod
    def _make(worker_env, monkeypatch, *, is_pooling, is_last_rank):
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        runner = _RunnerDouble()
        runner.is_pooling_model = is_pooling
        w.model_runner = runner
        pp = _PPGroupStub(is_first_rank=True, is_last_rank=is_last_rank)
        monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
        monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace())
        # The stub tree binds ModelRunnerOutput/AsyncModelRunnerOutput to
        # plain ``object``, which would make the isinstance early-return
        # match everything; replace both with non-matching sentinels so the
        # IntermediateTensors send tail is reachable.
        monkeypatch.setattr(worker, "ModelRunnerOutput", type("_MR", (), {}))
        monkeypatch.setattr(worker, "AsyncModelRunnerOutput", type("_AMR", (), {}))
        return w, worker, runner, pp

    def test_v2_pooling_none_output_pools(self, worker_env, monkeypatch):
        """A V2 pooling model whose runner returns None gets pool()'s result
        as the execute_model output (line 1378)."""
        w, worker, runner, pp = self._make(
            worker_env, monkeypatch, is_pooling=True, is_last_rank=True
        )
        runner.execute_model = lambda so, it: None
        # A real pooled output is a ModelRunnerOutput; the sentinel patched
        # above stands in for it so the isinstance early-return matches.
        pooled = worker.ModelRunnerOutput()
        pooled.pooled = True
        runner.pool = lambda: pooled
        result = w.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))
        assert result is pooled
        assert pp.isend_calls == []

    def test_v2_pooling_pool_after_send(self, worker_env, monkeypatch):
        """A V2 pooling model on a non-last rank pools after launching the
        non-blocking send of the intermediate tensors (lines 1398-1399)."""
        w, worker, runner, pp = self._make(
            worker_env, monkeypatch, is_pooling=True, is_last_rank=False
        )
        runner.execute_model = lambda so, it: worker.IntermediateTensors(
            {"hidden": torch.zeros(1)}
        )
        pooled = SimpleNamespace(after_send=True)
        runner.pool = lambda: pooled
        result = w.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))
        assert result is None
        # The send was launched with the runner's intermediate tensors.
        assert len(pp.isend_calls) == 1
        sent_tensors, _ = pp.isend_calls[0]
        assert "hidden" in sent_tensors
