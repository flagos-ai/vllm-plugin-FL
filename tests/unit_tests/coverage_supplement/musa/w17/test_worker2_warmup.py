# SPDX-License-Identifier: Apache-2.0
"""Real-import contracts for WorkerFL.compile_or_warm_up_model branches.

The real worker module is imported; vLLM helpers (kernel warmup, JIT monitor,
inductor lazy init, distributed groups) are monkeypatched.  Worker instances use
``object.__new__`` and plain doubles, so no accelerator work happens.
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


def make_worker(cls, **attrs):
    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


class _Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, value):
        return self.start <= value < self.end


@pytest.fixture
def env(monkeypatch):
    import vllm_fl.worker.worker as w

    rec = []
    monkeypatch.setattr(
        w, "CompilationMode", SimpleNamespace(VLLM_COMPILE="compile", NONE="none")
    )
    monkeypatch.setattr(w, "CUDAGraphMode", SimpleNamespace(NONE="none"))
    monkeypatch.setattr(w, "GiB_bytes", 1 << 30)
    monkeypatch.setattr(w, "format_gib", lambda value: "{:.2f}".format(value / (1 << 30)))
    monkeypatch.setattr(w, "set_random_seed", lambda seed: rec.append(("seed", seed)))
    monkeypatch.setattr(w, "freeze_gc_heap", lambda: rec.append("freeze"))
    monkeypatch.setattr(w, "maybe_attach_gc_debug_callback", lambda: rec.append("gc_debug"))
    monkeypatch.setattr(w, "enable_gpu_sync_check", lambda: rec.append("sync_check"))
    monkeypatch.setattr(
        w, "set_torch_threads_for_runtime", lambda: rec.append("threads")
    )
    monkeypatch.setattr(
        w,
        "maybe_save_startup_plan",
        lambda worker, limit: rec.append(("plan", limit)),
    )
    monkeypatch.setattr(w, "kernel_warmup", lambda worker: rec.append("kernel_warmup"))
    monkeypatch.setattr(
        w,
        "warmup_kernels",
        lambda runner, execute, sample: rec.append("warmup_kernels"),
    )
    monkeypatch.setattr(w, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(w, "CompilationTimes", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(w, "current_platform", SimpleNamespace(device_type="cuda", vendor_name=""))
    _install(
        monkeypatch,
        "vllm.utils.jit_monitor",
        activate=lambda **kwargs: rec.append(("jit", kwargs)),
    )
    return w, rec


def _build(w, rec, **over):
    compilation = SimpleNamespace(
        mode=over.pop("mode", "none"),
        backend=over.pop("backend", "eager"),
        compile_sizes=over.pop("compile_sizes", None),
        cudagraph_mode=over.pop("cudagraph_mode", "none"),
        cudagraph_capture_sizes=over.pop("cudagraph_capture_sizes", None),
        get_compile_ranges=over.pop("get_compile_ranges", lambda: []),
        compilation_time=1.5,
        encoder_compilation_time=0.5,
    )

    def dummy_run(*args, **kwargs):
        rec.append(("dummy", args, kwargs))
        return ("hidden", "last")

    model_runner = SimpleNamespace(
        _dummy_run=dummy_run,
        maybe_remove_all_loras=lambda cfg: rec.append(("loras", cfg)),
        lora_config="lora",
        capture_model=lambda: rec.append("capture") or 123,
        is_pooling_model=over.pop("pooling", False),
        _dummy_sampler_run=lambda **kwargs: rec.append(("sampler", kwargs)),
        _dummy_pooler_run=lambda hidden: rec.append(("pooler", hidden)),
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        kernel_config=SimpleNamespace(enable_jit_warmup=over.pop("jit_warmup", False)),
        parallel_config=SimpleNamespace(),
    )
    worker = make_worker(
        w.WorkerFL,
        model_runner=model_runner,
        model_config=SimpleNamespace(
            enforce_eager=over.pop("enforce_eager", False), seed=7
        ),
        vllm_config=vllm_config,
        compilation_config=compilation,
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=over.pop("kv_bytes", None),
            gpu_memory_utilization=0.9,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=16),
        device=SimpleNamespace(type=over.pop("device_type", "cuda")),
        observability_config=SimpleNamespace(
            jit_monitor_mode="warn", jit_monitor_verbose=False
        ),
        init_snapshot=SimpleNamespace(free_memory=1000, total_memory=2000),
        requested_memory=900,
        total_consumed=100,
        peak_activation_memory=50,
        available_kv_cache_memory_bytes=700,
        cudagraph_memory_estimate=over.pop("cudagraph_memory_estimate", 0),
        use_v2_model_runner=over.pop("use_v2", False),
    )
    return worker, model_runner


def _dummy_sizes(rec):
    return [
        entry[1][0]
        for entry in rec
        if isinstance(entry, tuple) and entry and entry[0] == "dummy" and entry[1]
    ]


def test_compile_range_end_is_appended_for_gap(env):
    w, rec = env
    worker, _ = _build(
        w,
        rec,
        mode="compile",
        compile_sizes=[4],
        cudagraph_mode="full",
        cudagraph_capture_sizes=[8],
        get_compile_ranges=lambda: [_Range(100, 200)],
    )
    worker.compile_or_warm_up_model()
    assert _dummy_sizes(rec) == [200, 4]


def test_none_sizes_do_not_warm_up(env):
    w, rec = env
    worker, _ = _build(w, rec, mode="compile", compile_sizes=None, cudagraph_mode="none")
    worker.compile_or_warm_up_model()
    # No batch-size warmup; the trailing V1 dummy run passes num_tokens as a
    # keyword, so it does not appear as a positional size here.
    assert _dummy_sizes(rec) == []
    # The V1 last-rank sampling path still runs a dummy sampler.
    assert ("sampler", {"hidden_states": "last"}) in rec


def test_kunlunxin_skips_generic_kernel_warmup(env, monkeypatch):
    w, rec = env
    monkeypatch.setattr(
        w,
        "current_platform",
        SimpleNamespace(device_type="cuda", vendor_name="kunlunxin"),
    )
    worker, _ = _build(w, rec)
    worker.compile_or_warm_up_model()
    assert "kernel_warmup" not in rec


def test_inductor_lazy_init_triggered(env, monkeypatch):
    w, rec = env
    _install(
        monkeypatch,
        "vllm.compilation.compiler_interface",
        trigger_inductor_lazy_init=lambda device: rec.append(("inductor", device)),
    )
    worker, _ = _build(w, rec, mode="compile", backend="inductor")
    worker.compile_or_warm_up_model()
    assert ("inductor", worker.device) in rec


def test_npu_with_jit_warmup_activates_monitor(env):
    w, rec = env
    worker, _ = _build(w, rec, device_type="npu", jit_warmup=True)
    worker.compile_or_warm_up_model()
    assert any(isinstance(entry, tuple) and entry[0] == "jit" for entry in rec)
    assert "threads" not in rec


def test_v1_non_last_rank_skips_sampler(env, monkeypatch):
    w, rec = env
    monkeypatch.setattr(
        w, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    worker, _ = _build(w, rec)
    worker.compile_or_warm_up_model()
    assert not any(
        isinstance(entry, tuple) and entry and entry[0] in ("sampler", "pooler")
        for entry in rec
    )
    assert "freeze" in rec


def test_v2_uses_warmup_kernels(env):
    w, rec = env
    worker, _ = _build(w, rec, use_v2=True)
    worker.compile_or_warm_up_model()
    assert "warmup_kernels" in rec
    assert not any(
        isinstance(entry, tuple) and entry and entry[0] == "sampler" for entry in rec
    )
