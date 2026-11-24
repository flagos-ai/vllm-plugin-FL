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

"""G2 sprint: coverage for vllm_fl/worker/nvidia_worker.py (all lines).

``nvidia_worker`` imports vLLM's native NVIDIA GPU worker at module import
time, which cannot be imported on a non-CUDA host. The native worker module
is therefore replaced in ``sys.modules`` with a stub *before* the target
module is imported; the stub captures the base class that ``NvidiaWorkerFL``
inherits. The remaining vLLM/plugin dependencies (``io_dumper``,
``qsa_cache_hint``, ``managed_inference_mode``) are injected or patched per
test so no CUDA device or real IO dumper is required.
"""

import sys
import types
from types import SimpleNamespace

import pytest

# The stub must be installed before ``vllm_fl.worker.nvidia_worker`` is
# imported, and the target import must happen at module scope so its
# module-level lines are collected exactly once.
_NATIVE_OUTPUT = {"sample": None, "execute": None}

_fake_native_mod = types.ModuleType("vllm.v1.worker.gpu_worker")


class _FakeNativeWorker:
    def __init__(self, *args, **kwargs):
        self.rank = kwargs.get("rank", args[0] if args else 0)
        self.init_device_calls = 0

    def init_device(self):
        self.init_device_calls += 1


def _impl_sample(self, grammar_output):
    return _NATIVE_OUTPUT["sample"]


def _impl_execute(self, scheduler_output):
    return _NATIVE_OUTPUT["execute"]


_impl_sample.__wrapped__ = _impl_sample
_impl_execute.__wrapped__ = _impl_execute
_FakeNativeWorker.sample_tokens = _impl_sample
_FakeNativeWorker.execute_model = _impl_execute
_fake_native_mod.Worker = _FakeNativeWorker

_saved_native = sys.modules.get("vllm.v1.worker.gpu_worker")
sys.modules["vllm.v1.worker.gpu_worker"] = _fake_native_mod
try:
    import vllm_fl.worker.nvidia_worker as nv
finally:
    if _saved_native is None:
        sys.modules.pop("vllm.v1.worker.gpu_worker", None)
    else:
        sys.modules["vllm.v1.worker.gpu_worker"] = _saved_native


class _Patch:
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


class _ModuleInject:
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


def _identity_managed():
    return lambda fn: fn


def _fake_dumper(**overrides):
    mod = types.ModuleType("vllm_fl.dispatch.io_dumper")
    mod.init_io_dump_from_env = overrides.get("init", lambda eager: None)
    mod.is_dump_enabled = overrides.get("enabled", lambda: False)
    mod.register_io_module_hooks = overrides.get("register", lambda model: None)
    mod.advance_io_step = overrides.get("advance", lambda: None)
    return mod


_DUMPER = "vllm_fl.dispatch.io_dumper"


def _worker(rank=0):
    """Construct a worker with the plugin runtime boot no-op'd."""
    with _Patch(nv, "_initialize_fl_runtime", lambda r: None):
        return nv.NvidiaWorkerFL(rank=rank)


# ---------------------------------------------------------------------------
# __init__ (75-78)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# init_device (80-86)
# ---------------------------------------------------------------------------


def _fake_qsa(returns):
    calls = []

    def _patch(model_config, device):
        calls.append((model_config, device))
        return returns

    mod = types.ModuleType("vllm_fl.patches.qsa_cache_hint")
    mod.patch_qsa_cache_hint = _patch
    return mod, calls


def test_init_device_applies_qsa_patch_and_logs():
    w = _worker()
    w.model_config = SimpleNamespace(architecture="Qwen4ExpForCausalLM")
    w.device = SimpleNamespace(type="cuda")
    info = []
    qsa, calls = _fake_qsa(True)
    with (
        _Patch(nv, "logger", SimpleNamespace(info=lambda *a: info.append(a))),
        _ModuleInject("vllm_fl.patches.qsa_cache_hint", qsa),
    ):
        w.init_device()
    assert w.init_device_calls == 1
    assert calls == [(w.model_config, w.device)]
    assert info and "QSA" in info[0][0]


def test_init_device_skips_log_when_qsa_not_applied():
    w = _worker()
    w.model_config = SimpleNamespace(architecture="Other")
    w.device = SimpleNamespace(type="cuda")
    info = []
    qsa, calls = _fake_qsa(False)
    with (
        _Patch(nv, "logger", SimpleNamespace(info=lambda *a: info.append(a))),
        _ModuleInject("vllm_fl.patches.qsa_cache_hint", qsa),
    ):
        w.init_device()
    assert w.init_device_calls == 1
    assert len(calls) == 1
    assert info == []


# ---------------------------------------------------------------------------
# _ensure_io_dump_initialized (88-104)
# ---------------------------------------------------------------------------


def test_ensure_io_dump_initialized_early_return_when_done():
    w = _worker()
    w._fl_io_dump_initialized = True

    def _must_not_run(*a, **k):
        raise AssertionError("dumper must not be imported after initialization")

    fake = _fake_dumper(init=_must_not_run)
    with _ModuleInject(_DUMPER, fake):
        assert w._ensure_io_dump_initialized() is None


def test_ensure_io_dump_disabled_sets_flag_without_hooks():
    w = _worker()
    w.model_config = SimpleNamespace(enforce_eager=True)
    seen = {}
    fake = _fake_dumper(
        init=lambda eager: seen.__setitem__("eager", eager),
        enabled=lambda: False,
        register=lambda model: seen.setdefault("hooks", model),
    )
    with _ModuleInject(_DUMPER, fake):
        w._ensure_io_dump_initialized()
    assert seen["eager"] is True
    assert "hooks" not in seen
    assert w._fl_io_dump_initialized is True


def test_ensure_io_dump_enabled_installs_methods_and_hooks():
    w = _worker()
    w.model_config = SimpleNamespace(enforce_eager=False)
    w.model_runner = SimpleNamespace(get_model=lambda: "MODEL")
    seen = {"installed": [], "hooks": []}
    info = []
    fake = _fake_dumper(
        init=lambda eager: None,
        enabled=lambda: True,
        register=lambda model: seen["hooks"].append(model),
    )
    with (
        _Patch(
            nv,
            "_install_native_runner_io_methods",
            lambda mr: seen["installed"].append(mr),
        ),
        _Patch(nv, "logger", SimpleNamespace(info=lambda *a: info.append(a))),
        _ModuleInject(_DUMPER, fake),
    ):
        w._ensure_io_dump_initialized()
    assert seen["installed"] == [w.model_runner]
    assert seen["hooks"] == ["MODEL"]
    assert info
    assert w._fl_io_dump_initialized is True


# ---------------------------------------------------------------------------
# sample_tokens / execute_model (106-130)
# ---------------------------------------------------------------------------


def test_sample_tokens_advances_step_on_output():
    w = _worker()
    w._fl_io_dump_initialized = True
    advanced = []
    fake = _fake_dumper(advance=lambda: advanced.append(1))
    _NATIVE_OUTPUT["sample"] = "SAMPLE_OUT"
    try:
        with (
            _Patch(nv, "managed_inference_mode", _identity_managed),
            _ModuleInject(_DUMPER, fake),
        ):
            out = nv.NvidiaWorkerFL.sample_tokens(w, "GRAMMAR")
    finally:
        _NATIVE_OUTPUT["sample"] = None
    assert out == "SAMPLE_OUT"
    assert advanced == [1]


def test_sample_tokens_none_output_skips_advance():
    w = _worker()
    w._fl_io_dump_initialized = True
    advanced = []
    fake = _fake_dumper(advance=lambda: advanced.append(1))
    _NATIVE_OUTPUT["sample"] = None
    with (
        _Patch(nv, "managed_inference_mode", _identity_managed),
        _ModuleInject(_DUMPER, fake),
    ):
        out = nv.NvidiaWorkerFL.sample_tokens(w, None)
    assert out is None
    assert advanced == []


def test_execute_model_advances_step_on_output():
    w = _worker()
    w._fl_io_dump_initialized = True
    advanced = []
    fake = _fake_dumper(advance=lambda: advanced.append(1))
    _NATIVE_OUTPUT["execute"] = "EXEC_OUT"
    try:
        with (
            _Patch(nv, "managed_inference_mode", _identity_managed),
            _ModuleInject(_DUMPER, fake),
        ):
            out = nv.NvidiaWorkerFL.execute_model(w, "SCHED")
    finally:
        _NATIVE_OUTPUT["execute"] = None
    assert out == "EXEC_OUT"
    assert advanced == [1]


def test_execute_model_none_output_skips_advance():
    w = _worker()
    w._fl_io_dump_initialized = True
    advanced = []
    fake = _fake_dumper(advance=lambda: advanced.append(1))
    _NATIVE_OUTPUT["execute"] = None
    with (
        _Patch(nv, "managed_inference_mode", _identity_managed),
        _ModuleInject(_DUMPER, fake),
    ):
        out = nv.NvidiaWorkerFL.execute_model(w, None)
    assert out is None
    assert advanced == []


# ---------------------------------------------------------------------------
# _install_native_runner_io_methods (27-50)
# ---------------------------------------------------------------------------


def test_install_native_runner_io_methods_replaces_wrapped_and_skips():
    class _Runner:
        def execute_model(self, *args):
            return "EX"

    def _native(self, *args):
        return "EX"

    _Runner.execute_model.__wrapped__ = _native

    def _pool(*args):
        return None

    _pool._vllm_fl_managed_io = True

    runner = _Runner()
    runner.pool = _pool  # already managed: must be skipped (line 37-38)
    # sample_tokens is absent -> skipped as None (line 36-38)

    with _Patch(nv, "managed_inference_mode", _identity_managed):
        nv._install_native_runner_io_methods(runner)

    assert runner.execute_model.__func__._vllm_fl_managed_io is True
    assert runner.execute_model.__func__ is _native
    assert runner.pool is _pool


def test_install_native_runner_io_methods_raises_without_wrapper():
    class _Runner:
        def execute_model(self, *args):
            return 1

    runner = _Runner()
    with pytest.raises(RuntimeError, match="no longer exposes"):
        nv._install_native_runner_io_methods(runner)


# ---------------------------------------------------------------------------
# _call_native_worker_with_managed_inference (53-63)
# ---------------------------------------------------------------------------


def test_call_native_worker_raises_without_wrapper():
    def _plain(self, *args, **kwargs):
        return None

    with pytest.raises(RuntimeError, match="no longer exposes"):
        nv._call_native_worker_with_managed_inference(_plain, object())


def test_call_native_worker_invokes_unwrapped_impl():
    def _impl(self, *args, **kwargs):
        return ("OK", args)

    _impl.__wrapped__ = _impl
    with _Patch(nv, "managed_inference_mode", _identity_managed):
        result = nv._call_native_worker_with_managed_inference(
            _impl, "WORKER", 1, 2, extra=3
        )
    assert result == ("OK", (1, 2))
