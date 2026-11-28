# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the NVIDIA worker IO integration.

``vllm_fl/worker/nvidia_worker.py`` wraps vLLM v0.32's native GPU worker to
install FL's IO-dump hooks around the runner's inference-mode-decorated entry
points.  The module does not import on a machine without VLLM, so these tests
install minimal fake ``vllm`` / ``vllm_fl.worker.runtime`` graphs (external
boundaries only) and load the real source via ``importlib``:

* ``_install_native_runner_io_methods``: unwrapping the runner methods, the
  managed-IO flag, idempotence and the missing-wrapper error;
* ``_call_native_worker_with_managed_inference``: delegation and its error;
* ``NvidiaWorkerFL.__init__``: FL runtime bootstrap argument forwarding;
* ``_ensure_io_dump_initialized`` plus ``sample_tokens`` / ``execute_model``:
  dump enablement (once), the enabled/disabled branches and
  ``advance_io_step`` only on non-None outputs.

The ``NvidiaPlatformFL`` vendor-gate surface (dist backend, config mutation,
communicator selection, FlagGems attention override) and the
``init_device`` QSA-patch ordering are covered by
``tests/unit_tests/worker/test_nvidia_worker.py``.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_WORKER_SOURCE = _ROOT / "vllm_fl/worker/nvidia_worker.py"

_WORKER_MODULE = "vllm_fl.worker.nvidia_worker"


def _io():
    """Import ``vllm_fl.dispatch.io_common`` lazily, at test time only.

    Importing it at module scope would run ``vllm_fl/__init__.py`` during
    pytest collection 鈥?before any test has set ``TRITON_INTERPRET=1`` 鈥?    which imports triton and permanently binds ``tl.sigmoid`` and friends
    as non-interpreted JITFunctions, breaking the interpreted-kernel tests
    in ``test_gdn_qsa_patches.py`` when both files run in one session.
    """
    import vllm_fl.dispatch.io_common as io_common

    return io_common


def _put(installed: dict, name: str, **attrs: Any) -> ModuleType:
    """Create a stub module and remember it for teardown."""
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    installed[name] = module
    return module


def _link(installed: dict, dotted: str) -> None:
    """Attach ``a.b.c`` to its parent ``a.b`` as attribute ``c``."""
    parent_name, _, child_name = dotted.rpartition(".")
    setattr(installed[parent_name], child_name, installed[dotted])


class _FakeLogger:
    """``vllm.logger.init_logger`` stand-in recording level calls."""

    def __init__(self):
        self.records = []

    def _log(self, level, msg, *args):
        self.records.append((level, msg % args if args else msg))

    def info(self, msg, *args):
        self._log("info", msg, *args)

    def debug(self, msg, *args):
        self._log("debug", msg, *args)

    def warning(self, msg, *args):
        self._log("warning", msg, *args)

    def warning_once(self, msg, *args):
        self._log("warning_once", msg, *args)


class _NativeRunnerStub:
    """Model runner whose entry points mimic vLLM's inference_mode wrappers."""

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        return "native-execute"

    @torch.inference_mode()
    def sample_tokens(self, grammar_output):
        return "native-sample"

    @torch.inference_mode()
    def pool(self):
        return "native-pool"

    def get_model(self):
        return SimpleNamespace(name="fake-model")


class _IoDumperStub:
    """Fake ``vllm_fl.dispatch.io_dumper`` recording every hook call."""

    def __init__(self, monkeypatch, enabled=True):
        self.calls = []
        self.enabled = enabled
        module = ModuleType("vllm_fl.dispatch.io_dumper")
        module.init_io_dump_from_env = self._init_io_dump_from_env
        module.is_dump_enabled = self.is_dump_enabled
        module.register_io_module_hooks = self._register_io_module_hooks
        module.advance_io_step = self._advance_io_step
        monkeypatch.setitem(sys.modules, "vllm_fl.dispatch.io_dumper", module)

    def _init_io_dump_from_env(self, enforce_eager):
        self.calls.append(("init", enforce_eager))

    def is_dump_enabled(self):
        return self.enabled

    def _register_io_module_hooks(self, model):
        self.calls.append(("hooks", model))

    def _advance_io_step(self):
        self.calls.append(("advance", None))

    def method_names(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def worker_env(fl_sys_modules, monkeypatch):
    """Load the worker module against a fake vllm graph.

    Returns a factory: each invocation installs a fresh stub set (a new
    ``NativeGPUWorker`` class and ``_initialize_fl_runtime`` recorder) and
    loads ``nvidia_worker.py`` under its real dotted name so the plugin
    import machinery resolves it once per call.
    """
    created = []

    def _install(native_init_device=None, sample_impl=None, execute_impl=None):
        installed: dict = {}
        logger = _FakeLogger()
        runtime_calls = []

        class _NativeGPUWorker:
            def __init__(self, *args, **kwargs):
                self.rank = kwargs.get("rank", 0)
                self.model_config = kwargs.get("model_config")
                self.device = kwargs.get("device", "cuda:0")

            def init_device(self):
                if native_init_device is not None:
                    return native_init_device(self)
                return "native-init"

            @torch.inference_mode()
            def sample_tokens(self, grammar_output):
                if sample_impl is not None:
                    return sample_impl(self)
                return "native-sample"

            @torch.inference_mode()
            def execute_model(self, scheduler_output):
                if execute_impl is not None:
                    return execute_impl(self)
                return "native-execute"

        _put(installed, "vllm")
        _put(installed, "vllm.logger", init_logger=lambda name: logger)
        _put(
            installed,
            "vllm.v1.core.sched.output",
            GrammarOutput=type("GrammarOutput", (), {}),
            SchedulerOutput=type("SchedulerOutput", (), {}),
        )
        _put(
            installed,
            "vllm.v1.outputs",
            AsyncModelRunnerOutput=type("AsyncModelRunnerOutput", (), {}),
            ModelRunnerOutput=type("ModelRunnerOutput", (), {}),
        )
        _put(installed, "vllm.v1.worker.gpu_worker", Worker=_NativeGPUWorker)
        # The FL runtime bootstrap is an external boundary of the worker.
        runtime_stub = _put(installed, "vllm_fl.worker.runtime")
        runtime_stub._initialize_fl_runtime = runtime_calls.append

        for dotted in (
            "vllm.v1",
            "vllm.v1.core",
            "vllm.v1.core.sched",
            "vllm.v1.worker",
        ):
            if dotted not in installed:
                _put(installed, dotted)
            _link(installed, dotted)
        for name, module in installed.items():
            monkeypatch.setitem(sys.modules, name, module)

        spec = importlib.util.spec_from_file_location(_WORKER_MODULE, _WORKER_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_WORKER_MODULE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_WORKER_MODULE, None)
            raise
        env = SimpleNamespace(
            module=module,
            logger=logger,
            runtime_calls=runtime_calls,
            native_cls=_NativeGPUWorker,
        )
        created.append(env)
        return env

    yield _install
    for _ in reversed(created):
        sys.modules.pop(_WORKER_MODULE, None)


def _make_worker(env, **kwargs: Any) -> Any:
    """Construct an NvidiaWorkerFL with the fixture's default kwargs."""
    kwargs.setdefault("rank", 0)
    return env.module.NvidiaWorkerFL(**kwargs)


# ---------------------------------------------------------------------------
# Runner IO method installation
# ---------------------------------------------------------------------------


class TestInstallNativeRunnerIoMethods:
    """``_install_native_runner_io_methods`` unwrapping rules."""

    def test_all_three_methods_get_managed_wrappers(self, worker_env):
        """execute_model/sample_tokens/pool are re-bound to managed versions."""
        env = worker_env()
        runner = _NativeRunnerStub()
        before = {
            name: getattr(runner, name)
            for name in ("execute_model", "sample_tokens", "pool")
        }

        env.module._install_native_runner_io_methods(runner)

        for name in ("execute_model", "sample_tokens", "pool"):
            managed = getattr(runner, name)
            assert managed is not before[name]
            assert getattr(managed.__func__, "_vllm_fl_managed_io", False)
        assert runner.execute_model(object()) == "native-execute"
        assert runner.sample_tokens(None) == "native-sample"
        assert runner.pool() == "native-pool"

    def test_missing_methods_are_skipped(self, worker_env):
        """A runner without all three names simply skips the absent ones."""
        env = worker_env()

        class Partial:
            @torch.inference_mode()
            def pool(self):
                return "pool"

        partial = Partial()
        env.module._install_native_runner_io_methods(partial)
        assert partial.pool() == "pool"

    def test_missing_wrapped_raises_runtime_error(self, worker_env):
        """A native method without the inference-mode wrapper is an error."""

        class Bare:
            def execute_model(self):
                pass

            @torch.inference_mode()
            def sample_tokens(self):
                pass

            @torch.inference_mode()
            def pool(self):
                pass

        env = worker_env()
        with pytest.raises(RuntimeError, match="no longer exposes"):
            env.module._install_native_runner_io_methods(Bare())

    def test_install_is_idempotent(self, worker_env):
        """A second pass leaves already-managed methods untouched."""
        env = worker_env()
        runner = _NativeRunnerStub()
        env.module._install_native_runner_io_methods(runner)
        first = runner.execute_model
        env.module._install_native_runner_io_methods(runner)
        assert runner.execute_model is first

    def test_managed_methods_run_without_io_active(self, worker_env):
        """With IO inactive the wrapper behaves like plain inference mode."""
        env = worker_env()
        runner = _NativeRunnerStub()
        env.module._install_native_runner_io_methods(runner)
        _io().set_io_active(False)
        try:
            output = runner.execute_model(object())
        finally:
            _io().set_io_active(False)
        assert output == "native-execute"

    def test_managed_methods_downgrade_when_io_active(self, worker_env):
        """With IO dumping active the wrapper runs under no_grad instead.

        The observable difference is autograd visibility: under
        ``torch.inference_mode`` a produced tensor is an inference tensor,
        while the managed path (``torch.no_grad``) keeps it a normal tensor.
        """
        env = worker_env()

        def producing(self, *args):
            return torch.arange(4.0)

        producing_wrapped = torch.inference_mode()(producing)

        class _Runner:
            pass

        runner = _Runner()
        runner.execute_model = producing_wrapped
        runner.sample_tokens = producing_wrapped
        runner.pool = producing_wrapped
        env.module._install_native_runner_io_methods(runner)

        _io().set_io_active(True)
        try:
            output = runner.execute_model(None)
        finally:
            _io().set_io_active(False)
        assert isinstance(output, torch.Tensor)
        assert not output.is_inference()


# ---------------------------------------------------------------------------
# _call_native_worker_with_managed_inference
# ---------------------------------------------------------------------------


class TestCallNativeWorkerWithManagedInference:
    """``_call_native_worker_with_managed_inference`` delegation."""

    def test_calls_the_unwrapped_native_method(self, worker_env):
        """The worker instance is passed through to the unwrapped function."""
        env = worker_env()
        worker = _make_worker(env, rank=3)
        result = env.module._call_native_worker_with_managed_inference(
            env.native_cls.sample_tokens, worker, "grammar"
        )
        assert result == "native-sample"

    def test_missing_wrapped_raises_runtime_error(self, worker_env):
        """A method lacking ``__wrapped__`` fails loudly."""

        def bare(self, *args):
            return None

        bare.__name__ = "bare"

        env = worker_env()
        worker = _make_worker(env)
        with pytest.raises(RuntimeError, match="no longer exposes"):
            env.module._call_native_worker_with_managed_inference(bare, worker, None)

    def test_respects_io_active_mode(self, worker_env):
        """With IO active the native body runs under no_grad."""
        env = worker_env()
        worker = _make_worker(env)

        @torch.inference_mode()
        def producing(self):
            return torch.arange(3.0)

        producing.__name__ = "producing"
        _io().set_io_active(True)
        try:
            output = env.module._call_native_worker_with_managed_inference(
                producing, worker
            )
        finally:
            _io().set_io_active(False)
        assert not output.is_inference()


# ---------------------------------------------------------------------------
# NvidiaWorkerFL lifecycle
# ---------------------------------------------------------------------------


class TestNvidiaWorkerInit:
    """``__init__`` FL bootstrap behaviour."""

    def test_init_initializes_fl_runtime_with_own_rank(self, worker_env):
        """super() runs, then _initialize_fl_runtime receives the rank."""
        env = worker_env()
        worker = _make_worker(env, rank=7)
        assert env.runtime_calls == [7]
        assert worker._fl_io_dump_initialized is False

    def test_init_forwards_arguments_to_native_worker(self, worker_env):
        """Constructor kwargs reach the native worker unchanged."""
        env = worker_env()
        model_config = SimpleNamespace(enforce_eager=True)
        worker = _make_worker(env, rank=2, model_config=model_config, device="cuda:3")
        assert worker.rank == 2
        assert worker.model_config is model_config
        assert worker.device == "cuda:3"


class TestEnsureIoDumpInitialized:
    """``_ensure_io_dump_initialized`` once-guard and enablement flow."""

    def test_enabled_flow_installs_methods_and_hooks(self, worker_env, monkeypatch):
        """Dump enabled: runner methods wrapped and module hooks registered."""
        env = worker_env()
        dumper = _IoDumperStub(monkeypatch, enabled=True)
        worker = _make_worker(env)
        runner = _NativeRunnerStub()
        worker.model_runner = runner
        worker.model_config = SimpleNamespace(enforce_eager=True)

        worker._ensure_io_dump_initialized()

        assert [name for name, _ in dumper.calls] == ["init", "hooks"]
        assert dumper.calls[0] == ("init", True)
        assert dumper.calls[1][0] == "hooks"
        assert getattr(runner.execute_model.__func__, "_vllm_fl_managed_io", False)
        assert worker._fl_io_dump_initialized is True

    def test_disabled_flow_initializes_but_installs_nothing(
        self, worker_env, monkeypatch
    ):
        """Dump disabled: init still runs, no methods or hooks installed."""
        env = worker_env()
        dumper = _IoDumperStub(monkeypatch, enabled=False)
        worker = _make_worker(env)
        runner = _NativeRunnerStub()
        worker.model_runner = runner
        worker.model_config = SimpleNamespace(enforce_eager=False)

        worker._ensure_io_dump_initialized()

        assert dumper.calls == [("init", False)]
        assert not getattr(runner.execute_model.__func__, "_vllm_fl_managed_io", False)
        assert worker._fl_io_dump_initialized is True

    def test_second_call_is_a_noop(self, worker_env, monkeypatch):
        """After the first call the flag short-circuits everything."""
        env = worker_env()
        dumper = _IoDumperStub(monkeypatch, enabled=True)
        worker = _make_worker(env)
        worker.model_runner = _NativeRunnerStub()
        worker.model_config = SimpleNamespace(enforce_eager=True)

        worker._ensure_io_dump_initialized()
        worker._ensure_io_dump_initialized()
        assert [name for name, _ in dumper.calls] == ["init", "hooks"]


class TestWorkerEntryPoints:
    """``sample_tokens`` / ``execute_model`` end-to-end flow."""

    def test_sample_tokens_advances_io_step_on_output(self, worker_env, monkeypatch):
        """A non-None output triggers advance_io_step after the native call."""
        env = worker_env()
        dumper = _IoDumperStub(monkeypatch, enabled=True)
        worker = _make_worker(env)
        worker.model_runner = _NativeRunnerStub()
        worker.model_config = SimpleNamespace(enforce_eager=True)

        result = worker.sample_tokens("grammar")

        assert result == "native-sample"
        assert dumper.method_names()[-1] == "advance"

    def test_sample_tokens_skips_advance_on_none(self, worker_env, monkeypatch):
        """A None output does not advance the IO step counter."""
        env = worker_env(sample_impl=lambda _self: None)
        dumper = _IoDumperStub(monkeypatch, enabled=True)
        worker = _make_worker(env)
        worker.model_runner = _NativeRunnerStub()
        worker.model_config = SimpleNamespace(enforce_eager=True)

        result = worker.sample_tokens("grammar")

        assert result is None
        assert "advance" not in dumper.method_names()

    def test_execute_model_advances_io_step_on_output(self, worker_env, monkeypatch):
        """execute_model mirrors sample_tokens' advance behaviour."""
        env = worker_env()
        dumper = _IoDumperStub(monkeypatch, enabled=True)
        worker = _make_worker(env)
        worker.model_runner = _NativeRunnerStub()
        worker.model_config = SimpleNamespace(enforce_eager=True)

        result = worker.execute_model("scheduler-output")

        assert result == "native-execute"
        assert dumper.method_names()[-1] == "advance"

    def test_execute_model_skips_advance_on_none(self, worker_env, monkeypatch):
        """A None execute_model output leaves the step counter alone."""
        env = worker_env(execute_impl=lambda _self: None)
        dumper = _IoDumperStub(monkeypatch, enabled=True)
        worker = _make_worker(env)
        worker.model_runner = _NativeRunnerStub()
        worker.model_config = SimpleNamespace(enforce_eager=True)

        result = worker.execute_model("scheduler-output")

        assert result is None
        assert "advance" not in dumper.method_names()

    def test_ensure_runs_before_first_execution(self, worker_env, monkeypatch):
        """The first entry-point call initializes the dump subsystem."""
        env = worker_env()
        dumper = _IoDumperStub(monkeypatch, enabled=True)
        worker = _make_worker(env)
        worker.model_runner = _NativeRunnerStub()
        worker.model_config = SimpleNamespace(enforce_eager=True)

        worker.execute_model("out")

        assert "init" in dumper.method_names()
        assert "hooks" in dumper.method_names()
        assert worker._fl_io_dump_initialized is True
