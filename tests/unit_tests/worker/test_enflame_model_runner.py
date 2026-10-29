# Copyright (c) 2026 BAAI. All rights reserved.

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

pytest.importorskip("vllm")

pytestmark = pytest.mark.skipif(
    Version(version("vllm")) < Version("0.32.0"),
    reason="Requires the upstream vLLM 0.32 V1 runner API",
)


@pytest.fixture
def runner_module():
    import vllm_fl.worker.enflame_model_runner as module

    return module


def test_enflame_uses_the_upstream_v1_runner(runner_module):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_fl.worker.runtime import get_fl_model_runner_cls

    assert issubclass(runner_module.EnflameModelRunner, GPUModelRunner)
    assert get_fl_model_runner_cls("gcu") is runner_module.EnflameModelRunner


@pytest.mark.parametrize("io_active", [False, True])
def test_runner_inference_context_preserves_dispatch_diagnostics(
    monkeypatch, runner_module, io_active
):
    import vllm_fl.dispatch.io_common as io_common

    monkeypatch.setattr(io_common, "is_io_active", lambda: io_active)

    @torch.inference_mode()
    def native(value):
        return value, torch.is_inference_mode_enabled(), torch.is_grad_enabled()

    wrapped = runner_module._io_aware_inference(native)
    assert wrapped("output") == ("output", not io_active, False)


@pytest.mark.parametrize("dummy", [False, True])
def test_model_loading_installs_io_hooks_around_native_loading(
    monkeypatch, runner_module, dummy
):
    events = []
    model = object()
    runner = object.__new__(runner_module.EnflameModelRunner)
    runner.model_config = SimpleNamespace(enforce_eager=True)
    monkeypatch.setattr(
        runner_module,
        "init_io_dump_from_env",
        lambda eager: events.append(("init", eager)),
    )
    monkeypatch.setattr(
        runner_module,
        "register_io_module_hooks",
        lambda value: events.append(("hooks", value)),
    )
    monkeypatch.setattr(runner, "get_model", lambda: model)

    def load(self, load_dummy_weights, *, model=None):
        events.append(("load", load_dummy_weights, model))
        return "loaded"

    monkeypatch.setattr(runner_module.GPUModelRunner, "load_model", load)
    assert runner.load_model(dummy, model=model) == "loaded"
    expected = [("init", True), ("load", dummy, model)]
    if not dummy:
        expected.append(("hooks", model))
    assert events == expected


@pytest.mark.parametrize("has_output", [False, True])
def test_sampling_advances_io_only_for_completed_output(
    monkeypatch, runner_module, has_output
):
    output = object() if has_output else None
    events = []

    @torch.inference_mode()
    def sample(self, grammar):
        events.append(("sample", grammar))
        return output

    monkeypatch.setattr(runner_module.GPUModelRunner, "sample_tokens", sample)
    monkeypatch.setattr(
        runner_module, "advance_io_step", lambda: events.append("advance")
    )
    runner = object.__new__(runner_module.EnflameModelRunner)
    assert runner.sample_tokens("grammar") is output
    assert events == [("sample", "grammar")] + (["advance"] if has_output else [])
