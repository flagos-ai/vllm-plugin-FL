# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Preserve native query-ready preparation and restrict the MetaX phase fence."""

from types import SimpleNamespace

import pytest

import vllm.config
import vllm.platforms

from vllm_fl.dispatch.backends.vendor.metax import patches
from vllm_fl.dispatch.backends.vendor.metax.patches.deepseek_v41 import graph as patch
from vllm_fl.worker import runtime
from vllm_fl.worker.deepseek_v41 import prepare_v41_graph_kernels


def graph_config(model="deepseek_v41", ep=True, dp=8, spec=True):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model)),
        speculative_config=SimpleNamespace(method="dspark") if spec else None,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=ep, data_parallel_size=dp
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=2),
    )


@pytest.mark.parametrize(
    "model,ep,dp,num_tokens,fenced",
    [
        ("deepseek_v41", True, 8, 5, True),
        ("deepseek_v41", True, 8, 40, True),
        ("deepseek_v41", False, 8, 40, False),
        ("deepseek_v41", True, 1, 40, False),
        ("qwen3", True, 8, 40, False),
    ],
)
def test_query_and_replay_boundaries(monkeypatch, model, ep, dp, num_tokens, fenced):
    events = []
    group = SimpleNamespace(barrier=lambda: events.append("barrier"))
    monkeypatch.setattr(patch, "get_ep_group", lambda: group)
    monkeypatch.setattr(patch.torch.cuda, "synchronize", lambda: events.append("sync"))
    manager = SimpleNamespace(vllm_config=graph_config(model, ep, dp))
    desc, result = object(), object()

    def prepare(owner, tokens):
        assert owner is manager and tokens == num_tokens
        events.append("prepare")
        return result

    def replay(owner, actual):
        assert owner is manager and actual is desc
        events.append("replay")
        return result

    assert patch._wrap_prepare_forward(prepare)(manager, num_tokens) is result
    assert patch._wrap_replay(replay)(manager, desc) is result
    assert events == (
        ["prepare", "barrier", "sync", "barrier", "replay", "sync", "barrier"]
        if fenced
        else ["prepare", "replay"]
    )


def test_installation_is_idempotent_and_rejects_unknown_contract(monkeypatch):
    def original(self, num_unpadded_tokens):
        return num_unpadded_tokens

    fake_class = SimpleNamespace(_prepare_eplb_forward=original)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="metax")
    )
    monkeypatch.setattr(patch, "DSparkSpeculator", fake_class)
    patches._install_v41_patches()
    installed = patch.DSparkSpeculator._prepare_eplb_forward
    patches._install_v41_patches()
    assert patch.DSparkSpeculator._prepare_eplb_forward is installed
    monkeypatch.setattr(
        patch.DSparkSpeculator, "_prepare_eplb_forward", lambda self: None
    )
    with pytest.raises(RuntimeError, match="graph boundary"):
        patches._install_v41_patches()


def test_required_patch_failure_is_not_hidden_by_runtime(monkeypatch):
    from vllm_fl.ops import custom_ops

    error = RuntimeError("incompatible upstream signature")
    monkeypatch.setattr(patches, "ensure_patches", lambda: {"deepseek_v41": error})
    monkeypatch.setattr(custom_ops, "register_oot_ops", lambda: None)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="metax")
    )
    cfg = graph_config()
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)
    with pytest.raises(RuntimeError, match="required V4.1 MetaX patch group") as caught:
        runtime._initialize_fl_runtime(1)
    assert caught.value.__cause__ is error


def test_installation_contract_is_checked_before_mutation(monkeypatch):
    def original(self, desc):
        return desc

    fake = SimpleNamespace(run_fullgraph=original)
    monkeypatch.setattr(patch, "CudaGraphManager", fake)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="metax")
    )
    patches._install_v41_patches()
    installed = fake.run_fullgraph
    patches._install_v41_patches()
    assert fake.run_fullgraph is installed
    fake.run_fullgraph = lambda self: None
    with pytest.raises(RuntimeError, match="graph boundary"):
        patches._install_v41_patches()


@pytest.mark.parametrize("failure_phase", ["prepare", "install"])
def test_dependent_patch_group_rolls_back_and_retries(monkeypatch, failure_phase):
    old, new = object(), object()
    first = SimpleNamespace(binding=old)

    class Second:
        binding = old
        fail = failure_phase == "install"

        def __setattr__(self, name, value):
            if name == "binding" and value is new and self.fail:
                self.fail = False
                raise RuntimeError("injected installation failure")
            object.__setattr__(self, name, value)

    second = Second()
    fail_prepare = failure_phase == "prepare"

    def prepare_second():
        nonlocal fail_prepare
        if fail_prepare:
            fail_prepare = False
            raise RuntimeError("injected preflight failure")
        return [(second, "binding", new)]

    modules = {
        "first": SimpleNamespace(prepare=lambda: [(first, "binding", new)]),
        "second": SimpleNamespace(prepare=prepare_second),
    }
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="metax")
    )
    monkeypatch.setattr(patches, "_V41_PATCH_MODULES", tuple(modules))
    monkeypatch.setattr(
        patches.importlib, "import_module", lambda name: modules[name.rsplit(".", 1)[1]]
    )
    with pytest.raises(RuntimeError, match="injected"):
        patches._install_v41_patches()
    assert first.binding is old and second.binding is old
    patches._install_v41_patches()
    assert first.binding is new and second.binding is new


def test_other_vendor_does_not_import_or_mutate_patch_group(monkeypatch):
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="nvidia")
    )
    monkeypatch.setattr(
        patches.importlib,
        "import_module",
        lambda name: pytest.fail("unexpected vendor import"),
    )
    patches._install_v41_patches()


def make_worker(model_type="deepseek_v41", dp=8, spec=True):
    target = SimpleNamespace(_graphs_captured=False)
    draft = SimpleNamespace(_graphs_captured=False)
    return SimpleNamespace(
        use_v2_model_runner=True,
        vllm_config=graph_config(model_type, dp=dp, spec=spec),
        model_runner=SimpleNamespace(
            cudagraph_manager=target,
            speculator=SimpleNamespace(query_cudagraph_manager=draft),
        ),
        execute_model=object(),
        sample_tokens=object(),
    )


def test_preparation_uses_native_callbacks_and_preserves_managers(monkeypatch):
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="metax")
    )
    worker = make_worker()
    runner = worker.model_runner
    target, draft = runner.cudagraph_manager, runner.speculator.query_cudagraph_manager
    events = []
    monkeypatch.setattr(
        vllm.distributed,
        "get_ep_group",
        lambda: SimpleNamespace(barrier=lambda: events.append("barrier")),
    )

    def warmup(actual, execute, sample):
        assert (
            actual is runner
            and execute is worker.execute_model
            and sample is worker.sample_tokens
        )
        assert not target._graphs_captured and not draft._graphs_captured
        events.append("warmup")

    assert prepare_v41_graph_kernels(worker, warmup)
    assert events == ["warmup", "barrier"]
    assert (
        runner.cudagraph_manager is target
        and runner.speculator.query_cudagraph_manager is draft
    )
    assert worker.vllm_config.compilation_config.cudagraph_mode == 2


@pytest.mark.parametrize(
    "model,dp,spec",
    [("qwen3", 8, True), ("deepseek_v41", 1, True), ("deepseek_v41", 8, False)],
)
def test_other_profiles_retain_native_order(monkeypatch, model, dp, spec):
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="metax")
    )
    assert not prepare_v41_graph_kernels(
        make_worker(model, dp, spec), lambda *args: pytest.fail("unexpected warmup")
    )


def test_live_graphs_are_rejected_before_preparation(monkeypatch):
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="metax")
    )
    worker = make_worker()
    worker.model_runner.cudagraph_manager._graphs_captured = True
    with pytest.raises(RuntimeError, match="precede graph capture"):
        prepare_v41_graph_kernels(
            worker, lambda *args: pytest.fail("unexpected warmup")
        )


def test_warmup_errors_propagate_without_capturing_or_retrying(monkeypatch):
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="metax")
    )
    worker = make_worker()
    error = RuntimeError("kernel preparation failed")

    def warmup(*args):
        raise error

    with pytest.raises(RuntimeError) as caught:
        prepare_v41_graph_kernels(worker, warmup)
    assert (
        caught.value is error
        and not worker.model_runner.cudagraph_manager._graphs_captured
    )
