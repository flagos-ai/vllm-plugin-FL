# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the model-scoped runtime policy registry."""

import os
from types import SimpleNamespace

import pytest

from vllm_fl.dispatch.policy import SelectionPolicy
from vllm_fl.runtime import model_policy
from vllm_fl.runtime.model_policy import (
    ModelPolicyError,
    ModelPolicyFactory,
    RuntimePlan,
    activate_runtime_plan,
    build_model_runtime_plan,
    get_active_runtime_plan,
    register_model_policy_factory,
    reset_model_policy_for_tests,
    validate_model_config,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_model_policy_for_tests()
    yield
    reset_model_policy_for_tests()


def _config(architectures=(), model_type=None, text_config=None, scheduler=None):
    hf_text_config = text_config
    if hf_text_config is None:
        hf_text_config = SimpleNamespace(
            model_type=model_type,
            architectures=list(architectures),
        )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=list(architectures),
            hf_config=SimpleNamespace(
                model_type=model_type, architectures=list(architectures)
            ),
            hf_text_config=hf_text_config,
        ),
        scheduler_config=scheduler,
    )


def _factory(**overrides):
    values = {
        "build": lambda cfg, caps, policy: RuntimePlan(
            selection_policy=policy,
            attention_backend="vllm_fl.attention.ModelBackend",
            native_aten_ops=frozenset({"aten::mm"}),
        ),
        "architectures": ("ModelForCausalLM",),
        "model_types": ("model_type",),
        "name": "test-model",
    }
    values.update(overrides)
    return ModelPolicyFactory(**values)


def test_factory_selected_by_architecture():
    register_model_policy_factory(_factory())
    plan = build_model_runtime_plan(
        _config(architectures=["ModelForCausalLM"]), None, SelectionPolicy()
    )
    assert plan.attention_backend == "vllm_fl.attention.ModelBackend"
    assert plan.native_aten_ops == frozenset({"aten::mm"})


def test_factory_selected_by_model_type_when_architecture_absent():
    register_model_policy_factory(
        _factory(architectures=(), model_types=("model_type",))
    )
    plan = build_model_runtime_plan(_config(model_type="model_type"), None, None)
    assert isinstance(plan, RuntimePlan)
    assert plan.attention_backend == "vllm_fl.attention.ModelBackend"


def test_architecture_match_wins_over_model_type():
    register_model_policy_factory(
        _factory(
            architectures=("Other",),
            model_types=("model_type",),
            build=lambda cfg, caps, policy: RuntimePlan(
                selection_policy=policy, attention_backend="BY_TYPE"
            ),
        )
    )
    register_model_policy_factory(
        _factory(
            architectures=("ModelForCausalLM",),
            model_types=(),
            build=lambda cfg, caps, policy: RuntimePlan(
                selection_policy=policy, attention_backend="BY_ARCH"
            ),
        )
    )
    plan = build_model_runtime_plan(
        _config(architectures=["ModelForCausalLM"], model_type="model_type"),
        None,
        SelectionPolicy(),
    )
    assert plan.attention_backend == "BY_ARCH"


def test_unknown_model_returns_default_plan():
    register_model_policy_factory(_factory())
    policy = SelectionPolicy.from_dict(prefer="reference")
    plan = build_model_runtime_plan(
        _config(architectures=["Unknown"], model_type="unknown"), None, policy
    )
    assert plan.selection_policy is policy
    assert plan.attention_backend is None
    assert plan.native_aten_ops == frozenset()


def test_build_plan_is_pure_and_does_not_touch_env(monkeypatch):
    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    register_model_policy_factory(_factory())
    env_before = dict(os.environ)
    config = _config(architectures=["ModelForCausalLM"])

    first = build_model_runtime_plan(config, None, SelectionPolicy())
    second = build_model_runtime_plan(config, None, SelectionPolicy())

    assert first == second
    assert dict(os.environ) == env_before


def test_validate_model_config_dispatches_factory_validation():
    calls: list[object] = []
    register_model_policy_factory(_factory(validate=lambda cfg: calls.append(cfg)))
    config = _config(architectures=["ModelForCausalLM"])
    validate_model_config(config)
    assert calls == [config]


def _ple_config(async_scheduling):
    text_config = SimpleNamespace(model_type="ple_model", ple_layer_ids=[1, 2, 3])
    scheduler = SimpleNamespace(async_scheduling=async_scheduling)
    return _config(
        architectures=["PleForCausalLM"],
        text_config=text_config,
        scheduler=scheduler,
    )


def _register_cpu_history_factory(cpu_history=True):
    register_model_policy_factory(
        _factory(
            architectures=("PleForCausalLM",),
            model_types=("ple_model",),
            cpu_token_history=lambda cfg: cpu_history,
            name="PLE",
        )
    )


def test_final_async_guard_raises_on_resolved_true():
    _register_cpu_history_factory()
    with pytest.raises(ModelPolicyError, match="--no-async-scheduling"):
        validate_model_config(_ple_config(async_scheduling=True))


def test_final_async_guard_allows_none_and_false():
    _register_cpu_history_factory()
    validate_model_config(_ple_config(async_scheduling=None))
    validate_model_config(_ple_config(async_scheduling=False))


def test_final_async_guard_skips_without_ple_layer_ids():
    _register_cpu_history_factory()
    config = _ple_config(async_scheduling=True)
    config.model_config.hf_text_config.ple_layer_ids = []
    validate_model_config(config)


def test_final_async_guard_skips_without_cpu_token_history():
    _register_cpu_history_factory(cpu_history=False)
    validate_model_config(_ple_config(async_scheduling=True))


def test_final_async_guard_skips_unknown_model():
    validate_model_config(_ple_config(async_scheduling=True))


def test_runtime_activation_is_idempotent_for_equal_plans():
    original = RuntimePlan(SelectionPolicy())
    assert activate_runtime_plan(original) is None
    assert activate_runtime_plan(RuntimePlan(SelectionPolicy())) is original
    assert get_active_runtime_plan() is original


@pytest.mark.parametrize(
    "change",
    [
        {"selection_policy": SelectionPolicy.from_dict(prefer="reference")},
        {"attention_backend": "another.Backend"},
        {"native_aten_ops": frozenset({"aten::mm"})},
    ],
)
def test_runtime_activation_rejects_conflicts_without_replacing_plan(change):
    original = RuntimePlan(SelectionPolicy())
    activate_runtime_plan(original)
    values = {"selection_policy": SelectionPolicy(), **change}
    with pytest.raises(ModelPolicyError, match="already active"):
        activate_runtime_plan(RuntimePlan(**values))
    assert get_active_runtime_plan() is original


def test_runtime_preflight_does_not_publish_and_checks_existing_plan():
    plan = RuntimePlan(SelectionPolicy())
    model_policy.preflight_runtime_plan(plan)
    assert get_active_runtime_plan() is None
    activate_runtime_plan(plan)
    with pytest.raises(ModelPolicyError, match="already active"):
        model_policy.preflight_runtime_plan(
            RuntimePlan(SelectionPolicy(), attention_backend="another.Backend")
        )
    assert get_active_runtime_plan() is plan


@pytest.fixture
def worker_startup(monkeypatch):
    """Exercise the actual worker startup boundary without loading a model."""
    from vllm_fl import activation, dispatch
    from vllm_fl.attention import utils as attention_utils
    from vllm_fl.dispatch.policy import PolicyManager
    from vllm_fl.dispatch.types import BackendImplKind
    from vllm_fl.worker import worker

    base_policy = SelectionPolicy.from_dict(per_op_order={"op": ["flagos"]})
    manager = PolicyManager()
    monkeypatch.setattr(PolicyManager, "_instance", manager)
    monkeypatch.setattr(manager, "policy_for_plan", lambda _: base_policy)
    monkeypatch.setattr(worker.fl_envs, "USE_FLAGGEMS", False)
    monkeypatch.setattr(worker, "register_oot_ops", lambda: None)
    monkeypatch.setattr(worker, "_probe_device_capability", lambda: None)
    monkeypatch.setattr(attention_utils, "patch_mm_encoder_attention", lambda: None)
    monkeypatch.setattr(
        worker.WorkerBase,
        "__init__",
        lambda self, vllm_config, **kwargs: setattr(self, "vllm_config", vllm_config),
    )
    applied = []
    plan = activation.ActivationPlan(
        name="test",
        fingerprint="test@1",
        apply=lambda: None,
        moe_defaults=activation.MoEDispatchDefaults(
            per_op_order=(("op", ("flagos",)),),
            required_impls=(("op", ("flagos",)),),
        ),
    )
    monkeypatch.setattr(activation, "preflight_activation_config", lambda *a: plan)
    monkeypatch.setattr(
        activation, "activate_for_model", lambda cfg: applied.append(cfg)
    )

    def resolve(op):
        policy = manager.get_policy()
        vendor = policy.prefer == "vendor"
        return [
            SimpleNamespace(
                kind=BackendImplKind.VENDOR if vendor else BackendImplKind.DEFAULT,
                vendor="cuda" if vendor else None,
                impl_id="vendor.cuda" if vendor else "default.flagos",
            )
        ]

    monkeypatch.setattr(
        dispatch,
        "get_default_manager",
        lambda: SimpleNamespace(resolve_candidates=resolve),
    )
    config = _config(architectures=["ModelForCausalLM"])
    config.num_speculative_tokens = 0
    config.profiler_config = SimpleNamespace(profiler=None)
    return lambda: worker.WorkerFL(config, 0, 0, "unused"), applied, base_policy


@pytest.mark.parametrize(
    "policy_change",
    [
        {"per_op_order": {"op": ["vendor.cuda"]}},
        {"prefer": "vendor"},
    ],
)
def test_worker_validates_factory_policy_before_activation(
    worker_startup, policy_change
):
    from vllm_fl.activation import ActivationConflict

    start, applied, _ = worker_startup
    policy = SelectionPolicy.from_dict(**policy_change)
    register_model_policy_factory(_factory(build=lambda *a: RuntimePlan(policy)))
    with pytest.raises(ActivationConflict):
        start()
    assert applied == []
    assert get_active_runtime_plan() is None


def test_worker_rejects_runtime_conflict_before_applying_model_patches(worker_startup):
    start, applied, _ = worker_startup
    original = RuntimePlan(SelectionPolicy(), attention_backend="existing.Backend")
    activate_runtime_plan(original)
    with pytest.raises(ModelPolicyError, match="already active"):
        start()
    assert applied == []
    assert get_active_runtime_plan() is original


@pytest.mark.parametrize("use_mla", [False, True])
def test_static_runtime_backend_is_not_limited_to_mla(monkeypatch, use_mla):
    from vllm_fl import dispatch
    from vllm_fl.platform import PlatformFL

    activate_runtime_plan(
        RuntimePlan(SelectionPolicy(), attention_backend="model.StaticBackend")
    )
    monkeypatch.setattr(dispatch, "call_op", lambda *a, **kw: "generic.Backend")
    selector = SimpleNamespace(use_mla=use_mla, use_sparse=False)
    assert PlatformFL.get_attn_backend_cls(None, selector) == "model.StaticBackend"
