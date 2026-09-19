# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the model activation plan infrastructure."""

from types import SimpleNamespace

import pytest

from vllm_fl.activation import (
    ActivationConflict,
    ActivationPlan,
    MoEDispatchDefaults,
    PendingPatch,
    activate,
    activate_for_model,
    bind_patches,
    get_active_plan,
    install_patch,
    merge_per_op_defaults,
    preflight_activation_config,
    preflight_patches,
    register_plan_provider,
    reset_activation_for_tests,
    validate_flaggems_whitelist,
    validate_plan_capability,
)
from vllm_fl.dispatch.types import BackendImplKind


@pytest.fixture(autouse=True)
def _clean_activation():
    reset_activation_for_tests()
    yield
    reset_activation_for_tests()


def _plan(name: str, fingerprint: str, calls: list) -> ActivationPlan:
    return ActivationPlan(
        name=name,
        fingerprint=fingerprint,
        apply=lambda: calls.append(fingerprint),
    )


def _defaults() -> MoEDispatchDefaults:
    return MoEDispatchDefaults(
        whitelist_ops=("moe_align_block_size",),
        per_op_order=(("moe_align_block_size", ("flagos", "reference")),),
        required_impls=(("moe_align_block_size", ("flagos", "reference")),),
    )


def test_same_plan_activation_is_idempotent():
    calls: list[str] = []
    plan = _plan("glm", "glm@1", calls)
    assert activate(plan) is True
    assert activate(plan) is False
    assert calls == ["glm@1"]
    assert get_active_plan() is plan


def test_conflicting_plan_raises_before_side_effect():
    calls: list[str] = []
    activate(_plan("glm", "glm@1", calls))
    with pytest.raises(ActivationConflict):
        activate(_plan("qwen", "qwen@1", calls))
    assert calls == ["glm@1"]
    assert get_active_plan().fingerprint == "glm@1"


def test_activate_for_model_rejects_a_different_model():
    """An already-active plan must not silently accept another model."""
    plan = ActivationPlan(name="glm", fingerprint="glm@1", apply=lambda: None)
    register_plan_provider(lambda cfg: plan if cfg == "glm" else None)

    assert activate_for_model("glm") is plan
    with pytest.raises(ActivationConflict):
        activate_for_model("other-model-needing-no-plan")


def test_activate_for_model_same_model_is_idempotent():
    plan = ActivationPlan(name="glm", fingerprint="glm@1", apply=lambda: None)
    register_plan_provider(lambda cfg: plan)
    assert activate_for_model("glm") is plan
    assert activate_for_model("glm") is plan


def test_activate_invalidates_cached_policy(monkeypatch):
    """Finding 2: cached policy must observe the plan's defaults."""
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_FL_PER_OP", raising=False)

    from vllm_fl.dispatch.policy import PolicyManager

    manager = PolicyManager.get_instance()
    manager.reset_global_policy()
    before_epoch = manager.get_policy_epoch()
    assert manager.get_policy().get_per_op_order("moe_align_block_size") is None

    activate(
        ActivationPlan(
            name="policy",
            fingerprint="policy@1",
            apply=lambda: None,
            moe_defaults=_defaults(),
        )
    )

    assert manager.get_policy_epoch() > before_epoch
    assert manager.get_policy().get_per_op_order("moe_align_block_size") == [
        "flagos",
        "reference",
    ]


def test_install_patch_idempotent_then_conflict():
    class Target:
        attr = "original"

    def get_current():
        return Target.attr

    assert install_patch(
        "Target.attr",
        "owner-a",
        get_current=get_current,
        pristine="original",
        expected_signature="attr()",
        apply=lambda: setattr(Target, "attr", "a"),
    )
    assert not install_patch(
        "Target.attr",
        "owner-a",
        get_current=get_current,
        pristine="original",
        expected_signature="attr()",
        apply=lambda: setattr(Target, "attr", "a"),
    )
    with pytest.raises(ActivationConflict):
        install_patch(
            "Target.attr",
            "owner-b",
            get_current=get_current,
            pristine="original",
            expected_signature="attr()",
            apply=lambda: setattr(Target, "attr", "b"),
        )
    assert Target.attr == "a"


def test_install_patch_refuses_foreign_modification():
    class Target:
        attr = "original"

    Target.attr = "foreign"
    with pytest.raises(ActivationConflict):
        install_patch(
            "Target.attr",
            "owner-a",
            get_current=lambda: Target.attr,
            pristine="original",
            expected_signature="attr()",
            apply=lambda: setattr(Target, "attr", "a"),
        )
    assert Target.attr == "foreign"


def test_bind_patches_preflight_aborts_without_side_effect():
    """Finding 4: a late preflight failure must leave earlier targets pristine."""

    class A:
        x = "a"

    class B:
        y = "b"

    patches = [
        PendingPatch("A.x", A, "x", "a2", "fp", pristine="a"),
        PendingPatch("B.y", B, "y", "b2", "fp", pristine="WRONG"),
    ]
    with pytest.raises(ActivationConflict):
        bind_patches(patches)
    assert A.x == "a" and B.y == "b"


def test_bind_patches_validates_signature():
    class A:
        def f(self, x):
            return x

    pristine = A.f
    patch = PendingPatch(
        "A.f",
        A,
        "f",
        (lambda: None),
        "fp",
        pristine=pristine,
        expected_params=("self", "x"),
    )
    with pytest.raises(ActivationConflict, match="does not accept"):
        bind_patches([patch])
    assert A.f is pristine


def test_preflight_patches_skips_already_owned():
    class A:
        x = "a"

    patch = PendingPatch("A.x", A, "x", "a2", "fp", pristine="a")
    assert len(preflight_patches([patch])) == 1
    bind_patches([patch])
    assert preflight_patches([patch]) == []


def test_preflight_detects_foreign_replacement_after_owned_install():
    class Target:
        attr = object()

    patch = PendingPatch("Target.attr", Target, "attr", object(), "fp", Target.attr)
    bind_patches([patch])
    foreign = object()
    Target.attr = foreign
    with pytest.raises(ActivationConflict, match="modified"):
        bind_patches([patch])
    assert Target.attr is foreign


@pytest.mark.parametrize("same_target_name", [True, False])
def test_duplicate_patch_destination_is_rejected_before_writes(same_target_name):
    class Target:
        attr = object()

    original = Target.attr
    patches = [
        PendingPatch("Target.attr", Target, "attr", object(), "a", original),
        PendingPatch(
            "Target.attr" if same_target_name else "alias.attr",
            Target,
            "attr",
            object(),
            "b",
            original,
        ),
    ]
    with pytest.raises(ActivationConflict, match="Duplicate"):
        bind_patches(patches)
    assert Target.attr is original


def test_merge_per_op_defaults_plan_over_fallback():
    """Finding 7: plan defaults must beat an auto-detected platform fallback."""
    merged = merge_per_op_defaults(
        _defaults(),
        explicit_order=None,
        fallback_order={"moe_align_block_size": ["vendor.cuda"]},
    )
    assert merged["moe_align_block_size"] == ["flagos", "reference"]


def test_merge_per_op_defaults_fills_only_unspecified():
    merged = merge_per_op_defaults(
        _defaults(),
        explicit_order={"fused_moe": ["vendor.cuda"]},
        fallback_order={"rms_norm": ["vendor"]},
    )
    assert merged["fused_moe"] == ["vendor.cuda"]
    assert merged["rms_norm"] == ["vendor"]
    assert merged["moe_align_block_size"] == ["flagos", "reference"]


def test_merge_per_op_defaults_keeps_valid_explicit_order():
    merged = merge_per_op_defaults(
        _defaults(), explicit_order={"moe_align_block_size": ["flagos"]}
    )
    assert merged["moe_align_block_size"] == ["flagos"]


def test_merge_per_op_defaults_rejects_incompatible_primary():
    """Finding 7: a later fallback entry does not validate a bad primary."""
    with pytest.raises(ActivationConflict):
        merge_per_op_defaults(
            _defaults(),
            explicit_order={"moe_align_block_size": ["vendor.cuda", "flagos"]},
        )


def test_validate_flaggems_whitelist():
    defaults = _defaults()
    assert validate_flaggems_whitelist(None, defaults) is None
    assert validate_flaggems_whitelist(["moe_align_block_size"], defaults) == [
        "moe_align_block_size"
    ]
    with pytest.raises(ActivationConflict, match="excludes operators"):
        validate_flaggems_whitelist(["grouped_topk"], defaults)


def _ssl(kind, vendor=None, impl_id="x"):
    return SimpleNamespace(kind=kind, vendor=vendor, impl_id=impl_id)


def test_policy_for_plan_prefers_plan_over_platform_fallback(monkeypatch):
    """Finding P2-1: a platform-default order must not outrank the plan."""
    from vllm_fl.dispatch.policy import PolicyManager, SelectionPolicy

    manager = PolicyManager.get_instance()
    platform_order = {"moe_align_block_size": ["vendor.cuda", "flagos", "reference"]}
    monkeypatch.setattr(manager, "_explicit_policy", None)
    monkeypatch.setattr(
        manager,
        "_global_policy",
        SelectionPolicy.from_dict(
            prefer="flagos", strict=False, per_op_order=platform_order
        ),
    )
    monkeypatch.setattr(manager, "_env_explicit_per_op", None)
    monkeypatch.setattr(manager, "_env_fallback_per_op", platform_order)

    policy = manager.policy_for_plan(_defaults())
    assert policy.get_per_op_order("moe_align_block_size") == ["flagos", "reference"]

    # An explicit user order keeps priority and must satisfy the requirement.
    monkeypatch.setattr(
        manager,
        "_explicit_policy",
        SelectionPolicy.from_dict(
            prefer="flagos", strict=False, per_op_order=platform_order
        ),
    )
    with pytest.raises(ActivationConflict):
        manager.policy_for_plan(_defaults())


def test_validate_plan_capability_uses_real_candidates():
    """Finding 1: selectable implementations must satisfy semantics."""
    plan = ActivationPlan(
        name="p", fingerprint="p@1", apply=lambda: None, moe_defaults=_defaults()
    )
    good = _ssl(BackendImplKind.DEFAULT, impl_id="default.flagos")
    bad = _ssl(BackendImplKind.VENDOR, vendor="cuda", impl_id="vendor.cuda")

    # Plan order (flagos|reference) only reaches the compatible impl.
    validate_plan_capability(plan, lambda op: [good])
    validate_plan_capability(plan, lambda op: [good, bad])
    # Nothing compatible under the plan order -> abort.
    with pytest.raises(ActivationConflict):
        validate_plan_capability(plan, lambda op: [bad])
    # An explicit order that reaches an incompatible impl -> abort.
    with pytest.raises(ActivationConflict):
        validate_plan_capability(
            plan,
            lambda op: [good, bad],
            policy_order_for=lambda op: ["flagos", "vendor:cuda"],
        )


def test_preflight_config_conflict_happens_before_activation():
    """Finding 3: config validation must precede plan.apply()."""
    plan = ActivationPlan(
        name="p", fingerprint="p@1", apply=lambda: None, moe_defaults=_defaults()
    )
    register_plan_provider(lambda cfg: plan)

    with pytest.raises(ActivationConflict, match="excludes operators"):
        preflight_activation_config("glm", ["grouped_topk"])
    assert get_active_plan() is None


def test_plain_model_then_plan_model_is_rejected():
    """Finding 4: a plain model must occupy the process too."""
    plan = ActivationPlan(name="glm", fingerprint="glm@1", apply=lambda: None)
    register_plan_provider(lambda cfg: plan if cfg == "glm" else None)

    assert activate_for_model("plain") is None
    with pytest.raises(ActivationConflict):
        activate_for_model("glm")


def test_activate_preserves_explicit_policy(monkeypatch):
    """Finding 2: activation must not discard an explicit global policy."""
    from vllm_fl.dispatch.policy import PolicyManager, SelectionPolicy

    manager = PolicyManager.get_instance()
    manager.reset_global_policy()
    explicit = SelectionPolicy.from_dict(
        prefer="reference",
        strict=True,
        per_op_order={"moe_align_block_size": ["reference"]},
        deny_vendors={"cuda"},
    )
    manager.set_global_policy(explicit)
    try:
        activate(
            ActivationPlan(
                name="policy",
                fingerprint="policy@1",
                apply=lambda: None,
                moe_defaults=_defaults(),
            )
        )
        policy = manager.get_policy()
        assert policy.prefer == "reference"
        assert policy.strict is True
        assert "cuda" in policy.deny_vendors
        assert policy.get_per_op_order("moe_align_block_size") == ["reference"]
    finally:
        manager.reset_global_policy()


def test_constructor_patch_transaction_restores_on_failure():
    from vllm_fl.activation import patch_inventory, temporary_patches

    owner = SimpleNamespace(capability=lambda: False)
    original = owner.capability
    patch = PendingPatch(
        target="test.constructor.capability",
        owner=owner,
        attr="capability",
        replacement=lambda: True,
        pristine=original,
        fingerprint="glm-test",
        phase="construction",
    )
    with (
        pytest.raises(RuntimeError, match="constructor failed"),
        temporary_patches([patch]),
    ):
        assert owner.capability() is True
        assert any(
            p["target"] == patch.target and p["phase"] == "construction"
            for p in patch_inventory()
        )
        raise RuntimeError("constructor failed")
    assert owner.capability is original
    assert not any(p["target"] == patch.target for p in patch_inventory())
