# SPDX-License-Identifier: Apache-2.0
"""GLM5 model-policy registration tests (C3 runtime plan registry)."""

from types import SimpleNamespace

import pytest

from vllm_fl.activation import reset_activation_for_tests
from vllm_fl.dispatch.policy import SelectionPolicy
from vllm_fl.kernels.glm5_next import provider
from vllm_fl.patches import glm5_next_runtime as glm_patch
from vllm_fl.runtime.model_policy import (
    build_model_runtime_plan,
    reset_model_policy_for_tests,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_activation_for_tests()
    reset_model_policy_for_tests()
    provider._has_nvidia_reference_kernels.cache_clear()
    yield
    reset_activation_for_tests()
    reset_model_policy_for_tests()
    provider._has_nvidia_reference_kernels.cache_clear()


def _glm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm5_next_text", architectures=["Glm5NextForCausalLM"]
            ),
            hf_text_config=SimpleNamespace(model_type="glm5_next_text"),
            architectures=["Glm5NextForCausalLM"],
        )
    )


def _other_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_8_flash_next", architectures=["Qwen3Model"]
            ),
            hf_text_config=SimpleNamespace(model_type="qwen3_8_flash_next"),
            architectures=["Qwen3Model"],
        )
    )


def test_glm_registration_binds_runtime_policy_factory(monkeypatch):
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: False)
    provider._has_nvidia_reference_kernels.cache_clear()

    assert glm_patch.register_glm5_next_support() is True

    plan = build_model_runtime_plan(_glm_config(), None, SelectionPolicy())
    assert plan.attention_backend is None
    assert plan.native_aten_ops == frozenset()
    assert plan.selection_policy.get_per_op_order("moe_align_block_size") == [
        "flagos",
        "reference",
    ]


def test_glm_runtime_plan_preserves_explicit_user_order(monkeypatch):
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: False)
    provider._has_nvidia_reference_kernels.cache_clear()
    assert glm_patch.register_glm5_next_support() is True

    user_policy = SelectionPolicy.from_dict(
        prefer="flagos",
        per_op_order={"moe_align_block_size": ["flagos"]},
    )
    plan = build_model_runtime_plan(_glm_config(), None, user_policy)
    assert plan.selection_policy.get_per_op_order("moe_align_block_size") == ["flagos"]


def test_glm_runtime_plan_ignores_other_models(monkeypatch):
    monkeypatch.setattr(provider, "_has_vllm_native_extension", lambda: False)
    provider._has_nvidia_reference_kernels.cache_clear()
    assert glm_patch.register_glm5_next_support() is True

    policy = SelectionPolicy.from_dict(prefer="reference")
    plan = build_model_runtime_plan(_other_config(), None, policy)
    assert plan.selection_policy is policy
    assert plan.attention_backend is None


@pytest.mark.parametrize("sparse", [False, True])
def test_attention_override_tracks_public_policy_and_native_capability(
    monkeypatch, sparse
):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.policy import policy_context

    available_checks = []
    native_capability = [True]
    monkeypatch.setattr(glm_patch, "use_nvidia_reference", lambda: native_capability[0])
    monkeypatch.setattr(
        FlagGemsBackend, "is_available", lambda self: available_checks.append(1) or True
    )
    op = "flash_mla_sparse_fwd" if sparse else "attention"
    vendor = SelectionPolicy.from_dict(prefer="vendor")
    flagos = SelectionPolicy.from_dict(prefer="vendor", per_op_order={op: ["flagos"]})
    denied = SelectionPolicy.from_dict(prefer="vendor", deny_vendors={"cuda"})
    expected = "FlagGemsSparseMLABackend" if sparse else "MLAFLBackend"
    # Begin with a valid native selection, then change only one public condition.
    for policy, native, uses_library in (
        (vendor, True, False),
        (flagos, True, True),
        (vendor, True, False),
        (denied, True, True),
        (vendor, False, True),
    ):
        native_capability[0] = native
        before = len(available_checks)
        with policy_context(policy):
            result = glm_patch._glm5_attention_override(True, sparse)
        if uses_library:
            assert result.endswith(expected)
            assert len(available_checks) == before + 1
        else:
            assert result is None
            assert len(available_checks) == before
    # Non-MLA must bypass model-specific capability/dependency work entirely.
    before = len(available_checks)
    with policy_context(flagos):
        assert glm_patch._glm5_attention_override(False, sparse) is None
    assert len(available_checks) == before
