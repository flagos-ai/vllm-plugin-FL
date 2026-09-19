# SPDX-License-Identifier: Apache-2.0
"""GLM5 model-policy registration tests (C3 runtime plan registry)."""

from types import SimpleNamespace

import pytest

from vllm_fl.activation import reset_activation_for_tests
from vllm_fl.dispatch.policy import SelectionPolicy
from vllm_fl.kernels.glm5_next import provider
from vllm_fl.patches import glm5_next_v024 as glm_patch
from vllm_fl.runtime.model_policy import (
    build_model_runtime_plan,
    reset_model_policy_for_tests,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_activation_for_tests()
    reset_model_policy_for_tests()
    provider.get_glm5_provider.cache_clear()
    yield
    reset_activation_for_tests()
    reset_model_policy_for_tests()
    provider.get_glm5_provider.cache_clear()


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
    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    provider.get_glm5_provider.cache_clear()

    assert glm_patch.apply_glm5_next_v024_patches() is True

    plan = build_model_runtime_plan(_glm_config(), None, SelectionPolicy())
    assert plan.attention_backend is None
    assert plan.native_aten_ops == frozenset()
    assert plan.selection_policy.get_per_op_order("moe_align_block_size") == [
        "flagos",
        "reference",
    ]


def test_glm_runtime_plan_preserves_explicit_user_order(monkeypatch):
    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    provider.get_glm5_provider.cache_clear()
    assert glm_patch.apply_glm5_next_v024_patches() is True

    user_policy = SelectionPolicy.from_dict(
        prefer="flagos",
        per_op_order={"moe_align_block_size": ["flagos"]},
    )
    plan = build_model_runtime_plan(_glm_config(), None, user_policy)
    assert plan.selection_policy.get_per_op_order("moe_align_block_size") == ["flagos"]


def test_glm_runtime_plan_ignores_other_models(monkeypatch):
    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    provider.get_glm5_provider.cache_clear()
    assert glm_patch.apply_glm5_next_v024_patches() is True

    policy = SelectionPolicy.from_dict(prefer="reference")
    plan = build_model_runtime_plan(_other_config(), None, policy)
    assert plan.selection_policy is policy
    assert plan.attention_backend is None
