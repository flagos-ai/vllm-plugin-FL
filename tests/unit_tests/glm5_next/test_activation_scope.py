# SPDX-License-Identifier: Apache-2.0
"""Scope tests: GLM registration must not rewrite process-global state, and the
GLM provider environment must not change the generic path of other models."""

import os
from types import SimpleNamespace

import pytest

from vllm_fl.activation import (
    ActivationConflict,
    activate,
    reset_activation_for_tests,
)
from vllm_fl.kernels.glm5_next import provider
from vllm_fl.patches import glm5_next_v024 as glm_patch


@pytest.fixture(autouse=True)
def _reset():
    reset_activation_for_tests()
    provider.get_glm5_provider.cache_clear()
    yield
    reset_activation_for_tests()
    provider.get_glm5_provider.cache_clear()


def test_registration_does_not_mutate_env_or_shared_classes(monkeypatch):
    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "grouped_topk,moe_sum")
    monkeypatch.setenv(
        "VLLM_FL_PER_OP", "moe_align_block_size=vendor.cuda;fused_moe=reference"
    )
    provider.get_glm5_provider.cache_clear()

    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    dispatch_keys = ("VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_PER_OP")
    env_before = {k: os.environ.get(k) for k in dispatch_keys}
    indexer_before = DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
        DeepseekV32IndexerBackend
    )
    mhc_before = None
    try:
        from vllm.model_executor.layers.mhc import MHCPreOp

        mhc_before = MHCPreOp.forward_oot
    except Exception:
        MHCPreOp = None

    assert glm_patch.apply_glm5_next_v024_patches() is True

    assert {k: os.environ.get(k) for k in dispatch_keys} == env_before
    assert (
        DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
            DeepseekV32IndexerBackend
        )
        is indexer_before
    )
    if MHCPreOp is not None:
        assert MHCPreOp.forward_oot is mhc_before


def test_is_glm5_model_matcher():
    glm_text = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="glm5_next_text"),
            hf_text_config=SimpleNamespace(model_type="glm5_next_text"),
        )
    )
    glm_arch = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Glm5NextForCausalLM"]),
            hf_text_config=None,
            architectures=None,
        )
    )
    other = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_8_flash_next", architectures=["Qwen3Model"]
            ),
            hf_text_config=SimpleNamespace(model_type="qwen3_8_flash_next"),
            architectures=["Qwen3Model"],
        )
    )
    assert glm_patch._is_glm5_model(glm_text)
    assert glm_patch._is_glm5_model(glm_arch)
    assert not glm_patch._is_glm5_model(other)
    assert not glm_patch._is_glm5_model(SimpleNamespace())


def test_plan_provider_returns_none_for_non_glm():
    other = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_8_flash_next", architectures=["Qwen3Model"]
            ),
            hf_text_config=SimpleNamespace(model_type="qwen3_8_flash_next"),
        )
    )
    assert glm_patch._glm5_plan_provider(other) is None


def test_platform_ignores_glm_provider_without_active_plan(monkeypatch):
    """A GLM provider env alone must not change a non-GLM MLA model's path.

    The generic dispatch result (``call_op``) must be reached verbatim when no
    plan is active, even with ``VLLM_FL_GLM5_PROVIDER=flaggems`` set.
    """
    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    provider.get_glm5_provider.cache_clear()
    reset_activation_for_tests()

    import vllm_fl.dispatch as dispatch

    monkeypatch.setattr(
        dispatch, "call_op", lambda op, **kwargs: "GENERIC_DISPATCH_SENTINEL"
    )
    from vllm_fl.platform import PlatformFL

    selector = SimpleNamespace(use_mla=True, use_sparse=False)
    assert (
        PlatformFL.get_attn_backend_cls(None, selector) == "GENERIC_DISPATCH_SENTINEL"
    )


def test_portable_moe_defaults_expose_required_impls(monkeypatch):
    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    provider.get_glm5_provider.cache_clear()

    defaults = glm_patch.glm5_portable_moe_defaults()
    assert "moe_align_block_size" in defaults.whitelist_ops
    assert "invoke_fused_moe_triton_kernel" in defaults.whitelist_ops
    assert defaults.required_for("moe_align_block_size") == ("flagos", "reference")


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


def _protect_patched_state(monkeypatch):
    """Restore any class/module attributes the real activation patches."""
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    monkeypatch.setattr(
        DeepseekV32IndexerBackend,
        "indexes_kv_by_block_stride",
        DeepseekV32IndexerBackend.indexes_kv_by_block_stride,
    )
    from vllm.model_executor.layers.mhc import (
        MHCFusedPostPreOp,
        MHCPostOp,
        MHCPreOp,
    )

    for mhc_cls in (MHCPreOp, MHCPostOp, MHCFusedPostPreOp):
        monkeypatch.setattr(mhc_cls, "forward_oot", mhc_cls.forward_oot)
    try:
        from vllm.model_executor.layers.activation import SiluAndMulWithClamp

        monkeypatch.setattr(
            SiluAndMulWithClamp, "forward_oot", SiluAndMulWithClamp.forward_oot
        )
    except Exception:
        pass
    from vllm import _custom_ops

    for name in ("concat_mla_q", "concat_and_cache_mla"):
        if hasattr(_custom_ops, name):
            monkeypatch.setattr(_custom_ops, name, getattr(_custom_ops, name))
    from vllm.v1.attention.backends.mla import indexer as indexer_backend
    from vllm.v1.worker import utils as worker_utils

    monkeypatch.setattr(
        worker_utils.AttentionGroup,
        "create_metadata_builders",
        worker_utils.AttentionGroup.create_metadata_builders,
    )
    monkeypatch.setattr(
        indexer_backend.DeepseekV32IndexerMetadataBuilder,
        "build",
        indexer_backend.DeepseekV32IndexerMetadataBuilder.build,
    )
    monkeypatch.setattr(
        worker_utils.KVBlockZeroer, "__init__", worker_utils.KVBlockZeroer.__init__
    )


def test_real_glm_activation_applies_before_model_construction(monkeypatch):
    _protect_patched_state(monkeypatch)
    provider.get_glm5_provider.cache_clear()

    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    plan = glm_patch._glm5_plan_provider(_glm_config())
    assert plan is not None

    assert activate(plan) is True
    from vllm_fl.models.glm5_next_kpool import Glm5NextIndexerAttentionBackend

    assert Glm5NextIndexerAttentionBackend.indexes_kv_by_block_stride() is True
    assert DeepseekV32IndexerBackend.indexes_kv_by_block_stride() is False
    assert activate(plan) is False

    reset_activation_for_tests()


def test_registration_leaves_worker_kpool_paths_pristine():
    """Item 1: the runner-side kpool hooks are plan-bound, not import-bound."""
    assert glm_patch.apply_glm5_next_v024_patches() is True

    from vllm_fl.patches import glm5_next_kpool_v024 as kpool

    patches = kpool.glm5_next_kpool_runtime_patches("test@1")
    assert {patch.attr for patch in patches} == {
        "create_metadata_builders",
        "build",
        "__init__",
    }
    for patch in patches:
        assert patch.current() is patch.pristine


def test_real_glm_activation_binds_kpool_runtime_patches(monkeypatch):
    _protect_patched_state(monkeypatch)
    provider.get_glm5_provider.cache_clear()

    from vllm_fl.patches import glm5_next_kpool_v024 as kpool

    plan = glm_patch._glm5_plan_provider(_glm_config())
    assert plan is not None
    assert activate(plan) is True
    for patch in kpool.glm5_next_kpool_runtime_patches(plan.fingerprint):
        assert patch.current() is not patch.pristine
    reset_activation_for_tests()


def test_foreign_prepatch_conflicts_without_partial_side_effect(monkeypatch):
    _protect_patched_state(monkeypatch)
    provider.get_glm5_provider.cache_clear()

    from vllm.model_executor.layers.mhc import MHCPreOp
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    # Simulate another plugin having patched mHC before GLM activation.
    monkeypatch.setattr(MHCPreOp, "forward_oot", lambda self, *a, **k: None)

    plan = glm_patch._glm5_plan_provider(_glm_config())
    indexer_before = DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
        DeepseekV32IndexerBackend
    )
    with pytest.raises(ActivationConflict):
        activate(plan)

    # Preflight aborted before any target was written.
    assert (
        DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
            DeepseekV32IndexerBackend
        )
        is indexer_before
    )
