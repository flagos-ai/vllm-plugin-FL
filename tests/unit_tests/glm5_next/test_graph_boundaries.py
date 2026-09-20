# SPDX-License-Identifier: Apache-2.0
"""Registration and config behavior for GLM5-Next graph boundaries."""

from types import SimpleNamespace


def test_kpool_custom_op_is_a_piecewise_split() -> None:
    from vllm.config.compilation import CompilationConfig

    from vllm_fl.activation import patch_inventory
    from vllm_fl.patches.glm5_next_v024 import apply_glm5_next_v024_patches

    apply_glm5_next_v024_patches()
    assert "vllm::sparse_attn_indexer_kpool" in CompilationConfig._attention_ops
    assert any(
        p["target"].endswith("CompilationConfig._attention_ops")
        and p["phase"] == "engine/config"
        for p in patch_inventory()
    )


def test_graph_config_disables_glm5_allreduce_fusion(monkeypatch) -> None:
    """Graph mode must not inherit the generic H100 O2 allreduce fusion."""
    from vllm.model_executor.models.config import HybridAttentionMambaModelConfig

    from vllm_fl.patches.glm5_next_v024 import Glm5NextForCausalLMConfig

    # Keep this a focused config test; the common hybrid checks are covered by
    # vLLM and are unrelated to the GLM5 graph safety override.
    monkeypatch.setattr(
        HybridAttentionMambaModelConfig,
        "verify_and_update_config",
        classmethod(lambda cls, _config: None),
    )

    def make_config(*, enforce_eager: bool, fuse_allreduce_rms):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                enforce_eager=enforce_eager,
                hf_text_config=SimpleNamespace(index_kpool_compress=False),
            ),
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(
                    fuse_allreduce_rms=fuse_allreduce_rms,
                )
            ),
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        )

    graph_config = make_config(enforce_eager=False, fuse_allreduce_rms=True)
    Glm5NextForCausalLMConfig.verify_and_update_config(graph_config)
    assert graph_config.compilation_config.pass_config.fuse_allreduce_rms is False

    eager_config = make_config(enforce_eager=True, fuse_allreduce_rms=True)
    Glm5NextForCausalLMConfig.verify_and_update_config(eager_config)
    assert eager_config.compilation_config.pass_config.fuse_allreduce_rms is True
