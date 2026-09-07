# SPDX-License-Identifier: Apache-2.0
"""Static contracts for GLM5-Next breakable CUDA-graph boundaries."""

import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


def _method_calls(path: Path, class_name: str, method_name: str) -> set[str]:
    module = ast.parse(path.read_text())
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return {
        ast.unparse(node.func)
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
    }


def test_kda_forward_calls_breakable_body_directly() -> None:
    calls = _method_calls(
        ROOT / "vllm_fl/models/glm5_next.py",
        "Glm5NextLinearAttention",
        "forward",
    )
    assert "self._forward" in calls
    assert "super().forward" not in calls
    assert not any(call.startswith("torch.ops.vllm.kda_attention") for call in calls)


def test_flagos_cuda_runner_honors_breakable_graph() -> None:
    source = (ROOT / "vllm_fl/worker/model_runner.py").read_text()
    assert "is_breakable_cudagraph_enabled()" in source
    assert "BreakableCUDAGraphWrapper(self.model, self.vllm_config)" in source


def test_kpool_custom_op_is_a_piecewise_split() -> None:
    patch_source = (ROOT / "vllm_fl/patches/glm5_next_v024.py").read_text()
    assert '"vllm::sparse_attn_indexer_kpool"' in patch_source


def test_indexer_translated_block_table_keeps_a_stable_base_address() -> None:
    patch_source = (ROOT / "vllm_fl/patches/glm5_next_kpool_v024.py").read_text()
    assert "scheduler_config.max_num_batched_tokens" in patch_source
    assert "block_table_tensor=translated_table" in patch_source
    assert "buffer.shape != compressed.shape" not in patch_source


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
