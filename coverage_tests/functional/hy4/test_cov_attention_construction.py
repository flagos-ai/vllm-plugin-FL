# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construction contracts of the HY4 lightning indexer and MLA attention.

``Indexer`` is built through a FlagGems runtime plan on a one-rank TP
group: the fused ``wk_weights_proj`` GEMM shapes, the FP8 k-cache row
width, the prefill buffer sizing and both module-level layer registries
are checked against independently computed expectations.
``HYV4MLAAttention`` is built on dense layers and the full attention
stack (gating GEMM, projections, MLA impl wiring) is verified.
"""

import pytest

from ..._distributed import single_rank_gloo
from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from ._construction_helpers import (
    HID,
    KV_LORA,
    N_HEADS,
    Q_LORA,
    QK_NOPE,
    QK_ROPE,
    V_HEAD,
    runtime_plan,
    set_current_vllm_config,
    snapshot_layer_registries,
    vllm_namespace,
)
from vllm_fl.attention.sparse_indexer import (
    _LAYER_OPERATIONS,
    PortableSparseAttnIndexer,
)
from vllm_fl.models.hy4.attention import HYV4MLAAttention, Indexer
from vllm_fl.models.hy4.config import HYV4Config

pytestmark = [pytest.mark.gpu]


@pytest.fixture()
def single_rank_tp():
    """Start a fresh one-rank gloo TP group; restore the previous one."""
    with single_rank_gloo("tp"):
        yield


@pytest.fixture()
def restored_layer_registries():
    """Restore the portable indexer's global layer registry after use."""
    teardown = snapshot_layer_registries()
    yield
    teardown()


def _hf_config(**overrides):
    """A validating HYV4Config with the production MLA head layout."""
    base = dict(
        vocab_size=32,
        hidden_size=HID,
        num_hidden_layers=2,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        qk_head_dim=QK_NOPE + QK_ROPE,
        v_head_dim=V_HEAD,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        hc_mult=4,
        n_routed_experts=2,
        max_position_embeddings=64,
    )
    base.update(overrides)
    return HYV4Config(**base)


class TestIndexerConstruction:
    """Indexer built through the FlagGems plan on a single rank."""

    PREFIX = "model.layers.0.self_attn.indexer"

    def _build(self, cfg, hf, prefix, plan):
        return Indexer(
            cfg,
            hf,
            hidden_size=HID,
            q_lora_rank=Q_LORA,
            quant_config=None,
            cache_config=cfg.cache_config,
            topk_indices_buffer=None,
            prefix=prefix,
            runtime_plan=plan,
        )

    def test_flaggems_plan_builds_portable_indexer(
        self, single_rank_tp, restored_layer_registries
    ):
        """FlagGems plan builds portable indexer."""
        hf = _hf_config(
            layer_types=["deepseek_sparse_attention", "dense"],
            indexer_types=["full", "full"],
        )
        cfg = vllm_namespace(hf)
        plan = runtime_plan()

        with set_current_vllm_config(cfg):
            indexer = self._build(cfg, hf, self.PREFIX, plan)

        # Portable (FlagGems) indexer selected, not vLLM's native CustomOp.
        assert isinstance(indexer.indexer_op, PortableSparseAttnIndexer)
        assert indexer.indexer_op.k_cache is indexer.k_cache
        # Defaults: index_head_dim=128, index_n_heads=32, quant_block=128.
        assert indexer.topk_tokens == hf.index_topk
        assert indexer.head_dim == 128
        assert indexer.n_head == 32
        # The fused wk+weights GEMM produces [head_dim, n_head] in one shot
        # (TP disabled: the indexer cache is replicated).
        assert list(indexer.wk_weights_proj.output_sizes) == [128, 32]
        assert indexer.wq_b.output_size == 128 * 32
        assert indexer.k_norm.weight.shape == (128,)
        assert indexer.softmax_scale == 128**-0.5
        assert indexer.rope_dim == QK_ROPE
        assert indexer.scale_fmt == "ue8m0"
        assert indexer.quant_block_size == 128
        # k_cache row width: 128 fp8 values + one fp32 scale per block of
        # 128 values.
        assert indexer.k_cache.head_dim == 128 + 128 // 128 * 4
        # The plugin delegates sizing to the installed vLLM indexer policy.
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        assert indexer.max_total_seq_len == get_max_prefill_buffer_size(cfg)
        # Both global layer registries hold this layer after construction.
        key = f"{self.PREFIX}.k_cache"
        assert cfg.compilation_config.static_forward_context[key] is (indexer.k_cache)
        assert _LAYER_OPERATIONS[key] is plan.operations

    def test_duplicate_layer_prefix_is_rejected(
        self, single_rank_tp, restored_layer_registries
    ):
        # DeepseekV32IndexerCache refuses a duplicate name in the static
        # forward context (the registry the compiler resolves layers by).
        """Duplicate layer prefix is rejected."""
        hf = _hf_config(layer_types=["deepseek_sparse_attention", "dense"])
        cfg = vllm_namespace(hf)
        with set_current_vllm_config(cfg):
            first = self._build(cfg, hf, self.PREFIX, runtime_plan())
            with pytest.raises(ValueError, match="Duplicate layer name"):
                self._build(cfg, hf, self.PREFIX, runtime_plan())
        assert first.k_cache is not None


class TestDenseAttentionConstruction:
    """HYV4MLAAttention on dense layers (non-sparse layer_types)."""

    @pytest.fixture
    def dense_backend(self, monkeypatch):
        from vllm.model_executor.layers.attention import mla_attention
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        backend = AttentionBackendEnum.TRITON_MLA.get_class()

        def select_dense_mla(
            head_size, dtype, kv_cache_dtype, *, use_mla, use_sparse, num_heads
        ):
            assert use_mla and not use_sparse
            assert head_size == KV_LORA + QK_ROPE
            assert num_heads == N_HEADS and kv_cache_dtype == "auto"
            return backend

        monkeypatch.setattr(mla_attention, "get_attn_backend", select_dense_mla)
        return backend

    def _build(self, cfg, hf, prefix="model.layers.1.self_attn"):
        return HYV4MLAAttention(
            vllm_config=cfg,
            config=hf,
            hidden_size=HID,
            num_heads=N_HEADS,
            qk_nope_head_dim=QK_NOPE,
            qk_rope_head_dim=QK_ROPE,
            v_head_dim=V_HEAD,
            q_lora_rank=Q_LORA,
            kv_lora_rank=KV_LORA,
            max_position_embeddings=64,
            cache_config=cfg.cache_config,
            quant_config=None,
            prefix=prefix,
            topk_indices_buffer=None,
            layer_idx=1,
            runtime_plan=runtime_plan(),
        )

    def test_dense_layer_builds_full_attention_stack(
        self, single_rank_tp, restored_layer_registries, dense_backend
    ):
        # Layer 1 is dense: no sparse probe, no indexer, no sink resolution.
        """Dense layer builds full attention stack."""
        hf = _hf_config(
            layer_types=["deepseek_sparse_attention", "dense"],
            learnable_sink=False,
        )
        cfg = vllm_namespace(hf)
        with set_current_vllm_config(cfg):
            attn = self._build(cfg, hf)

        assert attn.is_sparse is False
        assert attn.skip_topk is False
        assert attn.create_indexer is False
        assert attn.indexer is None
        assert attn.indexer_rope_emb is None
        assert attn.learnable_sink is False
        assert not hasattr(attn, "learnable_sink_param")
        assert attn.num_local_heads == N_HEADS
        assert attn.qk_head_dim == QK_NOPE + QK_ROPE
        assert attn.scaling == (QK_NOPE + QK_ROPE) ** -0.5
        # Gated MLA (elementwise default) builds the per-head gate GEMM.
        assert attn.gated_mla is True
        assert list(attn.linear_gate.output_sizes) == [N_HEADS * V_HEAD]
        # MLA core projections exist.
        assert attn.q_a_proj is not None
        assert attn.kv_a_proj_with_mqa is not None
        assert attn.q_b_proj is not None
        assert attn.kv_b_proj is not None
        assert attn.o_proj is not None
        # The MLA impl wires our indexer (None here) and kv_b_proj through.
        assert attn.mla_attn.kv_b_proj is attn.kv_b_proj
        assert attn.mla_attn.indexer is None
        assert attn.mla_attn.num_heads == N_HEADS
        assert attn.mla_attn.attn_backend is dense_backend
        # The attention layer registers itself in the static forward
        # context under its own prefix.
        key = "model.layers.1.self_attn.attn"
        assert cfg.compilation_config.static_forward_context[key] is (attn.mla_attn)

    def test_headwise_gating_builds_one_lane_per_head(
        self, single_rank_tp, restored_layer_registries, dense_backend
    ):
        """Headwise gating builds one lane per-head."""
        hf = _hf_config(
            layer_types=["dense", "dense"],
            learnable_sink=False,
            gating_type="headwise",
        )
        cfg = vllm_namespace(hf)
        with set_current_vllm_config(cfg):
            attn = self._build(cfg, hf)
        assert attn.gate_projection_size_per_head == 1
        assert list(attn.linear_gate.output_sizes) == [N_HEADS]

    def test_unknown_gating_type_is_rejected(
        self, single_rank_tp, restored_layer_registries
    ):
        """Unknown gating type is rejected."""
        hf = _hf_config(
            layer_types=["dense", "dense"],
            learnable_sink=False,
            gating_type="bogus",
        )
        cfg = vllm_namespace(hf)
        with (
            set_current_vllm_config(cfg),
            pytest.raises(ValueError, match="Unknown gating type"),
        ):
            self._build(cfg, hf)
