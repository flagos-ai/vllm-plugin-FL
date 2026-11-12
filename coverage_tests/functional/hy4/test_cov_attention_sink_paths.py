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

"""HY4 sink selection and sparse-backend failure contracts."""

from types import SimpleNamespace

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
from vllm_fl.models.hy4 import attention as hy4_attention
from vllm_fl.models.hy4.attention import HYV4MLAAttention
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


class TestSparseBackendFailure:
    """Sparse layers surface selector failures without a dense retry."""

    def _build(self, cfg, hf, layer_idx):
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
            prefix=f"model.layers.{layer_idx}.self_attn",
            topk_indices_buffer=None,
            layer_idx=layer_idx,
            runtime_plan=runtime_plan(),
        )

    @pytest.mark.parametrize(
        ("layer_idx", "indexer_types"), [(0, ["full", "full"]), (1, ["full", "shared"])]
    )
    def test_sparse_layer_refuses_to_fall_back_to_dense(
        self,
        single_rank_tp,
        restored_layer_registries,
        monkeypatch,
        layer_idx,
        indexer_types,
    ):
        """Preserve the sparse selector error without retrying a dense backend."""
        hf = _hf_config(
            layer_types=["deepseek_sparse_attention"] * 2,
            indexer_types=indexer_types,
            learnable_sink=False,
        )
        cfg = vllm_namespace(hf)
        requests = []

        def reject_sparse_backend(**kwargs):
            requests.append(kwargs)
            raise RuntimeError("no compatible sparse MLA backend")

        monkeypatch.setattr(hy4_attention, "get_attn_backend", reject_sparse_backend)
        with (
            set_current_vllm_config(cfg),
            pytest.raises(RuntimeError, match="Refusing to fall back") as error,
        ):
            self._build(cfg, hf, layer_idx)
        assert len(requests) == 1
        assert requests[0]["use_mla"] is True and requests[0]["use_sparse"] is True
        assert str(error.value.__cause__) == "no compatible sparse MLA backend"

    def test_learnable_sink_without_sink_backend_is_rejected(
        self, single_rank_tp, restored_layer_registries
    ):
        # The dense backends on SM80 support no sinks and the sink-capable
        # FlashMLA-sparse backend needs SM90+, so construction must refuse
        # rather than half-apply the sink.
        """Learnable sink without sink backend is rejected."""
        hf = _hf_config(layer_types=["dense", "dense"], learnable_sink=True)
        cfg = vllm_namespace(hf)
        with (
            set_current_vllm_config(cfg),
            pytest.raises(RuntimeError, match="attention sink support"),
        ):
            self._build(cfg, hf, 1)


class TestForceSparseMqa:
    """_force_sparse_mqa against both AttentionConfig generations."""

    def _stub(self):
        return object.__new__(HYV4MLAAttention)

    def test_missing_switch_leaves_vllm_024_path(self):
        # Stock vLLM 0.24 has no sparse_mla_force_mqa switch; the sink
        # layer logs and returns without touching anything.
        """Missing switch leaves vllm 024 path."""
        fake = SimpleNamespace(attention_config=SimpleNamespace())
        with set_current_vllm_config(fake):
            self._stub()._force_sparse_mqa()
        assert not hasattr(fake.attention_config, "sparse_mla_force_mqa")

    def test_false_switch_is_flipped_true(self):
        # The newer feature branch starts with the switch off; the sink
        # layer flips it on so dense prefill cannot drop the sink.
        """False switch is flipped true."""
        fake = SimpleNamespace(
            attention_config=SimpleNamespace(sparse_mla_force_mqa=False),
        )
        with set_current_vllm_config(fake):
            self._stub()._force_sparse_mqa()
        assert fake.attention_config.sparse_mla_force_mqa is True

    def test_already_true_stays_true(self):
        """Already true stays true."""
        fake = SimpleNamespace(
            attention_config=SimpleNamespace(sparse_mla_force_mqa=True),
        )
        with set_current_vllm_config(fake):
            self._stub()._force_sparse_mqa()
        assert fake.attention_config.sparse_mla_force_mqa is True
