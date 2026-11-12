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

"""QSA indexer: projection/normalization, metadata preparation and dispatch.

Projection and normalization are driven with recording stand-ins so the
documented split/reshape order is asserted directly; the metadata and forward
tests use explicit small tensors and recording kernel entries to pin the
cache-prepare ordering and fixed-width selection contract.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import indexer_qsa
from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa as qsa_ops


class _Linear:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, hidden_states):
        self.calls.append(hidden_states)
        return self.output, None


class _Norm:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.calls = []

    def __call__(self, values):
        self.calls.append(values)
        return values * self.scale


class TestQsaIndexerConstruction:
    def _install(self, monkeypatch):
        created = SimpleNamespace(raw=None, compressed=None, linear=None, norms=[])

        class FakeLinear:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                created.linear = self

        class FakeNorm:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                created.norms.append(self)

        class FakeRaw:
            def __init__(self, **kwargs):
                kwargs = dict(kwargs)
                self.kwargs = kwargs
                created.raw = self

        class FakeCompressed:
            def __init__(self, **kwargs):
                kwargs = dict(kwargs)
                self.kwargs = kwargs
                created.compressed = self

        monkeypatch.setattr(indexer_qsa, "ReplicatedLinear", FakeLinear)
        monkeypatch.setattr(indexer_qsa, "GemmaRMSNorm", FakeNorm)
        monkeypatch.setattr(indexer_qsa, "QSAKeyStateCache", FakeRaw)
        monkeypatch.setattr(indexer_qsa, "QSACompressedKeyCache", FakeCompressed)
        return created

    def _config(self, **overrides):
        values = dict(
            hidden_size=8,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=4,
            indexer_budget=8,
            indexer_compress_ratio=2,
            rms_norm_eps=1e-6,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def _vllm_config(self, **overrides):
        values = dict(
            cache_config=SimpleNamespace(block_size=16),
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                max_model_len=64,
                uses_mrope=True,
            ),
            kv_transfer_config=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_constructs_caches_and_projections(self, monkeypatch):
        created = self._install(monkeypatch)
        indexer = indexer_qsa.QSAIndexer(
            vllm_config=self._vllm_config(),
            config=self._config(),
            layer_id=3,
            rotary_emb="rope",
            prefix="p.indexer",
        )
        assert indexer.layer_id == 3
        assert indexer.index_n_heads == 2
        assert indexer.index_kv_heads == 1
        assert indexer.index_head_dim == 4
        assert indexer.token_topk == 8
        assert indexer.compress_ratio == 2
        assert indexer.max_model_len == 64
        assert indexer.select_all_tokens is False
        assert indexer.output_width == 9
        assert indexer.rotary_emb == "rope"
        assert created.linear.args == (8, (2 + 1) * 4)
        assert created.linear.kwargs["prefix"] == "p.indexer.index_qk_proj"
        assert [norm.args for norm in created.norms] == [(4,), (4,)]
        assert created.raw.kwargs["prefix"] == "p.indexer.raw_key_cache"
        assert created.raw.kwargs["cache_rope_positions"] is True
        assert created.raw.kwargs["head_size"] == 4
        assert created.compressed.kwargs["prefix"] == (
            "p.indexer.compressed_key_cache"
        )
        assert created.compressed.kwargs["compress_ratio"] == 2

    def test_unprefixed_projection_name(self, monkeypatch):
        created = self._install(monkeypatch)
        indexer = indexer_qsa.QSAIndexer(
            vllm_config=self._vllm_config(),
            config=self._config(),
            layer_id=0,
            rotary_emb="rope",
        )
        assert created.linear.kwargs["prefix"] == "index_qk_proj"
        assert created.raw.kwargs["prefix"] == "raw_key_cache"
        assert indexer.prefix == ""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"cache_config": None}, "paged KV cache"),
            (
                {
                    "model_config": SimpleNamespace(
                        dtype=torch.float16,
                        max_model_len=64,
                        uses_mrope=False,
                    )
                },
                "requires BF16",
            ),
        ],
    )
    def test_configuration_guards(self, monkeypatch, overrides, message):
        self._install(monkeypatch)
        with pytest.raises((ValueError, NotImplementedError), match=message):
            indexer_qsa.QSAIndexer(
                vllm_config=self._vllm_config(**overrides),
                config=self._config(),
                layer_id=0,
                rotary_emb="rope",
            )

    @pytest.mark.parametrize(
        ("max_model_len", "topk", "kv_transfer", "expected"),
        [
            (8, 8, None, True),
            (7, 8, None, True),
            (9, 8, None, False),
            (8, 8, object(), False),
        ],
    )
    def test_select_all_requires_context_within_budget_and_no_export(
        self, monkeypatch, max_model_len, topk, kv_transfer, expected
    ):
        self._install(monkeypatch)
        vllm_config = self._vllm_config(kv_transfer_config=kv_transfer)
        vllm_config.model_config.max_model_len = max_model_len
        indexer = indexer_qsa.QSAIndexer(
            vllm_config=vllm_config,
            config=self._config(indexer_budget=topk),
            layer_id=0,
            rotary_emb="rope",
        )
        assert indexer.select_all_tokens is expected


def _indexer(
    *,
    head_dim=4,
    compress_ratio=2,
    token_topk=8,
    max_model_len=64,
    rotary=None,
    select_all=False,
):
    indexer = object.__new__(indexer_qsa.QSAIndexer)
    nn.Module.__init__(indexer)
    indexer.layer_id = 0
    indexer.index_n_heads = 2
    indexer.index_kv_heads = 1
    indexer.index_head_dim = head_dim
    indexer.token_topk = token_topk
    indexer.compress_ratio = compress_ratio
    indexer.max_model_len = max_model_len
    indexer.select_all_tokens = select_all
    indexer.prefix = "p"
    indexer.rotary_emb = rotary or SimpleNamespace(
        rotary_dim=head_dim, mrope_section=None
    )
    indexer.raw_key_cache = SimpleNamespace(
        prefix="raw", key_cache=torch.zeros(1), rope_position_cache=None
    )
    indexer.compressed_key_cache = SimpleNamespace(
        prefix="compressed", kv_cache=torch.zeros(1)
    )
    indexer.index_qk_proj = _Linear(torch.zeros(2, 12))
    indexer.q_layernorm = _Norm(2.0)
    indexer.k_layernorm = _Norm(3.0)
    return indexer


class TestProjectQk:
    def test_splits_projects_normalizes_and_rotates(self, monkeypatch):
        indexer = _indexer()
        raw = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        indexer.index_qk_proj = _Linear(raw)
        rotation_calls = []

        def fake_rotate(rotary, positions, tensor):
            rotation_calls.append((rotary, positions, tensor))
            return tensor + 1

        monkeypatch.setattr(indexer_qsa, "fast_gemma_rmsnorm", lambda norm, x: None)
        monkeypatch.setattr(indexer_qsa, "apply_qsa_rope", fake_rotate)
        positions = torch.tensor([0, 1])
        q, token_k = indexer.project_qk(torch.zeros(2, 8), positions)
        assert q.shape == (2, 2, 4)
        assert token_k.shape == (2, 1, 4)
        # The Q half is normalized by the injected factor before rotation.
        expected_q = raw[:, :8].reshape(2, 2, 4) * 2.0 + 1
        assert q.tolist() == expected_q.tolist()
        assert torch.equal(token_k[:, 0], raw[:, 8:])
        assert rotation_calls[0][1] is positions

    def test_fast_norm_result_is_preferred_when_available(self, monkeypatch):
        indexer = _indexer()
        raw = torch.ones(2, 12)
        indexer.index_qk_proj = _Linear(raw)
        monkeypatch.setattr(
            indexer_qsa, "fast_gemma_rmsnorm", lambda norm, x: x * 5
        )
        monkeypatch.setattr(
            indexer_qsa, "apply_qsa_rope", lambda rotary, positions, tensor: tensor
        )
        q, _ = indexer.project_qk(torch.zeros(2, 8), torch.zeros(2))
        assert q.tolist() == torch.full((2, 2, 4), 5.0).tolist()


class TestNormalizeCompressedKeys:
    def test_plain_rope_uses_the_first_token_position(self, monkeypatch):
        indexer = _indexer(rotary=SimpleNamespace(rotary_dim=4, mrope_section=None))
        recorded = []
        monkeypatch.setattr(indexer_qsa, "fast_gemma_rmsnorm", lambda norm, x: None)
        monkeypatch.setattr(
            indexer_qsa,
            "apply_qsa_rope",
            lambda rotary, positions, tensor: recorded.append(positions) or tensor,
        )
        pooled = torch.ones(3, 4)
        first_positions = torch.tensor([[5, 6, 7], [8, 9, 10]])
        out = indexer.normalize_compressed_keys(pooled, first_positions)
        assert out.shape == (3, 1, 4)
        assert out.tolist() == torch.full((3, 1, 4), 3.0).tolist()
        assert recorded[0].tolist() == [5, 8]

    def test_mrope_uses_the_transposed_group_positions(self, monkeypatch):
        indexer = _indexer(
            rotary=SimpleNamespace(rotary_dim=4, mrope_section=(2, 1, 1))
        )
        recorded = []
        monkeypatch.setattr(indexer_qsa, "fast_gemma_rmsnorm", lambda norm, x: None)
        monkeypatch.setattr(
            indexer_qsa,
            "apply_qsa_rope",
            lambda rotary, positions, tensor: recorded.append(positions) or tensor,
        )
        first_positions = torch.tensor([[5, 6, 7], [8, 9, 10]])
        indexer.normalize_compressed_keys(torch.ones(2, 4), first_positions)
        assert recorded[0].tolist() == [[5, 8], [6, 9], [7, 10]]


def _metadata(raw_tokens=2, compressed_tokens=2, device_type="cpu"):
    raw = SimpleNamespace(
        num_actual_tokens=raw_tokens,
        logical_positions=torch.zeros(2),
        prepare_calls=0,
    )
    if device_type == "cpu":
        logical_positions = torch.zeros(2)
    else:
        logical_positions = torch.zeros(2, device=device_type)
    compressed = SimpleNamespace(
        num_actual_tokens=compressed_tokens,
        logical_positions=logical_positions,
        prepare_calls=0,
    )

    def raw_prepare():
        raw.prepare_calls += 1

    def compressed_prepare():
        compressed.prepare_calls += 1

    raw.prepare = raw_prepare
    compressed.prepare = compressed_prepare
    return raw, compressed


class TestMetadataPreparation:
    def test_non_dict_metadata_returns_none(self, monkeypatch):
        indexer = _indexer()
        monkeypatch.setattr(
            indexer_qsa,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=object()),
        )
        assert indexer._metadata() is None

    def test_list_metadata_uses_the_first_entry(self, monkeypatch):
        indexer = _indexer()
        raw, compressed = _metadata()
        monkeypatch.setattr(
            indexer_qsa,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata=[{"raw": raw, "compressed": compressed}]
            ),
        )
        assert indexer._metadata() == (raw, compressed)

    def test_prepares_compressed_then_validates_raw_on_cpu(self, monkeypatch):
        indexer = _indexer()
        raw, compressed = _metadata()
        monkeypatch.setattr(
            indexer_qsa,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata={"raw": raw, "compressed": compressed}
            ),
        )
        assert indexer._metadata() == (raw, compressed)
        assert compressed.prepare_calls == 1
        assert raw.prepare_calls == 1

    def test_non_cpu_metadata_skips_the_raw_comparison(self, monkeypatch):
        indexer = _indexer()
        raw, compressed = _metadata(device_type="meta")
        monkeypatch.setattr(
            indexer_qsa,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata={"raw": raw, "compressed": compressed}
            ),
        )
        assert indexer._metadata() == (raw, compressed)
        assert compressed.prepare_calls == 1
        assert raw.prepare_calls == 0

    def test_token_count_disagreement_is_rejected(self, monkeypatch):
        indexer = _indexer()
        raw, compressed = _metadata(raw_tokens=3)
        monkeypatch.setattr(
            indexer_qsa,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata={"raw": raw, "compressed": compressed}
            ),
        )
        with pytest.raises(RuntimeError, match="side-cache metadata token counts"):
            indexer._metadata()

    def test_position_disagreement_on_cpu_is_rejected(self, monkeypatch):
        indexer = _indexer()
        raw, compressed = _metadata()
        raw.logical_positions = torch.ones(2)
        monkeypatch.setattr(
            indexer_qsa,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata={"raw": raw, "compressed": compressed}
            ),
        )
        with pytest.raises(RuntimeError, match="positions disagree"):
            indexer._metadata()


class TestCompressionImpl:
    @pytest.mark.parametrize(
        ("head_dim", "rotary_dim", "sections", "neox", "expected"),
        [
            (128, 64, (16, 8, 8), True, "fused"),
            (128, 64, (16, 8, 8), False, "generic"),
            (128, 64, (16, 8, 4), True, "generic"),
            (64, 64, (32, 0, 0), True, "generic"),
        ],
    )
    def test_selects_fused_only_for_the_supported_layout(
        self, head_dim, rotary_dim, sections, neox, expected
    ):
        rotary = SimpleNamespace(
            rotary_dim=rotary_dim,
            mrope_section=sections,
            is_neox_style=neox,
        )
        indexer = _indexer(head_dim=head_dim, rotary=rotary)
        impl = indexer._compression_impl()
        target = (
            qsa_ops.qsa_compress_norm_mrope_store_groups
            if expected == "fused"
            else qsa_ops.qsa_compress_groups_with_ratio
        )
        assert impl is target

    def test_missing_mrope_section_uses_a_plain_one_axis_layout(self):
        rotary = SimpleNamespace(rotary_dim=64, is_neox_style=True)
        indexer = _indexer(head_dim=128, rotary=rotary)
        impl = indexer._compression_impl()
        # The synthesized (32, 0, 0) section sums to rotary_dim // 2.
        assert impl is qsa_ops.qsa_compress_norm_mrope_store_groups


class TestForwardDispatch:
    def test_no_metadata_returns_minus_one_padding(self, monkeypatch):
        indexer = _indexer(compress_ratio=2)
        monkeypatch.setattr(indexer, "_metadata", lambda: None)
        hidden = torch.zeros(3, 8)
        result = indexer.forward(hidden, torch.zeros(3))
        assert result.shape == (3, 9)
        assert result.dtype is torch.int32
        assert result.tolist() == torch.full((3, 9), -1).tolist()
        out = torch.zeros(3, 9, dtype=torch.int32)
        assert indexer.forward(hidden, torch.zeros(3), out=out) is out
        assert out.tolist() == torch.full((3, 9), -1).tolist()

    def test_select_all_skips_projection_and_cache_updates(self, monkeypatch):
        indexer = _indexer(select_all=True)
        raw, compressed = _metadata()
        compressed.logical_positions = torch.tensor([0, 0])
        compressed.seq_lens = torch.tensor([2])
        compressed.token_to_req = torch.tensor([0, 0])
        monkeypatch.setattr(indexer, "_metadata", lambda: (raw, compressed))
        calls = []
        monkeypatch.setattr(
            qsa_ops,
            "qsa_select_all_paged_tokens",
            lambda *args: calls.append(args) or "selected",
        )
        monkeypatch.setattr(
            indexer, "project_qk", lambda *a: pytest.fail("projection must be skipped")
        )
        out = object()
        assert indexer.forward(torch.zeros(2, 8), torch.zeros(2), out=out) == "selected"
        assert calls[0][0] is compressed.logical_positions
        assert calls[0][1] is compressed.seq_lens
        assert calls[0][2] is compressed.token_to_req
        assert calls[0][3] == 8
        assert calls[0][4] == 2
        assert calls[0][5] == 64
        assert calls[0][6] is out

    def test_standard_path_slices_tokens_and_threads_output(self, monkeypatch):
        indexer = _indexer(select_all=False)
        raw, compressed = _metadata()
        raw.num_actual_tokens = 5
        monkeypatch.setattr(indexer, "_metadata", lambda: (raw, compressed))
        calls = []
        monkeypatch.setattr(
            indexer, "project_qk", lambda hidden, positions: ("q", "token-k")
        )

        def update(token_k, positions, raw_meta, compressed_meta):
            calls.append(("update", token_k, positions))

        monkeypatch.setattr(indexer, "_update_and_compress", update)
        monkeypatch.setattr(
            indexer, "_select", lambda q, meta, out: calls.append(("select", q, out)) or "out"
        )
        hidden = torch.arange(30, dtype=torch.float32).view(10, 3)
        positions = torch.arange(10)
        result = indexer.forward(hidden, positions, out="buffer")
        assert result == "out"
        assert calls[0][1] == "token-k"
        assert calls[0][2].tolist() == [0, 1, 2, 3, 4]
        assert calls[1] == ("select", "q", "buffer")


class TestSelectAndCompressionUpdates:
    def test_select_forwards_the_fixed_width_contract(self, monkeypatch):
        indexer = _indexer()
        recorded = []
        monkeypatch.setattr(
            qsa_ops,
            "qsa_select_paged_tokens",
            lambda *args: recorded.append(args) or "selected",
        )
        metadata = SimpleNamespace(
            block_table="table",
            token_to_req="requests",
            logical_positions="positions",
            seq_lens="lens",
        )
        assert indexer._select("q", metadata, "out") == "selected"
        args = recorded[0]
        assert args[0] == "q"
        assert args[1] is indexer.compressed_key_cache.kv_cache
        assert args[2:6] == ("table", "requests", "positions", "lens")
        assert args[6] == 8
        assert args[7] == 2
        assert args[8] == "out"

    def test_generic_compression_normalizes_and_stores(self, monkeypatch):
        indexer = _indexer(rotary=SimpleNamespace(rotary_dim=4, mrope_section=None))
        indexer.raw_key_cache = SimpleNamespace(
            key_cache="raw-cache", rope_position_cache=None
        )
        indexer.compressed_key_cache = SimpleNamespace(kv_cache="compressed-cache")
        stores = []
        monkeypatch.setattr(
            qsa_ops,
            "qsa_store_cache_rows",
            lambda cache, slots, values: stores.append((cache, slots, values)),
        )
        monkeypatch.setattr(
            indexer, "_compression_impl", lambda: qsa_ops.qsa_compress_groups_with_ratio
        )
        monkeypatch.setattr(
            qsa_ops,
            "qsa_compress_groups_with_ratio",
            lambda *args: ("pooled", "first-positions"),
        )
        monkeypatch.setattr(
            indexer,
            "normalize_compressed_keys",
            lambda pooled, first: ("normalized", pooled, first),
        )
        raw_meta = SimpleNamespace(
            num_actual_tokens=2,
            slot_mapping=torch.tensor([1, 2]),
            block_table="table",
        )
        compressed_meta = SimpleNamespace(
            token_to_req="requests",
            logical_positions="positions",
            slot_mapping=torch.tensor([3, 4]),
        )
        token_k = torch.arange(8, dtype=torch.float32).view(2, 1, 4)
        indexer._update_and_compress(
            token_k, torch.tensor([0, 1]), raw_meta, compressed_meta
        )
        assert stores[0][0] == "raw-cache"
        assert torch.equal(stores[0][1], torch.tensor([1, 2]))
        assert torch.equal(stores[0][2], token_k)
        assert stores[1][0] == "compressed-cache"
        assert torch.equal(stores[1][1], torch.tensor([3, 4]))
        assert stores[1][2] == ("normalized", "pooled", "first-positions")

    def test_rope_position_cache_is_stored_with_canonical_positions(
        self, monkeypatch
    ):
        indexer = _indexer(rotary=SimpleNamespace(rotary_dim=4, mrope_section=None))
        position_cache = torch.zeros(8)
        indexer.raw_key_cache = SimpleNamespace(
            key_cache="raw-cache", rope_position_cache=position_cache
        )
        indexer.compressed_key_cache = SimpleNamespace(kv_cache="compressed-cache")
        stores = []
        monkeypatch.setattr(
            qsa_ops,
            "qsa_store_cache_rows",
            lambda cache, slots, values: stores.append((cache, slots, values)),
        )
        monkeypatch.setattr(
            indexer, "_compression_impl", lambda: qsa_ops.qsa_compress_groups_with_ratio
        )
        monkeypatch.setattr(
            qsa_ops,
            "qsa_compress_groups_with_ratio",
            lambda *args: ("pooled", "first-positions"),
        )
        monkeypatch.setattr(
            indexer, "normalize_compressed_keys", lambda *args: "normalized"
        )
        raw_meta = SimpleNamespace(
            num_actual_tokens=2,
            slot_mapping=torch.tensor([1, 2]),
            block_table="table",
        )
        compressed_meta = SimpleNamespace(
            token_to_req="requests",
            logical_positions="positions",
            slot_mapping=torch.tensor([3, 4]),
        )
        positions = torch.tensor([[0, 1], [0, 1], [0, 1]])
        indexer._update_and_compress(
            torch.zeros(2, 1, 4), positions, raw_meta, compressed_meta
        )
        assert stores[0][0] == "raw-cache"
        assert stores[1][0] is position_cache
        assert stores[2][0] == "compressed-cache"
        assert stores[1][2].tolist() == [[[0, 0, 0]], [[1, 1, 1]]]

    def test_fused_compression_passes_the_unit_norm_and_rope_parameters(
        self, monkeypatch
    ):
        rotary = SimpleNamespace(
            rotary_dim=64,
            mrope_section=(32, 16, 16),
            mrope_interleaved=True,
            is_neox_style=True,
            _match_cos_sin_cache_dtype=lambda tensor: "cos-sin",
        )
        indexer = _indexer(head_dim=128, rotary=rotary)
        indexer.raw_key_cache = SimpleNamespace(
            key_cache="raw-cache", rope_position_cache=None
        )
        indexer.compressed_key_cache = SimpleNamespace(kv_cache="compressed-cache")
        indexer.k_layernorm = SimpleNamespace(
            weight="norm-weight", variance_epsilon=1e-6
        )
        monkeypatch.setattr(qsa_ops, "qsa_store_cache_rows", lambda *args: None)
        monkeypatch.setattr(
            indexer,
            "_compression_impl",
            lambda: qsa_ops.qsa_compress_norm_mrope_store_groups,
        )
        recorded = []
        monkeypatch.setattr(
            qsa_ops,
            "qsa_compress_norm_mrope_store_groups",
            lambda *args: recorded.append(args),
        )
        raw_meta = SimpleNamespace(
            num_actual_tokens=2,
            slot_mapping=torch.tensor([1, 2]),
            block_table="table",
        )
        compressed_meta = SimpleNamespace(
            token_to_req="requests",
            logical_positions="positions",
            slot_mapping=torch.tensor([3, 4]),
        )
        indexer._update_and_compress(
            torch.zeros(2, 1, 128),
            torch.tensor([0, 1]),
            raw_meta,
            compressed_meta,
        )
        args = recorded[0]
        assert args[0] == "raw-cache"
        assert args[2] == "requests"
        assert args[3] == "positions"
        assert torch.equal(args[4], torch.tensor([3, 4]))
        assert args[5] == "compressed-cache"
        assert args[6] == "norm-weight"
        assert args[7] == "cos-sin"
        assert args[8] == 2
        assert args[9] == 1e-6
        assert args[10] == 64
        assert args[11] == (32, 16, 16)
        assert args[12] is True
        assert args[13] is None
