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

"""QSA attention owner: cache unpacking, construction guards and transactions.

Every distributed/upstream owner is injected at the module boundary, so the
tests observe the plugin's own validation and cache-update/attention ordering.
The cache transaction is driven with recording kernels and a metadata dict
whose tokens/positions are small and explicit.
"""

from types import SimpleNamespace

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import (
    QSAForwardMetadata,
    QSAMetadataBuilder,
)
from vllm_fl.models.qwen3_8_flash_next.gpu import qsa as qsa_module
from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa as qsa_ops


class TestAttentionBackend:
    def test_advertised_contract(self):
        backend = qsa_module.Qwen3_8FlashNextQSAAttentionBackend
        assert backend.get_name() == "QWEN38_FLASH_NEXT_QSA_FLAGTREE"
        assert backend.get_builder_cls() is QSAMetadataBuilder
        assert backend.indexes_kv_by_block_stride() is False
        assert backend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
        assert backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        ) == (0, 1, 2, 3, 4, 5)
        assert backend.supports_kv_connector() is False
        assert backend.supported_dtypes == [torch.bfloat16]
        assert backend.supported_kv_cache_dtypes == ["auto", "bfloat16"]

    def test_kv_cache_shape_keeps_the_kv_dimension(self):
        shape = qsa_module.Qwen3_8FlashNextQSAAttentionBackend.get_kv_cache_shape(
            5, 16, 1, 8, cache_dtype_str="bfloat16"
        )
        assert shape == (5, 2, 16, 1, 8)

    def test_impl_class_is_rejected(self):
        with pytest.raises(NotImplementedError, match="out-of-band"):
            qsa_module.Qwen3_8FlashNextQSAAttentionBackend.get_impl_cls()


class TestUnpackCache:
    def test_legacy_five_dimensional_cache_splits_into_views(self):
        storage = torch.zeros(2, 2, 3, 1, 4, dtype=torch.bfloat16)
        key, value = qsa_module._unpack_qsa_kv_cache(storage, head_size=4)
        assert key.shape == (2, 3, 1, 4)
        assert value.shape == (2, 3, 1, 4)
        assert key.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
        assert value.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
        assert key.stride() == (24, 4, 4, 1)
        assert value.stride() == (24, 4, 4, 1)

    @pytest.mark.parametrize(
        "shape",
        [(2, 3, 3, 1, 4), (2, 2, 3, 1, 8)],
    )
    def test_invalid_five_dimensional_geometry_is_rejected(self, shape):
        with pytest.raises(ValueError, match="invalid legacy QSA KV cache shape"):
            qsa_module._unpack_qsa_kv_cache(torch.zeros(shape), head_size=4)

    def test_packed_four_dimensional_layout_is_rejected(self):
        with pytest.raises(ValueError, match="packed 4-D"):
            qsa_module._unpack_qsa_kv_cache(torch.zeros(2, 1, 3, 8), head_size=4)

    def test_other_ranks_are_rejected(self):
        with pytest.raises(ValueError, match="invalid QSA KV cache rank"):
            qsa_module._unpack_qsa_kv_cache(torch.zeros(2, 3, 8), head_size=4)


class TestPreparedTokenToRequest:
    def _indexer(self):
        return SimpleNamespace(
            compressed_key_cache=SimpleNamespace(prefix="compressed")
        )

    def test_returns_the_prepared_request_rows(self):
        compressed = SimpleNamespace(
            num_actual_tokens=3, token_to_req=torch.tensor([0, 1, 2, 3])
        )
        got = qsa_module._prepared_qsa_token_to_req(
            {"compressed": compressed}, self._indexer(), num_tokens=3
        )
        assert got.tolist() == [0, 1, 2]

    def test_token_count_disagreement_is_rejected(self):
        compressed = SimpleNamespace(
            num_actual_tokens=2, token_to_req=torch.tensor([0, 1, 2])
        )
        with pytest.raises(RuntimeError, match="token counts disagree"):
            qsa_module._prepared_qsa_token_to_req(
                {"compressed": compressed}, self._indexer(), num_tokens=3
            )


def _attention_config(**overrides):
    values = dict(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        rope_parameters={},
        rms_norm_eps=1e-6,
        is_causal=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _attention_vllm_config(**overrides):
    values = dict(
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, multimodal_config=None
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_attention_doubles(monkeypatch, *, tp_size=1, kv_dtype=torch.bfloat16):
    created = SimpleNamespace(indexer=None, qkv=None, o_proj=None, rope=None)

    class FakeQkv:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            created.qkv = self

    class FakeProj:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            created.o_proj = self

    class FakeRope:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.is_neox_style = True
            created.rope = self

    class FakeNorm:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeIndexer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.output_width = 5
            created.indexer = self

    monkeypatch.setattr(
        qsa_module, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(qsa_module, "QKVParallelLinear", FakeQkv)
    monkeypatch.setattr(qsa_module, "RowParallelLinear", FakeProj)
    monkeypatch.setattr(qsa_module, "get_rope", FakeRope)
    monkeypatch.setattr(qsa_module, "GemmaRMSNorm", FakeNorm)
    monkeypatch.setattr(qsa_module, "QSAIndexer", FakeIndexer)
    monkeypatch.setattr(qsa_module, "set_default_quant_scales", lambda *a, **k: None)
    monkeypatch.setattr(
        qsa_module, "kv_cache_dtype_str_to_dtype", lambda *a, **k: kv_dtype
    )
    monkeypatch.setattr(
        qsa_module,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: False),
    )
    return created


def _build_attention(monkeypatch, *, config=None, vllm_config=None, **kwargs):
    created = _install_attention_doubles(monkeypatch, **kwargs)
    layer = qsa_module.Qwen3_8FlashNextQSAAttention(
        vllm_config=vllm_config or _attention_vllm_config(),
        config=config or _attention_config(),
        layer_id=2,
        quant_config=None,
        reduce_results=True,
        prefix="p.self_attn",
    )
    return layer, created


class TestQsaAttentionConstruction:
    def test_valid_configuration_wires_every_owner(self, monkeypatch):
        layer, created = _build_attention(monkeypatch)
        assert layer.hidden_size == 8
        assert layer.total_num_heads == 2
        assert layer.num_heads == 2
        assert layer.total_num_kv_heads == 1
        assert layer.num_kv_heads == 1
        assert layer.head_dim == 4
        assert layer.q_size == 8
        assert layer.kv_size == 4
        assert layer.scaling == 0.5
        assert layer.attn_output_gate is True
        assert layer.q_size + 2 * layer.kv_size == 16
        assert layer.layer_name == "p.self_attn.attn"
        assert layer.kv_cache.numel() == 0
        assert layer.topk_indices_buffer.shape == (16, 5)
        assert layer.topk_indices_buffer.dtype is torch.int32
        assert created.qkv.kwargs["prefix"] == "p.self_attn.qkv_proj"
        assert created.qkv.args[2] == 4  # total heads incl. the output gate
        assert created.o_proj.kwargs["reduce_results"] is True
        assert created.indexer.kwargs["prefix"] == "p.self_attn.indexer"
        assert created.indexer.kwargs["layer_id"] == 2
        assert created.indexer.kwargs["rotary_emb"] is layer.rotary_emb
        assert layer.get_attn_backend() is qsa_module.Qwen3_8FlashNextQSAAttentionBackend

    def test_layer_registers_in_the_static_forward_context(self, monkeypatch):
        vllm_config = _attention_vllm_config()
        _build_attention(monkeypatch, vllm_config=vllm_config)
        static = vllm_config.compilation_config.static_forward_context
        assert "p.self_attn.attn" in static
        with pytest.raises(ValueError, match="Duplicate layer name"):
            _build_attention(monkeypatch, vllm_config=vllm_config)

    def test_replicated_kv_heads_are_clamped_to_one(self, monkeypatch):
        layer, _ = _build_attention(
            monkeypatch,
            tp_size=4,
            config=_attention_config(num_attention_heads=4, num_key_value_heads=2),
        )
        assert layer.num_heads == 1
        assert layer.num_kv_heads == 1
        assert layer.kv_size == 4

    @pytest.mark.parametrize(
        ("overrides", "tp", "message"),
        [
            ({"cache_config": None}, 1, "paged KV cache"),
            (
                {
                    "model_config": SimpleNamespace(
                        dtype=torch.float16, multimodal_config=None
                    )
                },
                1,
                "requires BF16",
            ),
            (
                {"cache_config": SimpleNamespace(cache_dtype="fp8", block_size=16)},
                1,
                "BF16 main KV cache",
            ),
            (
                {
                    "parallel_config": SimpleNamespace(
                        prefill_context_parallel_size=2,
                        decode_context_parallel_size=1,
                    )
                },
                1,
                "context parallelism",
            ),
        ],
    )
    def test_vllm_configuration_guards(self, monkeypatch, overrides, tp, message):
        _install_attention_doubles(monkeypatch, tp_size=tp)
        with pytest.raises((ValueError, NotImplementedError), match=message):
            qsa_module.Qwen3_8FlashNextQSAAttention(
                vllm_config=_attention_vllm_config(**overrides),
                config=_attention_config(),
                layer_id=0,
                prefix="p",
            )

    def test_kv_quantization_scheme_is_rejected(self, monkeypatch):
        _install_attention_doubles(monkeypatch)
        with pytest.raises(NotImplementedError, match="KV quantization"):
            qsa_module.Qwen3_8FlashNextQSAAttention(
                vllm_config=_attention_vllm_config(),
                config=_attention_config(),
                layer_id=0,
                quant_config=SimpleNamespace(kv_cache_scheme=object()),
                prefix="p",
            )

    def test_non_causal_attention_is_rejected(self, monkeypatch):
        _install_attention_doubles(monkeypatch)
        with pytest.raises(NotImplementedError, match="causal"):
            qsa_module.Qwen3_8FlashNextQSAAttention(
                vllm_config=_attention_vllm_config(),
                config=_attention_config(is_causal=False),
                layer_id=0,
                prefix="p",
            )

    def test_head_divisibility_guards(self, monkeypatch):
        _install_attention_doubles(monkeypatch, tp_size=2)
        with pytest.raises(ValueError, match="heads must be divisible by TP"):
            qsa_module.Qwen3_8FlashNextQSAAttention(
                vllm_config=_attention_vllm_config(),
                config=_attention_config(num_attention_heads=3),
                layer_id=0,
                prefix="p",
            )
        with pytest.raises(ValueError, match="KV heads must be divisible"):
            qsa_module.Qwen3_8FlashNextQSAAttention(
                vllm_config=_attention_vllm_config(),
                config=_attention_config(num_key_value_heads=3),
                layer_id=0,
                prefix="p",
            )

    def test_replicated_kv_heads_guard(self, monkeypatch):
        _install_attention_doubles(monkeypatch, tp_size=4)
        with pytest.raises(ValueError, match="replicated QSA KV heads"):
            qsa_module.Qwen3_8FlashNextQSAAttention(
                vllm_config=_attention_vllm_config(),
                config=_attention_config(
                    num_attention_heads=4, num_key_value_heads=3
                ),
                layer_id=0,
                prefix="p",
            )

    def test_dual_chunk_rope_is_rejected(self, monkeypatch):
        _install_attention_doubles(monkeypatch)
        with pytest.raises(NotImplementedError, match="dual-chunk"):
            qsa_module.Qwen3_8FlashNextQSAAttention(
                vllm_config=_attention_vllm_config(),
                config=_attention_config(dual_chunk_attention_config={}),
                layer_id=0,
                prefix="p",
            )

    def test_non_bf16_cache_storage_is_rejected(self, monkeypatch):
        _install_attention_doubles(monkeypatch, kv_dtype=torch.float16)
        with pytest.raises(NotImplementedError, match="BF16 cache storage"):
            qsa_module.Qwen3_8FlashNextQSAAttention(
                vllm_config=_attention_vllm_config(),
                config=_attention_config(),
                layer_id=0,
                prefix="p",
            )

    def test_kv_cache_spec_uses_the_local_head_geometry(self, monkeypatch):
        layer, _ = _build_attention(monkeypatch)
        recorded = {}

        def fake_spec(**kwargs):
            recorded.update(kwargs)
            return "spec"

        monkeypatch.setattr(qsa_module, "FullAttentionSpec", fake_spec)
        monkeypatch.setattr(
            qsa_module, "get_kv_quant_mode", lambda name: ("mode", name)
        )
        assert layer.get_kv_cache_spec(_attention_vllm_config()) == "spec"
        assert recorded["block_size"] == 16
        assert recorded["num_kv_heads"] == 1
        assert recorded["head_size"] == 4
        assert recorded["head_size_v"] == 4
        assert recorded["dtype"] is torch.bfloat16
        assert recorded["kv_quant_mode"] == ("mode", "auto")


def _main_metadata(num_tokens=2):
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        block_table_tensor=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
    )
    return QSAForwardMetadata(
        common=common,
        token_to_req=torch.zeros(8, dtype=torch.int32),
        logical_positions=torch.zeros(8, dtype=torch.int64),
        qsa_slot_mapping=torch.tensor([4, 5], dtype=torch.int64),
        storage_block_size=4,
        compress_ratio=2,
    )


class _IndexerCallable:
    def __init__(self, width, *, selected=None):
        self.output_width = width
        self.raw_key_cache = SimpleNamespace(prefix="raw")
        self.compressed_key_cache = SimpleNamespace(prefix="compressed")
        self.selected = selected
        self.calls = []

    def __call__(self, hidden_states, positions, out):
        self.calls.append((hidden_states, positions, out))
        if self.selected is not None:
            return self.selected
        return torch.full(
            (out.shape[0], self.output_width), 3, dtype=torch.int32
        )


class TestRunQsa:
    def _layer(self, *, indexer=None, kv_cache=None):
        layer = object.__new__(qsa_module.Qwen3_8FlashNextQSAAttention)
        layer.layer_name = "p.attn"
        layer.kv_cache = (
            torch.zeros(2, 2, 3, 1, 4, dtype=torch.bfloat16)
            if kv_cache is None
            else kv_cache
        )
        layer.head_dim = 4
        layer.scaling = 0.5
        layer.kv_cache_dtype = "auto"
        layer._k_scale = torch.tensor(1.0)
        layer._v_scale = torch.tensor(2.0)
        layer.topk_indices_buffer = torch.zeros(8, 5, dtype=torch.int32)
        layer._qsa_split_workspaces = {}
        layer.indexer = indexer or _IndexerCallable(5)
        return layer

    def _metadata(self, *, main_tokens=2, side_tokens=2, compressed_tokens=2):
        main = _main_metadata(main_tokens)
        side = SimpleNamespace(num_actual_tokens=side_tokens)
        compressed = SimpleNamespace(
            num_actual_tokens=compressed_tokens,
            token_to_req=torch.tensor([0, 0, 9, 9], dtype=torch.int32),
        )
        return {"p.attn": main, "raw": side, "compressed": compressed}

    def test_fallback_store_then_sparse_attention(self, monkeypatch):
        layer = self._layer()
        metadata = self._metadata()
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=metadata),
        )
        monkeypatch.setattr(qsa_module, "has_native_cache_update", lambda: False)
        calls = []
        monkeypatch.setattr(
            qsa_ops,
            "qsa_store_kv_cache_rows",
            lambda *args: calls.append(("store", args)),
        )
        monkeypatch.setattr(
            qsa_ops,
            "qsa_prepare_split_workspace",
            lambda *args: calls.append(("prepare", args)) or (3, "workspace"),
        )

        def sparse(*args, **kwargs):
            calls.append(("sparse", args, kwargs))
            args[7].fill_(6)

        monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", sparse)
        query = torch.full((2, 2, 4), 2.0, dtype=torch.bfloat16)
        key = torch.full((2, 1, 4), 3.0, dtype=torch.bfloat16)
        value = torch.full((2, 1, 4), 4.0, dtype=torch.bfloat16)
        gate = torch.full((2, 2, 4), 5.0, dtype=torch.bfloat16)
        output = torch.zeros(2, 2, 4)
        layer._run_qsa(
            torch.zeros(2, 8), torch.zeros(2), query, key, value, gate, output
        )
        assert output.tolist() == torch.full((2, 2, 4), 6.0).tolist()
        assert calls[0][0] == "store"
        key_cache, value_cache = calls[0][1][0], calls[0][1][1]
        assert key_cache.shape == (2, 3, 1, 4)
        assert value_cache.shape == (2, 3, 1, 4)
        assert torch.equal(calls[0][1][2], torch.tensor([4, 5]))
        assert torch.equal(calls[0][1][3], key)
        assert calls[1][0] == "prepare"
        assert calls[1][1][2] == 5
        assert calls[1][1][3] is layer._qsa_split_workspaces
        sparse_call = calls[2]
        assert torch.equal(sparse_call[1][0], query)
        assert torch.equal(sparse_call[1][3], layer.topk_indices_buffer[:2])
        assert torch.equal(sparse_call[1][4], metadata["p.attn"].block_table)
        assert sparse_call[1][5].tolist() == [0, 0]
        assert sparse_call[1][6] == 0.5
        assert sparse_call[2]["split_workspace"] == "workspace"
        assert torch.equal(sparse_call[2]["gate"], gate)

    def test_native_cache_update_replaces_the_fallback_store(self, monkeypatch):
        layer = self._layer()
        metadata = self._metadata()
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=metadata),
        )
        monkeypatch.setattr(qsa_module, "has_native_cache_update", lambda: True)
        native_calls = []
        monkeypatch.setattr(
            qsa_module,
            "native_cache_update",
            lambda *args: native_calls.append(args),
        )
        monkeypatch.setattr(
            qsa_ops,
            "qsa_store_kv_cache_rows",
            lambda *args: pytest.fail("fallback store must not run"),
        )
        monkeypatch.setattr(
            qsa_ops, "qsa_prepare_split_workspace", lambda *args: (1, None)
        )
        monkeypatch.setattr(
            qsa_ops, "qsa_sparse_paged_attention", lambda *args, **kwargs: None
        )
        key = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
        value = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
        layer._run_qsa(
            torch.zeros(2, 8, dtype=torch.bfloat16),
            torch.zeros(2),
            torch.zeros(2, 2, 4, dtype=torch.bfloat16),
            key,
            value,
            torch.zeros(2, 2, 4, dtype=torch.bfloat16),
            torch.zeros(2, 2, 4),
        )
        assert len(native_calls) == 1
        args = native_calls[0]
        assert torch.equal(args[0], key)
        assert torch.equal(args[1], value)
        assert args[5] == "auto"
        assert args[6].item() == 1.0
        assert args[7].item() == 2.0

    def test_non_dict_metadata_zeroes_the_output(self, monkeypatch):
        layer = self._layer()
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=object()),
        )
        output = torch.full((2, 2, 4), 9.0)
        layer._run_qsa(
            torch.zeros(2, 8),
            torch.zeros(2),
            torch.zeros(2, 2, 4),
            torch.zeros(2, 1, 4),
            torch.zeros(2, 1, 4),
            torch.zeros(2, 2, 4),
            output,
        )
        assert output.tolist() == torch.zeros(2, 2, 4).tolist()

    def test_metadata_list_uses_the_first_entry(self, monkeypatch):
        layer = self._layer()
        metadata = self._metadata()
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=[metadata]),
        )
        monkeypatch.setattr(qsa_module, "has_native_cache_update", lambda: False)
        monkeypatch.setattr(qsa_ops, "qsa_store_kv_cache_rows", lambda *a: None)
        monkeypatch.setattr(
            qsa_ops, "qsa_prepare_split_workspace", lambda *a: (1, None)
        )
        monkeypatch.setattr(
            qsa_ops, "qsa_sparse_paged_attention", lambda *a, **k: None
        )
        layer._run_qsa(
            torch.zeros(2, 8, dtype=torch.bfloat16),
            torch.zeros(2),
            torch.zeros(2, 2, 4, dtype=torch.bfloat16),
            torch.zeros(2, 1, 4, dtype=torch.bfloat16),
            torch.zeros(2, 1, 4, dtype=torch.bfloat16),
            torch.zeros(2, 2, 4, dtype=torch.bfloat16),
            torch.zeros(2, 2, 4),
        )

    def test_empty_cache_is_rejected(self, monkeypatch):
        layer = self._layer(kv_cache=torch.tensor([]))
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=self._metadata()),
        )
        with pytest.raises(RuntimeError, match="not bound"):
            layer._run_qsa(
                torch.zeros(2, 8),
                torch.zeros(2),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 2, 4),
            )

    def test_side_cache_token_disagreement_is_rejected(self, monkeypatch):
        layer = self._layer()
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata=self._metadata(side_tokens=3)
            ),
        )
        with pytest.raises(RuntimeError, match="main and side metadata"):
            layer._run_qsa(
                torch.zeros(2, 8),
                torch.zeros(2),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 2, 4),
            )

    def test_indexer_selection_width_is_validated(self, monkeypatch):
        indexer = _IndexerCallable(5, selected=torch.zeros(2, 4, dtype=torch.int32))
        layer = self._layer(indexer=indexer)
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=self._metadata()),
        )
        with pytest.raises(RuntimeError, match="invalid selection shape"):
            layer._run_qsa(
                torch.zeros(2, 8),
                torch.zeros(2),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 2, 4),
            )

    def test_compressed_request_map_disagreement_is_rejected(self, monkeypatch):
        layer = self._layer()
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata=self._metadata(compressed_tokens=3)
            ),
        )
        with pytest.raises(RuntimeError, match="main and compressed"):
            layer._run_qsa(
                torch.zeros(2, 8),
                torch.zeros(2),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 2, 4),
            )

    def test_non_bf16_cache_is_rejected(self, monkeypatch):
        layer = self._layer(kv_cache=torch.zeros(2, 2, 3, 1, 4))
        metadata = self._metadata()
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=metadata),
        )
        monkeypatch.setattr(qsa_module, "has_native_cache_update", lambda: False)
        monkeypatch.setattr(qsa_ops, "qsa_store_kv_cache_rows", lambda *a: None)
        with pytest.raises(NotImplementedError, match="BF16 Q/K/V"):
            layer._run_qsa(
                torch.zeros(2, 8),
                torch.zeros(2),
                torch.zeros(2, 2, 4, dtype=torch.bfloat16),
                torch.zeros(2, 1, 4, dtype=torch.bfloat16),
                torch.zeros(2, 1, 4, dtype=torch.bfloat16),
                torch.zeros(2, 2, 4, dtype=torch.bfloat16),
                torch.zeros(2, 2, 4),
            )


class TestCustomOpBoundary:
    def test_op_dispatches_to_the_owner(self, monkeypatch):
        calls = []
        layer = object.__new__(qsa_module.Qwen3_8FlashNextQSAAttention)
        layer._run_qsa = lambda *args: calls.append(args)
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(no_compile_layers={"p.attn": layer}),
        )
        tensors = [torch.zeros(1) for _ in range(7)]
        qsa_module.qwen3_8_flash_next_qsa_with_output(
            *tensors, layer_name="p.attn"
        )
        assert len(calls) == 1
        assert calls[0][0] is tensors[0]
        assert calls[0][6] is tensors[6]

    def test_wrong_owner_type_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            qsa_module,
            "get_forward_context",
            lambda: SimpleNamespace(no_compile_layers={"p.attn": object()}),
        )
        with pytest.raises(TypeError, match="is not a Qwen3.8-Flash-Next QSA owner"):
            qsa_module.qwen3_8_flash_next_qsa_with_output(
                *[torch.zeros(1) for _ in range(7)], layer_name="p.attn"
            )

    def test_fake_impl_is_inert(self):
        assert (
            qsa_module.qwen3_8_flash_next_qsa_with_output_fake(
                *[torch.zeros(1) for _ in range(7)], layer_name="p.attn"
            )
            is None
        )


class _Projection:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.output, None


class TestQsaAttentionForward:
    def _layer(self, *, gate=True):
        layer = object.__new__(qsa_module.Qwen3_8FlashNextQSAAttention)
        layer.num_heads = 2
        layer.num_kv_heads = 1
        layer.head_dim = 4
        layer.layer_name = "p.attn"
        tokens = 2
        q = torch.full((tokens, 8), 1.0)
        k = torch.full((tokens, 4), 2.0)
        v = torch.full((tokens, 4), 3.0)
        g = torch.full((tokens, 8), 4.0) if gate else None
        layer.qkv_proj = _Projection(torch.zeros(tokens, 16))
        layer._project_qkv_gate = lambda qkv, positions: (q, k, v, g)
        layer.o_proj = _Projection("projected")
        return layer

    def test_forward_uses_the_registered_op(self, monkeypatch):
        layer = self._layer()
        recorded = []

        def fake_op(*args):
            recorded.append(args)
            args[6].fill_(5)

        monkeypatch.setattr(
            qsa_module,
            "current_platform",
            SimpleNamespace(opaque_attention_op=lambda: False),
        )
        monkeypatch.setattr(qsa_module, "_encode_layer_name", lambda name: ("enc", name))
        monkeypatch.setattr(
            qsa_module, "qwen3_8_flash_next_qsa_with_output", fake_op
        )
        hidden = torch.zeros(2, 8)
        positions = torch.tensor([0, 1])
        assert layer.forward(positions, hidden) == "projected"
        assert len(recorded) == 1
        args = recorded[0]
        assert args[0] is hidden
        assert args[1] is positions
        assert args[2].shape == (2, 2, 4)
        assert args[6].tolist() == torch.full((2, 2, 4), 5.0).tolist()
        assert args[7] == ("enc", "p.attn")
        projected = layer.o_proj.calls[0][0]
        assert projected.tolist() == torch.full((2, 8), 5.0).tolist()

    def test_opaque_attention_uses_torch_ops(self, monkeypatch):
        layer = self._layer()
        recorded = []

        def fake_op(*args):
            recorded.append(args)
            args[6].fill_(7)

        monkeypatch.setattr(
            qsa_module,
            "current_platform",
            SimpleNamespace(opaque_attention_op=lambda: True),
        )
        monkeypatch.setattr(qsa_module, "_encode_layer_name", lambda name: name)
        monkeypatch.setattr(
            torch.ops.vllm,
            "qwen3_8_flash_next_qsa_with_output",
            fake_op,
            raising=False,
        )
        layer.forward(torch.zeros(2), torch.zeros(2, 8))
        assert len(recorded) == 1
        assert recorded[0][7] == "p.attn"
        assert layer.o_proj.calls[0][0].tolist() == torch.full((2, 8), 7.0).tolist()

    def test_missing_output_gate_is_rejected(self, monkeypatch):
        layer = self._layer(gate=False)
        monkeypatch.setattr(
            qsa_module,
            "current_platform",
            SimpleNamespace(opaque_attention_op=lambda: False),
        )
        with pytest.raises(RuntimeError, match="requires its output gate"):
            layer.forward(torch.zeros(2), torch.zeros(2, 8))
