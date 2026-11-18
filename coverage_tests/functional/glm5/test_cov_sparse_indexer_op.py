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

"""GLM5 sparse-indexer operator orchestration and custom-op entry points.

A recording backend stands in for the platform kernel chain, so the tests
assert the documented prefill-then-decode ordering, kpool pool expansion and
top-k buffer slices.  Small explicit tensors make every expected count
derivable from the request boundaries alone.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5 import sparse_indexer as si


def _indexer_metadata(
    *,
    slot_mapping,
    num_decodes=0,
    num_prefills=0,
    num_decode_tokens=0,
    num_prefill_tokens=0,
    prefill=None,
    decode=None,
    max_seq_len=64,
):
    return DeepseekV32IndexerMetadata(
        seq_lens=torch.zeros(1, dtype=torch.int32),
        max_seq_len=max_seq_len,
        slot_mapping=slot_mapping,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=prefill,
        decode=decode,
    )


class _Workspace:
    def __init__(self):
        self.calls = []

    def get_simultaneous(self, values_spec, scales_spec, *extra):
        self.calls.append((values_spec, scales_spec, extra))
        if extra:
            return (torch.empty(1), torch.empty(1), torch.empty(1))
        values_shape, _ = values_spec
        scales_shape, _ = scales_spec
        return (
            torch.zeros(values_shape, dtype=torch.float32),
            torch.zeros(scales_shape, dtype=torch.uint8),
        )


class _Backend:
    use_fp8 = False
    is_nvidia = False
    has_nvidia_deep_gemm = False

    def __init__(self):
        self.calls = []

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def indexer_k_quant_and_cache(self, *args):
        self._record("insert", *args)

    def cp_gather_indexer_k_quant_cache(self, *args):
        self._record("gather", *args)

    def mqa_logits(self, q, k, weights, ks, ke, clean_logits=False):
        self._record("mqa", q, k, weights, ks, ke, clean_logits=clean_logits)
        return torch.zeros(weights.shape[0], 6)

    def topk_prefill(self, logits, ks, ke, dst, rows, s0, s1, k):
        self._record("topk_prefill", logits, dst, rows, s0, s1, k)
        dst.fill_(-1)

    def paged_mqa_logits(
        self,
        q,
        k_cache,
        weights,
        seq_lens,
        block_table,
        schedule,
        clean_logits=False,
        max_model_len=None,
    ):
        self._record(
            "paged",
            q,
            weights,
            seq_lens,
            block_table,
            schedule,
            max_model_len=max_model_len,
        )
        return torch.zeros(weights.shape[0], 8)

    def topk_decode(
        self, logits, next_n, seq_lens, dst, rows, s0, s1, k, max_seq_len=None
    ):
        self._record("topk_decode", next_n, dst, rows, s0, s1, k, max_seq_len)
        dst.fill_(-1)

    def pack_seq(self, tensor, lengths, pad_value=0, max_length=None):
        self._record("pack", tensor.shape, lengths.tolist(), pad_value, max_length)
        batch = len(lengths)
        width = max_length or int(lengths.max())
        out = torch.full(
            (batch, width, *tensor.shape[1:]), pad_value, dtype=tensor.dtype
        )
        start = 0
        for row, length in enumerate(lengths.tolist()):
            out[row, :length] = tensor[start : start + length]
            start += length
        return out

    def unpack_seq(self, tensor, lengths, total_tokens=None):
        self._record("unpack", tensor.shape, lengths.tolist(), total_tokens)
        flat = tensor.reshape(-1, *tensor.shape[2:])
        pieces = []
        start = 0
        for length in lengths.tolist():
            pieces.append(flat[start : start + length])
            start += length
        return torch.cat(pieces, dim=0)

    def expand_pools_and_append_tail(self, pool_ids, seq_lens, kpool):
        self._record("expand", pool_ids, seq_lens, kpool)
        return torch.full((pool_ids.shape[0], 6), 5, dtype=torch.int32)

    def expand_pools_to_tokens(self, pool_ids, valid, topk, kpool):
        self._record("expand_valid", pool_ids, valid, topk, kpool)
        return torch.full((pool_ids.shape[0], 6), 6, dtype=torch.int32)

    def kpool_decode_update_and_maybe_write_cache_batched(self, *args, **kwargs):
        self._record("decode_update", *args, **kwargs)

    def kpool_seed_tail_cache(self, *args):
        self._record("seed_tail", *args)

    def kpool_compress_and_write_cache(self, *args, **kwargs):
        self._record("kpool_write", *args, **kwargs)

    def gather_rows(self, tensor, index):
        self._record("gather_rows", tensor, index)
        return tensor

    def scatter_decode_tokens(self, tokens, pad_value, num_requests, lmax, indices):
        self._record("scatter", tokens.shape, num_requests, lmax)
        req_id, intra = indices
        out = torch.full(
            (num_requests, lmax, *tokens.shape[1:]),
            pad_value,
            dtype=tokens.dtype,
        )
        out[req_id, intra] = tokens
        return out

    def append_tail_to_topk(self, *args):
        self._record("append_tail", *args)


def _install_backend(monkeypatch):
    backend = _Backend()
    workspace = _Workspace()
    monkeypatch.setattr(si, "INDEXER_BACKEND", backend)
    monkeypatch.setattr(si, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(si.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    monkeypatch.setattr(
        si, "current_platform", SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn)
    )
    monkeypatch.setattr(si, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(si, "_resolve_layer_name", lambda name: name)
    return backend, workspace


def _call(buffer, attn_metadata, **overrides):
    values = dict(
        hidden_states=torch.zeros(buffer.shape[0], 8),
        k_cache_prefix="layer",
        kv_cache=torch.zeros(4, 2, 8),
        q_quant=torch.zeros(buffer.shape[0], 1, 4),
        q_scale=None,
        k=torch.zeros(buffer.shape[0], 1, 4),
        weights=torch.ones(buffer.shape[0], 1),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=4,
        head_dim=4,
        max_model_len=16,
        total_seq_lens=4,
        topk_indices_buffer=buffer,
        skip_k_cache_insert=True,
    )
    values.update(overrides)
    return si.sparse_attn_indexer_kpool(**values)


class TestProfilingBranch:
    def test_missing_metadata_reserves_workspace_and_returns_the_buffer(
        self, monkeypatch
    ):
        """Verifies missing metadata reserves workspace and returns the buffer."""
        backend, workspace = _install_backend(monkeypatch)
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
        )
        buffer = torch.zeros(1, 8, dtype=torch.int32)
        assert _call(buffer, None) is buffer
        assert backend.calls == []
        # values, scales plus the radix-top-k reservation.
        assert len(workspace.calls) == 1
        assert len(workspace.calls[0][2]) == 1

    def test_configured_scheduler_sizes_the_logits_sentinel(self, monkeypatch):
        """Verifies configured scheduler sizes the logits sentinel."""
        _install_backend(monkeypatch)
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
        )
        monkeypatch.setattr(
            si,
            "get_current_vllm_config_or_none",
            lambda: SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=2, max_num_batched_tokens=8
                )
            ),
        )
        buffer = torch.zeros(1, 8, dtype=torch.int32)
        assert _call(buffer, None) is buffer


class TestPrefillSelection:
    def _metadata(self, *, slots, short=True, chunks=()):
        prefill = SimpleNamespace(
            max_prefill_seq_len=4 if short else 16,
            chunks=list(chunks),
        )
        return {
            "layer": _indexer_metadata(
                slot_mapping=slots,
                num_prefills=1,
                num_prefill_tokens=3,
                prefill=prefill,
            )
        }

    def test_short_prefill_attends_to_every_token_causally(self, monkeypatch):
        """Verifies short prefill attends to every token causally."""
        _install_backend(monkeypatch)
        metadata = self._metadata(slots=torch.zeros(3, dtype=torch.int64))
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        buffer = torch.full((3, 8), -7, dtype=torch.int32)
        out = _call(buffer, metadata, positions=torch.tensor([0, 1, 2]))
        expected = torch.full((3, 8), -1, dtype=torch.int32)
        expected[0, :1] = 0
        expected[1, :2] = torch.arange(2)
        expected[2, :3] = torch.arange(3)
        assert out is buffer
        assert out.tolist() == expected.tolist()

    def test_long_prefill_gathers_scores_and_selects_topk(self, monkeypatch):
        """Verifies long prefill gathers scores and selects topk."""
        _install_backend(monkeypatch)
        chunk = SimpleNamespace(
            token_start=0,
            token_end=3,
            total_seq_lens=4,
            skip_kv_gather=False,
            block_table="table",
            cu_seq_lens="cu",
            cu_seqlen_ks=torch.tensor([0, 1, 2]),
            cu_seqlen_ke=torch.tensor([1, 2, 3]),
        )
        metadata = self._metadata(
            slots=torch.zeros(3, dtype=torch.int64), short=False, chunks=[chunk]
        )
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        buffer = torch.full((3, 4), 9, dtype=torch.int32)
        q = torch.ones(3, 1, 4)
        weights = torch.ones(3, 1)
        out = _call(
            buffer,
            metadata,
            positions=torch.tensor([10, 11, 12]),
            q_quant=q,
            weights=weights,
            total_seq_lens=4,
        )
        assert out is buffer
        backend = si.INDEXER_BACKEND
        names = [name for name, _, _ in backend.calls]
        assert names == ["gather", "mqa", "topk_prefill"]
        mqa = backend.calls[1][1]
        assert torch.equal(mqa[0][0], q[0:3])
        assert mqa[0][1] is None
        assert mqa[2] is not weights
        assert torch.equal(mqa[2], weights[0:3])
        assert backend.calls[0][1][3] == "table"
        assert out.tolist() == torch.full((3, 4), -1).tolist()

    def test_pool_prefill_expands_selected_pools(self, monkeypatch):
        """Verifies pool prefill expands selected pools."""
        _install_backend(monkeypatch)
        chunk = SimpleNamespace(
            token_start=0,
            token_end=3,
            total_seq_lens=4,
            skip_kv_gather=False,
            block_table="table",
            cu_seq_lens="cu",
            cu_seqlen_ks=torch.tensor([0, 1, 2]),
            cu_seqlen_ke=torch.tensor([1, 2, 3]),
        )
        metadata = self._metadata(
            slots=torch.zeros(3, dtype=torch.int64), short=False, chunks=[chunk]
        )
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        buffer = torch.zeros(3, 9, dtype=torch.int32)
        out = _call(
            buffer,
            metadata,
            positions=torch.tensor([10, 11, 12]),
            topk_tokens=8,
            index_kpool=2,
        )
        backend = si.INDEXER_BACKEND
        topk = next(call for call in backend.calls if call[0] == "topk_prefill")
        assert topk[1][5] == 4  # topk 8 // kpool 2
        expand = next(call for call in backend.calls if call[0] == "expand")
        assert expand[1][1].tolist() == [11, 12, 13]
        assert expand[1][2] == 2
        assert out[:3, :6].tolist() == torch.full((3, 6), 5).tolist()


class TestDecodeSelection:
    def _metadata(self, *, decode, slots, num_decode_tokens, num_decodes=1):
        return {
            "layer": _indexer_metadata(
                slot_mapping=slots,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                decode=decode,
            )
        }

    def test_uniform_decode_scores_selects_and_writes_the_buffer(self, monkeypatch):
        """Verifies uniform decode scores selects and writes the buffer."""
        _install_backend(monkeypatch)
        decode = SimpleNamespace(
            decode_lens=torch.tensor([2]),
            requires_padding=False,
            seq_lens=torch.tensor([12]),
            block_table="block-table",
            schedule_metadata="schedule",
        )
        metadata = self._metadata(
            decode=decode, slots=torch.tensor([1, 2]), num_decode_tokens=2
        )
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        buffer = torch.zeros(2, 4, dtype=torch.int32)
        out = _call(
            buffer,
            metadata,
            positions=torch.tensor([10, 11]),
            topk_tokens=4,
            max_model_len=16,
        )
        backend = si.INDEXER_BACKEND
        assert [name for name, _, _ in backend.calls] == ["paged", "topk_decode"]
        paged = backend.calls[0][1]
        assert paged[1].shape == (2, 1)
        assert paged[2].tolist() == [12]
        assert paged[3] == "block-table"
        assert backend.calls[0][2]["max_model_len"] == 16
        topk = backend.calls[1][1]
        assert topk[0] == 2
        assert topk[2] == 2
        assert out.tolist() == torch.full((2, 4), -1).tolist()

    def test_padded_decode_packs_and_unpacks_query_rows(self, monkeypatch):
        """Verifies padded decode packs and unpacks query rows."""
        _install_backend(monkeypatch)
        decode = SimpleNamespace(
            decode_lens=torch.tensor([1, 0, 1]),
            requires_padding=True,
            seq_lens=torch.tensor([3, 0, 4]),
            block_table="block-table",
            schedule_metadata="schedule",
        )
        metadata = self._metadata(
            decode=decode,
            slots=torch.tensor([1, 2]),
            num_decode_tokens=2,
            num_decodes=3,
        )
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        buffer = torch.zeros(3, 4, dtype=torch.int32)
        out = _call(
            buffer,
            metadata,
            positions=torch.tensor([2, 3]),
            topk_tokens=4,
        )
        backend = si.INDEXER_BACKEND
        names = [name for name, _, _ in backend.calls]
        assert names == ["pack", "pack", "paged", "topk_decode", "unpack"]
        pack = backend.calls[0][1]
        assert pack[0] == (2, 1, 4)
        assert pack[1] == [1, 0, 1]
        unpack = backend.calls[-1][1]
        assert unpack[1] == [1, 0, 1]
        assert unpack[2] == 2
        assert out.shape == (3, 4)
        assert out[:2].tolist() == torch.full((2, 4), -1).tolist()

    def test_pool_decode_uses_token_granular_tail_lengths(self, monkeypatch):
        """Verifies pool decode uses token granular tail lengths."""
        _install_backend(monkeypatch)
        decode = SimpleNamespace(
            decode_lens=torch.tensor([1]),
            requires_padding=False,
            seq_lens=torch.tensor([12]),
            block_table="block-table",
            schedule_metadata="schedule",
        )
        metadata = self._metadata(
            decode=decode, slots=torch.tensor([1]), num_decode_tokens=1
        )
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        buffer = torch.zeros(2, 9, dtype=torch.int32)
        gate = torch.zeros(1, 1, 4)
        out = _call(
            buffer,
            metadata,
            positions=torch.tensor([10]),
            topk_tokens=8,
            index_kpool=2,
            skip_k_cache_insert=False,
            gate_score=gate,
            compress_ape=torch.zeros(1, 4),
        )
        backend = si.INDEXER_BACKEND
        names = [name for name, _, _ in backend.calls]
        assert names == ["paged", "topk_decode", "expand"]
        expand = backend.calls[-1][1]
        assert expand[1].tolist() == [11]
        assert expand[2] == 2
        assert out[:1, :6].tolist() == torch.full((1, 6), 5).tolist()

    def test_non_uniform_pool_decode_scatters_and_updates_the_tail(self, monkeypatch):
        """Verifies non uniform pool decode scatters and updates the tail."""
        _install_backend(monkeypatch)
        decode = SimpleNamespace(
            decode_lens=torch.tensor([1, 0, 1]),
            requires_padding=True,
            seq_lens=torch.tensor([3, 0, 4]),
            block_table="block-table",
            schedule_metadata="schedule",
        )
        tail_meta = _indexer_metadata(
            slot_mapping=torch.tensor([30, 31]), num_decodes=3
        )
        metadata = self._metadata(
            decode=decode,
            slots=torch.tensor([1, 2]),
            num_decode_tokens=2,
            num_decodes=3,
        )
        metadata["tail"] = tail_meta
        monkeypatch.setattr(
            si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        buffer = torch.zeros(3, 9, dtype=torch.int32)
        out = _call(
            buffer,
            metadata,
            positions=torch.tensor([2, 3]),
            topk_tokens=8,
            index_kpool=2,
            skip_k_cache_insert=False,
            gate_score=torch.zeros(2, 1, 4),
            compress_ape=torch.zeros(1, 4),
            tail_kv_cache=torch.zeros(1),
            tail_prefix="tail",
        )
        backend = si.INDEXER_BACKEND
        update = next(call for call in backend.calls if call[0] == "decode_update")
        tail_slots, dec_k, dec_gate, dec_slots, dec_pos = (
            update[1][2],
            update[1][3],
            update[1][4],
            update[1][6],
            update[1][7],
        )
        assert tail_slots.tolist() == [[30], [-1], [31]]
        assert dec_k.shape == (3, 1, 1, 4)
        assert dec_gate.shape == (3, 1, 1, 4)
        assert dec_slots.tolist() == [[1], [-1], [2]]
        assert dec_pos.tolist() == [[2], [-1], [3]]
        unpack = next(call for call in backend.calls if call[0] == "unpack")
        assert unpack[1][1] == [1, 0, 1]
        assert out.shape == (3, 9)


def _custop_config():
    return SimpleNamespace(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, custom_ops=["all"]
        )
    )


class TestFakeAndClassEntryPoints:
    def test_fake_returns_the_buffer(self):
        """Verifies fake returns the buffer."""
        buffer = torch.zeros(1, 4, dtype=torch.int32)
        assert (
            si.sparse_attn_indexer_kpool_fake(
                torch.zeros(1, 8),
                "layer",
                torch.zeros(1),
                torch.zeros(1),
                None,
                torch.zeros(1),
                torch.zeros(1),
                128,
                "ue8m0",
                4,
                4,
                16,
                4,
                buffer,
                False,
            )
            is buffer
        )

    def _op_layer(self, monkeypatch, *, use_fp4=False):
        backend = _Backend()
        monkeypatch.setattr(si, "INDEXER_BACKEND", backend)
        k_cache = SimpleNamespace(prefix="k", kv_cache=torch.zeros(1))
        with set_current_vllm_config(_custop_config()):
            layer = si.SparseAttnIndexerKpool(
                k_cache,
                128,
                "ue8m0",
                4,
                4,
                16,
                32,
                torch.zeros(2, 4, dtype=torch.int32),
                skip_k_cache_insert=False,
                use_fp4_cache=use_fp4,
                tail_cache=SimpleNamespace(prefix="tail", kv_cache="tail-cache"),
            )
        return layer, backend

    def test_nvidia_without_deep_gemm_is_rejected(self, monkeypatch):
        """Verifies nvidia without deep gemm is rejected."""
        backend = _Backend()
        backend.is_nvidia = True
        backend.has_nvidia_deep_gemm = False
        monkeypatch.setattr(si, "INDEXER_BACKEND", backend)
        with (
            set_current_vllm_config(_custop_config()),
            pytest.raises(RuntimeError, match="requires DeepGEMM"),
        ):
            si.SparseAttnIndexerKpool(
                SimpleNamespace(prefix="k", kv_cache=torch.zeros(1)),
                128,
                "ue8m0",
                4,
                4,
                16,
                32,
                torch.zeros(2, 4, dtype=torch.int32),
            )

    def test_native_forward_delegates_to_cuda(self, monkeypatch):
        """Verifies native forward delegates to cuda."""
        layer, _ = self._op_layer(monkeypatch)
        calls = []
        monkeypatch.setattr(
            si.SparseAttnIndexerKpool,
            "forward_cuda",
            lambda self, *args, **kwargs: calls.append((args, kwargs)) or "out",
        )
        assert (
            layer.forward_native(
                "hidden",
                "q",
                "k",
                "weights",
                gate_score="gate",
                compress_ape="ape",
                index_kpool=2,
                positions="positions",
            )
            == "out"
        )
        assert calls[0][0] == ("hidden", "q", "k", "weights")
        assert calls[0][1] == {
            "gate_score": "gate",
            "compress_ape": "ape",
            "index_kpool": 2,
            "positions": "positions",
        }

    def test_cuda_forward_dispatches_the_registered_operator(self, monkeypatch):
        """Verifies cuda forward dispatches the registered operator."""
        layer, _ = self._op_layer(monkeypatch, use_fp4=True)
        recorded = []
        monkeypatch.setattr(
            torch.ops.vllm,
            "sparse_attn_indexer_kpool",
            lambda *args: recorded.append(args) or "result",
            raising=False,
        )
        monkeypatch.setattr(si, "_encode_layer_name", lambda name: ("enc", name))
        q_values = torch.zeros(1)
        q_scale = torch.zeros(1)
        assert (
            layer.forward_cuda(
                "hidden",
                (q_values, q_scale),
                "k",
                "weights",
                gate_score="gate",
                compress_ape="ape",
                index_kpool=2,
                positions="positions",
            )
            == "result"
        )
        args = recorded[0]
        assert args[0] == "hidden"
        assert args[1] == ("enc", "k")
        assert args[2] is layer.k_cache.kv_cache
        assert args[3] is q_values
        assert args[4] is q_scale
        assert args[13] is layer.topk_indices_buffer
        assert args[14] is False
        assert args[15] is True
        assert args[16:20] == ("gate", "ape", 2, "positions")
        assert args[20] == "tail-cache"
        assert args[21] == "tail"

    def test_fp8_cuda_forward_passes_a_none_scale(self, monkeypatch):
        """Verifies fp8 cuda forward passes a none scale."""
        layer, _ = self._op_layer(monkeypatch, use_fp4=False)
        recorded = []
        monkeypatch.setattr(
            torch.ops.vllm,
            "sparse_attn_indexer_kpool",
            lambda *args: recorded.append(args) or "result",
            raising=False,
        )
        layer.forward_cuda("hidden", "q", "k", "weights")
        assert recorded[0][4] is None
        assert recorded[0][20] == "tail-cache"
        assert recorded[0][21] == "tail"

    def test_hip_forward_guards_and_dispatches(self, monkeypatch):
        """Verifies hip forward guards and dispatches."""
        layer, _ = self._op_layer(monkeypatch, use_fp4=True)
        with pytest.raises(AssertionError, match="doesn't support fp4"):
            layer.forward_hip("hidden", torch.zeros(1), "k", "weights")
        layer.use_fp4_cache = False
        monkeypatch.setattr(
            si, "rocm_aiter_ops", SimpleNamespace(is_enabled=lambda: False)
        )
        with pytest.raises(RuntimeError, match="only supported on AITER"):
            layer.forward_hip("hidden", torch.zeros(1), "k", "weights")
        monkeypatch.setattr(
            si, "rocm_aiter_ops", SimpleNamespace(is_enabled=lambda: True)
        )
        monkeypatch.setattr(si, "_encode_layer_name", lambda name: name)
        recorded = []
        monkeypatch.setattr(
            torch.ops.vllm,
            "rocm_aiter_sparse_attn_indexer",
            lambda *args, **kwargs: recorded.append((args, kwargs)) or "hip",
            raising=False,
        )
        assert layer.forward_hip("hidden", torch.zeros(1), "k", "weights") == "hip"
        assert recorded[0][0][0] == "hidden"
        assert recorded[0][1]["skip_k_cache_insert"] is False

    def test_hip_forward_rejects_a_fp4_tuple(self, monkeypatch):
        """Verifies hip forward rejects a fp4 tuple."""
        layer, _ = self._op_layer(monkeypatch, use_fp4=False)
        with pytest.raises(AssertionError, match="single FP8"):
            layer.forward_hip(
                "hidden",
                (torch.zeros(1), torch.zeros(1)),
                "k",
                "weights",
            )
