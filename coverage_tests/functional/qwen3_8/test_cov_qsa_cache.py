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

"""Pure-CPU contracts for the QSA side-cache bookkeeping.

``common/qsa_cache.py`` owns the exact logical-position and compressed-slot
math used by the Triton metadata kernel. These tests exercise the pure-Torch
reference implementations against loop-based references derived from the
documented layout: token ``i`` of request ``r`` has logical position
``seq_len[r] - query_len[r] + (i - query_start[r])``; a compressed slot exists
only at a group boundary and addresses ``physical_block * block_size +
compressed_position % block_size``.
"""

from types import SimpleNamespace

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.common import qsa_cache as qc
from vllm_fl.models.qwen3_8_flash_next.common.ple import (
    compute_ple_shard_overlap,
    copy_ple_embedding_shard_,
)


def _compressed_slots_reference(table, requests, positions, block_size, ratio):
    out = []
    for req, pos in zip(requests.tolist(), positions.tolist()):
        if pos < 0 or req < 0 or req >= table.shape[0]:
            out.append(-1)
            continue
        compressed = pos // ratio
        block = compressed // block_size
        if block >= table.shape[1]:
            out.append(-1)
            continue
        physical = int(table[req, block])
        boundary = (pos + 1) % ratio == 0
        out.append(
            physical * block_size + compressed % block_size
            if boundary and physical >= 0
            else -1
        )
    return out


class TestCanonicalRopePositions:
    """Positions are packed as exact ``[tokens, 1, 3]`` int64 rows."""

    def test_one_dimensional_positions_are_replicated_on_all_axes(self):
        """Verifies one dimensional positions are replicated on all axes."""
        positions = torch.tensor([4, 7], dtype=torch.int32)
        got = qc.canonical_qsa_rope_positions(positions)
        assert got.shape == (2, 1, 3)
        assert got.dtype == torch.int64
        assert got.tolist() == [[[4, 4, 4]], [[7, 7, 7]]]

    def test_single_axis_rows_are_broadcast_to_three_axes(self):
        """Verifies single axis rows are broadcast to three axes."""
        positions = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        got = qc.canonical_qsa_rope_positions(positions)
        assert got.tolist() == [[[1, 1, 1]], [[2, 2, 2]], [[3, 3, 3]]]

    def test_three_axis_rows_are_transposed_into_token_rows(self):
        """Verifies three axis rows are transposed into token rows."""
        positions = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
        got = qc.canonical_qsa_rope_positions(positions)
        assert got.tolist() == [[[1, 3, 5]], [[2, 4, 6]]]

    def test_unsupported_rank_and_axis_count_are_rejected(self):
        """Verifies unsupported rank and axis count are rejected."""
        with pytest.raises(ValueError, match=r"\[tokens\] or \[1\|3, tokens\]"):
            qc.canonical_qsa_rope_positions(torch.zeros(2, 3))
        with pytest.raises(ValueError, match="RoPE positions"):
            qc.canonical_qsa_rope_positions(torch.zeros(2, 2, 2))


class TestLogicalPositions:
    """Logical positions are request-relative absolute sequence positions."""

    def test_matches_loop_reference(self):
        """Verifies matches loop reference."""
        query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
        seq_lens = torch.tensor([5, 6], dtype=torch.int32)
        token_to_req = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
        got = qc._logical_positions(
            query_start_loc, seq_lens, token_to_req, num_tokens=5
        )
        expected = [5 - 2 + 0, 5 - 2 + 1, 6 - 3 + 0, 6 - 3 + 1, 6 - 3 + 2]
        assert got.dtype == torch.int64
        assert got.tolist() == expected

    def test_zero_tokens_returns_empty_int64(self):
        """Verifies zero tokens returns empty int64."""
        got = qc._logical_positions(
            torch.tensor([0, 0], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            num_tokens=0,
        )
        assert got.shape == (0,)
        assert got.dtype == torch.int64

    def test_precomputed_arange_is_sliced_to_token_count(self):
        """Verifies precomputed arange is sliced to token count."""
        query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
        seq_lens = torch.tensor([10], dtype=torch.int32)
        token_to_req = torch.zeros(3, dtype=torch.int32)
        arange = torch.tensor([0, 1, 2, 99], dtype=torch.int64)
        got = qc._logical_positions(
            query_start_loc, seq_lens, token_to_req, 3, arange=arange
        )
        assert got.tolist() == [7, 8, 9]


class TestPopulateTokenToReq:
    """The pure-Torch fallback maps real tokens and zeroes padding rows."""

    def test_maps_query_tokens_and_zeroes_padding(self):
        """Verifies maps query tokens and zeroes padding."""
        common = SimpleNamespace(
            num_actual_tokens=6,
            query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        )
        out = torch.full((6,), 7, dtype=torch.int32)
        got = qc._populate_token_to_req(common, out)
        assert got.tolist() == [0, 0, 1, 1, 1, 0]
        assert got.data_ptr() == out.data_ptr()

    def test_all_padding_when_no_query_tokens(self):
        """Verifies all padding when no query tokens."""
        common = SimpleNamespace(
            num_actual_tokens=3,
            query_start_loc_cpu=torch.tensor([0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
        )
        out = torch.full((3,), 5, dtype=torch.int32)
        assert qc._populate_token_to_req(common, out).tolist() == [0, 0, 0]


class TestLogicalToPhysicalSlots:
    """Logical positions are translated through the two-dimensional table."""

    def test_broadcasts_request_indices_across_position_rows(self):
        """Verifies broadcasts request indices across position rows."""
        table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
        requests = torch.tensor([[0], [1]], dtype=torch.int64)
        positions = torch.tensor([[0, 3], [4, 7]], dtype=torch.int64)
        got = qc._logical_to_physical_qsa_slots(table, requests, positions, 4)
        assert got.tolist() == [
            [5 * 4 + 0, 5 * 4 + 3],
            [8 * 4 + 0, 8 * 4 + 3],
        ]

    def test_invalid_requests_negative_positions_and_blocks_are_masked(self):
        """Verifies invalid requests negative positions and blocks are masked."""
        table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
        requests = torch.tensor([0, 5, 1, 0], dtype=torch.int64)
        positions = torch.tensor([-1, 0, 8, 0], dtype=torch.int64)
        got = qc._logical_to_physical_qsa_slots(table, requests, positions, 4)
        # row 1: request out of range; row 2: block 2 >= table width;
        # row 0: negative position.
        assert got.tolist() == [-1, -1, -1, 20]

    def test_empty_table_returns_all_minus_one(self):
        """Verifies empty table returns all minus one."""
        table = torch.zeros((0, 2), dtype=torch.int32)
        requests = torch.tensor([0, 0], dtype=torch.int64)
        positions = torch.tensor([0, 1], dtype=torch.int64)
        got = qc._logical_to_physical_qsa_slots(table, requests, positions, 4)
        assert got.tolist() == [-1, -1]
        assert got.dtype == torch.int64

    def test_rejects_non_positive_block_size_and_rank(self):
        """Verifies rejects non positive block size and rank."""
        table = torch.zeros((1, 2), dtype=torch.int32)
        positions = torch.tensor([0], dtype=torch.int64)
        requests = torch.tensor([0], dtype=torch.int64)
        with pytest.raises(ValueError, match="positive"):
            qc._logical_to_physical_qsa_slots(table, requests, positions, 0)
        with pytest.raises(ValueError, match="two-dimensional"):
            qc._logical_to_physical_qsa_slots(
                torch.zeros(2, dtype=torch.int32), requests, positions, 4
            )


class TestCompressedSlotMapping:
    """Only complete groups receive a slot; everything else is ``-1``."""

    def test_matches_loop_reference_for_partial_groups(self):
        """Verifies matches loop reference for partial groups."""
        table = torch.tensor([[2, 0, 4, 6], [1, 3, 5, 7]], dtype=torch.int32)
        requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        positions = torch.tensor([0, 3, 4, 7], dtype=torch.int64)
        got = qc.compressed_qsa_slot_mapping(table, requests, positions, 4, 4)
        assert got.tolist() == _compressed_slots_reference(
            table, requests, positions, 4, 4
        )
        assert got.tolist() == [-1, 8, -1, 5]

    def test_out_buffer_is_filled_and_tail_reset_to_minus_one(self):
        """Verifies out buffer is filled and tail reset to minus one."""
        table = torch.tensor([[2, 0, 4, 6]], dtype=torch.int32)
        requests = torch.zeros(2, dtype=torch.int32)
        positions = torch.tensor([3, 7], dtype=torch.int64)
        out = torch.full((5,), 9, dtype=torch.int64)
        got = qc.compressed_qsa_slot_mapping(table, requests, positions, 4, 4, out=out)
        assert got.tolist() == [8, 9]
        assert out.tolist() == [8, 9, -1, -1, -1]
        assert got.data_ptr() == out.data_ptr()

    def test_rejects_non_positive_block_size_or_ratio(self):
        """Verifies rejects non positive block size or ratio."""
        table = torch.zeros((1, 2), dtype=torch.int32)
        requests = torch.zeros(1, dtype=torch.int32)
        positions = torch.zeros(1, dtype=torch.int64)
        with pytest.raises(ValueError, match="must be positive"):
            qc.compressed_qsa_slot_mapping(table, requests, positions, 0, 4)
        with pytest.raises(ValueError, match="must be positive"):
            qc.compressed_qsa_slot_mapping(table, requests, positions, 4, 0)


def _common_metadata(num_tokens=5):
    return SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2, -1, -1, -1], dtype=torch.int64),
        block_table_tensor=torch.tensor(
            [[2, 0, 4, 6], [1, 3, 5, 7]], dtype=torch.int32
        ),
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
    )


class TestForwardMetadataPrepare:
    """The metadata object produces replay-sensitive outputs exactly once."""

    def test_ratio_one_uses_common_slots_and_loop_reference(self):
        """Verifies ratio one uses common slots and loop reference."""
        common = _common_metadata()
        token_to_req = torch.zeros(5, dtype=torch.int32)
        logical = torch.zeros(5, dtype=torch.int64)
        metadata = qc.QSAForwardMetadata(
            common=common,
            token_to_req=token_to_req,
            logical_positions=logical,
            qsa_slot_mapping=common.slot_mapping[:5],
            storage_block_size=4,
            compress_ratio=1,
        )
        assert metadata.slot_mapping.data_ptr() == common.slot_mapping.data_ptr()
        assert metadata.block_table is common.block_table_tensor
        assert metadata.seq_lens is common.seq_lens
        assert metadata.query_start_loc is common.query_start_loc
        assert metadata.num_actual_tokens == 5

        metadata.prepare()
        assert token_to_req.tolist() == [0, 0, 1, 1, 1]
        assert logical.tolist() == [3, 4, 3, 4, 5]
        assert metadata._prepared is True
        # A second prepare must not recompute over mutated outputs.
        logical.fill_(99)
        metadata.prepare()
        assert logical.tolist() == [99, 99, 99, 99, 99]

    def test_ratio_one_marks_graph_padding_rows_invalid(self):
        """Verifies ratio one marks graph padding rows invalid."""
        common = _common_metadata(num_tokens=6)
        metadata = qc.QSAForwardMetadata(
            common=common,
            token_to_req=torch.zeros(6, dtype=torch.int32),
            logical_positions=torch.zeros(6, dtype=torch.int64),
            qsa_slot_mapping=common.slot_mapping[:6],
            storage_block_size=4,
            compress_ratio=1,
        )
        metadata.prepare()
        assert metadata.token_to_req.tolist() == [0, 0, 1, 1, 1, 0]
        assert metadata.logical_positions.tolist() == [3, 4, 3, 4, 5, -1]

    def test_compressed_fallback_matches_compressed_reference(self):
        """Verifies compressed fallback matches compressed reference."""
        common = _common_metadata()
        token_to_req = torch.zeros(5, dtype=torch.int32)
        logical = torch.zeros(5, dtype=torch.int64)
        slots = torch.zeros(5, dtype=torch.int64)
        metadata = qc.QSAForwardMetadata(
            common=common,
            token_to_req=token_to_req,
            logical_positions=logical,
            qsa_slot_mapping=slots,
            storage_block_size=4,
            compress_ratio=4,
        )
        metadata.prepare()
        assert token_to_req.tolist() == [0, 0, 1, 1, 1]
        assert logical.tolist() == [3, 4, 3, 4, 5]
        expected = _compressed_slots_reference(
            common.block_table_tensor,
            token_to_req,
            logical,
            4,
            4,
        )
        # Common slot mapping -1 masks tokens 3 and 4 after compression.
        expected = [
            -1 if common.slot_mapping[i] < 0 else value
            for i, value in enumerate(expected)
        ]
        assert slots.tolist() == expected
        assert slots.tolist() == [8, -1, 4, -1, -1]


def _builder(monkeypatch, *, defer=False):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    if defer:
        import vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa as qsa_ops

        monkeypatch.setattr(qc, "_global_full_cudagraph_enabled", lambda cfg: True)
        monkeypatch.setattr(
            qsa_ops, "qsa_forward_metadata_triton_supported", lambda device: True
        )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    return qc.QSAMetadataBuilder(spec, ["qsa"], config, torch.device("cpu"))


class TestMetadataBuilder:
    """Builders allocate fixed-address buffers and prepare only when eager."""

    def test_buffers_start_zeroed_and_logical_positions_are_invalid(self, monkeypatch):
        """Verifies buffers start zeroed and logical positions are invalid."""
        builder = _builder(monkeypatch)
        assert builder.compress_ratio == 4
        assert builder.defer_metadata_to_forward is False
        assert torch.count_nonzero(builder.token_to_req_buffer) == 0
        assert bool((builder.logical_positions_buffer == -1).all())
        assert builder.token_to_req_buffer.dtype == torch.int32
        assert builder.logical_positions_buffer.dtype == torch.int64

    def test_eager_build_prepares_compressed_metadata(self, monkeypatch):
        """Verifies eager build prepares compressed metadata."""
        builder = _builder(monkeypatch)
        common = _common_metadata()
        metadata = builder.build(0, common)
        assert metadata.common is common
        assert metadata.token_to_req.tolist() == [0, 0, 1, 1, 1]
        assert metadata.logical_positions.tolist() == [3, 4, 3, 4, 5]
        expected = _compressed_slots_reference(
            common.block_table_tensor,
            metadata.token_to_req,
            metadata.logical_positions,
            builder.storage_block_size,
            4,
        )
        expected = [
            -1 if common.slot_mapping[i] < 0 else value
            for i, value in enumerate(expected)
        ]
        assert metadata.qsa_slot_mapping.tolist() == expected
        assert metadata._prepared is True

    def test_ratio_one_build_aliases_common_slot_mapping(self, monkeypatch):
        """Verifies ratio one build aliases common slot mapping."""
        builder = _builder(monkeypatch)
        builder.compress_ratio = 1
        common = _common_metadata()
        metadata = builder.build(0, common)
        assert metadata.qsa_slot_mapping.data_ptr() == common.slot_mapping.data_ptr()
        assert metadata._prepared is False

    def test_deferred_build_leaves_prepare_to_model_forward(self, monkeypatch):
        """Verifies deferred build leaves prepare to model forward."""
        builder = _builder(monkeypatch, defer=True)
        assert builder.defer_metadata_to_forward is True
        metadata = builder.build(0, _common_metadata())
        assert metadata._prepared is False
        assert metadata.qsa_slot_mapping.data_ptr() == (
            builder.slot_mapping_buffer.data_ptr()
        )


class TestStateBackends:
    """Static cache-owner contracts used by the allocator."""

    def test_backend_names_and_builders(self):
        """Verifies backend names and builders."""
        assert qc.QSAStateBackend.get_name() == "QWEN38_FLASH_NEXT_EXP_QSA_STATE"
        assert qc.QSAStateBackend.get_builder_cls() is qc.QSAMetadataBuilder
        assert qc.QSAStateBackend.indexes_kv_by_block_stride() is True
        assert qc.QSAStateBackend.get_kv_cache_stride_order() == (0, 1, 2, 3)
        assert qc.QSAStateBackend.get_kv_cache_stride_order(True) == (0, 1, 2, 3, 4)

    def test_impl_is_out_of_band(self):
        """Verifies impl is out of band."""
        with pytest.raises(NotImplementedError):
            qc.QSAStateBackend.get_impl_cls()

    def test_kv_cache_shape_requires_one_kv_head(self):
        """Verifies kv cache shape requires one kv head."""
        assert qc.QSAStateBackend.get_kv_cache_shape(4, 16, 1, 8) == (4, 16, 1, 8)
        with pytest.raises(ValueError, match="exactly one KV head"):
            qc.QSAStateBackend.get_kv_cache_shape(4, 16, 2, 8)


def _state_config():
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )


def _state_cache(**overrides):
    values = dict(
        head_size=8,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=16),
        prefix="qsa.state",
        vllm_config=_state_config(),
    )
    values.update(overrides)
    # Use the concrete key owner; the shared base class is abstract in some
    # vLLM builds while its constructor/validation logic is identical.
    return qc.QSAKeyStateCache(**values)


class TestStateCache:
    """Base cache owner validation, registration and binding."""

    def test_registers_in_static_forward_context(self):
        """Verifies registers in static forward context."""
        vllm_config = _state_config()
        cache = _state_cache(vllm_config=vllm_config)
        assert (
            vllm_config.compilation_config.static_forward_context["qsa.state"] is cache
        )
        assert cache.forward() is None
        assert cache.get_attn_backend() is qc.QSAStateBackend
        assert cache.kv_cache.numel() == 0

    def test_bind_replaces_storage(self):
        """Verifies bind replaces storage."""
        cache = _state_cache()
        tensor = torch.zeros(2, 4, 1, 8, dtype=torch.bfloat16)
        cache.bind_kv_cache(tensor)
        assert cache.kv_cache is tensor

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"head_size": 0}, "head size must be positive"),
            ({"compress_ratio": 0}, "compression ratio must be positive"),
            (
                {
                    "cache_config": SimpleNamespace(block_size=15),
                    "compress_ratio": 4,
                },
                "divisible",
            ),
        ],
    )
    def test_rejects_invalid_geometry(self, overrides, message):
        """Verifies rejects invalid geometry."""
        with pytest.raises(ValueError, match=message):
            _state_cache(**overrides)

    def test_duplicate_prefix_is_rejected(self):
        """Verifies duplicate prefix is rejected."""
        vllm_config = _state_config()
        _state_cache(vllm_config=vllm_config)
        with pytest.raises(ValueError, match="Duplicate layer name"):
            _state_cache(vllm_config=vllm_config)


def _key_cache(**overrides):
    values = dict(
        head_size=8,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=16),
        prefix="qsa.key",
        vllm_config=_state_config(),
        cache_rope_positions=False,
    )
    values.update(overrides)
    return qc.QSAKeyStateCache(**values)


class TestKeyStateCache:
    """Raw key storage pads to int64 alignment when positions are cached."""

    @pytest.mark.parametrize(("head", "offset"), [(8, 8), (10, 12), (4, 4), (5, 8)])
    def test_position_offset_rounds_up_to_int64(self, head, offset):
        """Verifies position offset rounds up to int64."""
        cache = _key_cache(head_size=head)
        assert cache.key_head_size == head
        assert cache.rope_position_offset == offset

    def test_rope_positions_add_three_aligned_axes(self):
        """Verifies rope positions add three aligned axes."""
        cache = _key_cache(head_size=8, cache_rope_positions=True)
        assert cache.head_size == 8 + 3 * 4
        assert cache.cache_rope_positions is True

    def test_bind_without_positions_creates_key_view_only(self):
        """Verifies bind without positions creates key view only."""
        cache = _key_cache()
        storage = torch.zeros(2, 4, 1, 8, dtype=torch.bfloat16)
        cache.bind_kv_cache(storage)
        assert cache.rope_position_cache is None
        assert cache.key_cache.shape == (2, 4, 1, 8)
        assert cache.key_cache.untyped_storage().data_ptr() == (
            storage.untyped_storage().data_ptr()
        )

    def test_bind_with_positions_creates_int64_tail_view(self):
        """Verifies bind with positions creates int64 tail view."""
        cache = _key_cache(head_size=8, cache_rope_positions=True)
        storage = torch.zeros(2, 4, 1, 20, dtype=torch.bfloat16)
        cache.bind_kv_cache(storage)
        assert cache.key_cache.shape == (2, 4, 1, 8)
        assert cache.rope_position_cache.dtype == torch.int64
        assert cache.rope_position_cache.shape == (2, 4, 1, 3)
        assert cache.rope_position_cache.untyped_storage().data_ptr() == (
            storage.untyped_storage().data_ptr()
        )

    @pytest.mark.parametrize(
        ("storage", "message"),
        [
            (torch.zeros(2, 4, 8, dtype=torch.bfloat16), "must be"),
            (torch.zeros(2, 4, 2, 8, dtype=torch.bfloat16), r"\[blocks, block_size, 1"),
            (torch.zeros(2, 4, 1, 8, dtype=torch.float32), "does not match"),
            (torch.zeros(2, 4, 1, 7, dtype=torch.bfloat16), "does not match"),
        ],
    )
    def test_validation_rejects_bad_layouts(self, storage, message):
        """Verifies validation rejects bad layouts."""
        cache = _key_cache()
        with pytest.raises(ValueError, match=message):
            cache.validate_kv_cache(storage)

    def test_validation_rejects_non_contiguous_last_dimension(self):
        """Verifies validation rejects non contiguous last dimension."""
        cache = _key_cache(head_size=20)
        flat = torch.zeros(2 * 4 * 20 * 2, dtype=torch.bfloat16)
        storage = torch.as_strided(flat, (2, 4, 1, 20), (160, 40, 40, 2))
        assert storage.shape[-1] == cache.head_size
        with pytest.raises(ValueError, match="contiguous last dimension"):
            cache.validate_kv_cache(storage)

    def test_validation_rejects_misaligned_position_pack(self):
        """Verifies validation rejects misaligned position pack."""
        cache = _key_cache(head_size=20, cache_rope_positions=True)
        # head_size 20 packs to a 32-element storage width; storage_offset 1
        # makes the position tail start at an odd bf16 element.
        base = torch.zeros(2, 4, 1, 33, dtype=torch.bfloat16)
        storage = base[..., 1:]
        assert storage.shape[-1] == cache.head_size
        with pytest.raises(ValueError, match="align to"):
            cache.validate_kv_cache(storage)

    def test_key_spec_is_a_padded_full_attention_spec(self):
        """Verifies key spec is a padded full attention spec."""
        cache = _key_cache()
        spec = cache.get_kv_cache_spec(SimpleNamespace())
        assert spec.block_size == 16
        assert spec.num_kv_heads == 1
        assert spec.head_size == cache.head_size
        assert spec.dtype == torch.bfloat16

    def test_compressed_spec_carries_the_compression_ratio(self):
        """Verifies compressed spec carries the compression ratio."""
        cache = qc.QSACompressedKeyCache(
            head_size=8,
            dtype=torch.bfloat16,
            cache_config=SimpleNamespace(block_size=16),
            prefix="qsa.compressed",
            vllm_config=_state_config(),
            compress_ratio=4,
        )
        spec = cache.get_kv_cache_spec(SimpleNamespace())
        assert spec.compress_ratio == 4
        assert spec.block_size == 16
        assert spec.head_size == 8


class TestPleShardCopy:
    """Checkpoint shards are intersected with the local TP vocabulary range."""

    def test_overlap_intersects_checkpoint_and_tp_ranges(self):
        """Verifies overlap intersects checkpoint and tp ranges."""
        overlap = compute_ple_shard_overlap(
            checkpoint_start=0, checkpoint_rows=10, tp_start=3, tp_end=7
        )
        assert (overlap.source_start, overlap.destination_start, overlap.row_count) == (
            3,
            0,
            4,
        )

    def test_disjoint_ranges_have_no_overlap(self):
        """Verifies disjoint ranges have no overlap."""
        assert (
            compute_ple_shard_overlap(
                checkpoint_start=10, checkpoint_rows=5, tp_start=0, tp_end=10
            )
            is None
        )

    def test_overlap_rejects_invalid_bounds(self):
        """Verifies overlap rejects invalid bounds."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_ple_shard_overlap(
                checkpoint_start=-1, checkpoint_rows=1, tp_start=0, tp_end=2
            )
        with pytest.raises(ValueError, match="invalid TP"):
            compute_ple_shard_overlap(
                checkpoint_start=0, checkpoint_rows=1, tp_start=3, tp_end=2
            )

    def test_copy_places_only_the_overlapping_rows(self):
        """Verifies copy places only the overlapping rows."""
        destination = torch.full((4, 2), -1.0)
        loaded = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        copied = copy_ple_embedding_shard_(
            destination,
            loaded,
            checkpoint_start=2,
            tp_start=0,
            tp_end=4,
        )
        assert copied == 2
        assert destination.tolist() == [
            [-1.0, -1.0],
            [-1.0, -1.0],
            [0.0, 1.0],
            [2.0, 3.0],
        ]

    def test_copy_reports_zero_for_disjoint_shard(self):
        """Verifies copy reports zero for disjoint shard."""
        destination = torch.zeros((2, 2))
        loaded = torch.ones((2, 2))
        assert (
            copy_ple_embedding_shard_(
                destination,
                loaded,
                checkpoint_start=10,
                tp_start=0,
                tp_end=2,
            )
            == 0
        )

    def test_copy_rejects_incompatible_shapes(self):
        """Verifies copy rejects incompatible shapes."""
        with pytest.raises(ValueError, match="matching ranks"):
            copy_ple_embedding_shard_(
                torch.zeros(2),
                torch.zeros(2, 2),
                checkpoint_start=0,
                tp_start=0,
                tp_end=2,
            )
        with pytest.raises(ValueError, match="dimensions do not match"):
            copy_ple_embedding_shard_(
                torch.zeros(2, 3),
                torch.zeros(2, 2),
                checkpoint_start=0,
                tp_start=0,
                tp_end=2,
            )
        with pytest.raises(ValueError, match="does not cover"):
            copy_ple_embedding_shard_(
                torch.zeros(2, 2),
                torch.zeros(4, 2),
                checkpoint_start=0,
                tp_start=0,
                tp_end=4,
            )
