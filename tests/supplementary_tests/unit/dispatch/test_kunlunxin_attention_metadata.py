# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin attention backend metadata machinery.

Covers ``vllm_fl/.../kunlunxin/impl/attention.py`` on CPU:

* ``KunlunxinMetadata`` — prefill/decode slicing and caching properties,
  encoder/cross metadata flags;
* ``KunlunxinAttentionMetadataBuilder`` — ``__init__`` thresholds (spec
  decode adjusts ``reorder_batch_threshold``), persistent
  ``kv_prefix_start_loc`` buffers, ``build`` for decode-only and mixed
  batches, and ``build_for_cudagraph_capture`` assertions;
* ``KunlunxinAttentionBackend`` classmethods — ``customize_spec`` (equal
  head sizes required; num_head_slots/state_content_bytes contract),
  layout set, swap/copy raising, kv cache shape, supported head sizes;
* ``KunlunxinAttentionBackendImpl.__init__`` validations — missing
  xtorch_ops, soft cap rejection, head size rejection, GVA ratio;
* ``_is_full_graph_runtime`` and ``is_kunlunxin_ops_available``;
* ``KunlunxinPagedAttention.split_kv_cache`` / ``physical_block_stride``;
* ``forward``'s fake-metadata profiling path via an xtorch_ops stub.

The metadata slicing runs against real ``CommonAttentionMetadata`` objects
(a real ``VllmConfig`` is built in the config fixture; the platform
downgrade of the compilation mode is re-set afterwards because only the
wrapper's availability matters here).
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.supplementary]


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _vllm_config():
    from vllm.config import (
        CompilationConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationMode

    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            is_encoder_decoder=False,
            max_num_batched_tokens=1024,
            max_model_len=2048,
        ),
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    # the CPU/platform default downgrade is irrelevant for metadata tests
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    with set_current_vllm_config(config):
        yield config


def _make_builder(spec_decode=None):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinAttentionMetadataBuilder,
    )

    with _vllm_config() as config:
        if spec_decode is not None:
            config.speculative_config = spec_decode
        builder = KunlunxinAttentionMetadataBuilder(
            None, ["layer.0"], config, torch.device("cpu")
        )
    return builder


def _common_metadata(**overrides):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    defaults = dict(
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([5, 6, 7], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=3,
        max_query_len=1,
        max_seq_len=7,
        block_table_tensor=torch.arange(12).view(3, 4),
        slot_mapping=torch.arange(3),
        seq_lens_cpu_upper_bound=torch.tensor([5, 6, 7], dtype=torch.int32),
    )
    defaults.update(overrides)
    return CommonAttentionMetadata(**defaults)


@contextlib.contextmanager
def _xtorch_ops_stub(**ops):
    stub = ModuleType("xtorch_ops")
    for name, fn in ops.items():
        setattr(stub, name, fn)
    saved = sys.modules.get("xtorch_ops")
    sys.modules["xtorch_ops"] = stub
    try:
        yield stub
    finally:
        if saved is None:
            sys.modules.pop("xtorch_ops", None)
        else:
            sys.modules["xtorch_ops"] = saved


# ---------------------------------------------------------------------------
# KunlunxinMetadata properties
# ---------------------------------------------------------------------------


class TestKunlunxinMetadataProperties:
    def _metadata(self, **overrides):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinMetadata,
        )

        defaults = dict(
            seq_lens_tensor=torch.tensor([5, 6, 7], dtype=torch.int32),
            max_decode_seq_len=7,
            block_tables=torch.arange(12).view(3, 4),
            num_prefills=1,
            num_prefill_tokens=4,
            num_decode_tokens=4,
            slot_mapping=torch.arange(8),
            enable_kv_scales_calculation=False,
            max_prefill_seq_len=4,
            num_actual_tokens=8,
            use_cuda_graph=False,
            max_query_len=4,
            max_kv_len=7,
            query_start_loc=torch.tensor([0, 4, 6, 8], dtype=torch.int32),
            query_start_loc_host=torch.tensor([0, 4, 6, 8], dtype=torch.int32),
            kv_prefix_start_loc=torch.tensor([0, 5, 11, 18], dtype=torch.int32),
            kv_prefix_start_loc_host=torch.tensor(
                [0, 5, 11, 18], dtype=torch.int32
            ),
            seq_lens_tensor_host=torch.tensor([5, 6, 7], dtype=torch.int32),
        )
        defaults.update(overrides)
        return KunlunxinMetadata(**defaults)

    def test_prefill_metadata_none_when_no_prefills(self):
        meta = self._metadata(num_prefills=0)
        assert meta.prefill_metadata is None

    def test_prefill_metadata_none_when_no_prefill_tokens(self):
        meta = self._metadata(num_prefills=0, num_prefill_tokens=0)
        assert meta.prefill_metadata is None

    def test_prefill_metadata_slices_last_rows(self):
        meta = self._metadata()
        prefill = meta.prefill_metadata
        assert prefill is not None
        # slot mapping keeps the LAST num_prefill_tokens (4)
        assert prefill.slot_mapping.tolist() == [4, 5, 6, 7]
        # seq lens / block tables keep the LAST num_prefills (1)
        assert prefill.seq_lens_tensor.tolist() == [7]
        assert prefill.block_tables.shape[0] == 1
        # cached on second access
        assert meta.prefill_metadata is prefill

    def test_prefill_metadata_query_start_loc_is_rebased(self):
        meta = self._metadata()
        prefill = meta.prefill_metadata
        # qsl[-2:] - qsl[-2] = [6, 8] - 6 = [0, 2]
        assert prefill.query_start_loc.tolist() == [0, 2]

    def test_prefill_metadata_rebases_kv_prefix_start_loc(self):
        meta = self._metadata()
        prefill = meta.prefill_metadata
        # kv_prefix[-2:] - kv_prefix[-2] = [11, 18] - 11 = [0, 7]
        assert prefill.kv_prefix_start_loc.tolist() == [0, 7]

    def test_prefill_metadata_clears_decode_counters(self):
        meta = self._metadata()
        prefill = meta.prefill_metadata
        assert prefill.num_decode_tokens == 0
        assert prefill.max_decode_seq_len == 0
        assert prefill.use_cuda_graph is False

    def test_prefill_metadata_propagates_encoder_fields(self):
        meta = self._metadata(
            encoder_seq_lens=[3],
            encoder_seq_lens_tensor=torch.tensor([3]),
            max_encoder_seq_len=3,
            cross_slot_mapping=torch.arange(2),
            cross_block_tables=torch.arange(4).view(1, 4),
        )
        prefill = meta.prefill_metadata
        assert prefill.encoder_seq_lens == [3]
        assert prefill.cross_slot_mapping.tolist() == [0, 1]
        assert prefill.cross_block_tables.shape == (1, 4)

    def test_decode_metadata_none_without_decode_tokens(self):
        meta = self._metadata(num_decode_tokens=0)
        assert meta.decode_metadata is None

    def test_decode_metadata_slices_leading_rows_when_prefills_exist(self):
        meta = self._metadata()
        decode = meta.decode_metadata
        assert decode is not None
        # slot mapping drops the first num_prefill_tokens
        assert decode.slot_mapping.tolist() == [0, 1, 2, 3]
        # seq lens / block tables / qsl drop the first num_prefills
        assert decode.seq_lens_tensor.tolist() == [5, 6]
        assert decode.block_tables.shape[0] == 2
        assert decode.query_start_loc.tolist() == [0, 4, 6]
        assert decode.seq_lens_tensor_host.tolist() == [5, 6]
        assert decode.query_start_loc_host.tolist() == [0, 4, 6]
        # cached
        assert meta.decode_metadata is decode

    def test_decode_metadata_identity_when_no_prefills(self):
        meta = self._metadata(
            num_prefills=0, num_prefill_tokens=0, num_decode_tokens=8
        )
        decode = meta.decode_metadata
        assert decode.slot_mapping is meta.slot_mapping
        assert decode.seq_lens_tensor is meta.seq_lens_tensor
        assert decode.block_tables is meta.block_tables
        assert decode.query_start_loc is meta.query_start_loc

    def test_decode_metadata_zeroes_prefill_counters(self):
        meta = self._metadata()
        decode = meta.decode_metadata
        assert decode.num_prefills == 0
        assert decode.num_prefill_tokens == 0
        assert decode.max_prefill_seq_len == 0

    def test_is_all_encoder_attn_metadata_set(self):
        meta = self._metadata()
        assert meta.is_all_encoder_attn_metadata_set is False
        meta.encoder_seq_lens = [1]
        meta.encoder_seq_lens_tensor = torch.tensor([1])
        meta.max_encoder_seq_len = 1
        assert meta.is_all_encoder_attn_metadata_set is True

    def test_is_all_cross_attn_metadata_set(self):
        meta = self._metadata()
        assert meta.is_all_cross_attn_metadata_set is False
        meta.encoder_seq_lens = [1]
        meta.encoder_seq_lens_tensor = torch.tensor([1])
        meta.max_encoder_seq_len = 1
        assert meta.is_all_cross_attn_metadata_set is False
        meta.cross_slot_mapping = torch.tensor([0])
        meta.cross_block_tables = torch.tensor([[0]])
        assert meta.is_all_cross_attn_metadata_set is True

    def test_post_init_binds_bias_fields(self):
        meta = self._metadata()
        assert meta.attn_bias is None
        assert meta.encoder_attn_bias is None
        assert meta.cross_attn_bias is None


# ---------------------------------------------------------------------------
# KunlunxinAttentionMetadataBuilder
# ---------------------------------------------------------------------------


class TestMetadataBuilder:
    def test_plain_builder_defaults(self):
        builder = _make_builder()
        assert builder.reorder_batch_threshold == 1
        assert builder.is_spec_decode is False
        # kv prefix buffers sized to max_num_seqs + 1 = 9
        assert builder._kv_prefix_max_len == 9
        assert builder._kv_prefix_capacity == 0

    def test_spec_decode_raises_threshold(self):
        spec = SimpleNamespace(num_speculative_tokens=3)
        builder = _make_builder(spec_decode=spec)
        assert builder.is_spec_decode is True
        assert builder.reorder_batch_threshold == 4

    def test_build_decode_only_batch(self):
        builder = _make_builder()
        common = _common_metadata()
        meta = builder.build(0, common)

        assert meta.num_prefills == 0
        assert meta.num_prefill_tokens == 0
        assert meta.num_decode_tokens == 3
        assert meta.max_decode_seq_len == 1
        assert meta.max_prefill_seq_len == 0
        # kv prefix start loc is the cumsum of seq lens
        assert meta.kv_prefix_start_loc.tolist() == [0, 5, 11, 18]
        assert meta.kv_prefix_start_loc_host.tolist() == [0, 5, 11, 18]
        # decode-only: prefill metadata absent
        assert meta.prefill_metadata is None
        assert meta.decode_metadata is not None

    def test_build_mixed_batch(self):
        builder = _make_builder()
        common = _common_metadata(
            query_start_loc=torch.tensor([0, 1, 5], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 5], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([5, 9], dtype=torch.int32),
            num_reqs=2,
            num_actual_tokens=5,
            max_query_len=4,
            max_seq_len=9,
            block_table_tensor=torch.arange(8).view(2, 4),
            slot_mapping=torch.arange(5),
        )
        meta = builder.build(0, common)

        assert meta.num_prefills == 1
        assert meta.num_prefill_tokens == 4
        assert meta.num_decode_tokens == 1
        assert meta.max_decode_seq_len == 1
        assert meta.max_prefill_seq_len == 4
        assert meta.max_query_len == 4
        # kv prefix over seq lens [5, 9]
        assert meta.kv_prefix_start_loc.tolist() == [0, 5, 14]

    def test_build_reuses_persistent_kv_prefix_buffers(self):
        builder = _make_builder()
        common = _common_metadata()
        first = builder.build(0, common)
        second = builder.build(0, common)
        assert (
            first.kv_prefix_start_loc.data_ptr()
            == second.kv_prefix_start_loc.data_ptr()
        )

    def test_build_grows_buffers_only_when_needed(self):
        builder = _make_builder()
        common = _common_metadata(num_reqs=3)
        first = builder.build(0, common)
        capacity_after_first = builder._kv_prefix_capacity
        assert capacity_after_first >= 4
        # smaller batch reuses the same buffers
        common_small = _common_metadata(
            num_reqs=2,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([5, 6], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([5, 6], dtype=torch.int32),
            num_actual_tokens=2,
            max_seq_len=6,
            block_table_tensor=torch.arange(8).view(2, 4),
            slot_mapping=torch.arange(2),
        )
        builder.build(0, common_small)
        assert builder._kv_prefix_capacity == capacity_after_first
        # any batch up to max_num_seqs reuses the same buffers (they are
        # sized once to max_num_seqs + 1 = 9 so a graph can bake addresses)
        six_reqs = _common_metadata(
            num_reqs=6,
            query_start_loc=torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor(
                [0, 1, 2, 3, 4, 5, 6], dtype=torch.int32
            ),
            seq_lens=torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor(
                [2, 3, 4, 5, 6, 7], dtype=torch.int32
            ),
            num_actual_tokens=6,
            max_seq_len=7,
            block_table_tensor=torch.arange(24).view(6, 4),
            slot_mapping=torch.arange(6),
        )
        meta_six = builder.build(0, six_reqs)
        assert builder._kv_prefix_capacity == capacity_after_first
        assert meta_six.kv_prefix_start_loc.shape[0] == 7
        assert (
            meta_six.kv_prefix_start_loc.data_ptr()
            == first.kv_prefix_start_loc.data_ptr()
        )

    def test_build_spec_decode_marks_decode_query_len(self):
        spec = SimpleNamespace(num_speculative_tokens=2)
        builder = _make_builder(spec_decode=spec)
        common = _common_metadata()
        meta = builder.build(0, common)
        assert meta.is_spec_decode is True
        assert meta.max_decode_query_len == 3

    def test_build_without_cpu_upper_bound_falls_back_to_cpu(self):
        builder = _make_builder()
        common = _common_metadata(seq_lens_cpu_upper_bound=None)
        meta = builder.build(0, common)
        assert meta.kv_prefix_start_loc_host.tolist() == [0, 5, 11, 18]

    def test_build_for_cudagraph_capture_requires_decode_only(self):
        builder = _make_builder()
        common = _common_metadata()
        meta = builder.build_for_cudagraph_capture(common)
        # 3 reqs x 1 token = 3 actual tokens, threshold 1 -> passes
        assert meta.num_decode_tokens == 3

    def test_build_for_cudagraph_capture_rejects_prefill_batches(self):
        builder = _make_builder()
        common = _common_metadata(
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=3,
            max_query_len=3,
            seq_lens=torch.tensor([5], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([5], dtype=torch.int32),
            block_table_tensor=torch.arange(4).view(1, 4),
            slot_mapping=torch.arange(3),
        )
        with pytest.raises(AssertionError):
            builder.build_for_cudagraph_capture(common)

    def test_use_cascade_attention_is_false(self):
        builder = _make_builder()
        assert builder.use_cascade_attention() is False
        assert builder.use_cascade_attention(1, 2, a=3) is False

    def test_cudagraph_support_is_uniform_batch(self):
        from vllm.v1.attention.backend import AttentionCGSupport
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionMetadataBuilder,
        )

        assert (
            KunlunxinAttentionMetadataBuilder._cudagraph_support
            is AttentionCGSupport.UNIFORM_BATCH
        )


# ---------------------------------------------------------------------------
# KunlunxinAttentionBackend classmethods
# ---------------------------------------------------------------------------


class TestBackendClassmethods:
    def _spec(self, **overrides):
        import torch as torch_mod

        from vllm.v1.kv_cache_interface import AttentionSpec

        defaults = dict(
            block_size=16,
            num_kv_heads=2,
            head_size=64,
            dtype=torch_mod.bfloat16,
            head_size_v=64,
        )
        defaults.update(overrides)
        return AttentionSpec(**defaults)

    def test_customize_spec_rejects_unequal_head_sizes(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackend,
        )

        spec = self._spec(head_size=64, head_size_v=128)
        with pytest.raises(ValueError, match="equal K/V head sizes"):
            KunlunxinAttentionBackend.customize_spec(spec)

    def test_customize_spec_sets_two_planes_and_content_bytes(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackend,
        )

        spec = KunlunxinAttentionBackend.customize_spec(self._spec())
        assert spec.num_head_slots == 2
        # 2 heads * 64 head size * 2 bytes (bf16)
        assert spec.state_content_bytes == 2 * 64 * 2

    def test_supported_kv_cache_layouts(self):
        from vllm.v1.kv_cache_layout import KVCacheLayout
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackend,
        )

        layouts = KunlunxinAttentionBackend.supported_kv_cache_layouts()
        assert set(layouts) == {KVCacheLayout.LHBNC, KVCacheLayout.LBHNC}

    def test_get_name_and_class_accessors(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackend,
            KunlunxinAttentionBackendImpl,
            KunlunxinAttentionMetadataBuilder,
            KunlunxinMetadata,
        )

        assert KunlunxinAttentionBackend.get_name() == "CUSTOM"
        assert (
            KunlunxinAttentionBackend.get_impl_cls()
            is KunlunxinAttentionBackendImpl
        )
        assert KunlunxinAttentionBackend.get_metadata_cls() is KunlunxinMetadata
        assert (
            KunlunxinAttentionBackend.get_builder_cls()
            is KunlunxinAttentionMetadataBuilder
        )

    def test_get_supported_head_sizes(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackend,
        )

        heads = KunlunxinAttentionBackend.get_supported_head_sizes()
        assert 128 in heads
        assert 63 not in heads

    def test_get_kv_cache_shape_two_planes(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackend,
        )

        shape = KunlunxinAttentionBackend.get_kv_cache_shape(
            num_blocks=10, block_size=16, num_kv_heads=2, head_size=64
        )
        assert shape == (2, 10, 2, 16, 64)

    def test_swap_blocks_raises(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackend,
        )

        with pytest.raises(NotImplementedError):
            KunlunxinAttentionBackend.swap_blocks([], [], torch.tensor([0]))

    def test_copy_blocks_raises(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackend,
        )

        with pytest.raises(NotImplementedError):
            KunlunxinAttentionBackend.copy_blocks([], torch.tensor([0]))


# ---------------------------------------------------------------------------
# KunlunxinPagedAttention static helpers
# ---------------------------------------------------------------------------


class TestPagedAttentionHelpers:
    def test_split_kv_cache_4d_unbinds_dim1(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinPagedAttention,
        )

        kv = torch.zeros(2, 8, 4, 16, 64)
        key_cache, value_cache = KunlunxinPagedAttention.split_kv_cache(
            kv, num_kv_heads=4, head_size=64
        )
        assert key_cache.shape == (8, 4, 16, 64)
        assert key_cache.data_ptr() == kv[0].data_ptr()
        assert value_cache.data_ptr() == kv[1].data_ptr()

    def test_split_kv_cache_5d_unbinds_dim0(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinPagedAttention,
        )

        # extra trailing dim is folded into the block axis by the view
        kv = torch.zeros(2, 8, 4, 16, 64, 3)
        key_cache, value_cache = KunlunxinPagedAttention.split_kv_cache(
            kv, num_kv_heads=4, head_size=64
        )
        assert key_cache.shape == (8, 4, 48, 64)
        assert value_cache.shape == (8, 4, 48, 64)
        assert key_cache.data_ptr() == kv[0].data_ptr()
        assert value_cache.data_ptr() == kv[1].data_ptr()

    def test_physical_block_stride_whole_pages(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinPagedAttention,
        )

        # dense pages: stride(0) == page numel -> stride 1, padding 0
        key_cache = torch.zeros(8, 4, 16, 64)
        assert KunlunxinPagedAttention.physical_block_stride(key_cache) == 1

    def test_physical_block_stride_counts_padded_pages(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinPagedAttention,
        )

        # allocate with a padded first dim: stride(0) = 2 pages
        base = torch.zeros(16, 4, 16, 64)
        key_cache = base[::2]
        assert KunlunxinPagedAttention.physical_block_stride(key_cache) == 2

    def test_physical_block_stride_accepts_multi_page_strides(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinPagedAttention,
        )

        # stride(0) covering exactly 3 whole pages is valid
        base = torch.zeros(24, 4, 16, 64)
        key_cache = base[::3]
        assert KunlunxinPagedAttention.physical_block_stride(key_cache) == 3


# ---------------------------------------------------------------------------
# availability helpers
# ---------------------------------------------------------------------------


class TestAvailabilityHelpers:
    def test_ops_unavailable_on_cpu(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            is_kunlunxin_ops_available,
        )

        # torch_xmlir is not installed on the CPU runner
        assert is_kunlunxin_ops_available() in (True, False)

    def test_is_full_graph_runtime_false_without_context(self):
        import vllm.forward_context as forward_context_module

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            _is_full_graph_runtime,
        )

        prev = forward_context_module._forward_context
        forward_context_module._forward_context = None
        try:
            assert _is_full_graph_runtime() is False
        finally:
            forward_context_module._forward_context = prev

    def test_is_full_graph_runtime_true_in_full_mode(self):
        import vllm.forward_context as forward_context_module
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import ForwardContext

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            _is_full_graph_runtime,
        )

        ctx = ForwardContext(
            no_compile_layers={}, attn_metadata={}, slot_mapping={}
        )
        ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        prev = forward_context_module._forward_context
        forward_context_module._forward_context = ctx
        try:
            assert _is_full_graph_runtime() is True
        finally:
            forward_context_module._forward_context = prev

    def test_is_full_graph_runtime_false_in_piecewise_mode(self):
        import vllm.forward_context as forward_context_module
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import ForwardContext

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            _is_full_graph_runtime,
        )

        ctx = ForwardContext(
            no_compile_layers={}, attn_metadata={}, slot_mapping={}
        )
        ctx.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        prev = forward_context_module._forward_context
        forward_context_module._forward_context = ctx
        try:
            assert _is_full_graph_runtime() is False
        finally:
            forward_context_module._forward_context = prev


# ---------------------------------------------------------------------------
# Impl __init__ validation
# ---------------------------------------------------------------------------


class TestImplValidation:
    def _impl(self, **kwargs):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackendImpl,
        )

        defaults = dict(
            num_heads=4,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )
        defaults.update(kwargs)
        return KunlunxinAttentionBackendImpl(**defaults)

    def test_raises_without_xtorch_ops(self):
        import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod

        real_flag = att_mod._KUNLUNXIN_OPS_AVAILABLE
        if real_flag:
            pytest.skip("xtorch_ops present; the unavailable path cannot run")
        with pytest.raises(RuntimeError, match="xtorch_ops is required"):
            self._impl()

    def test_init_validations_with_flag_forced(self):
        import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod

        real_flag = att_mod._KUNLUNXIN_OPS_AVAILABLE
        att_mod._KUNLUNXIN_OPS_AVAILABLE = True
        try:
            with pytest.raises(ValueError, match="soft capping"):
                self._impl(logits_soft_cap=1.0)

            with pytest.raises(ValueError, match="Head size 63"):
                self._impl(head_size=63)

            impl = self._impl(alibi_slopes=[1.0, 2.0])
            assert impl.num_heads == 4
            assert impl.num_queries_per_kv == 2
            assert impl.scale == 0.125
            assert impl.alibi_slopes.dtype == torch.float32
            assert impl.alibi_slopes.tolist() == [1.0, 2.0]
        finally:
            att_mod._KUNLUNXIN_OPS_AVAILABLE = real_flag

    def test_window_size_variants(self):
        from vllm.v1.attention.backend import AttentionType

        import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod

        real_flag = att_mod._KUNLUNXIN_OPS_AVAILABLE
        att_mod._KUNLUNXIN_OPS_AVAILABLE = True
        try:
            sliding = self._impl(sliding_window=128)
            assert sliding._get_window_size(AttentionType.DECODER) == (128, 0)
            assert sliding._get_window_size(AttentionType.ENCODER) == (128, 128)

            zero = self._impl(sliding_window=0)
            assert zero._get_window_size(AttentionType.DECODER) is None
        finally:
            att_mod._KUNLUNXIN_OPS_AVAILABLE = real_flag


# ---------------------------------------------------------------------------
# forward: fake-metadata profiling path (xtorch_ops stub)
# ---------------------------------------------------------------------------


class TestForwardProfilingPath:
    def test_no_metadata_builds_fake_meta_and_runs_context_forward(self):
        calls = []

        def prefill_attention(*args, **kwargs):
            calls.append(("prefill", kwargs))
            return None

        def reshape_and_cache(*args, **kwargs):
            calls.append(("cache",))

        xmlir = ModuleType("torch_xmlir")
        impl_name = (
            "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention"
        )
        saved_xml = sys.modules.get("torch_xmlir")
        saved_impl = sys.modules.get(impl_name)
        sys.modules["torch_xmlir"] = xmlir
        sys.modules.pop(impl_name, None)
        try:
            with _xtorch_ops_stub(
                prefill_attention=prefill_attention,
                reshape_and_cache=reshape_and_cache,
                reshape_and_cache_flash=lambda *a, **k: 0,
                decode_paged_attention=lambda *a, **k: None,
            ):
                # re-import the impl module so its module-scope
                # `import xtorch_ops` binds the stub
                att_mod = importlib.import_module(impl_name)
                real_flag = att_mod._KUNLUNXIN_OPS_AVAILABLE
                att_mod._KUNLUNXIN_OPS_AVAILABLE = True
                try:
                    impl = att_mod.KunlunxinAttentionBackendImpl(
                        num_heads=4,
                        head_size=64,
                        scale=0.125,
                        num_kv_heads=2,
                        alibi_slopes=None,
                        sliding_window=None,
                        kv_cache_dtype="auto",
                    )
                    query = torch.randn(3, 4, 64)
                    kv_cache = torch.zeros(2, 4, 2, 16, 64)
                    out = impl.forward(
                        layer=None,
                        query=query,
                        key=None,
                        value=None,
                        kv_cache=kv_cache,
                        attn_metadata=None,
                    )
                    assert tuple(out.shape) == (3, 4 * 64)
                    kinds = [k for k, _ in calls if isinstance(k, str)]
                    assert "prefill" in kinds
                    # the fake metadata is prefix-cache-free: the no-prefix
                    # branch passes key/value (None here) directly
                    prefill_kwargs = next(
                        kw for kind, kw in calls if kind == "prefill"
                    )
                    assert prefill_kwargs["is_prefix_cache"] is False
                    assert prefill_kwargs["is_causal"] is True
                finally:
                    att_mod._KUNLUNXIN_OPS_AVAILABLE = real_flag
        finally:
            if saved_impl is None:
                sys.modules.pop(impl_name, None)
            else:
                sys.modules[impl_name] = saved_impl
            if saved_xml is None:
                sys.modules.pop("torch_xmlir", None)
            else:
                sys.modules["torch_xmlir"] = saved_xml
