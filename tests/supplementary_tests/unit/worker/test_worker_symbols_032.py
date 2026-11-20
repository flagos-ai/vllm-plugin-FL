# Copyright (c) 2025 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl.patches.worker_symbols_032``.

The module re-publishes vLLM v0.28 worker helpers that vLLM 0.32 removed,
using "install only when absent" semantics so a vLLM build that still ships
the original keeps it.  These tests drive both directions against fake vllm
module trees installed in ``sys.modules`` (the external boundary), and test
the restored helpers' behaviour directly on CPU tensors:

* ``is_residual_scattered_for_sp`` SP predicate and its assertions;
* ``supports_xdrope`` / ``SupportsXDRoPE`` capability probe;
* ``XDRotaryEmbedding`` position writing;
* ``_reshape_attention_kv_cache`` view construction (contiguous, packed,
  padded-pages, page-aligned-blocks, permuted) and its layout assertions;
* ``RoutedExpertsLists`` / ``RoutedExpertsTensors`` carriers.
"""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_fl.patches.worker_symbols_032 as ws

pytestmark = pytest.mark.supplementary


class _VllmTree:
    """Install/restore the fake ``vllm`` modules the patch targets."""

    def __init__(self):
        self._installed = []
        self._previous = {}

        def put(name, **attrs):
            module = ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            self._previous[name] = sys.modules.get(name)
            sys.modules[name] = module
            self._installed.append(name)
            return module

        put("vllm")
        put("vllm.v1")
        self.worker_utils = put("vllm.v1.worker.utils")
        put("vllm.v1.worker")
        put("vllm.v1.worker.gpu")
        self.attn_utils = put("vllm.v1.worker.gpu.attn_utils")
        self.outputs = put("vllm.v1.outputs")
        put("vllm.model_executor")
        put("vllm.model_executor.layers")
        self.rotary_embedding = put("vllm.model_executor.layers.rotary_embedding")
        put("vllm.model_executor.models")
        self.interfaces = put("vllm.model_executor.models.interfaces")

    def restore(self):
        """Undo every installed sys.modules entry."""
        for name in self._installed:
            previous = self._previous[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        self._installed.clear()
        self._previous.clear()


@pytest.fixture
def vllm_tree(fl_sys_modules):
    """Install a fresh fake vllm tree; restore after the test."""
    tree = _VllmTree()
    yield tree
    tree.restore()


def _make_vllm_config(
    enable_sp,
    tensor_parallel_size,
    use_inductor_graph_partition=False,
    splitting_ops=(),
):
    """Build the vllm_config shape ``_is_residual_scattered_for_sp`` reads."""
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp),
            use_inductor_graph_partition=use_inductor_graph_partition,
            splitting_ops=list(splitting_ops),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tensor_parallel_size),
    )


def _kv_spec(dtype=torch.int32, page_size_padded=None, page_size_bytes=None):
    """Build the kv_cache_spec shape the KV view builder reads."""
    return SimpleNamespace(
        dtype=dtype,
        page_size_padded=page_size_padded,
        page_size_bytes=page_size_bytes,
    )


class TestApplyWorkerSymbols:
    """The publish-on-missing installer itself."""

    def test_missing_names_are_installed(self, vllm_tree):
        """Absent names receive the plugin-provided equivalents."""
        ws.apply_worker_symbols_032.cache_clear()
        ws.apply_worker_symbols_032()
        assert (
            vllm_tree.worker_utils.is_residual_scattered_for_sp
            is ws._is_residual_scattered_for_sp
        )
        assert vllm_tree.rotary_embedding.XDRotaryEmbedding is ws._XDRotaryEmbedding
        assert vllm_tree.interfaces.SupportsXDRoPE is ws._SupportsXDRoPE
        assert vllm_tree.interfaces.supports_xdrope is ws._supports_xdrope
        assert (
            vllm_tree.attn_utils._reshape_attention_kv_cache
            is ws._reshape_attention_kv_cache
        )
        assert vllm_tree.outputs.RoutedExpertsLists is ws._RoutedExpertsLists
        assert vllm_tree.outputs.RoutedExpertsTensors is ws._RoutedExpertsTensors

    def test_existing_names_are_left_alone(self, vllm_tree):
        """A name the installed vLLM already provides is not overwritten."""
        ws.apply_worker_symbols_032.cache_clear()
        sentinel_sp = object()
        sentinel_xd = object()
        sentinel_lists = object()
        vllm_tree.worker_utils.is_residual_scattered_for_sp = sentinel_sp
        vllm_tree.rotary_embedding.XDRotaryEmbedding = sentinel_xd
        vllm_tree.outputs.RoutedExpertsLists = sentinel_lists
        ws.apply_worker_symbols_032()
        assert vllm_tree.worker_utils.is_residual_scattered_for_sp is sentinel_sp
        assert vllm_tree.rotary_embedding.XDRotaryEmbedding is sentinel_xd
        assert vllm_tree.outputs.RoutedExpertsLists is sentinel_lists
        # Untouched names still get installed.
        assert vllm_tree.interfaces.supports_xdrope is ws._supports_xdrope

    def test_apply_is_cached(self, vllm_tree):
        """The second call short-circuits through lru_cache."""
        ws.apply_worker_symbols_032.cache_clear()
        assert ws.apply_worker_symbols_032() is True
        first_hits = ws.apply_worker_symbols_032.cache_info().hits
        ws.apply_worker_symbols_032()
        assert ws.apply_worker_symbols_032.cache_info().hits == first_hits + 1


class TestIsResidualScatteredForSp:
    """SP predicate ported from vLLM v0.28."""

    def test_false_when_sp_disabled(self):
        """Without sequence parallelism the residual is not scattered."""
        cfg = _make_vllm_config(enable_sp=False, tensor_parallel_size=8)
        assert ws._is_residual_scattered_for_sp(cfg, 16) is False

    def test_false_when_tensor_parallel_size_one(self):
        """SP with TP=1 has nothing to scatter across."""
        cfg = _make_vllm_config(enable_sp=True, tensor_parallel_size=1)
        assert ws._is_residual_scattered_for_sp(cfg, 4) is False

    def test_true_for_full_graph_sp_tp_gt_1(self):
        """SP + TP>1 + full-graph compilation scatters the residual."""
        cfg = _make_vllm_config(enable_sp=True, tensor_parallel_size=2)
        assert ws._is_residual_scattered_for_sp(cfg, 4) is True

    def test_true_with_inductor_graph_partition(self):
        """use_inductor_graph_partition also satisfies full-graph."""
        cfg = _make_vllm_config(
            enable_sp=True,
            tensor_parallel_size=2,
            use_inductor_graph_partition=True,
            splitting_ops=["aten::op"],
        )
        assert ws._is_residual_scattered_for_sp(cfg, 4) is True

    def test_asserts_when_splitting_ops_without_full_graph(self):
        """SP with splitting ops and no full-graph fails an assertion."""
        cfg = _make_vllm_config(
            enable_sp=True, tensor_parallel_size=2, splitting_ops=["aten::op"]
        )
        with pytest.raises(AssertionError, match="full-graph compilation"):
            ws._is_residual_scattered_for_sp(cfg, 4)

    def test_asserts_when_tokens_not_padded_to_tp(self):
        """num_input_tokens must be padded to a multiple of TP size."""
        cfg = _make_vllm_config(enable_sp=True, tensor_parallel_size=2)
        with pytest.raises(AssertionError):
            ws._is_residual_scattered_for_sp(cfg, 3)


class TestXDRoPE:
    """XDRoPE capability probe and position helper."""

    def test_supports_xdrope_detects_position_hook(self):
        """The probe checks for get_xdrope_input_positions."""

        class WithHook:
            def get_xdrope_input_positions(self, *args, **kwargs):
                return None

        assert ws._supports_xdrope(WithHook()) is True
        assert ws._supports_xdrope(object()) is False

    def test_protocol_instance_check_matches_probe(self):
        """The runtime-checkable Protocol agrees with supports_xdrope."""

        class WithHook:
            def get_xdrope_input_positions(self, *args, **kwargs):
                return None

        assert isinstance(WithHook(), ws._SupportsXDRoPE)
        assert not isinstance(object(), ws._SupportsXDRoPE)

    def test_next_input_positions_advance_with_context_len(self):
        """Positions are context_len..context_len+num_new_tokens-1."""
        buffer = np.zeros((2, 6), dtype=np.int64)
        ws._XDRotaryEmbedding.get_next_input_positions_tensor(
            buffer, out_offset=1, context_len=7, num_new_tokens=3
        )
        assert buffer[0, 1:4].tolist() == [7, 8, 9]
        assert buffer[1, 1:4].tolist() == [7, 8, 9]
        # Outside the written window stays untouched.
        assert buffer[0, 0] == 0 and buffer[0, 4] == 0


class TestReshapeAttentionKvCache:
    """KV-cache view construction for every supported layout."""

    def test_contiguous_view_preserves_values(self):
        """Default layout yields a plain contiguous typed view."""
        raw = torch.arange(24, dtype=torch.int32)
        view = ws._reshape_attention_kv_cache(
            raw, _kv_spec(), (2, 2, 6), (0, 1, 2), num_blocks=2, packing=None
        )
        assert tuple(view.shape) == (2, 2, 6)
        assert view.stride() == (12, 6, 1)
        assert torch.equal(view.flatten(), raw)

    def test_packing_branch_selects_offset_within_blocks(self):
        """Packed views slice (offset, block_stride) out of the raw tensor.

        The packing tuple is in *elements of the raw dtype*: the view takes
        ``raw.view(-1, block_stride)[:, offset : offset + page_bytes]``, so
        with a uint8 raw tensor and int32 storage, each packed block reads
        ``block_stride`` bytes starting at byte ``offset``.  Asserting through
        the resulting int32 words verifies both the offset and the
        block-stride handling.
        """
        raw = torch.arange(128, dtype=torch.uint8)
        view = ws._reshape_attention_kv_cache(
            raw,
            _kv_spec(),
            (2, 3, 4),
            (0, 1, 2),
            num_blocks=3,
            packing=(8, 64),
        )
        assert tuple(view.shape) == (2, 3, 4)
        # page_bytes = prod(shape[1:]) * 4 = 48; each row r covers raw bytes
        # [64r + 8, 64r + 56); each int32 groups four consecutive bytes.
        assert view[0, 0, 0].item() == 0x0B0A0908
        assert view[0, 0, 1].item() == 0x0F0E0D0C
        assert view[0, 1, 0].item() == 0x1B1A1918
        assert view[1, 0, 0].item() == 0x4B4A4948
        assert view[1, 2, 0].item() == 0x6B6A6968

    def test_padded_pages_skip_padding_between_pages(self):
        """page_size_padded builds a strided view skipping pad bytes."""
        spec = _kv_spec(page_size_padded=3, page_size_bytes=32)
        raw = torch.arange(64, dtype=torch.int32)
        view = ws._reshape_attention_kv_cache(
            raw, spec, (2, 2, 4), (0, 1, 2), num_blocks=2, packing=None
        )
        assert tuple(view.shape) == (2, 2, 4)
        # page_stride = 32 bytes / 4 = 8 int32 elements per physical page.
        assert view.stride() == (8, 4, 1)
        assert view[0, 0].tolist() == [0, 1, 2, 3]
        assert view[0, 1].tolist() == [4, 5, 6, 7]
        assert view[1, 1].tolist() == [12, 13, 14, 15]

    def test_padded_pages_rejects_kv_first_layout(self):
        """Padded pages with a kv-first layout fail the layout assertion."""
        spec = _kv_spec(page_size_padded=3, page_size_bytes=32)
        raw = torch.arange(64, dtype=torch.int32)
        with pytest.raises(AssertionError, match="num-blocks-first"):
            ws._reshape_attention_kv_cache(
                raw, spec, (2, 2, 4), (0, 1, 2), num_blocks=3, packing=None
            )

    def test_page_aligned_blocks_swap_kv_and_block_dims(self):
        """ROCm-style (kv, num_blocks, ...) layouts are made page-first."""
        raw = torch.arange(24, dtype=torch.int32)
        view = ws._reshape_attention_kv_cache(
            raw,
            _kv_spec(),
            (2, 2, 6),
            (0, 1, 2),
            num_blocks=2,
            packing=None,
            page_aligned_blocks=True,
        )
        assert tuple(view.shape) == (2, 2, 6)
        # Block b's K/V halves come from page b: K[0]=raw page 0, V[0]=raw
        # page 1, K[1]=raw page 2, V[1]=raw page 3 (page = 6 elements).
        assert view[0, 0].tolist() == [0, 1, 2, 3, 4, 5]
        assert view[1, 0].tolist() == [6, 7, 8, 9, 10, 11]
        assert view[0, 1].tolist() == [12, 13, 14, 15, 16, 17]
        assert view[1, 1].tolist() == [18, 19, 20, 21, 22, 23]

    def test_page_aligned_blocks_require_default_stride_order(self):
        """A non-default stride order fails the page-aligned assertion."""
        raw = torch.arange(24, dtype=torch.int32)
        with pytest.raises(AssertionError, match="default-strided"):
            ws._reshape_attention_kv_cache(
                raw,
                _kv_spec(),
                (2, 2, 6),
                (2, 0, 1),
                num_blocks=2,
                packing=None,
                page_aligned_blocks=True,
            )

    def test_stride_order_permutation_round_trips(self):
        """A permuted stride order yields the permuted logical shape."""
        raw = torch.arange(24, dtype=torch.int32)
        view = ws._reshape_attention_kv_cache(
            raw, _kv_spec(), (2, 2, 6), (2, 0, 1), num_blocks=2, packing=None
        )
        # Permute back to logical order and compare with the raw values.
        assert tuple(view.shape) == (2, 2, 6)
        back = view.permute(2, 0, 1)
        assert torch.equal(back.reshape(-1), raw.reshape(-1))


class TestRoutedExpertsCarriers:
    """The routed-experts output carriers."""

    def test_cpu_tensors_passthrough_nonblocking(self):
        """CPU tensors return the same carrier unchanged."""
        routing = torch.arange(12).reshape(3, 2, 2)
        slots = torch.arange(3)
        carrier = ws._RoutedExpertsTensors(routing, slots)
        assert carrier.to_cpu_nonblocking() is carrier

    def test_tolists_converts_to_numpy_backed_lists(self):
        """tolists moves tensors to CPU numpy arrays."""
        routing = torch.arange(12).reshape(3, 2, 2)
        slots = torch.arange(3)
        carrier = ws._RoutedExpertsTensors(routing, slots)
        lists = carrier.tolists()
        assert isinstance(lists, ws._RoutedExpertsLists)
        assert isinstance(lists.routing_data, np.ndarray)
        assert np.array_equal(lists.routing_data, routing.numpy())
        assert np.array_equal(lists.slot_mapping, slots.numpy())

    def test_tolist_field_names(self):
        """The NamedTuple fields keep their documented names."""
        assert ws._RoutedExpertsLists._fields == ("routing_data", "slot_mapping")
        assert ws._RoutedExpertsTensors._fields == ("routing_data", "slot_mapping")
