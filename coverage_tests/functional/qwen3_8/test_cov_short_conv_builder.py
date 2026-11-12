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

"""PLE short-conv metadata-builder construction and speculative-decode paths.

The inherited Mamba builder is substituted with a recording double; the test
metadata, block tables and request boundaries are explicit, so every expected
count/slice follows from the documented [spec | decode | prefill] grouping.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.common import short_conv_attn as sc


def _cudagraph_mode(full):
    return SimpleNamespace(has_full_cudagraphs=lambda: full)


def _builder_config(*, full_graph=False, capture_size=None, num_spec=1):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec),
        compilation_config=SimpleNamespace(
            cudagraph_mode=_cudagraph_mode(full_graph),
            max_cudagraph_capture_size=capture_size,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
    )


def _new_builder(monkeypatch, *, full_graph=False, capture_size=None, num_spec=1):
    config = _builder_config(
        full_graph=full_graph, capture_size=capture_size, num_spec=num_spec
    )

    def fake_base_init(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens
        self.compilation_config = vllm_config.compilation_config

    monkeypatch.setattr(
        BaseMambaAttentionMetadataBuilder, "__init__", fake_base_init
    )
    builder = sc.PleShortConvAttentionMetadataBuilder(
        kv_cache_spec="spec",
        layer_names=["p"],
        vllm_config=config,
        device=torch.device("cpu"),
    )
    return builder


class TestBuilderConstruction:
    def test_capture_geometry_and_persistent_buffers(self, monkeypatch):
        builder = _new_builder(monkeypatch, num_spec=2)
        assert builder.num_spec == 2
        assert builder.use_full_cuda_graph is False
        assert builder.decode_cudagraph_max_bs == 4
        assert builder.decode_cudagraph_max_tokens == 12
        assert builder.spec_state_indices_tensor.shape == (4,)
        assert builder.spec_state_indices_tensor.dtype is torch.int32
        assert builder.spec_sequence_masks.shape == (4,)
        assert builder.spec_sequence_masks.dtype is torch.bool
        assert builder.spec_token_indx.shape == (12,)
        assert builder.non_spec_token_indx.shape == (12,)
        assert builder.spec_query_start_loc.shape == (5,)
        assert builder.num_accepted_tokens.shape == (4,)
        assert builder.has_initial_states_d.shape == (4,)
        assert builder.has_initial_states_d.dtype is torch.bool

    def test_capture_size_caps_requests_and_tokens(self, monkeypatch):
        builder = _new_builder(monkeypatch, capture_size=5, num_spec=2)
        assert builder.decode_cudagraph_max_bs == 4
        assert builder.decode_cudagraph_max_tokens == 5
        builder = _new_builder(monkeypatch, capture_size=2, num_spec=2)
        assert builder.decode_cudagraph_max_bs == 2
        assert builder.decode_cudagraph_max_tokens == 2

    def test_full_graph_flag_comes_from_the_compilation_config(self, monkeypatch):
        builder = _new_builder(monkeypatch, full_graph=True)
        assert builder.use_full_cuda_graph is True


class _Common:
    def __init__(
        self,
        *,
        query_start_loc,
        block_table,
        seq_lens,
        num_actual_tokens,
        computed,
        num_reqs=None,
    ):
        self.query_start_loc = query_start_loc
        self.query_start_loc_cpu = query_start_loc.clone()
        self.block_table_tensor = block_table
        self.seq_lens = seq_lens
        self.num_actual_tokens = num_actual_tokens
        self.num_reqs = (
            num_reqs if num_reqs is not None else query_start_loc.numel() - 1
        )
        self._computed = computed

    def compute_num_computed_tokens(self):
        return self._computed

    def replace(self, **changes):
        values = dict(
            query_start_loc=self.query_start_loc,
            block_table_tensor=self.block_table_tensor,
            seq_lens=self.seq_lens,
            num_actual_tokens=self.num_actual_tokens,
            num_reqs=self.num_reqs,
            computed=self._computed,
        )
        values.update(changes)
        return SimpleNamespace(**values)


def _spec_builder(monkeypatch, **kwargs):
    builder = _new_builder(monkeypatch, **kwargs)
    builder.use_spec_decode = True
    builder.vllm_config = _builder_config(**kwargs)
    return builder


class TestSpeculativeBuild:
    def test_pure_spec_batch_uses_every_token_as_spec(self, monkeypatch):
        builder = _spec_builder(monkeypatch, num_spec=1)
        table = torch.tensor(
            [[10, 0, 0], [20, 0, 0]], dtype=torch.int32
        )
        monkeypatch.setattr(
            sc, "mamba_get_block_table_tensor", lambda *args: table
        )
        common = _Common(
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
            block_table=table,
            seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            num_actual_tokens=4,
            computed=torch.tensor([0, 0], dtype=torch.int32),
        )
        metadata = builder.build(
            0,
            common,
            num_accepted_tokens=torch.tensor([2, 1], dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.tensor([1, 0]),
        )
        assert metadata.num_spec_decodes == 2
        assert metadata.num_spec_decode_tokens == 4
        assert metadata.num_decodes == 0
        assert metadata.num_prefills == 0
        assert metadata.num_decode_tokens == 0
        assert metadata.num_prefill_tokens == 0
        assert metadata.spec_token_indx.tolist() == [0, 1, 2, 3]
        assert metadata.non_spec_token_indx.numel() == 0
        assert metadata.spec_state_indices_tensor.tolist() == [10, 20]
        assert metadata.state_indices_tensor.numel() == 0
        assert metadata.spec_query_start_loc.tolist() == [0, 2, 4]
        assert metadata.non_spec_query_start_loc is None
        assert metadata.num_accepted_tokens.tolist() == [2, 1]
        assert metadata.spec_query_len == 2
        assert metadata.max_prefill_query_len == 0
        assert metadata.num_actual_tokens == 4
        assert metadata.seq_lens is common.seq_lens

    def test_accepted_counts_follow_the_spec_request_order(self, monkeypatch):
        builder = _spec_builder(monkeypatch, num_spec=1)
        table = torch.tensor([[10, 0], [20, 0], [30, 0]], dtype=torch.int32)
        monkeypatch.setattr(
            sc, "mamba_get_block_table_tensor", lambda *args: table
        )
        common = _Common(
            query_start_loc=torch.tensor([0, 2, 3, 4], dtype=torch.int32),
            block_table=table,
            seq_lens=torch.tensor([2, 1, 1], dtype=torch.int32),
            num_actual_tokens=4,
            computed=torch.tensor([0, 0, 0], dtype=torch.int32),
        )
        metadata = builder.build(
            0,
            common,
            num_accepted_tokens=torch.tensor([1, 2, 3], dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.tensor([1, -1, 0]),
        )
        # Requests 0 and 2 are speculative; accepted counts are gathered.
        assert metadata.num_spec_decodes == 2
        assert metadata.spec_state_indices_tensor.tolist() == [10, 30]
        assert metadata.num_accepted_tokens.tolist() == [1, 3]
        assert metadata.spec_query_start_loc.tolist() == [0, 2, 3]
        assert metadata.spec_sequence_masks.tolist() == [True, False, True]

    def test_mixed_batch_groups_spec_then_decode_then_prefill(
        self, monkeypatch
    ):
        builder = _spec_builder(monkeypatch, num_spec=1)
        table = torch.tensor(
            [[10, 0], [20, 0], [30, 0]], dtype=torch.int32
        )
        monkeypatch.setattr(
            sc, "mamba_get_block_table_tensor", lambda *args: table
        )
        common = _Common(
            query_start_loc=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
            block_table=table,
            seq_lens=torch.tensor([2, 1, 3], dtype=torch.int32),
            num_actual_tokens=6,
            computed=torch.tensor([0, 1, 0], dtype=torch.int32),
        )
        metadata = builder.build(
            0,
            common,
            num_accepted_tokens=torch.tensor([2, 1, 1], dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.tensor([1, -1, -1]),
        )
        assert metadata.num_spec_decodes == 1
        assert metadata.num_decodes == 1
        assert metadata.num_prefills == 1
        assert metadata.num_decode_tokens == 1
        assert metadata.num_prefill_tokens == 3
        assert metadata.num_spec_decode_tokens == 2
        assert metadata.spec_token_indx.tolist() == [0, 1]
        assert metadata.non_spec_token_indx.tolist() == [2, 3, 4, 5]
        assert metadata.spec_state_indices_tensor.tolist() == [10]
        assert metadata.state_indices_tensor.tolist() == [20, 30]
        assert metadata.state_indices_tensor_d.tolist() == [20]
        assert metadata.state_indices_tensor_p.tolist() == [30]
        assert metadata.spec_query_start_loc.tolist() == [0, 2]
        assert metadata.non_spec_query_start_loc.tolist() == [0, 1, 4]
        assert metadata.query_start_loc_d.tolist() == [0, 1]
        assert metadata.query_start_loc_p.tolist() == [0, 3]
        assert metadata.has_initial_states_d.tolist() == [True]
        assert metadata.has_initial_states_p.tolist() == [False]
        assert metadata.max_prefill_query_len == 3
        assert metadata.nums_dict is None
        assert metadata.batch_ptr is None
        assert metadata.num_accepted_tokens.tolist() == [2]

    def test_full_graph_spec_batch_aliases_the_persistent_buffers(
        self, monkeypatch
    ):
        builder = _spec_builder(monkeypatch, full_graph=True, num_spec=1)
        table = torch.tensor([[10, 0], [20, 0]], dtype=torch.int32)
        monkeypatch.setattr(
            sc, "mamba_get_block_table_tensor", lambda *args: table
        )
        common = _Common(
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
            block_table=table,
            seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            num_actual_tokens=4,
            computed=torch.tensor([0, 0], dtype=torch.int32),
        )
        metadata = builder.build(
            0,
            common,
            num_accepted_tokens=torch.tensor([2, 1], dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.tensor([1, 0]),
        )
        assert (
            metadata.spec_state_indices_tensor.data_ptr()
            == builder.spec_state_indices_tensor.data_ptr()
        )
        assert (
            metadata.spec_sequence_masks.data_ptr()
            == builder.spec_sequence_masks.data_ptr()
        )
        assert (
            metadata.spec_query_start_loc.data_ptr()
            == builder.spec_query_start_loc.data_ptr()
        )
        assert (
            metadata.num_accepted_tokens.data_ptr()
            == builder.num_accepted_tokens.data_ptr()
        )
        assert metadata.spec_state_indices_tensor.tolist() == [10, 20]
        assert metadata.spec_sequence_masks.tolist() == [True, True]
        assert metadata.num_accepted_tokens.tolist() == [2, 1]

    def test_missing_accepted_tokens_is_rejected(self, monkeypatch):
        builder = _spec_builder(monkeypatch)
        table = torch.tensor([[10, 0]], dtype=torch.int32)
        monkeypatch.setattr(
            sc, "mamba_get_block_table_tensor", lambda *args: table
        )
        common = _Common(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            block_table=table,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            num_actual_tokens=1,
            computed=torch.tensor([0], dtype=torch.int32),
        )
        with pytest.raises(AssertionError):
            builder.build(
                0,
                common,
                num_decode_draft_tokens_cpu=torch.tensor([0]),
            )


class TestNonSpecOneDimensionalStateIndices:
    def test_decode_only_one_dimensional_indices_are_kept(
        self, monkeypatch
    ):
        builder = _new_builder(monkeypatch)
        builder.use_spec_decode = False
        metadata = object.__new__(sc.PleShortConvAttentionMetadata)
        metadata.num_prefills = 0
        metadata.num_decodes = 2
        metadata.num_decode_tokens = 2
        metadata.state_indices_tensor_d = torch.tensor([7, 8], dtype=torch.int32)
        metadata.state_indices_tensor_p = None

        def fake_base(self, *args, **kwargs):
            return metadata

        monkeypatch.setattr(BaseMambaAttentionMetadataBuilder, "build", fake_base)
        captured = {}
        monkeypatch.setattr(
            sc,
            "replace",
            lambda instance, **changes: captured.update(changes) or "metadata",
        )
        common = SimpleNamespace(
            num_actual_tokens=2,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            compute_num_computed_tokens=lambda: torch.tensor([0, 4]),
        )
        assert (
            builder._build_non_spec_metadata(0, common, False, None)
            == "metadata"
        )
        assert captured["state_indices_tensor"].tolist() == [7, 8]
        assert captured["has_initial_states_d"].tolist() == [False, True]
