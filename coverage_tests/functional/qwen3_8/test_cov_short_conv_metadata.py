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

"""PLE short-conv metadata assembly without a live model runner.

The builder wraps vLLM's Mamba metadata with spec-decode tensors and
graph-padding buffers. These tests substitute the inherited builder and the
dataclass ``replace`` so the module's own classification and buffer-filling
logic is observed directly; every expected index/flag comes from the
documented decode-then-prefill ordering.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.common import short_conv_attn as sc


class TestBackendStatics:
    def test_short_conv_backend_names_and_flags(self):
        assert sc.ShortConvAttentionBackend.get_name() == "SHORT_CONV_ATTN"
        assert (
            sc.ShortConvAttentionBackend.get_builder_cls()
            is sc.ShortConvAttentionMetadataBuilder
        )
        assert sc.ShortConvAttentionBackend.is_ssm() is True

    def test_ple_backend_uses_uniform_batch_graphs(self):
        assert sc.PleShortConvAttentionBackend.get_name() == "PLE_SHORT_CONV_ATTN"
        assert (
            sc.PleShortConvAttentionBackend.get_builder_cls()
            is sc.PleShortConvAttentionMetadataBuilder
        )
        builder = sc.PleShortConvAttentionMetadataBuilder
        assert builder._cudagraph_support is AttentionCGSupport.UNIFORM_BATCH
        assert builder.reorder_batch_threshold == 1
        assert builder.supports_update_block_table is False

    def test_metadata_inherits_the_generic_short_conv_metadata(self):
        assert issubclass(
            sc.PleShortConvAttentionMetadata, sc.ShortConvAttentionMetadata
        )
        fields = sc.PleShortConvAttentionMetadata.__dataclass_fields__
        assert fields["num_spec_decodes"].default == 0
        assert fields["num_spec_decode_tokens"].default == 0
        assert fields["num_actual_tokens"].default == 0
        assert fields["spec_query_len"].default == 1
        assert fields["max_prefill_query_len"].default == 0


def _builder(**overrides):
    builder = object.__new__(sc.PleShortConvAttentionMetadataBuilder)
    builder.decode_cudagraph_max_bs = 4
    builder.decode_cudagraph_max_tokens = 8
    builder.use_spec_decode = False
    builder.use_full_cuda_graph = False
    builder.num_spec = 2
    builder.has_initial_states_d = torch.zeros(4, dtype=torch.bool)
    for name, value in overrides.items():
        setattr(builder, name, value)
    return builder


class TestBuildForCudagraphCapture:
    def test_decode_only_capture_calls_build_without_spec_fields(self, monkeypatch):
        calls = []

        def fake_build(self, prefix, common, fast_build=False, **kwargs):
            calls.append((prefix, common, fast_build, kwargs))
            return "metadata"

        monkeypatch.setattr(
            sc.PleShortConvAttentionMetadataBuilder, "build", fake_build
        )
        builder = _builder(use_spec_decode=False)
        common = SimpleNamespace(
            num_reqs=3,
            num_actual_tokens=3,
            query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )
        assert builder.build_for_cudagraph_capture(common) == "metadata"
        assert calls == [(0, common, False, {})]

    def test_spec_capture_derives_accepted_and_draft_counts(self, monkeypatch):
        calls = []

        def fake_build(self, prefix, common, fast_build=False, **kwargs):
            calls.append((prefix, common, kwargs))
            return "metadata"

        monkeypatch.setattr(
            sc.PleShortConvAttentionMetadataBuilder, "build", fake_build
        )
        builder = _builder(use_spec_decode=True)
        common = SimpleNamespace(
            num_reqs=2,
            num_actual_tokens=5,
            query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        )
        assert builder.build_for_cudagraph_capture(common) == "metadata"
        _, _, kwargs = calls[0]
        assert kwargs["num_accepted_tokens"].tolist() == [2, 3]
        assert kwargs["num_decode_draft_tokens_cpu"].tolist() == [1, 2]

    def test_over_capture_sized_batches_are_rejected(self):
        builder = _builder()
        too_many_reqs = SimpleNamespace(
            num_reqs=5,
            num_actual_tokens=5,
            query_start_loc=torch.zeros(6, dtype=torch.int32),
        )
        with pytest.raises(AssertionError, match="ShortConv only supports"):
            builder.build_for_cudagraph_capture(too_many_reqs)
        too_many_tokens = SimpleNamespace(
            num_reqs=1,
            num_actual_tokens=9,
            query_start_loc=torch.zeros(2, dtype=torch.int32),
        )
        with pytest.raises(AssertionError, match="token capture size"):
            builder.build_for_cudagraph_capture(too_many_tokens)


class TestBuildRouting:
    def test_non_spec_decode_route_forwards_to_non_spec_builder(self, monkeypatch):
        calls = []

        def fake_non_spec(
            self, prefix, common, fast_build, num_decode_draft_tokens_cpu, **kwargs
        ):
            calls.append(
                (prefix, common, fast_build, num_decode_draft_tokens_cpu, kwargs)
            )
            return "metadata"

        monkeypatch.setattr(
            sc.PleShortConvAttentionMetadataBuilder,
            "_build_non_spec_metadata",
            fake_non_spec,
        )
        builder = _builder(use_spec_decode=False)
        common = SimpleNamespace(num_reqs=2)
        draft = torch.tensor([-1, -1])
        assert builder.build(0, common, False, num_decode_draft_tokens_cpu=draft) == (
            "metadata"
        )
        assert calls == [(0, common, False, draft, {})]

    def test_all_minus_one_draft_counts_stay_on_the_non_spec_route(self, monkeypatch):
        calls = []

        def fake_non_spec(
            self, prefix, common, fast_build, num_decode_draft_tokens_cpu, **kwargs
        ):
            calls.append(num_decode_draft_tokens_cpu)
            return "metadata"

        monkeypatch.setattr(
            sc.PleShortConvAttentionMetadataBuilder,
            "_build_non_spec_metadata",
            fake_non_spec,
        )
        builder = _builder(use_spec_decode=True)
        draft = torch.tensor([-1, -1, 3])
        common = SimpleNamespace(num_reqs=2)
        assert builder.build(0, common, num_decode_draft_tokens_cpu=draft) == "metadata"
        assert calls == [draft]


def _fake_metadata(**overrides):
    metadata = object.__new__(sc.PleShortConvAttentionMetadata)
    metadata.num_prefills = 0
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.num_prefill_tokens = 0
    metadata.state_indices_tensor_d = None
    metadata.state_indices_tensor_p = None
    for name, value in overrides.items():
        setattr(metadata, name, value)
    return metadata


@pytest.fixture()
def captured_build(monkeypatch):
    """Substitute the inherited build and dataclass replace, recording inputs."""
    captured = {}

    def install(metadata):
        def fake_base(
            self, prefix, common, fast_build, num_accepted_tokens=None, **kwargs
        ):
            captured["base_args"] = (prefix, common, fast_build)
            return metadata

        def fake_replace(instance, **changes):
            captured["instance"] = instance
            captured.update(changes)
            return "metadata"

        monkeypatch.setattr(BaseMambaAttentionMetadataBuilder, "build", fake_base)
        monkeypatch.setattr(sc, "replace", fake_replace)

    return captured, install


class TestNonSpecMetadata:
    """Decode/prefill state indices, initial flags and PLE buffer sizing."""

    def test_pure_decode_keeps_padded_state_rows_and_initial_flags(
        self, captured_build
    ):
        captured, install = captured_build
        metadata = _fake_metadata(
            num_decodes=2,
            num_decode_tokens=2,
            state_indices_tensor_d=torch.tensor([[7], [8]], dtype=torch.int32),
        )
        install(metadata)
        common = SimpleNamespace(
            num_actual_tokens=2,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            compute_num_computed_tokens=lambda: torch.tensor([0, 3]),
        )
        builder = _builder()
        result = builder._build_non_spec_metadata(0, common, False, None)
        assert result == "metadata"
        assert captured["state_indices_tensor"].tolist() == [7, 8]
        assert captured["has_initial_states_d"].tolist() == [False, True]
        assert captured["max_prefill_query_len"] == 0
        assert captured["spec_query_len"] == 3
        assert captured["query_start_loc"] is common.query_start_loc
        assert captured["non_spec_query_start_loc"] is common.query_start_loc

    def test_pure_prefill_uses_prefill_indices_and_max_query_length(
        self, captured_build
    ):
        captured, install = captured_build
        metadata = _fake_metadata(
            num_prefills=2,
            num_prefill_tokens=4,
            state_indices_tensor_p=torch.tensor([5, 6, 7], dtype=torch.int32),
        )
        install(metadata)
        common = SimpleNamespace(
            num_actual_tokens=4,
            query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 4], dtype=torch.int32),
            compute_num_computed_tokens=lambda: torch.tensor([1, 1]),
        )
        builder = _builder()
        builder._build_non_spec_metadata(0, common, False, None)
        assert captured["state_indices_tensor"].tolist() == [5, 6]
        assert captured["has_initial_states_d"] is None
        assert captured["max_prefill_query_len"] == 3

    def test_mixed_batch_concatenates_decodes_then_prefills(self, captured_build):
        captured, install = captured_build
        metadata = _fake_metadata(
            num_decodes=1,
            num_prefills=1,
            num_decode_tokens=1,
            num_prefill_tokens=3,
            state_indices_tensor_d=torch.tensor([7], dtype=torch.int32),
            state_indices_tensor_p=torch.tensor([8, 9], dtype=torch.int32),
        )
        install(metadata)
        common = SimpleNamespace(
            num_actual_tokens=4,
            query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 4], dtype=torch.int32),
            compute_num_computed_tokens=lambda: torch.tensor([0, 2]),
        )
        builder = _builder()
        builder._build_non_spec_metadata(0, common, False, None)
        assert captured["state_indices_tensor"].tolist() == [7, 8]
        assert captured["has_initial_states_d"].tolist() == [False]
        assert captured["max_prefill_query_len"] == 3
        assert captured["num_decode_draft_tokens_cpu"] is None

    def test_full_graph_padding_rows_get_null_initial_state(self, captured_build):
        captured, install = captured_build
        metadata = _fake_metadata(
            num_decodes=2,
            num_decode_tokens=2,
            state_indices_tensor_d=torch.tensor([7, 8, -1, -1], dtype=torch.int32),
        )
        install(metadata)
        common = SimpleNamespace(
            num_actual_tokens=2,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            compute_num_computed_tokens=lambda: torch.tensor([0, 3]),
        )
        builder = _builder(use_full_cuda_graph=True)
        builder._build_non_spec_metadata(0, common, False, None)
        flags = captured["has_initial_states_d"]
        assert flags.data_ptr() == builder.has_initial_states_d.data_ptr()
        assert flags.tolist() == [False, True, False, False]
        assert captured["state_indices_tensor"].tolist() == [7, 8, -1, -1]
