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

"""Coverage tests for the optional branches of
ModelRunnerFL._build_attention_metadata.

The base assembly is covered in test_cov_build_attn_meta.py. This file
drives the remaining paths: cascade-prefix lengths, async spec decode's
CPU-metadata suppression, multimodal bidirectional document ranges with
the sliding-window filter, routed-experts slot snapshots, encoder-only
KV-cache groups, microbatch slicing, spec-decode-specific mamba/GDN
builder kwargs, drafter group selection with unpadding, and the
"rows are current" padding rule.

The method is invoked unbound on a namespace runner holding real
tensors, as in the sibling file; metadata builders and the drafter are
inputs replaced with recorders (CONTRACT R2).
"""

from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


def _spec():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )


def _group(layer_names, spec=None):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    return KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec or _spec())


class _Builder:
    """Recording metadata builder; records the common-prefix length."""

    supports_update_block_table = False

    def __init__(self, name="b0"):
        self.name = name
        self.built = []
        self.captured = None

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.built.append((common_prefix_len, common_attn_metadata, kwargs))
        return {"builder": self.name}

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.captured = common_attn_metadata
        return {"builder": self.name, "capture": True}


def _runner(num_reqs=2, num_tokens=5, attn_group=None, builder=None):
    """Namespace runner with the buffers _build_attention_metadata reads."""
    group = _group(["layers.0", "layers.1"])
    builder = builder or _Builder()
    if attn_group is None:
        attn_group = SimpleNamespace(
            layer_names=["layers.0", "layers.1"],
            get_metadata_builder=lambda ubid=0: builder,
        )
    block_tensor = torch.arange(num_reqs * 4, dtype=torch.int32).reshape(num_reqs, 4)
    block_table = SimpleNamespace(
        get_device_tensor=lambda padded: block_tensor[:padded],
        slot_mapping=SimpleNamespace(gpu=torch.arange(num_tokens + 3)),
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group]),
        attn_groups=[[attn_group]],
        input_batch=SimpleNamespace(
            block_table=[block_table],
            num_computed_tokens_cpu_tensor=torch.tensor(
                [2, 1] + [0] * (max(num_reqs - 2, 0)), dtype=torch.int32
            )[:num_reqs],
            num_prompt_tokens_cpu_tensor=torch.tensor(
                [4, 4] + [4] * (max(num_reqs - 2, 0)), dtype=torch.int32
            )[:num_reqs],
            req_ids=["r0", "r1"][:num_reqs],
            req_id_to_index={"r0": 0, "r1": 1},
        ),
        requests={f"r{i}": SimpleNamespace(mm_features=[]) for i in range(num_reqs)},
        optimistic_seq_lens_cpu=torch.tensor(
            [5, 3] + [0] * (max(num_reqs - 2, 0)), dtype=torch.int32
        )[:num_reqs],
        max_model_len=64,
        device=torch.device("cpu"),
        common_attention_metadata_graph=None,
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(sliding_window=None)
        ),
        routed_experts_initialized=False,
        speculative_config=None,
        drafter=None,
        use_ascend_attention=False,
        dcp_world_size=1,
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        num_accepted_tokens=SimpleNamespace(
            gpu=torch.ones(num_reqs, dtype=torch.int32)
        ),
        num_decode_draft_tokens=SimpleNamespace(
            cpu=torch.zeros(num_reqs, dtype=torch.int32)
        ),
        mamba_prev_last_scheduled_idx=None,
        query_start_loc=SimpleNamespace(
            gpu=torch.tensor([0, 3, 5, 5, 5], dtype=torch.int32),
            cpu=torch.tensor([0, 3, 5, 5, 5], dtype=torch.int32),
        ),
        seq_lens=torch.tensor([5, 3, 0, 0], dtype=torch.int32)[: num_reqs + 1],
        num_computed_tokens=torch.tensor([2, 1, 0, 0], dtype=torch.int32)[
            : num_reqs + 1
        ],
        positions=torch.arange(num_tokens + 3, dtype=torch.int64),
    )
    runner._get_encoder_seq_lens = ModelRunnerFL._get_encoder_seq_lens.__get__(
        runner, ModelRunnerFL
    )
    return runner, builder, block_tensor


def test_build_attention_metadata_passes_cascade_prefix_len_to_builder():
    """Per-group cascade prefix lengths reach the builder as common_prefix_len."""
    runner, builder, _ = _runner()
    ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        cascade_attn_prefix_lens=[[7]],
        slot_mappings={0: torch.arange(8)},
    )
    assert builder.built[0][0] == 7


def test_build_attention_metadata_async_spec_decode_suppresses_cpu_metadata():
    """Async spec decode trusts device seq-lens: CPU fields are dropped."""
    runner, builder, _ = _runner()
    runner.use_async_spec_decode = True
    ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        slot_mappings={0: torch.arange(8)},
    )
    _, cm, _ = builder.built[0]
    assert cm._seq_lens_cpu is None
    assert cm._num_computed_tokens_cpu is None
    # The independent upper bound keeps the optimistic values.
    assert cm.seq_lens_cpu_upper_bound.tolist() == [5, 3]


def test_build_attention_metadata_mm_prefix_doc_ranges_filters_audio_and_sw():
    """Bidirectional ranges skip audio features and ranges wider than the window."""
    from vllm.multimodal.inputs import PlaceholderRange

    runner, builder, _ = _runner()
    runner.is_mm_prefix_lm = True
    runner.model_config.hf_text_config = SimpleNamespace(sliding_window=8)
    runner.requests["r0"] = SimpleNamespace(
        mm_features=[
            SimpleNamespace(
                modality="image",
                mm_position=PlaceholderRange(offset=0, length=4),
            ),
            SimpleNamespace(
                modality="audio",
                mm_position=PlaceholderRange(offset=4, length=4),
            ),
            SimpleNamespace(
                modality="image",
                mm_position=PlaceholderRange(offset=20, length=10),
            ),
        ]
    )

    ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        slot_mappings={0: torch.arange(8)},
    )
    _, cm, _ = builder.built[0]
    # r0 keeps only the width-4 image; audio is skipped and the width-10
    # image exceeds the sliding window.
    assert cm.mm_req_doc_ranges == {0: [(0, 3)], 1: []}


def test_build_attention_metadata_snapshots_routed_experts_slot_mapping():
    """Initialized routed experts copy this step's attention slots privately."""
    runner, _, _ = _runner()
    runner.routed_experts_initialized = True
    runner.routed_experts_attn_gid = 0
    runner.routed_experts_slot_mapping_device = torch.zeros(8, dtype=torch.int64)
    ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        slot_mappings={0: torch.arange(8) + 10},
    )
    assert runner.routed_experts_slot_mapping_device[:5].tolist() == [
        10,
        11,
        12,
        13,
        14,
    ]


def test_build_attention_metadata_encoder_only_group_uses_zero_block_table():
    """Encoder-only KV-cache groups pass a zero block table, not the batch's."""
    from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

    enc_spec = EncoderOnlyAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    group = _group(["enc.l0"], enc_spec)
    builder = _Builder("enc")
    runner, _, _ = _runner(
        attn_group=SimpleNamespace(
            layer_names=["enc.l0"],
            get_metadata_builder=lambda ubid=0: builder,
        ),
        builder=builder,
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])

    attn_meta, _ = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        slot_mappings={0: torch.arange(8)},
    )
    _, cm, _ = builder.built[0]
    assert attn_meta["enc.l0"] == {"builder": "enc"}
    assert cm.block_table_tensor.shape == (2, 1)
    assert cm.block_table_tensor.dtype == torch.int32
    assert cm.block_table_tensor.eq(0).all()


def test_build_attention_metadata_keeps_padding_rows_when_current():
    """With current block-table rows, graph padding is left untouched."""
    runner, builder, block_tensor = _runner(num_reqs=2)
    padded_table = torch.cat([block_tensor, torch.full((2, 4), 99, dtype=torch.int32)])
    runner.input_batch.block_table[0].get_device_tensor = lambda padded: padded_table[
        :padded
    ]
    ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        num_reqs_padded=4,
        slot_mappings={0: torch.arange(8)},
        block_table_rows_are_current=True,
    )
    _, cm, _ = builder.built[0]
    # The caller already wrote padding rows; the NULL_BLOCK_ID fill is skipped.
    assert cm.block_table_tensor[2:].tolist() == [[99, 99, 99, 99]] * 2


def test_build_attention_metadata_microbatch_slices_build_each_ubatch():
    """Microbatch slices produce one metadata dict per ubatch."""
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    runner, builder, _ = _runner()
    slices = [
        UBatchSlice(slice(0, 1), slice(0, 3)),
        UBatchSlice(slice(1, 2), slice(3, 5)),
    ]
    attn_meta, _ = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        ubatch_slices=slices,
        slot_mappings={0: torch.arange(8)},
    )
    assert isinstance(attn_meta, list) and len(attn_meta) == 2
    # Each ubatch dict carries every layer of the group.
    for per_ubatch in attn_meta:
        assert set(per_ubatch) == {"layers.0", "layers.1"}
    # The builder saw one request / 3 tokens, then one request / 2 tokens.
    assert builder.built[0][1].num_reqs == 1
    assert builder.built[0][1].num_actual_tokens == 3
    assert builder.built[1][1].num_reqs == 1
    assert builder.built[1][1].num_actual_tokens == 2


def test_build_attention_metadata_mamba2_builder_gets_spec_decode_tensors():
    """Spec decode adds accepted/draft tensors and the mamba state index."""
    from vllm.v1.attention.backends.mamba2_attn import (
        Mamba2AttentionMetadataBuilder,
    )

    builder = object.__new__(Mamba2AttentionMetadataBuilder)
    builder.supports_update_block_table = False
    calls = []
    builder.build = lambda **kw: (calls.append(kw), {})[1]
    runner, _, _ = _runner(
        attn_group=SimpleNamespace(
            layer_names=["layers.0", "layers.1"],
            get_metadata_builder=lambda ubid=0: builder,
        )
    )
    runner.mamba_prev_last_scheduled_idx = SimpleNamespace(
        gpu=torch.arange(4, dtype=torch.int32) + 50
    )

    ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        use_spec_decode=True,
        slot_mappings={0: torch.arange(8)},
    )
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["num_accepted_tokens"].tolist() == [1, 1]
    assert kwargs["num_decode_draft_tokens_cpu"].tolist() == [0, 0]
    assert kwargs["prev_last_scheduled_idx"].tolist() == [50, 51]
    assert kwargs["common_prefix_len"] == 0


def test_build_attention_metadata_gdn_builder_omits_mamba_state_index():
    """GDN builders get the spec tensors but never the mamba last index."""
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    builder = object.__new__(GDNAttentionMetadataBuilder)
    calls = []
    builder.build = lambda **kw: (calls.append(kw), {})[1]
    runner, _, _ = _runner(
        attn_group=SimpleNamespace(
            layer_names=["layers.0", "layers.1"],
            get_metadata_builder=lambda ubid=0: builder,
        )
    )

    ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        use_spec_decode=True,
        slot_mappings={0: torch.arange(8)},
    )
    kwargs = calls[0]
    assert set(kwargs) == {
        "common_prefix_len",
        "common_attn_metadata",
        "num_accepted_tokens",
        "num_decode_draft_tokens_cpu",
    }


def test_build_attention_metadata_drafter_group_selection_and_unpadding():
    """The drafter's KV group owns the spec metadata, unpadded to real sizes."""
    from vllm.v1.spec_decode.eagle import EagleProposer

    runner, _, _ = _runner()
    drafter = object.__new__(EagleProposer)
    drafter.kv_cache_gid = 0
    runner.speculative_config = SimpleNamespace()
    runner.drafter = drafter

    _, spec_meta = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        num_tokens_padded=8,
        num_reqs_padded=4,
        max_query_len=3,
        slot_mappings={0: torch.arange(8)},
    )
    # Padded metadata is unpadded back to the real request/token counts.
    assert spec_meta is not None
    assert spec_meta.num_reqs == 2
    assert spec_meta.num_actual_tokens == 5

    # A drafter bound to another group does not claim group 0's metadata.
    drafter.kv_cache_gid = 1
    _, spec_meta_other = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        slot_mappings={0: torch.arange(8)},
    )
    assert spec_meta_other is None
