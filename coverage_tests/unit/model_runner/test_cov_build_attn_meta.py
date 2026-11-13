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

"""Coverage tests for ModelRunnerFL._build_attention_metadata.

The method assembles a CommonAttentionMetadata from the runner's buffers
and dispatches per-group builders. Following the dev-test pattern
(tests/unit_tests/worker/test_model_runner.py), it is invoked as an
unbound method on a SimpleNamespace runner whose buffers are real
tensors with hand-derived expected values. The attention-group builders
and spec-decode drafter are inputs (CONTRACT R2) replaced with
recorders; the two caching paths of the nested
``_build_attn_group_metadata`` closure (build vs update_block_table)
are both driven.
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


def _group(layer_names):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    return KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=_spec())


class _Builder:
    """Recording metadata builder with an updatable block table."""

    supports_update_block_table = True

    def __init__(self, name):
        self.name = name
        self.built = []
        self.updates = []
        self.captured = None

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.built.append((common_prefix_len, common_attn_metadata))
        return {"builder": self.name}

    def update_block_table(self, cached, block_table, slot_mapping):
        self.updates.append((cached, block_table, slot_mapping))
        return {"builder": self.name, "updated": True}

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.captured = common_attn_metadata
        return {"builder": self.name, "capture": True}


def _runner(num_reqs=2, num_tokens=5):
    """Namespace runner with the buffers _build_attention_metadata reads."""
    group = _group(["layers.0", "layers.1"])
    builder = _Builder("b0")
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
        requests={f"r{i}": object() for i in range(num_reqs)},
        optimistic_seq_lens_cpu=torch.tensor(
            [5, 3] + [0] * (max(num_reqs - 2, 0)), dtype=torch.int32
        )[:num_reqs],
        max_model_len=64,
        device=torch.device("cpu"),
        common_attention_metadata_graph=None,
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        routed_experts_initialized=False,
        speculative_config=None,
        use_ascend_attention=False,
        dcp_world_size=1,
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        num_accepted_tokens=SimpleNamespace(
            gpu=torch.ones(num_reqs, dtype=torch.int32)
        ),
        num_decode_draft_tokens=SimpleNamespace(
            cpu=torch.zeros(num_reqs, dtype=torch.int32)
        ),
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
    # Bind the real encoder-seq-lens helper (a W1b target): with a
    # non-cross-attention spec it returns the (None, None) pair.
    runner._get_encoder_seq_lens = ModelRunnerFL._get_encoder_seq_lens.__get__(
        runner, ModelRunnerFL
    )
    return runner, builder, block_tensor


def test_build_attention_metadata_empty_groups_short_circuits():
    """An empty KV-cache group list returns empty metadata and no spec metadata."""
    runner = SimpleNamespace(kv_cache_config=SimpleNamespace(kv_cache_groups=[]))
    attn_meta, spec_meta = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=4,
        num_reqs=2,
        max_query_len=2,
        slot_mappings={},
    )
    assert attn_meta == {}
    assert spec_meta is None


def test_build_attention_metadata_populates_common_and_layers():
    """Layers share one builder object; common metadata fields come from buffers."""
    runner, builder, block_tensor = _runner()
    attn_meta, spec_meta = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        slot_mappings={0: torch.arange(8)},
    )
    assert spec_meta is None  # no speculative config
    # Both layers share the builder's metadata object.
    assert attn_meta["layers.0"] is attn_meta["layers.1"]
    assert attn_meta["layers.0"] == {"builder": "b0"}

    prefix_len, cm = builder.built[0]
    assert prefix_len == 0  # no cascade prefix lens passed
    # CommonAttentionMetadata fields come straight from the buffers.
    assert cm.num_reqs == 2
    assert cm.num_actual_tokens == 5
    assert cm.max_query_len == 3
    # max_seq_len from optimistic seq lens max (5).
    assert cm.max_seq_len == 5
    assert cm.query_start_loc.tolist() == [0, 3, 5]
    assert cm.seq_lens.tolist() == [5, 3]
    torch.testing.assert_close(cm.block_table_tensor, block_tensor)
    # slot_mapping is the full padded buffer (8 slots for 5 tokens).
    assert cm.slot_mapping.tolist() == list(range(8))
    # is_prefilling: computed < prompt tokens for both requests.
    assert cm.is_prefilling.tolist() == [True, True]
    # positions slice covers exactly the scheduled tokens.
    assert cm.positions.tolist() == [0, 1, 2, 3, 4]


def test_build_attention_metadata_cudagraph_capture_uses_max_model_len():
    """Capture mode swaps the optimistic seq-len max for max_model_len."""
    runner, builder, _ = _runner()
    attn_meta, _ = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        for_cudagraph_capture=True,
        slot_mappings={0: torch.arange(8)},
    )
    assert attn_meta["layers.0"] == {"builder": "b0", "capture": True}
    # Capture mode uses max_model_len so sliding-window kernels select
    # against the full context.
    _, cm = builder.captured and (None, builder.captured)
    assert cm.max_seq_len == 64


def test_build_attention_metadata_pads_with_null_blocks():
    """Graph padding zero-fills block table rows past num_reqs."""
    runner, builder, block_tensor = _runner(num_reqs=2)
    # Give the block table four physical rows: rows 2..3 exist on device
    # so the null-block fill is observable.
    padded_table = torch.cat([block_tensor, torch.full((2, 4), 99, dtype=torch.int32)])
    runner.input_batch.block_table[0].get_device_tensor = lambda padded: padded_table[
        :padded
    ]
    attn_meta, _ = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        num_reqs_padded=4,
        slot_mappings={0: torch.arange(8)},
    )
    _, cm = builder.built[0]
    # Rows past num_reqs are filled with the null block id (0): the
    # sentinel 99s are overwritten in place.
    assert cm.block_table_tensor[2:].tolist() == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Sanity: the padded tensor reports the padded request count.
    assert cm.num_reqs == 4


def test_build_attention_metadata_reuses_cached_build_across_groups():
    """Two groups with the same spec+builder type hit update_block_table."""
    group0 = _group(["g0.l0"])
    group1 = _group(["g1.l0"])
    builder = _Builder("shared")
    runner, _, block_tensor = _runner()
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group0, group1])
    runner.attn_groups = [
        [
            SimpleNamespace(
                layer_names=["g0.l0"],
                get_metadata_builder=lambda ubid=0: builder,
            )
        ],
        [
            SimpleNamespace(
                layer_names=["g1.l0"],
                get_metadata_builder=lambda ubid=0: builder,
            )
        ],
    ]
    second_table = block_tensor.clone()
    runner.input_batch.block_table = [
        runner.input_batch.block_table[0],
        SimpleNamespace(
            get_device_tensor=lambda padded: second_table[:padded],
            slot_mapping=SimpleNamespace(gpu=torch.arange(8) + 100),
        ),
    ]

    attn_meta, _ = ModelRunnerFL._build_attention_metadata(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        slot_mappings={0: torch.arange(8), 1: torch.arange(8) + 100},
    )
    # First group built; second reused via update_block_table.
    assert len(builder.built) == 1
    assert len(builder.updates) == 1
    cached, table, slots = builder.updates[0]
    assert cached == {"builder": "shared"}
    torch.testing.assert_close(table, second_table)
    assert slots.tolist() == [100, 101, 102, 103, 104, 105, 106, 107]
    # Both layers keep the (updated) metadata object.
    assert attn_meta["g1.l0"] == {"builder": "shared", "updated": True}
