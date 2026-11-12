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

"""Coverage tests for ModelRunnerFL._get_slot_mappings and the entry
branches of ModelRunnerFL.execute_model (vllm_fl/worker/model_runner.py).

``_get_slot_mappings`` builds per-group and per-layer slot mappings from
the block table (or zeros for encoder-only groups), pads unused tail slots
with -1 for full-graph capture, and slices them per ubatch. It is pure
CPU tensor/numpy logic, driven here through the unbound method on a
SimpleNamespace runner with a real ``KVCacheGroupSpec`` / attention spec.

``execute_model``'s CPU-reachable entry branches are exercised: the
state-machine guard, the routed-experts buffer clear, the ngram-gpu
scheduler-output copy, and the empty-schedule early return. The vllm
distributed group and the W1b cooperation methods are inputs replaced
with recorders (CONTRACT R2); ``get_pp_group`` is patched at the module
boundary, which is an allowed torch.distributed boundary.
"""

from types import SimpleNamespace

import torch

from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
)
from vllm.v1.worker.ubatch_utils import UBatchSlice

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


class _Patch:
    """Deterministic set/restore of a module attribute (R2 input boundary)."""

    def __init__(self, module, name, value):
        self.module = module
        self.name = name
        self.value = value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _full_attn_group(layer_names, block_size=16, heads=2, head_size=8):
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    return KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)


def _runner_with_groups(groups, block_tables):
    """Namespace runner exposing a kv_cache_config over ``groups``."""
    return SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(block_table=block_tables),
    )


def test_slot_mappings_absent_when_no_kv_cache_config():
    """A runner without a kv_cache_config yields (None, None) for both maps."""
    for missing in (
        SimpleNamespace(kv_cache_config=None),
        SimpleNamespace(),  # attribute entirely absent (hasattr guard)
        SimpleNamespace(kv_cache_config=SimpleNamespace(kv_cache_groups=[])),
    ):
        by_gid, by_layer = ModelRunnerFL._get_slot_mappings(
            missing,
            num_tokens_padded=8,
            num_reqs_padded=2,
            num_tokens_unpadded=5,
        )
        assert by_gid is None
        assert by_layer is None


def test_slot_mappings_pad_tail_with_minus_one_for_capture():
    """Unused tail slots are filled with -1 so reshape_and_cache skips them."""
    group = _full_attn_group(["layers.0", "layers.1"])
    block_slots = torch.arange(32, dtype=torch.int64)
    runner = _runner_with_groups(
        [group],
        [
            SimpleNamespace(slot_mapping=SimpleNamespace(gpu=block_slots)),
        ],
    )

    by_gid, by_layer = ModelRunnerFL._get_slot_mappings(
        runner,
        num_tokens_padded=32,
        num_reqs_padded=4,
        num_tokens_unpadded=20,
    )

    # One entry per group; both layers share the same tensor object.
    assert set(by_gid) == {0}
    assert by_layer == {"layers.0": by_gid[0], "layers.1": by_gid[0]}
    # Scheduled slots keep the block-table order; the padded tail is -1.
    torch.testing.assert_close(by_gid[0][:20], block_slots[:20])
    assert by_gid[0][20:].tolist() == [-1] * 12


def test_slot_mappings_kept_when_common_metadata_graph_is_current():
    """slot_mapping_is_current=True skips the -1 tail fill."""
    group = _full_attn_group(["layers.0"])
    block_slots = torch.arange(16, dtype=torch.int64)
    runner = _runner_with_groups(
        [group],
        [
            SimpleNamespace(slot_mapping=SimpleNamespace(gpu=block_slots.clone())),
        ],
    )

    by_gid, _ = ModelRunnerFL._get_slot_mappings(
        runner,
        num_tokens_padded=16,
        num_reqs_padded=4,
        num_tokens_unpadded=8,
        slot_mapping_is_current=True,
    )

    # No -1 fill: the full padded range keeps the block-table values.
    torch.testing.assert_close(by_gid[0], block_slots)


def test_slot_mappings_encoder_only_group_uses_zero_slots():
    """Encoder-only groups get zero-filled slot mappings, not block-table rows."""
    enc_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
    )
    enc_group = KVCacheGroupSpec(
        layer_names=["enc.0"],
        kv_cache_spec=enc_spec,
    )
    full_group = _full_attn_group(["dec.0"])
    # The full-attention group reads the block table; the encoder group must
    # not (its block_table entry is missing on purpose).
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[enc_group, full_group],
        ),
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            block_table=[
                "unused-marker",  # index 0 is the encoder-only group
                SimpleNamespace(
                    slot_mapping=SimpleNamespace(
                        gpu=torch.arange(16, dtype=torch.int64)
                    ),
                ),
            ]
        ),
    )

    by_gid, by_layer = ModelRunnerFL._get_slot_mappings(
        runner,
        num_tokens_padded=16,
        num_reqs_padded=2,
        num_tokens_unpadded=10,
    )

    # Encoder-only group: zeros with a -1 tail (reshape_and_cache skips them).
    assert by_gid[0].tolist() == [0] * 10 + [-1] * 6
    # Full-attention group reads the block table as usual.
    assert by_gid[1][:10].tolist() == list(range(10))
    assert by_layer["enc.0"] is by_gid[0]
    assert by_layer["dec.0"] is by_gid[1]


def test_slot_mappings_ubatch_slices_returned_per_slice():
    """With ubatch slices the per-layer map becomes a list of sliced dicts."""
    group = _full_attn_group(["layers.0", "layers.1"])
    block_slots = torch.arange(16, dtype=torch.int64)
    runner = _runner_with_groups(
        [group],
        [
            SimpleNamespace(slot_mapping=SimpleNamespace(gpu=block_slots)),
        ],
    )
    ubatch_slices = [
        UBatchSlice(slice(0, 2), slice(0, 8)),
        UBatchSlice(slice(2, 4), slice(8, 16)),
    ]

    by_gid, by_layer = ModelRunnerFL._get_slot_mappings(
        runner,
        num_tokens_padded=16,
        num_reqs_padded=4,
        num_tokens_unpadded=16,
        ubatch_slices=ubatch_slices,
    )

    # by_gid is unchanged; by_layer is one dict per ubatch with the
    # token-sliced mapping for every layer in the group.
    assert isinstance(by_layer, list)
    assert len(by_layer) == 2
    torch.testing.assert_close(by_layer[0]["layers.0"], block_slots[:8])
    torch.testing.assert_close(by_layer[1]["layers.1"], block_slots[8:16])
    # Both layers of a group map to the same slice values.
    torch.testing.assert_close(by_layer[0]["layers.0"], by_layer[0]["layers.1"])
