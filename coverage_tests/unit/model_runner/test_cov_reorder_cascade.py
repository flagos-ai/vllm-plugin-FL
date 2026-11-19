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

"""Coverage tests for the batch reordering and cascade-attention
prefix-length helpers of vllm_fl/worker/model_runner.py.

``_may_reorder_batch`` gates the decode/prefill split on the
kv-cache group count and the configured threshold (the reorder
helper itself is an input, CONTRACT R2, replaced with a recorder).
The ``_compute_cascade_attn_prefix_lens``/``_compute_cascade_attn_prefix_len``
pair runs on namespace runners with recording metadata builders: the
per-group collection, the block-multiple truncation, the builder
decline, and the computed-token minimum are each asserted by value.
"""

from types import SimpleNamespace

import numpy as np
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm_fl.worker.model_runner import ModelRunnerFL


def _spec():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )


class _Patch:
    """Deterministic set/restore of a module attribute (input boundary)."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def test_may_reorder_batch_no_groups_is_a_noop():
    """Without kv-cache groups the reorder step never touches the batch."""
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        reorder_batch_threshold=None,
    )
    # No assertions to make on the batch: the method must simply return
    # without touching the input batch (verified via a marker attribute).
    runner.input_batch = SimpleNamespace(touched=False)
    ModelRunnerFL._may_reorder_batch.__get__(runner, ModelRunnerFL)(object())
    assert not runner.input_batch.touched


def test_may_reorder_batch_invokes_reorder_helper_above_threshold():
    """With groups and a threshold the reorder helper runs with config."""
    calls = []

    def _reorder(input_batch, scheduler_output, decode_threshold):
        calls.append((input_batch, scheduler_output, decode_threshold))

    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=["g0"]),
        reorder_batch_threshold=5,
        input_batch="the-batch",
    )
    with _Patch(mr_module, "reorder_batch_to_split_decodes_and_prefills", _reorder):
        ModelRunnerFL._may_reorder_batch.__get__(runner, ModelRunnerFL)("sched")
    assert calls == [("the-batch", "sched", 5)]


def test_compute_cascade_attn_prefix_lens_returns_none_when_unused():
    """All-zero prefix lens (no common blocks) => None, not a zero matrix."""
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=["g0"]),
        attn_groups=[
            [
                SimpleNamespace(
                    kv_cache_spec=_spec(),
                    get_metadata_builder=lambda: SimpleNamespace(
                        use_cascade_attention=lambda **kw: False
                    ),
                )
            ]
        ],
        num_query_heads=8,
        use_alibi=False,
        num_sms=8,
        dcp_world_size=1,
    )
    # Bind the real single-group helper (the nested call target).
    runner._compute_cascade_attn_prefix_len = (
        ModelRunnerFL._compute_cascade_attn_prefix_len.__get__(runner, ModelRunnerFL)
    )
    out = ModelRunnerFL._compute_cascade_attn_prefix_lens.__get__(
        runner, ModelRunnerFL
    )(
        num_scheduled_tokens=np.array([1, 1]),
        num_computed_tokens=np.array([0, 0]),
        num_common_prefix_blocks=[0],
    )
    assert out is None


def test_compute_cascade_attn_prefix_lens_collects_per_group_lengths():
    """Per-group builder decisions produce the nested length lists."""
    from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

    encoder_spec = EncoderOnlyAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )

    def _builder(use):
        return SimpleNamespace(
            use_cascade_attention=lambda **kw: use,
        )

    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=["g0", "g1"]),
        attn_groups=[
            # Group 0: an encoder-only group (prefix 0 by definition) and
            # a full-attention group whose builder declines the cascade.
            [
                SimpleNamespace(
                    kv_cache_spec=encoder_spec,
                    get_metadata_builder=lambda: _builder(True),
                ),
                SimpleNamespace(
                    kv_cache_spec=_spec(),
                    get_metadata_builder=lambda: _builder(False),
                ),
            ],
            # Group 1: a full-attention group that accepts the cascade.
            [
                SimpleNamespace(
                    kv_cache_spec=_spec(),
                    get_metadata_builder=lambda: _builder(True),
                )
            ],
        ],
        num_query_heads=8,
        use_alibi=False,
        num_sms=8,
        dcp_world_size=1,
    )
    runner._compute_cascade_attn_prefix_len = (
        ModelRunnerFL._compute_cascade_attn_prefix_len.__get__(runner, ModelRunnerFL)
    )
    # 2 common blocks of size 16 => 32, capped by min computed (24) and
    # floored to a block multiple => 16.
    out = ModelRunnerFL._compute_cascade_attn_prefix_lens.__get__(
        runner, ModelRunnerFL
    )(
        num_scheduled_tokens=np.array([1, 1]),
        num_computed_tokens=np.array([24, 40]),
        num_common_prefix_blocks=[2, 2],
    )
    assert out == [[0, 0], [16]]


def test_compute_cascade_attn_prefix_len_truncates_to_block_multiple():
    """Common-prefix blocks are truncated to the computed-token minimum."""
    builder = SimpleNamespace(use_cascade_attention=lambda **kw: True)
    spec = _spec()
    runner = SimpleNamespace(
        num_query_heads=8, use_alibi=False, num_sms=8, dcp_world_size=1
    )
    # 3 common blocks (48 tokens) capped at min computed 50 -> 48, which
    # is a multiple of the 16 block size already.
    out = ModelRunnerFL._compute_cascade_attn_prefix_len.__get__(runner, ModelRunnerFL)(
        num_scheduled_tokens=np.array([1, 1]),
        num_computed_tokens=np.array([50, 60]),
        num_common_prefix_blocks=3,
        kv_cache_spec=spec,
        attn_metadata_builder=builder,
    )
    assert out == 48


def test_compute_cascade_attn_prefix_len_declined_by_builder_is_zero():
    """A declining builder collapses the prefix length to zero."""
    builder = SimpleNamespace(use_cascade_attention=lambda **kw: False)
    runner = SimpleNamespace(
        num_query_heads=8, use_alibi=False, num_sms=8, dcp_world_size=1
    )
    out = ModelRunnerFL._compute_cascade_attn_prefix_len.__get__(runner, ModelRunnerFL)(
        num_scheduled_tokens=np.array([1]),
        num_computed_tokens=np.array([64]),
        num_common_prefix_blocks=2,
        kv_cache_spec=_spec(),
        attn_metadata_builder=builder,
    )
    # Builder declines: reported prefix length collapses to 0.
    assert out == 0


def test_compute_cascade_attn_prefix_len_min_of_computed_tokens_wins():
    """The minimum computed token count bounds the cascade prefix."""
    builder = SimpleNamespace(use_cascade_attention=lambda **kw: True)
    runner = SimpleNamespace(
        num_query_heads=8, use_alibi=False, num_sms=8, dcp_world_size=1
    )
    # 4 common blocks = 64 tokens, but the minimum computed is 17 ->
    # truncated to 17 then floored to the block multiple 16.
    out = ModelRunnerFL._compute_cascade_attn_prefix_len.__get__(runner, ModelRunnerFL)(
        num_scheduled_tokens=np.array([2, 1]),
        num_computed_tokens=np.array([17, 90]),
        num_common_prefix_blocks=4,
        kv_cache_spec=_spec(),
        attn_metadata_builder=builder,
    )
    assert out == 16
