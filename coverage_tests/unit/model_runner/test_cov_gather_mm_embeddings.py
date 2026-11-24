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

"""Coverage tests for ModelRunnerFL._gather_mm_embeddings.

The gather splices encoder-cache embeddings into the scheduled token
window of each request and marks the covered positions in the
``is_mm_embed`` mask. Features carry real ``PlaceholderRange`` mm
positions so window arithmetic (`get_embeds_indices_in_range`) follows
the production path; the runner is a namespace holding exactly the
fields the method reads.
"""

from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm_fl.worker.model_runner import ModelRunnerFL


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


def _range(**kw):
    """A real PlaceholderRange (lazy import keeps the shim first)."""
    from vllm.multimodal.inputs import PlaceholderRange

    return PlaceholderRange(**kw)


def _sched(**kw):
    """SchedulerOutput-shaped namespace with token counts."""
    return SimpleNamespace(
        total_num_scheduled_tokens=kw.get("total", 0),
        num_scheduled_tokens=kw.get("num_scheduled_tokens", {}),
    )


def _gather(
    features_by_req, num_scheduled, computed, encoder_cache, total=None, shift=0
):
    """Run _gather_mm_embeddings with the local PIN_MEMORY probe forced off.

    ``PIN_MEMORY`` (the host's pinned-allocation capability probe) is True
    on this CPU-only host's platform but ``torch.zeros(pin_memory=True)``
    fails without an accelerator; on CUDA hosts the patch is inert.
    """
    runner = SimpleNamespace(
        requests={
            req_id: SimpleNamespace(
                mm_features=feats,
                num_computed_tokens=computed.get(req_id, 0),
                mrope_positions=None,
            )
            for req_id, feats in features_by_req.items()
        },
        input_batch=SimpleNamespace(req_ids=list(features_by_req)),
        encoder_cache=encoder_cache,
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
    )
    total = total if total is not None else sum(num_scheduled.values())
    with _Patch(mr_module, "PIN_MEMORY", False):
        return ModelRunnerFL._gather_mm_embeddings(
            runner,
            _sched(total=total, num_scheduled_tokens=num_scheduled),
            shift_computed_tokens=shift,
        )


def test_gather_mm_embeddings_no_features_yields_empty_mask():
    """A request without features produces no embeds and an all-False mask."""
    mm_embeds, is_mm_embed = _gather({"r0": []}, {"r0": 3}, {"r0": 0}, {})
    assert mm_embeds == []
    assert is_mm_embed.tolist() == [False, False, False]


def test_gather_mm_embeddings_splices_window_slice_into_mask():
    """The computed window selects which cached embed rows are gathered."""
    embeds = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    feature = SimpleNamespace(
        identifier="h0",
        mm_position=_range(offset=2, length=4),
    )
    # Request computed 2 tokens, schedules 3 more: the window is
    # [2, 5) which overlaps the feature at [2, 6): start_idx 0, end_idx 3.
    mm_embeds, is_mm_embed = _gather(
        {"r0": [feature]}, {"r0": 3}, {"r0": 2}, {"h0": embeds}
    )
    assert len(mm_embeds) == 1
    torch.testing.assert_close(mm_embeds[0], embeds[0:3])
    assert is_mm_embed.tolist() == [True, True, True]


def test_gather_mm_embeddings_shifts_window_with_shift_computed_tokens():
    """shift_computed_tokens moves the gather window by one embed row."""
    embeds = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    feature = SimpleNamespace(
        identifier="h0",
        mm_position=_range(offset=0, length=4),
    )
    # Same state but the caller shifts the computed window by one.
    mm_embeds, is_mm_embed = _gather(
        {"r0": [feature]}, {"r0": 2}, {"r0": 0}, {"h0": embeds}, shift=1
    )
    assert len(mm_embeds) == 1
    torch.testing.assert_close(mm_embeds[0], embeds[1:3])
    assert is_mm_embed.tolist() == [True, True]


def test_gather_mm_embeddings_partial_window_overlap():
    """Only the window part overlapping the feature range is marked."""
    embeds = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    feature = SimpleNamespace(
        identifier="h0",
        mm_position=_range(offset=1, length=3),
    )
    # Window [0, 4) overlaps the feature [1, 4) on [1, 4).
    mm_embeds, is_mm_embed = _gather(
        {"r0": [feature]}, {"r0": 4}, {"r0": 0}, {"h0": embeds}
    )
    # Only the overlapping part [1, 4) is gathered: embeds[0:3].
    torch.testing.assert_close(mm_embeds[0], embeds[0:3])
    # Positions 0 (outside) and 3 (outside) stay False.
    assert is_mm_embed.tolist() == [False, True, True, True]


def test_gather_mm_embeddings_ahead_of_cache_miss_is_skipped():
    """Drafter look-ahead reaching an un-encoded feature falls back."""
    embeds = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    feature = SimpleNamespace(
        identifier="future",
        mm_position=_range(offset=4, length=2),
    )
    # The feature starts at/after the processed boundary: skipped without
    # raising, and no mask position is marked.
    mm_embeds, is_mm_embed = _gather(
        {"r0": [feature]}, {"r0": 2}, {"r0": 2}, {"h0": embeds}
    )
    assert mm_embeds == []
    assert is_mm_embed.tolist() == [False, False]


def test_gather_mm_embeddings_cache_miss_inside_window_raises():
    """An un-encoded feature inside the window raises naming the hash."""
    feature = SimpleNamespace(
        identifier="missing",
        mm_position=_range(offset=0, length=2),
    )
    try:
        _gather({"r0": [feature]}, {"r0": 2}, {"r0": 0}, {})
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on encoder cache miss")


def test_gather_mm_embeddings_two_requests_offsets_masks():
    """Each request's scheduled segment is marked by its own feature."""
    embeds0 = torch.zeros(2, 2)
    embeds1 = torch.ones(2, 2)
    feat0 = SimpleNamespace(
        identifier="a",
        mm_position=_range(offset=0, length=2),
    )
    feat1 = SimpleNamespace(
        identifier="b",
        mm_position=_range(offset=0, length=2),
    )
    mm_embeds, is_mm_embed = _gather(
        {"r0": [feat0], "r1": [feat1]},
        {"r0": 2, "r1": 2},
        {"r0": 0, "r1": 0},
        {"a": embeds0, "b": embeds1},
    )
    # Each request's mask segment marks its own two positions.
    assert is_mm_embed.tolist() == [True, True, True, True]
    assert len(mm_embeds) == 2
    torch.testing.assert_close(mm_embeds[0], embeds0[0:2])
    torch.testing.assert_close(mm_embeds[1], embeds1[0:2])


def test_gather_mm_embeddings_embed_mask_partial_positions():
    """An is_embed mask marks only the embedding-carrying positions."""
    embeds = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    is_embed = torch.tensor([True, False, True])
    feature = SimpleNamespace(
        identifier="h0",
        mm_position=_range(offset=0, length=3, is_embed=is_embed),
    )
    mm_embeds, out_mask = _gather(
        {"r0": [feature]}, {"r0": 3}, {"r0": 0}, {"h0": embeds}
    )
    # is_embed restricts which scheduled positions become embeddings.
    assert out_mask.tolist() == [True, False, True]
    # The embeds slice covers the embed positions of the window.
    assert len(mm_embeds) == 1
