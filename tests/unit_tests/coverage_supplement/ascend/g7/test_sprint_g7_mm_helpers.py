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

"""G7-B (batch 2) coverage: multimodal / position / encoder-length helpers of
ModelRunnerFL (814b line).

Targets (814b): _init_mrope_positions 1845-1873, _extract_mm_kwargs 1877-1899,
_dummy_mm_kwargs 1901-1912, _compute_prev_positions 1940-1953,
_get_encoder_seq_lens 2091-2134, _get_nans_in_logits 6004-6020,
_to_list 8152-8165.

Pure CPU: tensors are tiny numpy/torch objects, and all collaborators
(group_and_batch_mm_kwargs, CrossAttentionSpec, count_nans_per_row, ...) are
patched at the module boundary.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import torch

from vllm_fl.worker import model_runner as mr


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


# --------------------------------------------------------------------------
# _init_mrope_positions
# --------------------------------------------------------------------------
def test_init_mrope_positions_with_token_ids():
    captured = {}
    model = SimpleNamespace(
        get_mrope_input_positions=lambda tokens, feats: (
            captured.update(tokens=tokens, feats=feats) or ("POS", 7)
        )
    )
    runner = SimpleNamespace(get_model=lambda: model)
    req = SimpleNamespace(
        mm_features=[
            SimpleNamespace(modality="image"),
            SimpleNamespace(modality="prompt_embeds"),
        ],
        prompt_token_ids=[11, 12, 13],
        prompt_embeds=None,
    )
    with _stack((mr, "supports_mrope", lambda m: True)):
        mr.ModelRunnerFL._init_mrope_positions(runner, req)
    assert captured["tokens"] == [11, 12, 13]
    assert len(captured["feats"]) == 1
    assert req.mrope_positions == "POS"
    assert req.mrope_position_delta == 7


def test_init_mrope_positions_with_prompt_embeds():
    captured = {}
    model = SimpleNamespace(
        get_mrope_input_positions=lambda tokens, feats: (
            captured.update(tokens=tokens) or ("POS2", 3)
        )
    )
    runner = SimpleNamespace(get_model=lambda: model)
    req = SimpleNamespace(
        mm_features=[],
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(4, 8),
    )
    with _stack((mr, "supports_mrope", lambda m: True)):
        mr.ModelRunnerFL._init_mrope_positions(runner, req)
    assert captured["tokens"] == [0, 1, 2, 3]
    assert req.mrope_positions == "POS2"


def test_init_mrope_positions_raises_without_inputs():
    runner = SimpleNamespace(get_model=lambda: SimpleNamespace())
    req = SimpleNamespace(mm_features=[], prompt_token_ids=None, prompt_embeds=None)
    with _stack((mr, "supports_mrope", lambda m: True)):
        try:
            mr.ModelRunnerFL._init_mrope_positions(runner, req)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------
# _extract_mm_kwargs
# --------------------------------------------------------------------------
def test_extract_mm_kwargs_early_return_without_output():
    runner = SimpleNamespace(is_multimodal_raw_input_only_model=True)
    assert mr.ModelRunnerFL._extract_mm_kwargs(runner, None) == {}


def test_extract_mm_kwargs_early_return_when_not_raw_only():
    runner = SimpleNamespace(is_multimodal_raw_input_only_model=False)
    sched = SimpleNamespace(scheduled_new_reqs=[])
    assert mr.ModelRunnerFL._extract_mm_kwargs(runner, sched) == {}


def test_extract_mm_kwargs_groups_features():
    seen = []
    feat_ok = SimpleNamespace(data={"pixel": 1}, modality="image")
    feat_skip = SimpleNamespace(data=None, modality="audio")
    req = SimpleNamespace(mm_features=[feat_ok, feat_skip])
    sched = SimpleNamespace(scheduled_new_reqs=[req])
    runner = SimpleNamespace(
        is_multimodal_raw_input_only_model=True,
        device=torch.device("cpu"),
    )

    def _group(items, device, pin_memory):
        seen.append(list(items))
        return [("image", 1, {"mm_kwargs": 5})]

    with _stack(
        (mr, "group_and_batch_mm_kwargs", _group),
        (mr, "PIN_MEMORY", False),
    ):
        result = mr.ModelRunnerFL._extract_mm_kwargs(runner, sched)
    assert result == {"mm_kwargs": 5}
    assert seen[0] == [("image", {"pixel": 1})]


# --------------------------------------------------------------------------
# _dummy_mm_kwargs
# --------------------------------------------------------------------------
def test_dummy_mm_kwargs_non_raw_returns_empty():
    runner = SimpleNamespace(is_multimodal_raw_input_only_model=False)
    assert mr.ModelRunnerFL._dummy_mm_kwargs(runner, 2) == {}


def test_dummy_mm_kwargs_no_tower_returns_empty():
    runner = SimpleNamespace(
        is_multimodal_raw_input_only_model=True,
        mm_budget=SimpleNamespace(mm_max_toks_per_item=0),
    )
    assert mr.ModelRunnerFL._dummy_mm_kwargs(runner, 2) == {}


def test_dummy_mm_kwargs_delegates_to_dummy_batch():
    runner = SimpleNamespace(
        is_multimodal_raw_input_only_model=True,
        mm_budget=SimpleNamespace(
            mm_max_toks_per_item=5,
            get_modality_with_max_tokens=lambda: "video",
        ),
    )
    runner._get_mm_dummy_batch = lambda modality, num_seqs: {
        "modality": modality,
        "num": num_seqs,
    }
    result = mr.ModelRunnerFL._dummy_mm_kwargs(runner, 3)
    assert result == {"modality": "video", "num": 3}


# --------------------------------------------------------------------------
# _compute_prev_positions
# --------------------------------------------------------------------------
def test_compute_prev_positions_fills_minus_one_without_prev():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            prev_req_id_to_index=None,
            req_ids=["a", "b"],
        ),
        prev_positions=SimpleNamespace(np=np.full(3, -2, dtype=np.int64)),
    )
    mr.ModelRunnerFL._compute_prev_positions(runner, 2)
    assert runner.prev_positions.np[:2].tolist() == [-1, -1]
    assert runner.prev_positions.np[2] == -2


def test_compute_prev_positions_maps_previous_indices():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            prev_req_id_to_index={"a": 1},
            req_ids=["a", "b"],
        ),
        prev_positions=SimpleNamespace(np=np.zeros(2, dtype=np.int64)),
    )
    mr.ModelRunnerFL._compute_prev_positions(runner, 2)
    assert runner.prev_positions.np.tolist() == [1, -1]


# --------------------------------------------------------------------------
# _get_encoder_seq_lens
# --------------------------------------------------------------------------
class _CrossAttentionSpec:
    pass


def test_get_encoder_seq_lens_non_cross_returns_none():
    runner = SimpleNamespace()
    with _stack((mr, "CrossAttentionSpec", _CrossAttentionSpec)):
        gpu, cpu = mr.ModelRunnerFL._get_encoder_seq_lens(
            runner, {}, object(), 1
        )
    assert gpu is None and cpu is None


def _encoder_runner():
    npbuf = np.zeros(3, dtype=np.int32)
    gpubuf = torch.zeros(3, dtype=torch.int32)
    encoder_seq_lens = SimpleNamespace(
        np=npbuf,
        gpu=gpubuf,
        copy_to_gpu=lambda n: None,
    )
    runner = SimpleNamespace(
        encoder_seq_lens=encoder_seq_lens,
        input_batch=SimpleNamespace(req_id_to_index={"r0": 0, "r1": 1}),
        requests={
            "r0": SimpleNamespace(
                mm_features=[
                    SimpleNamespace(mm_position=SimpleNamespace(length=4)),
                    SimpleNamespace(mm_position=SimpleNamespace(length=2)),
                ]
            ),
            "r1": SimpleNamespace(mm_features=None),
        },
    )
    return runner






# --------------------------------------------------------------------------
# _get_nans_in_logits
# --------------------------------------------------------------------------
def test_get_nans_in_logits_none_returns_empty_mapping():
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_id_to_index={}))
    fake_envs = SimpleNamespace(VLLM_RAISE_ON_LOGIT_NANS=False)
    with _stack((mr, "envs", fake_envs)):
        assert mr.ModelRunnerFL._get_nans_in_logits(runner, None) == {}


def test_get_nans_in_logits_counts_and_may_raise():
    calls = []
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_id_to_index={"r0": 0})
    )
    logits = torch.zeros(1, 4)

    def _count(t):
        return SimpleNamespace(tolist=lambda: [2])

    class _Env:
        VLLM_RAISE_ON_LOGIT_NANS = True

    with _stack(
        (mr, "count_nans_per_row", _count),
        (mr, "nans_to_dict", lambda counts, mapping: {"r0": 2}),
        (mr, "envs", _Env),
        (mr, "raise_if_nan_logits", lambda d: calls.append(d)),
    ):
        result = mr.ModelRunnerFL._get_nans_in_logits(runner, logits)
    assert result == {"r0": 2}
    assert calls == [{"r0": 2}]


# --------------------------------------------------------------------------
# _to_list
# --------------------------------------------------------------------------
def test_to_list_copies_and_synchronizes_event():
    calls = []
    runner = SimpleNamespace(
        sampled_token_ids_pinned_cpu=torch.zeros(2, 3, dtype=torch.int64),
        transfer_event=SimpleNamespace(
            record=lambda: calls.append("record"),
            synchronize=lambda: calls.append("sync"),
        ),
    )
    src = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    result = mr.ModelRunnerFL._to_list(runner, src)
    assert result == [[1, 2, 3], [4, 5, 6]]
    assert calls == ["record", "sync"]
