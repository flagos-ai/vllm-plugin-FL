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

"""Coverage tests for the method-dispatch half of
ModelRunnerFL.propose_draft_token_ids (vllm_fl/worker/model_runner.py):
the ``ngram`` / ``custom_class`` / ``suffix`` / ``medusa`` /
``extract_hidden_states`` routes.

Each route hands a different slice of the runner's CPU-visible batch state
to the drafter. The drafters are subclasses of the real proposer classes
(the dispatch asserts the concrete types) with recording ``propose`` /
``prepare_next_token_ids_padded`` overrides; the runner is an input
namespace (CONTRACT R2 — nothing on ModelRunnerFL is patched).

The ``use_ngram_gpu`` route (GPU token tensors, dedicated copy stream)
is not reachable on CPU and is not covered here; the eagle/draft-model
routes are covered in test_cov_propose_eagle.py.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.extract_hidden_states import (
    ExtractHiddenStatesProposer,
)
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


class _NgramProposer(NgramProposer):
    def __init__(self, calls, result):
        self._calls, self._result = calls, result

    def propose(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return self._result


class _SuffixProposer(SuffixDecodingProposer):
    def __init__(self, calls, result):
        self._calls, self._result = calls, result

    def propose(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return self._result


class _MedusaProposer(MedusaProposer):
    def __init__(self, calls, result):
        self._calls, self._result = calls, result

    def propose(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return self._result


class _ExtractProposer(ExtractHiddenStatesProposer):
    def __init__(self, calls, result, prep_result):
        self._calls, self._result = calls, result
        self._prep_result = prep_result

    def propose(self, *args, **kwargs):
        self._calls.append(("propose", args, kwargs))
        return self._result

    def prepare_next_token_ids_padded(self, *args, **kwargs):
        self._calls.append(("prepare_padded", args, kwargs))
        return self._prep_result


def _spec_config(method, uses_extract=False):
    cfg = SimpleNamespace(method=method, disable_padded_drafter_batch=False)
    cfg.use_ngram_gpu = lambda: False
    cfg.uses_extract_hidden_states = lambda: uses_extract
    return cfg


def _runner(spec_config, drafter, **extra):
    """Namespace runner carrying the CPU-visible batch state."""
    runner = SimpleNamespace(
        speculative_config=spec_config,
        drafter=drafter,
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            num_tokens_no_spec=[4, 2],
            token_ids_cpu=[[1, 2, 3, 4], [5, 6]],
            req_ids=["r0", "r1"],
        ),
        # Stale draft state: the entry of propose_draft_token_ids resets it.
        _draft_probs="stale-probs",
        _draft_prob_req_ids="stale-ids",
    )
    for key, value in extra.items():
        setattr(runner, key, value)
    return runner


def _propose(
    runner,
    *,
    sampled_token_ids,
    sample_hidden_states,
    hidden_states=None,
    aux_hidden_states=None,
    spec_decode_metadata=None,
    slot_mappings=None,
):
    """Invoke propose_draft_token_ids with the recorded defaults."""
    sched = SimpleNamespace(
        total_num_scheduled_tokens=6,
        num_spec_tokens_to_schedule=3,
    )
    return ModelRunnerFL.propose_draft_token_ids(
        runner,
        scheduler_output=sched,
        sampled_token_ids=sampled_token_ids,
        sampling_metadata="sampling-md",
        hidden_states=hidden_states if hidden_states is not None else torch.zeros(6, 2),
        sample_hidden_states=sample_hidden_states,
        aux_hidden_states=aux_hidden_states,
        spec_decode_metadata=spec_decode_metadata,
        common_attn_metadata="attn-md",
        slot_mappings=slot_mappings,
    )


def test_propose_ngram_routes_cpu_token_state():
    """method='ngram': spec-token count, sampled ids, per-request token
    counts and the CPU token block are handed to the drafter."""
    calls = []
    drafter = _NgramProposer(calls, result=[[1, 2], [3]])
    slots = {"layer.0": torch.zeros(6, dtype=torch.int64)}
    runner = _runner(_spec_config("ngram"), drafter)

    result = _propose(
        runner,
        sampled_token_ids=[[7], [8]],
        sample_hidden_states=torch.zeros(2, 2),
        slot_mappings=slots,
    )

    assert result == [[1, 2], [3]]
    assert calls == [
        (
            (3, [[7], [8]], [4, 2], [[1, 2, 3, 4], [5, 6]]),
            {"slot_mappings": slots},
        )
    ]
    # Entry of the dispatch cleared the stale draft state.
    assert runner._draft_probs is None
    assert runner._draft_prob_req_ids is None


def test_propose_custom_class_routes_without_spec_token_count():
    """method='custom_class': same CPU token state, no spec count."""
    calls = []
    drafter = SimpleNamespace(
        propose=lambda *a, **k: calls.append((a, k)) or [[9]],
    )
    runner = _runner(_spec_config("custom_class"), drafter)

    result = _propose(
        runner,
        sampled_token_ids=[[7]],
        sample_hidden_states=torch.zeros(1, 2),
    )

    assert result == [[9]]
    assert calls == [
        (
            ([[7]], [4, 2], [[1, 2, 3, 4], [5, 6]]),
            {"slot_mappings": None},
        )
    ]


def test_propose_suffix_routes_input_batch_and_sampled_ids():
    """method='suffix': the input batch object itself is forwarded."""
    calls = []
    drafter = _SuffixProposer(calls, result=[[4]])
    runner = _runner(_spec_config("suffix"), drafter)

    result = _propose(
        runner,
        sampled_token_ids=[[7], [8]],
        sample_hidden_states=torch.zeros(2, 2),
    )

    assert result == [[4]]
    assert calls == [
        (
            (3, runner.input_batch, [[7], [8]]),
            {"slot_mappings": None},
        )
    ]


def test_propose_medusa_uses_aligned_sample_hidden_states():
    """medusa: when the target forward ran over sampled tokens only, the
    sample hidden states are the drafter input unchanged."""
    calls = []
    drafter = _MedusaProposer(calls, result=[[5]])
    sample_hidden = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    runner = _runner(_spec_config("medusa"), drafter)

    result = _propose(
        runner,
        sampled_token_ids=[[7], [8]],
        sample_hidden_states=sample_hidden,
    )

    assert result == [[5]]
    # All-keyword contract of MedusaProposer.propose.
    assert calls == [
        (
            (),
            {
                "target_hidden_states": sample_hidden,
                "sampling_metadata": "sampling-md",
                "slot_mappings": None,
            },
        )
    ]


def test_propose_medusa_gathers_last_row_of_draft_groups():
    """medusa: when draft tokens were part of the forward input, the last
    row of each request's draft group is gathered instead."""
    calls = []
    drafter = _MedusaProposer(calls, result=[[5]])
    # Five forward rows: request 0 (2 scheduled: 1 target + 1 draft),
    # request 1 (3 scheduled: 1 target + 2 drafts).
    sample_hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    runner = _runner(_spec_config("medusa"), drafter)

    result = _propose(
        runner,
        sampled_token_ids=[[7], [8, 9]],
        sample_hidden_states=sample_hidden,
        spec_decode_metadata=SimpleNamespace(num_draft_tokens=[1, 2]),
    )

    assert result == [[5]]
    args, kwargs = calls[0]
    # Request 0's group spans row 0 (offset 0); request 1's group spans
    # rows 2..4 (offset 2 + 1 draft + 1 target), so its last row is 3.
    torch.testing.assert_close(
        kwargs["target_hidden_states"],
        sample_hidden[[0, 3]],
    )
    assert kwargs["sampling_metadata"] == "sampling-md"


def test_propose_extract_hidden_states_requires_aux_states():
    """extract_hidden_states without aux states raises ValueError."""
    drafter = _ExtractProposer([], result=None, prep_result=None)
    runner = _runner(
        _spec_config("extract_hidden_states", uses_extract=True),
        drafter,
        use_aux_hidden_state_outputs=False,
    )

    with pytest.raises(ValueError, match="aux_hidden_states"):
        _propose(
            runner,
            sampled_token_ids=torch.zeros(2, 1, dtype=torch.int32),
            sample_hidden_states=torch.zeros(2, 2),
        )


def test_propose_extract_hidden_states_slices_and_pads_aux_states():
    """extract_hidden_states: aux states are truncated to the scheduled
    token count, proposed, then padded next tokens are copied back."""
    calls = []
    drafter = _ExtractProposer(
        calls,
        result=[[6]],
        prep_result=("next_tokens", 7),
    )
    copy_count = []
    aux = [
        torch.arange(16, dtype=torch.float32).reshape(8, 2),
        torch.arange(16, 32, dtype=torch.float32).reshape(8, 2),
    ]
    sampled = torch.tensor([[7], [8]], dtype=torch.int32)
    runner = _runner(
        _spec_config("extract_hidden_states", uses_extract=True),
        drafter,
        use_aux_hidden_state_outputs=True,
        requests="reqs",
        discard_request_mask=SimpleNamespace(gpu="mask"),
        _copy_valid_sampled_token_count=(
            lambda nt, count: copy_count.append((nt, count))
        ),
    )

    result = _propose(
        runner,
        sampled_token_ids=sampled,
        sample_hidden_states=torch.zeros(2, 2),
        aux_hidden_states=aux,
    )

    assert result == [[6]]
    kinds = [c[0] for c in calls]
    assert kinds == ["propose", "prepare_padded"]
    _, args, kwargs = calls[0]
    # Each aux state truncated to the 6 scheduled tokens.
    assert len(kwargs["target_hidden_states"]) == 2
    torch.testing.assert_close(kwargs["target_hidden_states"][0], aux[0][:6])
    torch.testing.assert_close(kwargs["target_hidden_states"][1], aux[1][:6])
    assert kwargs["sampled_token_ids"] is sampled
    assert kwargs["common_attn_metadata"] == "attn-md"
    assert kwargs["slot_mappings"] is None
    # The padded prep ran over the full input-batch state...
    assert calls[1][1] == (sampled, "reqs", runner.input_batch, "mask")
    # ...and its result was copied to the valid-sample-count buffer.
    assert copy_count == [("next_tokens", 7)]
