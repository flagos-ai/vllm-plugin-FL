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

"""Coverage tests for the eagle / dflash / draft-model routes of
ModelRunnerFL.propose_draft_token_ids (vllm_fl/worker/model_runner.py):
the padded and unpadded drafter-input preparation.

The drafter is a subclass of the real EagleProposer (the route's isinstance
union reads the concrete type) with recording overrides; the runner is an
input namespace (CONTRACT R2). ``take_last_draft_probs`` is the real
inherited method reading ``_last_draft_probs`` — only its payload is set
up. ``_get_positions`` is bound for real (pure CPU tensor slicing).
"""

from types import SimpleNamespace

import torch

from vllm.v1.spec_decode.eagle import EagleProposer

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


class _EagleProposer(EagleProposer):
    """Recording EagleProposer: never touches a GPU."""

    def __init__(self, calls, result, last_draft_probs):
        self._calls = calls
        self._result = result
        self._last_draft_probs = last_draft_probs
        self.supports_mm_inputs = False

    def prepare_next_token_ids_padded(self, *args, **kwargs):
        self._calls.append(("prepare_padded", args, kwargs))
        return "next_tokens", 7

    def prepare_next_token_ids_cpu(self, *args, **kwargs):
        self._calls.append(("prepare_cpu", args, kwargs))
        return "next_tokens_cpu"

    def prepare_inputs_padded(self, *args, **kwargs):
        self._calls.append(("prepare_inputs_padded", args, kwargs))
        # Replaced common attention metadata exposing 4 actual tokens.
        return (
            SimpleNamespace(num_actual_tokens=4),
            "indices_to_sample",
            "num_rejected_gpu",
        )

    def prepare_inputs(self, *args, **kwargs):
        self._calls.append(("prepare_inputs", args, kwargs))
        return "attn-md-2", torch.tensor([0, 2])

    def propose(self, *args, **kwargs):
        self._calls.append(("propose", args, kwargs))
        return self._result


def _spec_config(disable_padded=False):
    cfg = SimpleNamespace(
        method="eagle",
        disable_padded_drafter_batch=disable_padded,
    )
    cfg.use_eagle = lambda: True
    cfg.use_dflash = lambda: False
    cfg.uses_draft_model = lambda: False
    cfg.uses_extract_hidden_states = lambda: False
    cfg.use_ngram_gpu = lambda: False
    return cfg


def _runner(spec_config, drafter, **extra):
    runner = SimpleNamespace(
        speculative_config=spec_config,
        drafter=drafter,
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            num_tokens_no_spec=[4, 2],
            token_ids_cpu=[[1, 2, 3, 4], [5, 6]],
            req_ids=["r0", "r1"],
        ),
        requests="reqs",
        discard_request_mask=SimpleNamespace(gpu="mask"),
        _draft_probs="stale-probs",
        _draft_prob_req_ids="stale-ids",
        supports_mm_inputs=False,
        use_aux_hidden_state_outputs=False,
        get_model=lambda: SimpleNamespace(),
        # Real position lookup: plain tensor slicing on the CPU.
        uses_mrope=False,
        uses_xdrope_dim=0,
        positions=torch.arange(24, dtype=torch.int64).reshape(6, 4),
        input_ids=SimpleNamespace(
            gpu=torch.arange(12, dtype=torch.int64).reshape(6, 2)
        ),
    )
    runner._get_positions = ModelRunnerFL._get_positions.__get__(
        runner,
        ModelRunnerFL,
    )
    runner._copy_valid_sampled_token_count = lambda nt, count: runner.calls.setdefault(
        "copy_count", []
    ).append((nt, count))
    runner.calls = {}
    for key, value in extra.items():
        setattr(runner, key, value)
    return runner


def _propose(
    runner,
    *,
    sampled_token_ids,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states=None,
    spec_decode_metadata=None,
    slot_mappings=None,
):
    return ModelRunnerFL.propose_draft_token_ids(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=6,
            num_spec_tokens_to_schedule=3,
            num_scheduled_tokens={"r0": 4, "r1": 2},
        ),
        sampled_token_ids=sampled_token_ids,
        sampling_metadata="sampling-md",
        hidden_states=hidden_states,
        sample_hidden_states=sample_hidden_states,
        aux_hidden_states=aux_hidden_states,
        spec_decode_metadata=spec_decode_metadata,
        common_attn_metadata="attn-md",
        slot_mappings=slot_mappings,
    )


def test_propose_eagle_padded_first_sample_schedules_drafter_inputs():
    """First spec step (no draft metadata): padded prep, then a range slice
    of input ids / positions / hidden states is proposed verbatim."""
    calls = []
    drafter = _EagleProposer(calls, result=[[1, 2, 3]], last_draft_probs="dp")
    runner = _runner(_spec_config(), drafter)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    sampled = torch.zeros(2, 2, dtype=torch.int32)

    result = _propose(
        runner,
        sampled_token_ids=sampled,
        hidden_states=hidden_states,
        sample_hidden_states=hidden_states,
    )

    assert result == [[1, 2, 3]]
    kinds = [c[0] for c in calls]
    assert kinds == ["prepare_padded", "propose"]
    # Padded prep saw the sampled GPU tensor and the request state.
    assert calls[0][1] == (sampled, "reqs", runner.input_batch, "mask")
    assert runner.calls["copy_count"] == [("next_tokens", 7)]
    _, _, kwargs = calls[1]
    assert kwargs["num_speculative_tokens"] == 3
    assert kwargs["next_token_ids"] == "next_tokens"
    assert kwargs["token_indices_to_sample"] is None
    assert kwargs["num_rejected_tokens_gpu"] is None
    assert kwargs["mm_embed_inputs"] is None
    # Range slices of the target model's forward outputs.
    torch.testing.assert_close(kwargs["target_token_ids"], runner.input_ids.gpu)
    torch.testing.assert_close(kwargs["target_positions"], runner.positions)
    torch.testing.assert_close(kwargs["target_hidden_states"], hidden_states)
    # The drafter's last draft probs were cached for the scheduler.
    assert runner._draft_probs == "dp"
    assert runner._draft_prob_req_ids == ["r0", "r1"]
    assert runner._draft_prob_req_ids is not runner.input_batch.req_ids


def test_propose_eagle_padded_with_draft_metadata_and_mm_inputs():
    """Follow-up spec steps: prepare_inputs_padded rewires the attention
    metadata, aux hidden states are concatenated, and multimodal inputs
    are gathered for mm-capable drafters."""
    calls = []
    drafter = _EagleProposer(calls, result=[[1]], last_draft_probs=None)
    drafter.supports_mm_inputs = True
    gathered = []
    runner = _runner(
        _spec_config(),
        drafter,
        supports_mm_inputs=True,
        use_aux_hidden_state_outputs=True,
        _gather_mm_embeddings=lambda sched, shift_computed_tokens: (
            gathered.append((sched, shift_computed_tokens)) or "mm-embed"
        ),
    )
    aux = [
        torch.arange(12, dtype=torch.float32).reshape(6, 2),
        torch.arange(12, 24, dtype=torch.float32).reshape(6, 2),
    ]
    hidden_states = torch.zeros(6, 2)

    result = _propose(
        runner,
        sampled_token_ids=torch.zeros(2, 2, dtype=torch.int32),
        hidden_states=hidden_states,
        sample_hidden_states=hidden_states,
        aux_hidden_states=aux,
        spec_decode_metadata=SimpleNamespace(num_draft_tokens=[1, 2]),
    )

    assert result == [[1]]
    kinds = [c[0] for c in calls]
    assert kinds == ["prepare_padded", "prepare_inputs_padded", "propose"]
    _, _, kwargs = calls[1]
    # The padded input prep took the original attention metadata, the
    # draft metadata and the valid sample count from the padded prep.
    attn_md_arg, spec_md_arg, count_arg = calls[1][1]
    assert attn_md_arg == "attn-md"
    assert spec_md_arg.num_draft_tokens == [1, 2]
    assert count_arg == 7
    # Multimodal embeddings were gathered with the token shift.
    assert len(gathered) == 1 and gathered[0][1] == 1
    _, _, kwargs = calls[2]
    # Rewired metadata, gathered mm inputs and rejected-token count ride
    # along into the proposal.
    assert kwargs["common_attn_metadata"].num_actual_tokens == 4
    assert kwargs["mm_embed_inputs"] == "mm-embed"
    assert kwargs["token_indices_to_sample"] == "indices_to_sample"
    assert kwargs["num_rejected_tokens_gpu"] == "num_rejected_gpu"
    assert kwargs["next_token_ids"] == "next_tokens"
    # The target slices follow the rewired token count (4 of 6 rows).
    torch.testing.assert_close(
        kwargs["target_hidden_states"],
        torch.cat([aux[0][:4], aux[1][:4]], dim=-1),
    )
    torch.testing.assert_close(kwargs["target_token_ids"], runner.input_ids.gpu[:4])


def test_propose_eagle_unpadded_uses_prepare_inputs_token_indices():
    """disable_padded_drafter_batch: CPU sampled lists flow through
    prepare_next_token_ids_cpu / prepare_inputs with a token-index gather,
    and the MTP hidden-state override replaces the forward output."""
    calls = []
    drafter = _EagleProposer(calls, result=[[2]], last_draft_probs=None)
    mtp_hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2) + 100
    runner = _runner(
        _spec_config(disable_padded=True),
        drafter,
        get_model=lambda: SimpleNamespace(
            get_mtp_target_hidden_states=lambda: mtp_hidden,
        ),
    )
    hidden_states = torch.zeros(6, 2)
    token_indices = torch.tensor([0, 2])

    result = _propose(
        runner,
        sampled_token_ids=[[7], [8]],
        hidden_states=hidden_states,
        sample_hidden_states=hidden_states,
        spec_decode_metadata=SimpleNamespace(num_draft_tokens=[1, 2]),
    )

    assert result == [[2]]
    kinds = [c[0] for c in calls]
    assert kinds == ["prepare_cpu", "prepare_inputs", "propose"]
    assert calls[0][1] == ([[7], [8]], "reqs", runner.input_batch, {"r0": 4, "r1": 2})
    # prepare_inputs got the original metadata and the draft-token counts.
    assert calls[1][1] == ("attn-md", [[7], [8]], [1, 2])
    _, _, kwargs = calls[2]
    assert kwargs["next_token_ids"] == "next_tokens_cpu"
    assert kwargs["common_attn_metadata"] == "attn-md-2"
    # Token-indexed gathers: input ids at [0, 2], positions at [0, 2], and
    # the MTP hidden-state override at those same indices.
    torch.testing.assert_close(
        kwargs["target_token_ids"], runner.input_ids.gpu[token_indices]
    )
    torch.testing.assert_close(
        kwargs["target_positions"], runner.positions[token_indices]
    )
    torch.testing.assert_close(
        kwargs["target_hidden_states"], mtp_hidden[token_indices]
    )
    # No draft probs were published (payload None).
    assert runner._draft_probs is None
    assert runner._draft_prob_req_ids is None
