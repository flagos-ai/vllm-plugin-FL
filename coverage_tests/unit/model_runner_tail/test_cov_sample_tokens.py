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

"""Coverage tests for ModelRunnerFL.sample_tokens (vllm_fl/worker/
model_runner.py): the no-forward passthrough, the synchronous sampling
pipeline assembled from recording collaborators, and the speculative
draft-zeroing when the drafter input does not fit.

The pipeline state is a real ``ExecuteModelState``; the sampler,
bookkeeping, EPLB and drafter are inputs replaced with recorders
(CONTRACT R2). ``get_pp_group`` is patched at the module boundary
(torch.distributed), the only module-level patch.
"""

from types import SimpleNamespace

import torch

import vllm_fl.worker.model_runner as model_runner_module
from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ExecuteModelState, ModelRunnerFL


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


def _pp_group_singleton():
    return SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        world_size=1,
        rank=0,
    )


def test_sample_tokens_without_forward_returns_kv_connector_passthrough():
    """No pending state: the stored connector output is returned alone."""
    for stored in (None, SimpleNamespace(is_empty=lambda: True)):
        runner = SimpleNamespace(
            execute_model_state=None,
            kv_connector_output=stored,
            use_async_scheduling=False,
        )
        with _Patch(model_runner_module, "get_pp_group", _pp_group_singleton()):
            result = ModelRunnerFL.sample_tokens(runner, grammar_output=None)
        # An absent/empty connector output maps to the shared empty output.
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

        assert result is EMPTY_MODEL_RUNNER_OUTPUT
        assert runner.kv_connector_output is None


def test_sample_tokens_pipeline_returns_model_runner_output():
    """The full sync pipeline packs bookkeeping values into the output.

    execute_model is first emulated by storing a real ExecuteModelState;
    sample_tokens then samples, updates post-execute state, bookkeeps, and
    returns the ModelRunnerOutput built from the recorded bookkeeping
    values.
    """
    calls = SimpleNamespace(
        sample=[],
        post_execute=[],
        bookkeeping=[],
        eplb=[],
        finalize=[],
        sampled_token_ids=None,
    )
    logits = torch.zeros(2, 4)
    hidden_states = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    sample_hidden_states = hidden_states[[2, 4]]
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=5,
        num_scheduled_tokens={"r0": 3, "r1": 2},
    )
    state = ExecuteModelState(
        scheduler_output=scheduler_output,
        logits=logits,
        spec_decode_metadata=None,
        spec_decode_common_attn_metadata=None,
        hidden_states=hidden_states,
        sample_hidden_states=sample_hidden_states,
        aux_hidden_states=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        slot_mappings=None,
    )

    class _SamplerOutput:
        sampled_token_ids = torch.tensor([[10], [20]], dtype=torch.int32)
        logprobs_tensors = None

    def _sample(received_logits, spec_decode_metadata):
        calls.sample.append(received_logits)
        return _SamplerOutput()

    def _update_after(sampled, sched):
        calls.post_execute.append((sampled, sched))

    def _bookkeeping(sched, sampler_out, rec_logits, rec_hidden, total):
        calls.bookkeeping.append((sched, rec_logits, rec_hidden, total))
        return (
            {"r0": 0},  # num_nans_in_logits
            {"r0": None},  # logprobs_lists
            [[10], [20]],  # valid_sampled_token_ids
            {},  # prompt_logprobs_dict
            ["r0", "r1"],  # req_ids
            {"r0": 0, "r1": 1},  # req_id_to_index
            [1],  # invalid_req_indices
        )

    runner = SimpleNamespace(
        execute_model_state=state,
        kv_connector_output=None,
        use_async_scheduling=False,
        _sample=_sample,
        _update_states_after_model_execute=_update_after,
        _draft_token_ids=object(),
        _draft_probs=object(),
        _draft_prob_req_ids=object(),
        _draft_token_req_ids=object(),
        valid_sampled_token_count_gpu=object(),
        input_batch=SimpleNamespace(
            prev_sampled_token_ids=object(),
            vocab_size=4,
        ),
        speculative_config=None,
        _bookkeeping_sync=_bookkeeping,
        finalize_kv_connector=lambda: calls.finalize.append(True),
        eplb_step=lambda **kw: calls.eplb.append(kw),
        supports_mm_inputs=False,
        routed_experts_initialized=False,
    )

    with _Patch(model_runner_module, "get_pp_group", _pp_group_singleton()):
        output = ModelRunnerFL.sample_tokens(runner, grammar_output=None)

    # The recorded pipeline ran in order with the state's tensors.
    assert calls.sample == [logits]
    assert calls.post_execute == [(_SamplerOutput.sampled_token_ids, scheduler_output)]
    assert calls.bookkeeping == [(scheduler_output, logits, hidden_states, 5)]
    assert calls.eplb == [{}]
    # No speculative config: the deferred KV-connector finalize is skipped.
    assert calls.finalize == []
    # Ephemeral draft state is cleared.
    assert runner._draft_token_ids is None
    assert runner._draft_probs is None
    assert runner.valid_sampled_token_count_gpu is None
    assert runner.input_batch.prev_sampled_token_ids is None
    # The returned output carries the bookkeeping values verbatim.
    assert output.req_ids == ["r0", "r1"]
    assert output.req_id_to_index == {"r0": 0, "r1": 1}
    assert output.sampled_token_ids == [[10], [20]]
    assert output.logprobs == {"r0": None}
    assert output.prompt_logprobs_dict == {}
    assert output.num_nans_in_logits == {"r0": 0}
    # State is consumed.
    assert runner.execute_model_state is None


def test_sample_tokens_zeroes_drafts_when_drafter_input_too_long():
    """A spec config whose context does not fit zeroes the draft tokens.

    With no EAGLE / draft-model / extract-hidden-states / ngram-gpu
    method and a drafter input longer than effective_drafter_max_model_len,
    sample_tokens must zero _draft_token_ids for every request and copy
    the zeros to the CPU buffer, then propose nothing after bookkeeping.
    """
    calls = SimpleNamespace(copy_draft=[], propose=[])

    class _SpecConfig:
        # Every method-based selector returns False so the sampled GPU
        # tokens are not used and the proposal is deferred (and then
        # skipped because the input does not fit the drafter).
        def use_eagle(self):
            return False

        def uses_draft_model(self):
            return False

        def uses_extract_hidden_states(self):
            return False

        def use_ngram_gpu(self):
            return False

        disable_padded_drafter_batch = False

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={"r0": 4},
        num_scheduled_tokens_dict=None,
    )
    logits = torch.zeros(1, 4)
    state = ExecuteModelState(
        scheduler_output=scheduler_output,
        logits=logits,
        spec_decode_metadata=None,
        spec_decode_common_attn_metadata=None,
        hidden_states=torch.zeros(4, 1),
        sample_hidden_states=torch.zeros(1, 1),
        aux_hidden_states=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        slot_mappings=None,
    )

    class _SamplerOutput:
        sampled_token_ids = torch.tensor([[7]], dtype=torch.int32)
        logprobs_tensors = None

    def _bookkeeping(*args, **kwargs):
        return ({}, None, [[7]], {}, ["r0", "r1"], {"r0": 0, "r1": 1}, [])

    runner = SimpleNamespace(
        execute_model_state=state,
        kv_connector_output=None,
        use_async_scheduling=False,
        _sample=lambda l, m: _SamplerOutput(),
        _update_states_after_model_execute=lambda sampled, sched: None,
        _draft_token_ids=None,
        _draft_probs=None,
        _draft_prob_req_ids=None,
        _draft_token_req_ids=None,
        valid_sampled_token_count_gpu=None,
        input_batch=SimpleNamespace(
            prev_sampled_token_ids=None,
            req_ids=["r0", "r1"],
            vocab_size=4,
        ),
        speculative_config=_SpecConfig(),
        num_spec_tokens=3,
        device=torch.device("cpu"),
        # Drafter context limit shorter than any real sequence:
        # _input_fits_in_drafter needs spec_decode_common_attn_metadata
        # (None here => False) and this bound.
        effective_drafter_max_model_len=16,
        _bookkeeping_sync=_bookkeeping,
        finalize_kv_connector=lambda: None,
        eplb_step=lambda **kw: None,
        supports_mm_inputs=False,
        routed_experts_initialized=False,
        _copy_draft_token_ids_to_cpu=lambda sched, **kw: calls.copy_draft.append(
            (sched, kw)
        ),
        # _input_fits_in_drafter is bound for real below; it also reads
        # the drafter method used only when the context check passes.
    )
    runner._input_fits_in_drafter = ModelRunnerFL._input_fits_in_drafter.__get__(
        runner,
        ModelRunnerFL,
    )

    with _Patch(model_runner_module, "get_pp_group", _pp_group_singleton()):
        output = ModelRunnerFL.sample_tokens(runner, grammar_output=None)

    # Draft tokens zeroed for both requests x num_spec_tokens and copied
    # to the CPU buffer with zeros_only=True; nothing proposed.
    assert calls.copy_draft == [(scheduler_output, {"zeros_only": True})]
    assert runner._draft_token_ids.shape == (2, 3)
    assert runner._draft_token_ids.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert runner._draft_probs is None
    assert output.sampled_token_ids == [[7]]
