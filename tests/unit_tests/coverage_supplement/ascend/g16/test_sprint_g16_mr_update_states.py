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

"""G16 coverage: ModelRunnerFL._update_states (model_runner.py L1387-1760).

Targets the state/persistent-batch reconciliation performed every scheduler
step: finished-request removal, KV block zeroing/copies, encoder-cache free,
unscheduled removal, new/resumed/streaming request insertion, sampling
generator construction, pooling prompt-logprob bookkeeping, M-RoPE init,
conditional spec-decode/computed-token updates, PP token propagation, block
id updates, ngram_gpu bookkeeping and the deferred spec-decode correction
closure.

Everything runs on CPU. InputBatch / scheduler output are SimpleNamespace
fakes; module-level collaborators (get_pp_group, copy_kv_cache_blocks_inplace,
update_scheduler_for_invalid_drafts, update_ngram_gpu_tensors_incremental,
length_from_prompt_token_ids_or_embeds) are patched at the module boundary.
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


def _cached(
    req_ids=(),
    num_computed=(),
    new_block_ids=(),
    resumed=(),
    num_output=(),
    new_token_ids=None,
    all_token_ids=None,
):
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_computed_tokens=list(num_computed),
        new_block_ids=list(new_block_ids),
        resumed_req_ids=list(resumed),
        num_output_tokens=list(num_output),
        new_token_ids=new_token_ids,
        all_token_ids=all_token_ids or {},
    )


def _sched(
    *,
    finished=None,
    new_reqs=None,
    num_scheduled=None,
    cached=None,
    spec_tokens=None,
    zero=None,
    copies=None,
    free_hashes=None,
):
    return SimpleNamespace(
        finished_req_ids=finished or [],
        scheduled_new_reqs=new_reqs or [],
        num_scheduled_tokens=num_scheduled or {},
        scheduled_cached_reqs=cached or _cached(),
        scheduled_spec_decode_tokens=spec_tokens or {},
        new_block_ids_to_zero=zero or [],
        kv_cache_block_copies=copies or [],
        free_encoder_mm_hashes=free_hashes or [],
    )


def _batch(**over):
    base = dict(
        req_id_to_index={},
        prev_req_id_to_index=None,
        num_computed_tokens_cpu=torch.zeros(8, dtype=torch.int32),
        num_tokens_no_spec=torch.zeros(8, dtype=torch.int64),
        token_ids_cpu=torch.zeros((8, 16), dtype=torch.int32),
        is_token_ids=torch.zeros((8, 16), dtype=torch.bool),
        num_prompt_tokens=[0] * 8,
        vocab_size=1000,
        block_table=SimpleNamespace(append_row=lambda *a: None),
        remove_request=lambda rid: None,
        add_request=lambda req: None,
        update_req_spec_token_ids=lambda req, spec: None,
        condense=lambda: None,
        refresh_metadata=lambda: None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _runner(batch, requests=None, **over):
    base = dict(
        requests={} if requests is None else requests,
        num_prompt_logprobs={},
        late_interaction_runner=SimpleNamespace(
            on_requests_finished=lambda ids: None,
            register_request=lambda rid, pooling: None,
        ),
        input_batch=batch,
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        is_pooling_model=False,
        uses_mrope=False,
        device=torch.device("cpu"),
        _on_request_state_removed=lambda rid, st: None,
        _process_encoder_cache_scheduler_output=lambda so: None,
        _may_reorder_batch=lambda so: None,
        _zero_block_ids=lambda ids: None,
        _init_mrope_positions=lambda st: None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _call(runner, sched, last_rank=True, extras=()):
    pairs = [(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=last_rank))]
    pairs.extend(extras)
    with _stack(*pairs):
        return mr.ModelRunnerFL._update_states(runner, sched)


def _new_req(
    req_id="n0",
    prompt=(1, 2),
    sampling=None,
    pooling=None,
    block_ids=((0,),),
):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        prompt_embeds=None,
        prompt_is_token_ids=True,
        mm_features=[],
        sampling_params=sampling,
        pooling_params=pooling,
        block_ids=[list(b) for b in block_ids],
        num_computed_tokens=0,
        lora_request=None,
    )


def _state(
    *,
    output=None,
    block_ids=((0,),),
    num_computed=0,
    num_tokens=1,
    prev_draft=0,
):
    return SimpleNamespace(
        prev_num_draft_len=prev_draft,
        output_token_ids=[] if output is None else output,
        block_ids=[list(b) for b in block_ids],
        num_computed_tokens=num_computed,
        num_tokens=num_tokens,
    )


# --------------------------------------------------------------------------
# finished / block / encoder housekeeping
# --------------------------------------------------------------------------
def test_update_states_removes_finished_requests():
    calls = []
    batch = _batch(
        remove_request=lambda rid: calls.append(("remove", rid)),
    )
    runner = _runner(batch, requests={"r1": _state()})
    runner.num_prompt_logprobs = {"r1": 5}
    runner._on_request_state_removed = lambda rid, st: calls.append(("hook", rid))
    finished = []
    runner.late_interaction_runner.on_requests_finished = lambda ids: finished.append(
        list(ids)
    )
    _call(runner, _sched(finished=["r1"]))
    assert "r1" not in runner.requests
    assert runner.num_prompt_logprobs == {}
    assert ("hook", "r1") in calls
    assert ("remove", "r1") in calls
    assert finished == [["r1"]]


def test_update_states_zero_blocks_and_copies_kv():
    calls = []
    batch = _batch()
    runner = _runner(batch)
    runner._zero_block_ids = lambda ids: calls.append(("zero", list(ids)))
    runner.kv_caches = {"layer": object()}
    runner.kv_cache_config = SimpleNamespace(num_blocks=10)
    copy_calls = []

    def _copy(kv_caches, num_blocks, copies):
        copy_calls.append((num_blocks, copies))

    _call(
        runner,
        _sched(zero=[3, 4], copies=[("src", "dst")]),
        extras=[(mr, "copy_kv_cache_blocks_inplace", _copy)],
    )
    assert calls == [("zero", [3, 4])]
    assert copy_calls == [(10, [("src", "dst")])]


def test_update_states_drops_unscheduled_requests():
    calls = []
    batch = _batch(
        req_id_to_index={"a": 0, "b": 1},
        remove_request=lambda rid: calls.append(rid),
    )
    runner = _runner(batch)
    _call(runner, _sched(num_scheduled={"a": 1}))
    assert calls == ["b"]


# --------------------------------------------------------------------------
# new requests
# --------------------------------------------------------------------------
def test_update_states_adds_new_request():
    calls = []
    batch = _batch(
        add_request=lambda req: calls.append(("add", req.req_id)),
        update_req_spec_token_ids=lambda req, spec: calls.append(("spec", req.req_id)),
    )
    runner = _runner(batch)
    new = _new_req(sampling=SimpleNamespace(
        sampling_type=None, prompt_logprobs=None, seed=None
    ))
    _call(runner, _sched(new_reqs=[new]))
    assert "n0" in runner.requests
    assert runner.requests["n0"].output_token_ids == []
    assert ("add", "n0") in calls
    assert ("spec", "n0") in calls


def test_update_states_new_request_random_seed_generator():
    batched = _batch()
    runner = _runner(batched)
    sampling = SimpleNamespace(
        sampling_type=mr.SamplingType.RANDOM_SEED, prompt_logprobs=None, seed=1234
    )
    _call(runner, _sched(new_reqs=[_new_req(sampling=sampling)]))
    gen = runner.requests["n0"].generator
    assert isinstance(gen, torch.Generator)
    assert gen.initial_seed() == 1234


def test_update_states_new_request_pooling_applies_task():
    applied = []
    poller = SimpleNamespace(
        get_pooling_updates=lambda task: SimpleNamespace(
            apply=lambda p: applied.append(task)
        )
    )
    batched = _batch()
    runner = _runner(
        batched,
        is_pooling_model=True,
        get_model=lambda: SimpleNamespace(pooler=poller),
    )
    pooling = SimpleNamespace(task="embed")
    _call(runner, _sched(new_reqs=[_new_req(pooling=pooling)]))
    assert applied == ["embed"]


def test_update_states_new_request_prompt_logprobs_sentinel():
    batched = _batch(vocab_size=777)
    runner = _runner(batched)
    sampling = SimpleNamespace(
        sampling_type=None, prompt_logprobs=-1, seed=None
    )
    _call(runner, _sched(new_reqs=[_new_req(sampling=sampling)]))
    assert runner.num_prompt_logprobs["n0"] == 777


def test_update_states_new_request_prompt_logprobs_explicit():
    batched = _batch()
    runner = _runner(batched)
    sampling = SimpleNamespace(
        sampling_type=None, prompt_logprobs=4, seed=None
    )
    _call(runner, _sched(new_reqs=[_new_req(sampling=sampling)]))
    assert runner.num_prompt_logprobs["n0"] == 4


def test_update_states_new_request_initializes_mrope():
    seen = []
    batched = _batch()
    runner = _runner(batched, uses_mrope=True)
    runner._init_mrope_positions = lambda st: seen.append(st)
    _call(runner, _sched(new_reqs=[_new_req()]))
    assert seen == [runner.requests["n0"]]






# --------------------------------------------------------------------------
# cached / running requests
# --------------------------------------------------------------------------




def test_update_states_async_scheduling_recovers_output_tokens():
    added = []
    batched = _batch(add_request=lambda req: added.append(req))
    state = _state(output=[])
    runner = _runner(
        batched,
        requests={"r0": state},
        use_async_scheduling=True,
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[4],
        new_block_ids=[None],
        num_output=[2],
        all_token_ids={"r0": [9, 8, 7, 6]},
    )
    _call(runner, _sched(num_scheduled={"r0": 2}, cached=cached))
    assert state.output_token_ids == [7, 6]
    assert added == [state]






def test_update_states_pp_async_no_new_tokens_advances_cursor():
    batched = _batch(req_id_to_index={"r0": 0})
    batched.num_tokens_no_spec[0] = 1
    state = _state(output=[], num_tokens=1)
    runner = _runner(batched, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[3],
        new_block_ids=[None],
        num_output=[0],
        new_token_ids=None,
    )
    _call(runner, _sched(num_scheduled={"r0": 2}, cached=cached), last_rank=False)
    assert state.output_token_ids == []
    assert batched.num_tokens_no_spec[0] == 3
    assert bool(batched.is_token_ids[0, 1:3].all())


def test_update_states_last_rank_truncates_discarded_tokens():
    batched = _batch(req_id_to_index={"r0": 0})
    batched.num_prompt_tokens[0] = 4
    state = _state(output=[1, 2, 3], num_tokens=4)
    runner = _runner(batched, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[4],
        new_block_ids=[None],
        num_output=[1],
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    assert state.output_token_ids == [1]
    assert batched.num_tokens_no_spec[0] == 5


def test_update_states_truncates_without_batch_index():
    added = []
    batched = _batch(add_request=lambda req: added.append(req))
    state = _state(output=[1, 2])
    runner = _runner(batched, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[1],
        new_block_ids=[None],
        num_output=[1],
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    assert state.output_token_ids == [1]
    assert added == [state]


def test_update_states_last_rank_keeps_full_output():
    appended = []
    batched = _batch(
        req_id_to_index={"r0": 0},
        block_table=SimpleNamespace(append_row=lambda ids, idx: appended.append(ids)),
    )
    state = _state(output=[1, 2, 3], num_tokens=4)
    runner = _runner(batched, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[4],
        new_block_ids=[None],
        num_output=[3],
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    assert state.output_token_ids == [1, 2, 3]
    assert appended == []


def test_update_states_async_spec_clears_draft_len_when_not_batched():
    added = []
    batched = _batch(add_request=lambda req: added.append(req))
    state = _state(prev_draft=2)
    runner = _runner(
        batched,
        requests={"r0": state},
        use_async_scheduling=True,
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[0],
        new_block_ids=[None],
        num_output=[0],
    )
    _call(runner, _sched(num_scheduled={"r0": 0}, cached=cached))
    assert state.prev_num_draft_len == 0
    assert added == [state]


def test_update_states_async_spec_defers_and_corrects_counts():
    batched = _batch(req_id_to_index={"r0": 0}, prev_req_id_to_index={"r0": 0})
    batched.num_computed_tokens_cpu[0] = 0
    state = _state(prev_draft=2, output=[])
    prev_np = np.zeros(4, dtype=np.int32)
    runner = _runner(
        batched,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=prev_np),
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[10],
        new_block_ids=[None],
        num_output=[2],
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 2}, cached=cached))
    assert callable(correct)
    assert state.output_token_ids == [-1, -1]
    assert prev_np[0] == 2
    runner._get_valid_sampled_token_count = lambda: [2]
    correct()
    assert state.num_computed_tokens == 9
    assert batched.num_computed_tokens_cpu[0] == 9


def test_update_states_correction_closure_skips_without_valid_counts():
    batched = _batch(req_id_to_index={"r0": 0}, prev_req_id_to_index={"r0": 0})
    state = _state(prev_draft=1, output=[])
    runner = _runner(
        batched,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=np.zeros(4, dtype=np.int32)),
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[5],
        new_block_ids=[None],
        num_output=[1],
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    runner._get_valid_sampled_token_count = lambda: []
    correct()
    assert state.num_computed_tokens == 5


def test_update_states_correction_closure_skips_without_prev_index():
    batched = _batch(req_id_to_index={"r0": 0}, prev_req_id_to_index=None)
    state = _state(prev_draft=1, output=[])
    runner = _runner(
        batched,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=np.zeros(4, dtype=np.int32)),
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[5],
        new_block_ids=[None],
        num_output=[1],
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    runner._get_valid_sampled_token_count = lambda: [4]
    correct()
    assert state.num_computed_tokens == 5


def test_update_states_ngram_gpu_restores_draft_len_and_syncs():
    scheduler_calls = []
    incremental = []
    batched = _batch(req_id_to_index={"r0": 0})
    state = _state(prev_draft=1, output=[])
    runner = _runner(
        batched,
        requests={"r0": state},
        speculative_config=SimpleNamespace(use_ngram_gpu=lambda: True),
        use_async_spec_decode=True,
        prev_num_draft_tokens=SimpleNamespace(
            np=np.array([5, 5], dtype=np.int32)
        ),
        _num_valid_draft_tokens_event=object(),
        _num_valid_draft_tokens_cpu=torch.zeros(8, dtype=torch.int32),
        token_ids_gpu_tensor=torch.zeros((8, 16), dtype=torch.int32),
        num_tokens_no_spec_gpu=torch.zeros(8, dtype=torch.int32),
        _ngram_pinned_idx_buf=None,
        _ngram_pinned_val_buf=None,
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[1],
        new_block_ids=[None],
        num_output=[0],
    )
    _call(
        runner,
        _sched(num_scheduled={"r0": 1}, cached=cached, spec_tokens={"r0": [1, 2]}),
        extras=[
            (
                mr,
                "update_scheduler_for_invalid_drafts",
                lambda *a, **k: scheduler_calls.append(a),
            ),
            (
                mr,
                "update_ngram_gpu_tensors_incremental",
                lambda *a, **k: incremental.append(a),
            ),
        ],
    )
    assert len(scheduler_calls) == 1
    assert runner.prev_num_draft_tokens.np.tolist() == [0, 0]
    assert state.prev_num_draft_len == 2
    assert len(incremental) == 1
