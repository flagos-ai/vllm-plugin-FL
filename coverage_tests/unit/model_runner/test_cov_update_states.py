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

"""Coverage tests for ModelRunnerFL._update_states scheduling state machine.

A real ModelRunnerFL is constructed once per test (CPU recipe from the
area shim); SchedulerOutput/NewRequestData/CachedRequestData are the
public dataclasses the scheduler sends, built with minimal fields. Every
test asserts the state machine's observable transitions: requests
removed from ``requests``/``num_prompt_logprobs``/the persistent batch,
new request state values (RANDOM_SEED generator, prompt_logprobs vocab
fallback), block id append vs replace on resume, and the returned
None/callable contract.
"""

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import make_runner, make_scheduler_output


def _arm_kv_cache(runner):
    """Give the runner the post-initialize_kv_cache state fields it reads.

    ``kv_cache_config`` / ``attn_groups`` are populated by
    initialize_kv_cache (a later lifecycle step outside these tests);
    _update_states and _prepare_inputs only need an empty group list
    here, exactly what an attention-free model would have.
    """
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    runner.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([], None)],  # type: ignore[arg-type]
    )
    runner.attn_groups = []


def _sched_output(
    *,
    new_reqs=(),
    cached=None,
    num_scheduled_tokens=None,
    total=None,
    finished=(),
    free_mm=(),
    spec_tokens=None,
):
    from vllm.v1.core.sched.output import (
        CachedRequestData,
    )

    num_scheduled_tokens = num_scheduled_tokens or {}
    return make_scheduler_output(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=(
            total if total is not None else sum(num_scheduled_tokens.values())
        ),
        scheduled_spec_decode_tokens=spec_tokens or {},
        scheduled_encoder_inputs={},
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=list(free_mm),
    )


def _new_req(req_id, prompt=None, sampling_params=None, block_ids=None, **kw):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt if prompt is not None else [5, 6, 7],
        mm_features=[],
        sampling_params=sampling_params or SamplingParams(),
        pooling_params=None,
        block_ids=block_ids if block_ids is not None else ([1, 2],),
        num_computed_tokens=0,
        lora_request=None,
        **kw,
    )


def _seeded_params(seed, prompt_logprobs=None):
    from vllm.sampling_params import SamplingParams

    params = SamplingParams(seed=seed)
    if prompt_logprobs is not None:
        params.prompt_logprobs = prompt_logprobs
    return params


def test_update_states_adds_new_request_to_cached_state():
    """A new request lands in cached state with a seeded generator."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        params = _seeded_params(1234)
        out = _sched_output(
            new_reqs=[_new_req("r1", prompt=[10, 11], sampling_params=params)],
            num_scheduled_tokens={"r1": 2},
        )
        assert runner._update_states(out) is None

        assert "r1" in runner.requests
        state = runner.requests["r1"]
        assert state.prompt_token_ids == [10, 11]
        assert state.num_prompt_tokens == 2
        assert state.num_tokens == 2
        assert state.output_token_ids == []
        assert state.block_ids == ([1, 2],)
        assert state.generator is not None
        # A seeded generator reproduces the same draw sequence.
        first = torch.rand(1, generator=state.generator).item()
        twin = torch.Generator().manual_seed(1234)
        assert first == torch.rand(1, generator=twin).item()
        # The request landed in the persistent batch too.
        assert runner.input_batch.req_id_to_index["r1"] == 0
        assert runner.input_batch.num_reqs == 1
    finally:
        patches.__exit__(None, None, None)


def test_update_states_unseeded_request_gets_no_generator():
    """Unseeded sampling params leave the request without a generator."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        out = _sched_output(
            new_reqs=[_new_req("r1", sampling_params=_seeded_params(None))],
            num_scheduled_tokens={"r1": 3},
        )
        runner._update_states(out)
        assert runner.requests["r1"].generator is None
    finally:
        patches.__exit__(None, None, None)


def test_update_states_prompt_logprobs_vocab_fallback_and_exact():
    """-1 prompt logprobs falls back to vocab; positive stays exact."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        out = _sched_output(
            new_reqs=[
                _new_req(
                    "r-all",
                    sampling_params=_seeded_params(None, prompt_logprobs=-1),
                ),
                _new_req(
                    "r-two",
                    sampling_params=_seeded_params(None, prompt_logprobs=2),
                ),
            ],
            num_scheduled_tokens={"r-all": 1, "r-two": 1},
        )
        runner._update_states(out)
        # -1 falls back to the input batch vocab size; 2 stays exact.
        assert runner.num_prompt_logprobs["r-all"] == runner.input_batch.vocab_size
        assert runner.num_prompt_logprobs["r-two"] == 2
    finally:
        patches.__exit__(None, None, None)


def test_update_states_removes_finished_and_frees_encoder_cache():
    """Finished requests leave all caches; freed hashes leave encoder."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        runner.requests["gone"] = object()
        runner.requests["kept"] = object()
        runner.num_prompt_logprobs["gone"] = 3
        runner.encoder_cache["mm1"] = torch.zeros(2)
        runner.encoder_cache["mm2"] = torch.zeros(2)
        # "kept" sits in the persistent batch so the unscheduled removal
        # path also runs; "gone" only exists in cached state.
        runner.input_batch.req_id_to_index["kept"] = 0
        runner.input_batch.req_ids.append("kept")

        out = _sched_output(
            num_scheduled_tokens={"kept": 1},
            finished=["gone"],
            free_mm=["mm1"],
        )
        runner._update_states(out)

        assert "gone" not in runner.requests
        assert "kept" in runner.requests
        assert "gone" not in runner.num_prompt_logprobs
        assert "mm1" not in runner.encoder_cache
        assert "mm2" in runner.encoder_cache
    finally:
        patches.__exit__(None, None, None)


def test_update_states_drops_unscheduled_requests_but_keeps_state():
    """Unscheduled requests drop from the batch but keep their state."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        out = _sched_output(
            new_reqs=[_new_req("a"), _new_req("b")],
            num_scheduled_tokens={"a": 1, "b": 1},
        )
        runner._update_states(out)
        assert set(runner.input_batch.req_id_to_index) == {"a", "b"}

        # Next step only schedules "a": "b" is preempted (removed from the
        # persistent batch) but its cached state survives for rescheduling.
        out = _sched_output(num_scheduled_tokens={"a": 1})
        runner._update_states(out)
        assert runner.input_batch.req_id_to_index == {"a": 0}
        assert runner.input_batch.num_reqs == 1
        assert "b" in runner.requests
    finally:
        patches.__exit__(None, None, None)


def test_update_states_appends_blocks_for_running_request():
    """A running request appends new block ids to its existing list."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        out = _sched_output(
            new_reqs=[_new_req("r1", block_ids=([10],))],
            num_scheduled_tokens={"r1": 2},
        )
        runner._update_states(out)
        assert runner.requests["r1"].block_ids == ([10],)

        # A running (not resumed) request: new blocks are appended.
        from vllm.v1.core.sched.output import CachedRequestData

        cached = CachedRequestData(
            req_ids=["r1"],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[([11, 12],)],
            num_computed_tokens=[2],
            num_output_tokens=[0],
        )
        out = _sched_output(cached=cached, num_scheduled_tokens={"r1": 1}, total=1)
        runner._update_states(out)
        assert runner.requests["r1"].block_ids == ([10, 11, 12],)
        assert runner.input_batch.num_computed_tokens_cpu[0] == 2
        # The request stayed at its persistent batch position.
        assert runner.input_batch.req_id_to_index["r1"] == 0
    finally:
        patches.__exit__(None, None, None)


def test_update_states_replaces_blocks_when_resumed_from_preemption():
    """A resumed request replaces its block list with a fresh one."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        out = _sched_output(
            new_reqs=[_new_req("r1", block_ids=([1, 2, 3],))],
            num_scheduled_tokens={"r1": 3},
        )
        runner._update_states(out)

        # Preempt the request (unscheduled this step).
        runner._update_states(_sched_output(num_scheduled_tokens={}))
        assert runner.input_batch.num_reqs == 0
        assert runner.requests["r1"].block_ids == ([1, 2, 3],)

        # Resume: the scheduler hands a fresh block list that replaces the
        # old one entirely.
        from vllm.v1.core.sched.output import CachedRequestData

        cached = CachedRequestData(
            req_ids=["r1"],
            resumed_req_ids={"r1"},
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[([7],)],
            num_computed_tokens=[1],
            num_output_tokens=[0],
        )
        out = _sched_output(cached=cached, num_scheduled_tokens={"r1": 2}, total=2)
        runner._update_states(out)
        assert runner.requests["r1"].block_ids == ([7],)
        assert runner.requests["r1"].num_computed_tokens == 1
        assert runner.input_batch.req_id_to_index["r1"] == 0
    finally:
        patches.__exit__(None, None, None)


def test_update_states_trims_output_tokens_on_discard():
    """Discarded output tokens are trimmed from the request state."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        out = _sched_output(
            new_reqs=[_new_req("r1", prompt=[1, 2])],
            num_scheduled_tokens={"r1": 2},
        )
        runner._update_states(out)
        state = runner.requests["r1"]
        state.output_token_ids.extend([50, 51, 52])
        # Simulate the batch already knowing the prompt+2 outputs.
        runner.input_batch.num_tokens_no_spec[0] = 2 + 3

        # Scheduler reports only 2 output tokens survived (1 discarded).
        from vllm.v1.core.sched.output import CachedRequestData

        cached = CachedRequestData(
            req_ids=["r1"],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[2],
            num_output_tokens=[2],
        )
        out = _sched_output(cached=cached, num_scheduled_tokens={"r1": 1}, total=1)
        runner._update_states(out)
        assert state.output_token_ids == [50, 51]
        # num_tokens_no_spec re-aligned to prompt + kept outputs.
        assert runner.input_batch.num_tokens_no_spec[0] == 4
    finally:
        patches.__exit__(None, None, None)


def test_update_states_returns_none_without_async_spec_corrections():
    """Plain sync scheduling never returns the correction callable."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        out = _sched_output(new_reqs=[_new_req("r1")], num_scheduled_tokens={"r1": 3})
        assert runner._update_states(out) is None

        from vllm.v1.core.sched.output import (
            CachedRequestData,
        )

        empty = make_scheduler_output(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids={"r1"},
            free_encoder_mm_hashes=[],
        )
        assert runner._update_states(empty) is None
        assert "r1" not in runner.requests
    finally:
        patches.__exit__(None, None, None)
