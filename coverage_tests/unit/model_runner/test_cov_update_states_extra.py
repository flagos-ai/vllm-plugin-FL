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

"""Coverage tests for the remaining ModelRunnerFL._update_states branches.

The base state-machine transitions are in test_cov_update_states.py; this
file drives the specialized bookkeeping: streaming-request replacement,
pooling-model task updates, M-RoPE/XD-RoPE position initialization, the
async-spec-decode deferred correction callable (including its guard
clauses), pipeline-parallel first-rank token propagation, ngram-gpu tensor
bookkeeping, block zeroing, and the async draft-count reset.

Every test asserts observable state transitions or the exact arguments a
boundary collaborator received.
"""

from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import (
    _model_config,
    _vllm_config,
    make_runner,
    make_scheduler_output,
    mr_module,
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


class _CpuEvent:
    """torch.Event stand-in; the cpu backend supports no real events."""

    def __init__(self):
        self.records = 0

    def record(self):
        self.records += 1

    def synchronize(self):
        pass


def _arm_kv_cache(runner):
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
    zero_blocks=None,
):
    from vllm.v1.core.sched.output import CachedRequestData

    num_scheduled_tokens = num_scheduled_tokens or {}
    kwargs = dict(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=(
            total if total is not None else sum(num_scheduled_tokens.values())
        ),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=list(free_mm),
    )
    if zero_blocks is not None:
        kwargs["new_block_ids_to_zero"] = list(zero_blocks)
    return make_scheduler_output(**kwargs)


def _new_req(req_id, prompt, sampling_params=None, pooling_params=None, **kw):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=kw.pop("mm_features", []),
        sampling_params=sampling_params or SamplingParams(),
        pooling_params=pooling_params,
        block_ids=kw.pop("block_ids", ([0],)),
        num_computed_tokens=kw.pop("num_computed_tokens", 0),
        lora_request=None,
        **kw,
    )


def _cached(
    req_ids,
    *,
    computed,
    new_block_ids,
    new_token_ids=None,
    output_tokens=None,
    resumed=(),
    all_token_ids=None,
):
    from vllm.v1.core.sched.output import CachedRequestData

    return CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(resumed),
        new_token_ids=(
            list(new_token_ids) if new_token_ids is not None else [[] for _ in req_ids]
        ),
        all_token_ids=all_token_ids or {},
        new_block_ids=list(new_block_ids),
        num_computed_tokens=list(computed),
        num_output_tokens=(
            list(output_tokens) if output_tokens is not None else [0] * len(req_ids)
        ),
    )


def test_update_states_streaming_request_reuses_state_and_clears_outputs():
    """A repeated req_id updates the cached request in place for streaming."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        runner._update_states(
            _sched_output(
                new_reqs=[_new_req("r1", [1, 2])],
                num_scheduled_tokens={"r1": 2},
            )
        )
        state = runner.requests["r1"]
        state.output_token_ids.extend([9])

        # The continuation carries the previously generated token inside
        # the prompt and a fresh block list.
        cont = _new_req(
            "r1",
            [1, 2, 9, 10],
            block_ids=([3],),
            num_computed_tokens=1,
        )
        runner._update_states(
            _sched_output(new_reqs=[cont], num_scheduled_tokens={"r1": 4})
        )

        assert runner.requests["r1"] is state
        assert state.prompt_token_ids == [1, 2, 9, 10]
        assert state.num_prompt_tokens == 4
        assert state.output_token_ids == []
        assert state.block_ids == ([3],)
        assert state.num_computed_tokens == 1
        # Removed and re-added through the persistent batch: one row.
        assert runner.input_batch.req_id_to_index == {"r1": 0}
        assert runner.input_batch.num_reqs == 1
    finally:
        patches.__exit__(None, None, None)


def test_update_states_pooling_request_applies_pooler_task_update():
    """A pooling request applies the pooler's task update to its params."""
    from vllm.pooling_params import PoolingParams

    runner, patches = make_runner(_vllm_config(_model_config(runner_type="pooling")))
    try:
        _arm_kv_cache(runner)
        applied = []

        class _Update:
            def apply(self, params):
                applied.append(params)

        class _Pooler:
            def get_pooling_updates(self, task):
                assert task == "embed"
                return _Update()

        class _Model:
            pooler = _Pooler()

        runner.model = _Model()
        params = PoolingParams(task="embed")
        runner._update_states(
            _sched_output(
                new_reqs=[_new_req("p1", [4, 5, 6], pooling_params=params)],
                num_scheduled_tokens={"p1": 3},
            )
        )

        assert applied == [params]
        assert runner.requests["p1"].pooling_params is params
        # Pooling params are not random sampling params: no generator.
        assert runner.requests["p1"].generator is None
    finally:
        patches.__exit__(None, None, None)


class _MRopeModel:
    """Fake multimodal model publishing M-RoPE input positions."""

    def __init__(self):
        self.calls = []

    def get_mrope_input_positions(self, input_tokens, mm_features):
        self.calls.append((list(input_tokens), list(mm_features)))
        positions = torch.arange(3 * len(input_tokens), dtype=torch.int64).reshape(
            3, len(input_tokens)
        )
        return positions, 7


class _XDRopeModel:
    """Fake model publishing XD-RoPE input positions."""

    def __init__(self):
        self.calls = []

    def get_xdrope_input_positions(self, input_tokens, mm_features):
        self.calls.append((list(input_tokens), list(mm_features)))
        return torch.arange(4 * len(input_tokens), dtype=torch.int64).reshape(
            4, len(input_tokens)
        )


def test_update_states_mrope_request_initializes_positions_and_delta():
    """M-RoPE configs compute the request's positions and delta."""
    runner, patches = make_runner(_vllm_config(_model_config(uses_mrope=True)))
    try:
        _arm_kv_cache(runner)
        model = _MRopeModel()
        runner.model = model

        with _Patch(mr_module, "supports_mrope", lambda m: True):
            runner._update_states(
                _sched_output(
                    new_reqs=[_new_req("r1", [5, 6])],
                    num_scheduled_tokens={"r1": 2},
                )
            )

        state = runner.requests["r1"]
        assert model.calls == [([5, 6], [])]
        assert state.mrope_position_delta == 7
        torch.testing.assert_close(
            state.mrope_positions,
            torch.arange(6, dtype=torch.int64).reshape(3, 2),
        )
    finally:
        patches.__exit__(None, None, None)


def test_update_states_xdrope_request_initializes_positions():
    """XD-RoPE configs compute the request's four-dimension positions."""
    runner, patches = make_runner(_vllm_config(_model_config(uses_xdrope_dim=4)))
    try:
        _arm_kv_cache(runner)
        model = _XDRopeModel()
        runner.model = model

        with _Patch(mr_module, "supports_xdrope", lambda m: True):
            runner._update_states(
                _sched_output(
                    new_reqs=[_new_req("r1", [1, 2, 3])],
                    num_scheduled_tokens={"r1": 3},
                )
            )

        state = runner.requests["r1"]
        assert model.calls == [([1, 2, 3], [])]
        torch.testing.assert_close(
            state.xdrope_positions,
            torch.arange(12, dtype=torch.int64).reshape(4, 3),
        )
    finally:
        patches.__exit__(None, None, None)


def test_update_states_async_spec_correction_callable_fixes_computed_tokens():
    """The returned corrector subtracts rejected optimistic draft tokens."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        runner._update_states(
            _sched_output(
                new_reqs=[_new_req("r1", [1, 2])],
                num_scheduled_tokens={"r1": 2},
            )
        )
        state = runner.requests["r1"]

        # Async spec decode saw two drafts accepted optimistically last
        # step; this step the sampler reports only one accepted token.
        runner.use_async_scheduling = True
        state.prev_num_draft_len = 2
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner.valid_sampled_token_count_event = _CpuEvent()
        runner.valid_sampled_token_count_cpu = torch.tensor([2], dtype=torch.int32)
        runner.input_batch.prev_sampled_token_ids = torch.zeros(1, 1)

        correction = runner._update_states(
            _sched_output(
                cached=_cached(
                    ["r1"],
                    computed=[5],
                    new_block_ids=[None],
                ),
                num_scheduled_tokens={"r1": 1},
                total=1,
            )
        )
        assert callable(correction)
        # _update_states installed the scheduler's computed count.
        assert state.num_computed_tokens == 5
        assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 5

        correction()
        # valid_sampled_token_count 2 -> accepted 1; optimistic 2, so the
        # correction is 2 - 1 = 1 token removed from both caches.
        assert state.num_computed_tokens == 4
        assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 4
    finally:
        patches.__exit__(None, None, None)


def test_update_states_async_spec_correction_guards_leave_state_untouched():
    """The corrector does nothing when counts or the previous mapping are absent."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        runner._update_states(
            _sched_output(
                new_reqs=[_new_req("r1", [1, 2])],
                num_scheduled_tokens={"r1": 2},
            )
        )
        state = runner.requests["r1"]
        runner.use_async_scheduling = True
        state.prev_num_draft_len = 1
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner.input_batch.prev_sampled_token_ids = torch.zeros(1, 1)

        correction = runner._update_states(
            _sched_output(
                cached=_cached(["r1"], computed=[6], new_block_ids=[None]),
                num_scheduled_tokens={"r1": 1},
                total=1,
            )
        )
        assert callable(correction)

        # No validity event: the count is unavailable, so nothing moves.
        runner.valid_sampled_token_count_event = None
        correction()
        assert state.num_computed_tokens == 6
        assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 6

        # Event present but the previous-iteration mapping is empty.
        runner.valid_sampled_token_count_event = _CpuEvent()
        runner.valid_sampled_token_count_cpu = torch.tensor([1], dtype=torch.int32)
        runner.input_batch.prev_req_id_to_index = None
        correction()
        assert state.num_computed_tokens == 6
        assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 6
    finally:
        patches.__exit__(None, None, None)


def test_update_states_first_pp_rank_appends_sampled_tokens():
    """A non-last PP rank copies the scheduler's sampled ids into the batch."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        pp = SimpleNamespace(
            is_first_rank=True,
            is_last_rank=False,
            world_size=1,
            ranks=[0],
        )
        with _Patch(mr_module, "get_pp_group", lambda: pp):
            runner._update_states(
                _sched_output(
                    new_reqs=[_new_req("r1", [1, 2, 3])],
                    num_scheduled_tokens={"r1": 3},
                )
            )
            runner._update_states(
                _sched_output(
                    cached=_cached(
                        ["r1"],
                        computed=[3],
                        new_block_ids=[None],
                        new_token_ids=[[9]],
                    ),
                    num_scheduled_tokens={"r1": 3},
                    total=3,
                )
            )

        state = runner.requests["r1"]
        # computed == prompt length: exactly one sampled token arrived.
        assert state.output_token_ids == [9]
        assert int(runner.input_batch.token_ids_cpu[0, 3]) == 9
        assert bool(runner.input_batch.is_token_ids[0, 3])
        assert int(runner.input_batch.num_tokens_no_spec[0]) == 4
    finally:
        patches.__exit__(None, None, None)


def test_update_states_first_pp_rank_async_step_has_no_new_token_ids():
    """Async PP has no per-request sampled ids and leaves the batch aligned."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        pp = SimpleNamespace(
            is_first_rank=True,
            is_last_rank=False,
            world_size=1,
            ranks=[0],
        )
        runner.use_async_scheduling = True
        with _Patch(mr_module, "get_pp_group", lambda: pp):
            runner._update_states(
                _sched_output(
                    new_reqs=[_new_req("r1", [1, 2, 3])],
                    num_scheduled_tokens={"r1": 3},
                )
            )
            runner._update_states(
                _sched_output(
                    cached=_cached(
                        ["r1"],
                        computed=[4],
                        new_block_ids=[None],
                        new_token_ids=[],
                    ),
                    num_scheduled_tokens={"r1": 1},
                    total=1,
                )
            )

        state = runner.requests["r1"]
        assert state.output_token_ids == []
        # No sampled ids are appended; num_tokens_no_spec advances only to
        # the scheduler's computed count (4 of the 3-token prompt + 1).
        assert int(runner.input_batch.num_tokens_no_spec[0]) == 4
    finally:
        patches.__exit__(None, None, None)


def test_update_states_ngram_gpu_updates_scheduler_and_tensors():
    """ngram-gpu requests trigger the scheduler fix-up and tensor refresh."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        calls = {"scheduler": [], "tensors": []}

        def _fix_scheduler(event, cpu_buf, sched, mapping):
            calls["scheduler"].append((event, cpu_buf, sched, mapping))

        def _update_tensors(batch, token_tensor, num_tokens, new_reqs, device, **kw):
            calls["tensors"].append(
                (batch, token_tensor, num_tokens, new_reqs, device, kw)
            )

        runner.speculative_config = SimpleNamespace(use_ngram_gpu=lambda: True)
        runner._num_valid_draft_tokens_event = "event"
        runner._num_valid_draft_tokens_cpu = "cpu-buf"
        runner.token_ids_gpu_tensor = "token-tensor"
        runner.num_tokens_no_spec_gpu = "num-tokens-tensor"
        runner._ngram_pinned_idx_buf = "idx-buf"
        runner._ngram_pinned_val_buf = "val-buf"

        with (
            _Patch(mr_module, "update_scheduler_for_invalid_drafts", _fix_scheduler),
            _Patch(
                mr_module,
                "update_ngram_gpu_tensors_incremental",
                _update_tensors,
            ),
        ):
            runner._update_states(
                _sched_output(
                    new_reqs=[_new_req("r1", [4, 5])],
                    num_scheduled_tokens={"r1": 2},
                )
            )

        state = runner.requests["r1"]
        assert len(calls["scheduler"]) == 1
        event, cpu_buf, sched, mapping = calls["scheduler"][0]
        assert (event, cpu_buf) == ("event", "cpu-buf")
        assert mapping is runner.input_batch.req_id_to_index
        assert len(calls["tensors"]) == 1
        batch, token_tensor, num_tensor, new_reqs, device, extra = calls["tensors"][0]
        assert batch is runner.input_batch
        assert (token_tensor, num_tensor) == ("token-tensor", "num-tokens-tensor")
        assert new_reqs == [state]
        assert device == runner.device
        assert extra["_pinned_idx_buf"] == "idx-buf"
        assert extra["_pinned_val_buf"] == "val-buf"
    finally:
        patches.__exit__(None, None, None)


def test_update_states_zeros_fresh_cache_blocks():
    """Scheduler-reported freshly allocated blocks are zeroed before use."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        zeroed = []
        runner._kv_block_zeroer = SimpleNamespace(
            zero_block_ids=lambda ids: zeroed.append(list(ids))
        )
        runner._update_states(
            _sched_output(num_scheduled_tokens={}, zero_blocks=[7, 8])
        )
        assert zeroed == [[7, 8]]
    finally:
        patches.__exit__(None, None, None)


def test_update_states_async_spec_decode_resets_prev_draft_counts():
    """Async spec decode clears the previous step's draft counters."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        runner.use_async_spec_decode = True
        runner.prev_num_draft_tokens.np[:] = 5
        runner._update_states(_sched_output(num_scheduled_tokens={}))
        assert runner.prev_num_draft_tokens.np.tolist() == [0, 0, 0, 0]
    finally:
        patches.__exit__(None, None, None)
