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

"""Coverage tests for ModelRunnerFL._bookkeeping_sync.

The method runs on real CPU-constructed runners (the area shim's
recipe) with a ``SamplerOutput``-shaped namespace: the sync
single-token path (valid rows kept, discarded rows emptied, token
ids appended to request state and the batch caches) and the async
path (prev_sampled_token_ids caching, -1 placeholders, invalid row
tracking).
"""

from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import make_runner, make_scheduler_output
from vllm_fl.worker.model_runner import ModelRunnerFL


def _arm_kv_cache(runner):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    runner.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([], None)],  # type: ignore[arg-type]
    )
    runner.attn_groups = []


def _sched(**kw):
    from vllm.v1.core.sched.output import CachedRequestData

    return make_scheduler_output(
        scheduled_new_reqs=kw.get("new_reqs", []),
        scheduled_cached_reqs=kw.get("cached", CachedRequestData.make_empty()),
        num_scheduled_tokens=kw.get("num_scheduled_tokens", {}),
        total_num_scheduled_tokens=kw.get(
            "total", sum(kw.get("num_scheduled_tokens", {}).values())
        ),
        scheduled_spec_decode_tokens=kw.get("spec_tokens", {}),
        scheduled_encoder_inputs=kw.get("encoder_inputs", {}),
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _prime(runner, prompts):
    """Add requests through the state machine; return a SchedulerOutput."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    new_reqs = []
    for rid, prompt in prompts.items():
        new_reqs.append(
            NewRequestData(
                req_id=rid,
                prompt_token_ids=prompt,
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0],),
                num_computed_tokens=0,
                lora_request=None,
            )
        )
    out = _sched(
        new_reqs=new_reqs,
        num_scheduled_tokens={rid: 0 for rid in prompts},
        total=0,
    )
    runner._update_states(out)
    return out


def _sampler_output(sampled, logprobs_tensors=None):
    return SimpleNamespace(
        sampled_token_ids=sampled,
        logprobs_tensors=logprobs_tensors,
    )


class _CpuEvent:
    """Stand-in for torch.Event; the cpu backend supports no events."""

    def record(self):
        pass

    def synchronize(self):
        pass


def _bookkeeping_runner(**kw):
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, kw.get("prompts", {"a": [10, 11]}))
    except BaseException:
        patches.close()
        raise
    runner.use_async_scheduling = kw.get("use_async_scheduling", False)
    runner.routed_experts_initialized = False
    # _to_list's event is a device capability this host lacks.
    runner.transfer_event = _CpuEvent()
    # The prompt logprobs dict is not requested: empty.
    runner._get_prompt_logprobs_dict = lambda hidden, num_tokens: {}
    return runner, patches


def test_bookkeeping_sync_single_token_appends_to_request_state():
    """Sync single-token steps append sampled ids to request state."""
    runner, patches = _bookkeeping_runner()
    try:
        out = _sched(num_scheduled_tokens={"a": 2}, total=2)
        state = runner.requests["a"]
        # 2 prompt tokens already counted by the batch.
        runner.input_batch.num_tokens_no_spec[0] = 2

        sampled = torch.tensor([[42]], dtype=torch.int64)
        result = ModelRunnerFL._bookkeeping_sync(
            runner,
            out,
            _sampler_output(sampled),
            logits=None,
            hidden_states=torch.zeros(2, 4),
            num_scheduled_tokens=2,
        )
        (
            num_nans,
            logprobs_lists,
            valid_ids,
            prompt_logprobs,
            req_ids_copy,
            req_id_to_index_copy,
            invalid_req_indices,
        ) = result
        assert num_nans == {}
        assert logprobs_lists is None
        assert valid_ids == [[42]]
        assert prompt_logprobs == {}
        assert req_ids_copy == ["a"]
        assert req_id_to_index_copy == {"a": 0}
        assert invalid_req_indices == []
        # The token was appended to the request's output ids.
        assert state.output_token_ids == [42]
        # And written into the batch's cpu caches.
        assert runner.input_batch.token_ids_cpu[0, 2].item() == 42
        assert runner.input_batch.num_tokens_no_spec[0] == 3
    finally:
        patches.__exit__(None, None, None)


def test_bookkeeping_sync_discarded_row_is_emptied():
    """Discarded rows yield empty valid ids and no state append."""
    runner, patches = _bookkeeping_runner()
    try:
        out = _sched(num_scheduled_tokens={"a": 2}, total=2)
        # The request's step output is discarded (e.g. partial prefill).
        runner.discard_request_mask.cpu[:1] = True

        sampled = torch.tensor([[5]], dtype=torch.int64)
        _, _, valid_ids, _, _, _, _ = ModelRunnerFL._bookkeeping_sync(
            runner,
            out,
            _sampler_output(sampled),
            logits=None,
            hidden_states=torch.zeros(2, 4),
            num_scheduled_tokens=2,
        )
        assert valid_ids == [[]]
        # Nothing appended to the request state.
        assert runner.requests["a"].output_token_ids == []
    finally:
        patches.__exit__(None, None, None)


def test_bookkeeping_sync_spec_tokens_parse_output():
    """Rejected draft markers are filtered out of the parsed valid ids."""
    runner, patches = _bookkeeping_runner()
    try:
        out = _sched(num_scheduled_tokens={"a": 3}, total=3)
        runner.input_batch.num_tokens_no_spec[0] = 2

        # Two spec tokens: the second draft rejected (-1 placeholder).
        sampled = torch.tensor([[4, -1]], dtype=torch.int64)
        _, _, valid_ids, _, _, _, _ = ModelRunnerFL._bookkeeping_sync(
            runner,
            out,
            _sampler_output(sampled),
            logits=None,
            hidden_states=torch.zeros(3, 4),
            num_scheduled_tokens=3,
        )
        # The rejected marker is filtered out by parse_output.
        assert valid_ids == [[4]]
        assert runner.requests["a"].output_token_ids == [4]
        assert runner.input_batch.num_tokens_no_spec[0] == 3
    finally:
        patches.__exit__(None, None, None)


def test_bookkeeping_sync_async_caches_prev_sampled_tokens():
    """Async steps cache sampled tokens and write -1 placeholders."""
    runner, patches = _bookkeeping_runner(use_async_scheduling=True)
    try:
        runner.use_async_scheduling = True
        out = _sched(num_scheduled_tokens={"a": 1}, total=1)
        runner.input_batch.num_tokens_no_spec[0] = 2

        sampled = torch.tensor([[33]], dtype=torch.int64)
        result = ModelRunnerFL._bookkeeping_sync(
            runner,
            out,
            _sampler_output(sampled),
            logits=None,
            hidden_states=torch.zeros(1, 4),
            num_scheduled_tokens=1,
        )
        (
            _,
            _,
            valid_ids,
            _,
            req_ids_copy,
            _,
            invalid_req_indices,
        ) = result
        # Async path: valid ids stay empty; the tensor is cached on the
        # batch for the next step's input gather.
        assert valid_ids == []
        assert invalid_req_indices == []
        torch.testing.assert_close(runner.input_batch.prev_sampled_token_ids, sampled)
        assert runner.input_batch.prev_req_id_to_index == {"a": 0}
        # Async writes a -1 placeholder into the request state (the real
        # token arrives with the next step's correction).
        assert runner.requests["a"].output_token_ids == [-1]
        assert req_ids_copy == ["a"]
    finally:
        patches.__exit__(None, None, None)


def test_bookkeeping_sync_async_discarded_row_not_in_prev_mapping():
    """Async discarded rows are excluded from the prev-request mapping."""
    runner, patches = _bookkeeping_runner(
        use_async_scheduling=True, prompts={"a": [10, 11], "b": [20, 21]}
    )
    try:
        runner.use_async_scheduling = True
        assert runner.input_batch.req_ids == ["a", "b"]
        out = _sched(num_scheduled_tokens={"a": 1, "b": 1}, total=2)
        # Request a (row 0) is discarded this step.
        runner.discard_request_mask.cpu[0] = True
        runner.input_batch.num_tokens_no_spec[:2] = 2

        sampled = torch.tensor([[1], [2]], dtype=torch.int64)
        result = ModelRunnerFL._bookkeeping_sync(
            runner,
            out,
            _sampler_output(sampled),
            logits=None,
            hidden_states=torch.zeros(2, 4),
            num_scheduled_tokens=2,
        )
        _, _, _, _, _, _, invalid = result
        assert invalid == [0]
        # Only the surviving request is mapped for the next step.
        assert runner.input_batch.prev_req_id_to_index == {"b": 1}
        # Async rows each get the -1 placeholder in request state (the
        # real ids are corrected on the next step).
        assert runner.requests["a"].output_token_ids == []
        assert runner.requests["b"].output_token_ids == [-1]
        # The token ids cache still holds both rows' placeholders.
        assert runner.input_batch.token_ids_cpu[1, 2].item() == -1
    finally:
        patches.__exit__(None, None, None)
