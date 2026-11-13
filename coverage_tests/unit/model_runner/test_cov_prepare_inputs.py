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

"""Coverage tests for ModelRunnerFL input preparation.

``_prepare_inputs`` and ``_prepare_input_ids`` run against a real
CPU-constructed runner holding two/three scheduled requests with known
prompt token ids. Expected values are derived first by hand: the cu-sum
request layout, token gather indices, positions, optimistic seq lens,
and the last-token logits indices. The async-scheduling scatter path of
``_prepare_input_ids`` (prev_sampled_token_ids) is exercised through the
input batch's public cache field.
"""

import numpy as np
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import make_runner, make_scheduler_output


def _arm_kv_cache(runner):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    runner.kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([], None)],  # type: ignore[arg-type]
    )
    runner.attn_groups = []


def _new_req(req_id, prompt, sampling_params=None):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=sampling_params or SamplingParams(),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )


def _sched(**kw):
    from vllm.v1.core.sched.output import (
        CachedRequestData,
    )

    return make_scheduler_output(
        scheduled_new_reqs=kw.get("new_reqs", []),
        scheduled_cached_reqs=kw.get("cached", CachedRequestData.make_empty()),
        num_scheduled_tokens=kw.get("num_scheduled_tokens", {}),
        total_num_scheduled_tokens=kw.get(
            "total", sum(kw.get("num_scheduled_tokens", {}).values())
        ),
        scheduled_spec_decode_tokens=kw.get("spec_tokens", {}),
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _prime(runner, prompts):
    """Add requests via the state machine; return the SchedulerOutput."""
    out = _sched(
        new_reqs=[_new_req(rid, prompt) for rid, prompt in prompts.items()],
        num_scheduled_tokens={rid: 0 for rid in prompts},
        total=0,
    )
    runner._update_states(out)
    return out


def test_prepare_inputs_gathers_tokens_positions_and_logits_indices():
    """Two prefills lay out tokens, positions, and logits indices."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        # Two requests: 3-token prompt and 2-token prompt, each fully
        # scheduled as prefill. Token ids chosen to be distinguishable.
        _prime(runner, {"a": [11, 12, 13], "b": [21, 22]})
        out = _sched(num_scheduled_tokens={"a": 3, "b": 2}, total=5)

        logits_indices, spec_meta = runner._prepare_inputs(
            out, np.array([3, 2], dtype=np.int32)
        )
        assert spec_meta is None  # no spec decode tokens scheduled

        # input_ids.cpu gathered in request order: a's tokens then b's.
        got_ids = runner.input_ids.gpu[:5].tolist()
        assert got_ids == [11, 12, 13, 21, 22]

        # positions restart per request at num_computed_tokens (0).
        assert runner.positions[:5].tolist() == [0, 1, 2, 0, 1]

        # query_start_loc: [0, 3, 5] with the tail padded to the last cumsum.
        assert runner.query_start_loc.gpu[:3].tolist() == [0, 3, 5]
        assert runner.query_start_loc.cpu[3:].tolist() == [5, 5]

        # optimistic seq lens = computed + scheduled.
        assert runner.optimistic_seq_lens_cpu[:2].tolist() == [3, 2]
        assert runner.optimistic_seq_lens_cpu[2:].tolist() == [0, 0]

        # seq_lens mirrors the optimistic values; tail zeroed.
        assert runner.seq_lens[:2].tolist() == [3, 2]
        assert runner.seq_lens[2:].tolist() == [0, 0]

        # req_indices maps each token to its request row.
        assert runner.req_indices.gpu[:5].tolist() == [0, 0, 0, 1, 1]

        # num_scheduled_tokens buffer updated for the batch only.
        assert runner.num_scheduled_tokens.gpu[:2].tolist() == [3, 2]
        assert runner.num_scheduled_tokens.gpu[2:].tolist() == [0, 0]

        # No spec tokens scheduled: logits at each request's last token.
        assert logits_indices.tolist() == [2, 4]

        # Nothing finished early: discard mask all False.
        assert runner.discard_request_mask.cpu[:2].tolist() == [False, False]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_inputs_partial_prefill_discards_finished_requests():
    """A partially scheduled prompt is marked for discard."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [1, 2, 3, 4], "b": [5, 6, 7]})
        # Chunked prefill: only 2 of a's 4 prompt tokens run this step;
        # b runs fully. a must not be sampled (mask True).
        out = _sched(num_scheduled_tokens={"a": 2, "b": 3}, total=5)
        logits_indices, _ = runner._prepare_inputs(
            out, np.array([2, 3], dtype=np.int32)
        )

        assert runner.discard_request_mask.cpu[:2].tolist() == [True, False]
        assert runner.optimistic_seq_lens_cpu[:2].tolist() == [2, 3]
        assert logits_indices.tolist() == [1, 4]
        # Input ids gathered from the start of each prompt.
        assert runner.input_ids.gpu[:5].tolist() == [1, 2, 5, 6, 7]
        assert runner.positions[:5].tolist() == [0, 1, 0, 1, 2]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_inputs_decode_step_uses_computed_tokens_offset():
    """Decode continues positions at the computed-token offset."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [7, 8, 9]})
        # First step computed the whole prompt.
        out = _sched(num_scheduled_tokens={"a": 3}, total=3)
        runner._prepare_inputs(out, np.array([3], dtype=np.int32))
        # Mark all 3 tokens computed, then decode 1 new token.
        runner.input_batch.num_computed_tokens_cpu[0] = 3

        out = _sched(num_scheduled_tokens={"a": 1}, total=1)
        logits_indices, _ = runner._prepare_inputs(out, np.array([1], dtype=np.int32))
        # Decode reads the token after the prompt: positions continue at 3.
        assert runner.positions[:1].tolist() == [3]
        assert runner.optimistic_seq_lens_cpu[:1].tolist() == [4]
        assert runner.seq_lens[:1].tolist() == [4]
        assert runner.query_start_loc.gpu[:2].tolist() == [0, 1]
        assert logits_indices.tolist() == [0]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_inputs_spec_decode_builds_draft_metadata():
    """Draft tokens produce spec metadata with bonus logits indices."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [3, 4, 5]})
        # Prompt fully computed, decode step with 2 draft tokens.
        runner.input_batch.num_computed_tokens_cpu[0] = 3
        out = _sched(
            num_scheduled_tokens={"a": 3},
            total=3,
            spec_tokens={"a": [8, 9]},
        )
        logits_indices, spec_meta = runner._prepare_inputs(
            out, np.array([3], dtype=np.int32)
        )
        assert spec_meta is not None
        # num_decode_draft_tokens buffer holds the draft length (prompt
        # already computed -> decode phase, not -1 prefill marker).
        assert runner.num_decode_draft_tokens.cpu[:1].tolist() == [2]
        assert runner.num_decode_draft_tokens.cpu[1:].tolist() == [-1] * 3
        # Spec metadata reports 2 draft tokens for the single request.
        assert spec_meta.num_draft_tokens == [2]
        # 3 scheduled tokens = 2 draft + 1 bonus: logits for all three.
        assert logits_indices.tolist() == [0, 1, 2]
        assert spec_meta.bonus_logits_indices.tolist() == [2]
        assert spec_meta.cu_num_sampled_tokens.tolist() == [3]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_inputs_resets_cached_states_between_runs():
    """Departed requests have their seq-len tails zeroed on later steps."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [1, 2], "b": [3, 4]})
        out = _sched(num_scheduled_tokens={"a": 2, "b": 2}, total=4)
        runner._prepare_inputs(out, np.array([2, 2], dtype=np.int32))
        first_seq_lens = runner.seq_lens[:2].tolist()

        # "a" leaves the batch; only "b" runs a decode step next.
        runner.input_batch.num_computed_tokens_cpu[1] = 2
        out = _sched(num_scheduled_tokens={"b": 1}, total=1)
        runner._update_states(out)
        out = _sched(num_scheduled_tokens={"b": 1}, total=1)
        runner._prepare_inputs(out, np.array([1], dtype=np.int32))
        # seq_lens tail is zeroed for departed requests.
        assert runner.seq_lens.tolist() == [3, 0, 0, 0]
        assert first_seq_lens == [2, 2]
        # b's decode continues at position 2 (past its 2-token prompt).
        assert runner.positions[:1].tolist() == [2]
        assert runner.input_batch.num_reqs == 1
        assert runner.input_batch.req_ids == ["b"]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_input_ids_normal_path_copies_cpu_buffer():
    """Without async scheduling the cpu upload populates the gpu ids."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [42, 43]})
        out = _sched(num_scheduled_tokens={"a": 2}, total=2)
        runner._update_states(out)
        # The upstream gather wrote this step's tokens into the cpu
        # buffer; _prepare_input_ids' normal path uploads it to the gpu.
        runner.input_ids.cpu[:2] = torch.tensor([42, 43], dtype=torch.int32)

        cu = np.array([2], dtype=np.int64)
        runner._prepare_input_ids(out, 1, 2, cu)
        assert runner.input_ids.gpu[:2].tolist() == [42, 43]
        assert runner.input_batch.prev_sampled_token_ids is None
    finally:
        patches.__exit__(None, None, None)


def test_prepare_input_ids_async_scatter_from_prev_sampled_tokens():
    """Async scheduling scatters previously sampled tokens into ids."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [10], "b": [20]})
        out = _sched(num_scheduled_tokens={"a": 1, "b": 1}, total=2)
        runner._update_states(out)

        # Simulate the async-scheduling cache: the previous step's sampled
        # token for each request sits in prev_sampled_token_ids, and both
        # requests continue (prev positions 0 and 1).
        prev = torch.tensor([[7], [8]], dtype=runner.input_ids.cpu.dtype)
        runner.input_batch.prev_sampled_token_ids = prev
        runner.prev_num_spec_tokens = 0
        runner.prev_positions.np[:2] = [0, 1]
        runner.input_batch.prev_req_id_to_index = {"a": 0, "b": 1}

        # cu_num_tokens [1, 2] means flattened sample indices [0, 1].
        cu = np.array([1, 2], dtype=np.int64)
        runner._prepare_input_ids(out, 2, 2, cu)
        # The scatter copied each request's previously sampled token into
        # its slot of this step's input_ids.
        assert runner.input_ids.gpu[:2].tolist() == [7, 8]
    finally:
        patches.__exit__(None, None, None)


def test_prepare_input_ids_async_no_common_requests():
    """New requests bypass the async scatter and use the plain upload."""
    runner, patches = make_runner()
    try:
        _arm_kv_cache(runner)
        _prime(runner, {"a": [30, 31]})
        out = _sched(num_scheduled_tokens={"a": 2}, total=2)
        runner._update_states(out)

        runner.input_batch.prev_sampled_token_ids = torch.tensor(
            [[99]], dtype=runner.input_ids.cpu.dtype
        )
        runner.prev_num_spec_tokens = 0
        # Current request is new: prev position -1 for every row.
        runner.prev_positions.np[:1] = [-1]
        # The cpu side holds the freshly gathered prompt tokens.
        runner.input_ids.cpu[:2] = torch.tensor([30, 31], dtype=torch.int32)

        cu = np.array([2], dtype=np.int64)
        runner._prepare_input_ids(out, 1, 2, cu)
        # No common requests: the plain cpu buffer upload wins.
        assert runner.input_ids.gpu[:2].tolist() == [30, 31]
    finally:
        patches.__exit__(None, None, None)
