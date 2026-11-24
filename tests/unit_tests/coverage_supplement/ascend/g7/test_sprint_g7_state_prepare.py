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

"""G7-C (batch 2) coverage: state update + input-id preparation.

Targets (814b): _update_states 1387-1760, _update_states_after_model_execute
1762-1807, _update_streaming_request 1810-1843, _prepare_input_ids 1955-2089,
_calc_spec_decode_metadata 3014-3086, _prepare_kv_sharing_fast_prefill 3089-3111,
_batch_mm_inputs_from_scheduler 3113-3154, _cache_encoder_output 3156-3166,
_allow_microbatching 4132-4175.

The InputBatch / scheduler output are faked with SimpleNamespace; module-level
collaborators (get_pp_group, async_tensor_h2d, CachedRequestState is real) and
PIN_MEMORY are patched at the module boundary. Everything runs on CPU.
"""

from __future__ import annotations

import contextlib
import types
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


def _scheduler(
    *,
    finished=None,
    new_reqs=None,
    num_scheduled=None,
    cached_reqs=None,
    spec_tokens=None,
):
    return SimpleNamespace(
        finished_req_ids=finished or [],
        scheduled_new_reqs=new_reqs or [],
        num_scheduled_tokens=num_scheduled if num_scheduled is not None else {},
        scheduled_cached_reqs=cached_reqs
        or SimpleNamespace(
            req_ids=[],
            num_computed_tokens=[],
            new_block_ids=[],
            resumed_req_ids=[],
            num_output_tokens=[],
        ),
        scheduled_spec_decode_tokens=spec_tokens or {},
        new_block_ids_to_zero=[],
        kv_cache_block_copies=[],
        free_encoder_mm_hashes=[],
    )


# --------------------------------------------------------------------------
# _update_states
# --------------------------------------------------------------------------
def test_update_states_updates_existing_request_no_deferred_corrections():
    calls = []
    req_state = SimpleNamespace(
        prev_num_draft_len=0,
        output_token_ids=[9],
        block_ids=[],
        num_computed_tokens=0,
    )
    input_batch = SimpleNamespace(
        req_id_to_index={"r0": 0},
        prev_req_id_to_index=None,
        num_computed_tokens_cpu=torch.zeros(2, dtype=torch.int32),
        remove_request=lambda rid: calls.append(("remove", rid)),
        update_req_spec_token_ids=lambda req, spec: calls.append(("spec", req)),
        condense=lambda: calls.append("condense"),
        refresh_metadata=lambda: calls.append("refresh"),
    )
    runner = SimpleNamespace(
        requests={"r0": req_state},
        num_prompt_logprobs={},
        late_interaction_runner=SimpleNamespace(
            on_requests_finished=lambda ids: calls.append(("finished", list(ids)))
        ),
        input_batch=input_batch,
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        _process_encoder_cache_scheduler_output=lambda so: calls.append("enc"),
        _may_reorder_batch=lambda so: calls.append("reorder"),
    )
    cached = SimpleNamespace(
        req_ids=["r0"],
        num_computed_tokens=[10],
        new_block_ids=[None],
        resumed_req_ids=[],
        num_output_tokens=[1],
    )
    sched = _scheduler(num_scheduled={"r0": 1}, cached_reqs=cached)
    with _stack((mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))):
        result = mr.ModelRunnerFL._update_states(runner, sched)
    assert result is None
    assert req_state.num_computed_tokens == 10
    assert calls == [("finished", []), "enc", ("spec", req_state), "condense", "reorder", "refresh"]


def test_update_states_adds_new_request_to_batch():
    calls = []
    input_batch = SimpleNamespace(
        req_id_to_index={},
        num_computed_tokens_cpu=torch.zeros(2, dtype=torch.int32),
        add_request=lambda req: calls.append(("add", req)),
        update_req_spec_token_ids=lambda req, spec: calls.append(("spec", req)),
        condense=lambda: calls.append("condense"),
        refresh_metadata=lambda: calls.append("refresh"),
    )
    runner = SimpleNamespace(
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=SimpleNamespace(
            on_requests_finished=lambda ids: None,
            register_request=lambda rid, pooling: calls.append(("li", rid)),
        ),
        input_batch=input_batch,
        is_pooling_model=False,
        uses_mrope=False,
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        device=torch.device("cpu"),
        _process_encoder_cache_scheduler_output=lambda so: None,
        _may_reorder_batch=lambda so: None,
    )
    new_req = SimpleNamespace(
        req_id="n0",
        prompt_token_ids=[1, 2],
        prompt_embeds=None,
        prompt_is_token_ids=True,
        mm_features=[],
        sampling_params=SimpleNamespace(
            sampling_type=None, prompt_logprobs=None, seed=None
        ),
        pooling_params=None,
        block_ids=[[0]],
        num_computed_tokens=0,
        lora_request=None,
    )
    sched = _scheduler(new_reqs=[new_req])
    with _stack((mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))):
        result = mr.ModelRunnerFL._update_states(runner, sched)
    assert result is None
    assert "n0" in runner.requests
    assert calls[0] == ("li", "n0")
    assert calls[1][0] == "add"
    assert calls[2] == ("spec", runner.requests["n0"])
    assert calls[3:] == ["condense", "refresh"]


# --------------------------------------------------------------------------
# _update_streaming_request
# --------------------------------------------------------------------------
def test_update_streaming_request_refreshes_fields():
    req_state = SimpleNamespace(
        prompt_token_ids=[0],
        mm_features=[],
        prompt_embeds=None,
        sampling_params=None,
        pooling_params=None,
        block_ids=[],
        num_computed_tokens=0,
        output_token_ids=[7, 8],
    )
    calls = []
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            remove_request=lambda rid: calls.append(("remove", rid))
        ),
        requests={"r0": req_state},
        late_interaction_runner=SimpleNamespace(
            register_request=lambda rid, p: calls.append(("register", rid))
        ),
        uses_mrope=False,
    )
    new_req = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        mm_features=["f"],
        prompt_embeds=None,
        sampling_params="sp",
        pooling_params="pp",
        block_ids=[[1]],
        num_computed_tokens=5,
    )
    with _stack(
        (mr, "length_from_prompt_token_ids_or_embeds", lambda ids, emb: 3)
    ):
        result = mr.ModelRunnerFL._update_streaming_request(runner, "r0", new_req)
    assert result is req_state
    assert req_state.prompt_token_ids == [1, 2, 3]
    assert req_state.num_prompt_tokens == 3
    assert req_state.output_token_ids == []
    assert req_state.num_computed_tokens == 5
    assert req_state.block_ids == [[1]]
    assert calls == [("remove", "r0"), ("register", "r0")]


def test_update_streaming_request_recomputes_mrope():
    req_state = SimpleNamespace(
        prompt_token_ids=[0],
        mm_features=[],
        prompt_embeds=None,
        sampling_params=None,
        pooling_params=None,
        block_ids=[],
        num_computed_tokens=0,
        output_token_ids=[],
    )
    calls = []
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(remove_request=lambda rid: None),
        requests={"r0": req_state},
        late_interaction_runner=SimpleNamespace(register_request=lambda r, p: None),
        uses_mrope=True,
        _init_mrope_positions=lambda st: calls.append(st),
    )
    new_req = SimpleNamespace(
        prompt_token_ids=[1],
        mm_features=[],
        prompt_embeds=None,
        sampling_params=None,
        pooling_params=None,
        block_ids=[],
        num_computed_tokens=0,
    )
    with _stack(
        (mr, "length_from_prompt_token_ids_or_embeds", lambda ids, emb: 1)
    ):
        mr.ModelRunnerFL._update_streaming_request(runner, "r0", new_req)
    assert calls == [req_state]


# --------------------------------------------------------------------------
# _update_states_after_model_execute
# --------------------------------------------------------------------------
def test_update_states_after_model_execute_early_return_non_hybrid():
    runner = SimpleNamespace(
        speculative_config=None, model_config=SimpleNamespace(is_hybrid=False)
    )
    mr.ModelRunnerFL._update_states_after_model_execute(
        runner, torch.zeros(2, 1, dtype=torch.int64), SimpleNamespace()
    )
    assert not hasattr(runner, "num_accepted_tokens")


def test_update_states_after_model_execute_non_align_copies_counts():
    calls = []
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(),
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(mamba_cache_mode="not_align"),
        num_accepted_tokens=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        input_batch=SimpleNamespace(
            num_accepted_tokens_cpu_tensor=torch.zeros(2, dtype=torch.int32)
        ),
        num_accepted_tokens_event=SimpleNamespace(
            record=lambda: calls.append("record")
        ),
    )
    out = torch.tensor([[1, 2, -1], [-1, -1, -1]], dtype=torch.int64)
    mr.ModelRunnerFL._update_states_after_model_execute(runner, out, SimpleNamespace())
    assert runner.num_accepted_tokens.gpu.tolist() == [2, 0]
    assert runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [2, 0]
    assert calls == ["record"]


# --------------------------------------------------------------------------
# _prepare_input_ids
# --------------------------------------------------------------------------
def _ids_buffer(size, calls=None, tag="ids"):
    def _copy(n):
        if calls is not None:
            calls.append((tag, n))

    return SimpleNamespace(
        gpu=torch.zeros(size, dtype=torch.int32),
        copy_to_gpu=_copy,
    )


def test_prepare_input_ids_normal_scheduling_copies_all():
    calls = []
    runner = SimpleNamespace(
        _pp_recv_work=None,
        input_ids=_ids_buffer(8, calls),
        enable_prompt_embeds=False,
        input_batch=SimpleNamespace(prev_sampled_token_ids=None),
    )
    mr.ModelRunnerFL._prepare_input_ids(
        runner,
        SimpleNamespace(scheduled_spec_decode_tokens={}),
        2,
        8,
        np.array([3, 5]),
    )
    assert calls == [("ids", 8)]


def test_prepare_input_ids_waits_for_pp_recv_work():
    calls = []
    runner = SimpleNamespace(
        _pp_recv_work=SimpleNamespace(wait=lambda: calls.append("wait")),
        input_ids=_ids_buffer(4),
        enable_prompt_embeds=False,
        input_batch=SimpleNamespace(prev_sampled_token_ids=None),
    )
    mr.ModelRunnerFL._prepare_input_ids(
        runner,
        SimpleNamespace(scheduled_spec_decode_tokens={}),
        1,
        4,
        np.array([4]),
    )
    assert calls == ["wait"]
    assert runner._pp_recv_work is None


def test_prepare_input_ids_no_common_tokens_returns_after_copy():
    calls = []
    prev_sampled = torch.tensor([[5], [6]], dtype=torch.int32)
    runner = SimpleNamespace(
        _pp_recv_work=None,
        input_ids=_ids_buffer(4, calls),
        enable_prompt_embeds=False,
        prev_positions=SimpleNamespace(np=np.array([-1, -1])),
        prev_num_spec_tokens=2,
        _draft_token_ids=None,
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            prev_sampled_token_ids=prev_sampled,
            req_ids=["a", "b"],
        ),
    )
    with _stack((mr, "PIN_MEMORY", False)):
        mr.ModelRunnerFL._prepare_input_ids(
            runner,
            SimpleNamespace(scheduled_spec_decode_tokens={}),
            2,
            2,
            np.array([1, 2]),
        )
    assert calls == [("ids", 2)]


def test_prepare_input_ids_common_path_copies_prev_sampled_tokens():
    prev_sampled = torch.tensor([[7], [8]], dtype=torch.int32)
    runner = SimpleNamespace(
        _pp_recv_work=None,
        input_ids=_ids_buffer(4),
        enable_prompt_embeds=False,
        prev_positions=SimpleNamespace(np=np.array([0, 1])),
        prev_num_spec_tokens=4,
        _draft_token_ids=None,
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            prev_sampled_token_ids=prev_sampled,
            req_ids=["a", "b"],
        ),
    )
    with _stack((mr, "PIN_MEMORY", False)):
        mr.ModelRunnerFL._prepare_input_ids(
            runner,
            SimpleNamespace(scheduled_spec_decode_tokens={}),
            2,
            2,
            np.array([1, 2]),
        )
    assert runner.input_ids.gpu[:2].tolist() == [7, 8]


def test_prepare_input_ids_scatter_sampled_and_draft_tokens():
    prev_sampled = torch.tensor([[7], [8], [9]], dtype=torch.int32)
    runner = SimpleNamespace(
        _pp_recv_work=None,
        input_ids=_ids_buffer(4),
        enable_prompt_embeds=False,
        prev_positions=SimpleNamespace(np=np.array([2])),
        prev_num_spec_tokens=2,
        _draft_token_ids=torch.arange(12, dtype=torch.int64).reshape(3, 4),
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            prev_sampled_token_ids=prev_sampled,
            req_ids=["a"],
        ),
    )
    sched = SimpleNamespace(scheduled_spec_decode_tokens={"a": [10, 11]})
    with _stack((mr, "PIN_MEMORY", False)):
        mr.ModelRunnerFL._prepare_input_ids(runner, sched, 1, 3, np.array([3]))
    gpu = runner.input_ids.gpu
    assert gpu[0].item() == 9  # prev_sampled[2, 0]
    assert gpu[1].item() == 4  # draft flatten[4]
    assert gpu[2].item() == 5  # draft flatten[5]


# --------------------------------------------------------------------------
# _calc_spec_decode_metadata
# --------------------------------------------------------------------------
def test_calc_spec_decode_metadata_builds_indices():
    runner = SimpleNamespace(
        _arange_scratch=np.zeros(32, dtype=np.int32),
        arange_np=np.arange(64),
        input_ids=SimpleNamespace(gpu=torch.arange(20, dtype=torch.int32)),
        device=torch.device("cpu"),
    )
    runner._get_cumsum_and_arange = types.MethodType(
        mr.ModelRunnerFL._get_cumsum_and_arange, runner
    )
    num_draft = np.array([3, 0, 2], dtype=np.int32)
    cu_scheduled = np.array([4, 5, 8], dtype=np.int32)
    with _stack((mr, "async_tensor_h2d", lambda arr, device: torch.as_tensor(arr))):
        meta = mr.ModelRunnerFL._calc_spec_decode_metadata(
            runner, num_draft, cu_scheduled
        )
    assert meta.logits_indices.tolist() == list(range(8))
    assert meta.num_draft_tokens == [3, 0, 2]
    assert meta.target_logits_indices.tolist() == [0, 1, 2, 5, 6]


# --------------------------------------------------------------------------
# _prepare_kv_sharing_fast_prefill
# --------------------------------------------------------------------------
def test_prepare_kv_sharing_fast_prefill_pads_and_dispatches():
    seen = {}
    runner = SimpleNamespace(
        kv_sharing_fast_prefill_logits_indices=torch.zeros(8, dtype=torch.int64),
        cudagraph_dispatcher=SimpleNamespace(
            dispatch=lambda n, invalid_modes: (
                seen.update(num=n, modes=invalid_modes)
                or ("mode", SimpleNamespace(num_tokens=6))
            )
        ),
    )
    logits = torch.tensor([10, 11, 12], dtype=torch.int64)
    result = mr.ModelRunnerFL._prepare_kv_sharing_fast_prefill(runner, logits)
    assert result.tolist() == [10, 11, 12, 12, 12, 12]
    assert seen["num"] == 3


# --------------------------------------------------------------------------
# _batch_mm_inputs_from_scheduler / _cache_encoder_output
# --------------------------------------------------------------------------
def test_batch_mm_inputs_from_scheduler_empty():
    runner = SimpleNamespace()
    sched = SimpleNamespace(scheduled_encoder_inputs={})
    assert mr.ModelRunnerFL._batch_mm_inputs_from_scheduler(runner, sched) == (
        [],
        [],
        [],
    )


def test_batch_mm_inputs_from_scheduler_collects_features():
    ok = SimpleNamespace(
        data={"x": 1}, identifier="h1", modality="image", mm_position="pos"
    )
    skip = SimpleNamespace(
        data=None, identifier="h2", modality="audio", mm_position="pos2"
    )
    req_state = SimpleNamespace(mm_features=[ok, skip])
    runner = SimpleNamespace(requests={"r0": req_state})
    sched = SimpleNamespace(scheduled_encoder_inputs={"r0": [0, 1]})
    hashes, kwargs, refs = mr.ModelRunnerFL._batch_mm_inputs_from_scheduler(
        runner, sched
    )
    assert hashes == ["h1"]
    assert kwargs == [("image", {"x": 1})]
    assert refs == [("r0", "pos")]


def test_cache_encoder_output_stores_and_notifies_connector():
    calls = []
    runner = SimpleNamespace(
        encoder_cache={},
        maybe_save_ec_to_connector=lambda cache, h: calls.append((cache, h)),
    )
    out = torch.zeros(2)
    mr.ModelRunnerFL._cache_encoder_output(runner, "h1", out, None, [])
    assert runner.encoder_cache == {"h1": out}
    assert calls == [(runner.encoder_cache, "h1")]


# --------------------------------------------------------------------------
# _allow_microbatching
# --------------------------------------------------------------------------
def test_allow_microbatching_disabled_returns_true():
    runner = SimpleNamespace(parallel_config=SimpleNamespace(use_ubatching=False))
    assert mr.ModelRunnerFL._allow_microbatching(runner, 4, np.array([1, 1])) is True


def test_allow_microbatching_single_request_returns_true():
    runner = SimpleNamespace(parallel_config=SimpleNamespace(use_ubatching=True))
    assert mr.ModelRunnerFL._allow_microbatching(runner, 1, np.array([1])) is True


def test_allow_microbatching_no_readers_returns_true():
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=True),
        reorder_batch_threshold=None,
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.array([0, 0]),
        ),
    )
    assert (
        mr.ModelRunnerFL._allow_microbatching(runner, 2, np.array([2, 2])) is True
    )


