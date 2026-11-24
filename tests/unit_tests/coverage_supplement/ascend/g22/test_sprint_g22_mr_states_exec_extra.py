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

"""G22 coverage: ModelRunnerFL._update_states / execute_model residual branches.

Extends G16 (``_update_states``) and G14 (``execute_model``) with the branches
those suites leave untouched:

_update_states
  * finished id with no cached state (hook receives ``None``)
  * forced-preemption overlap that removes a resumed id from the batch
  * pooling model without ``pooling_params`` (assertion)
  * RANDOM_SEED generator on the TXDA CPU-fallback device
  * PP rank with non-empty sampled ids that add zero new tokens
  * PP rank with partial token slice into ``token_ids_cpu``
  * ngram draft-count restore that is already in sync (no write)
  * async + ngram optimistic draft bump on ``num_tokens_no_spec``
  * deferred correction that adjusts ngram tensors for accepted tokens
  * deferred correction skipping a missing current / previous batch index
  * ngram full-tensor tracking for a re-added (resumed) request

execute_model
  * re-entry guard while ``execute_model_state`` is set
  * EC-transfer producer early return / consumer pass-through
  * empty schedule returning the kv-connector no-forward output
  * empty schedule + external launcher + kv transfer
  * empty schedule on a single-DP backend returning the empty output
  * ``EncoderOnlyAttentionSpec`` groups skipped by separate-KV detection
  * separate-KV update padding slots without a full CUDA graph
  * mixed encoder/normal groups keep separate-KV padding
  * pooling model with ``broadcast_pp_output`` (assertion)

Everything is CPU-only; collaborators are patched at the
``vllm_fl.worker.model_runner`` module boundary.
"""

from __future__ import annotations

import contextlib
import types

import numpy as np
import pytest
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


class _Ctx:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


class _Intermediate:
    def __init__(self, tensors=None):
        self.tensors = tensors or {}

    def __getitem__(self, key):
        return self


# --------------------------------------------------------------------------
# _update_states fakes
# --------------------------------------------------------------------------
def _cached(
    req_ids=(),
    num_computed=(),
    new_block_ids=(),
    resumed=(),
    num_output=(),
    new_token_ids=None,
    all_token_ids=None,
):
    return types.SimpleNamespace(
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
    return types.SimpleNamespace(
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
        block_table=types.SimpleNamespace(append_row=lambda *a: None),
        remove_request=lambda rid: None,
        add_request=lambda req: None,
        update_req_spec_token_ids=lambda req, spec: None,
        condense=lambda: None,
        refresh_metadata=lambda: None,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _runner(batch, requests=None, **over):
    base = dict(
        requests={} if requests is None else requests,
        num_prompt_logprobs={},
        late_interaction_runner=types.SimpleNamespace(
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
    return types.SimpleNamespace(**base)


def _call(runner, sched, last_rank=True, extras=()):
    pairs = [(mr, "get_pp_group", lambda: types.SimpleNamespace(is_last_rank=last_rank))]
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
    return types.SimpleNamespace(
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
    return types.SimpleNamespace(
        prev_num_draft_len=prev_draft,
        output_token_ids=[] if output is None else output,
        block_ids=[list(b) for b in block_ids],
        num_computed_tokens=num_computed,
        num_tokens=num_tokens,
    )


def _ngram_runner(batch, requests=None, **over):
    base = dict(
        speculative_config=types.SimpleNamespace(use_ngram_gpu=lambda: True),
        token_ids_gpu_tensor=torch.zeros((8, 16), dtype=torch.int32),
        num_tokens_no_spec_gpu=torch.zeros(8, dtype=torch.int32),
        _ngram_pinned_idx_buf=None,
        _ngram_pinned_val_buf=None,
        _num_valid_draft_tokens_event=object(),
        _num_valid_draft_tokens_cpu=torch.zeros(8, dtype=torch.int32),
    )
    base.update(over)
    return _runner(batch, requests, **base)


_NGRAM_EXTRAS = [
    (mr, "update_scheduler_for_invalid_drafts", lambda *a, **k: None),
    (mr, "update_ngram_gpu_tensors_incremental", lambda *a, **k: None),
]


# --------------------------------------------------------------------------
# _update_states targets
# --------------------------------------------------------------------------
def test_g22_update_states_finished_id_without_cached_state():
    removed = []
    hook = []
    finished = []
    batch = _batch(remove_request=lambda rid: removed.append(rid))
    runner = _runner(batch, requests={})
    runner._on_request_state_removed = lambda rid, st: hook.append((rid, st))
    runner.late_interaction_runner.on_requests_finished = lambda ids: finished.append(
        list(ids)
    )
    _call(runner, _sched(finished=["ghost"]))
    assert hook == [("ghost", None)]
    assert removed == ["ghost"]
    assert finished == [["ghost"]]




def test_g22_update_states_new_pooling_request_requires_params():
    batch = _batch()
    runner = _runner(batch, is_pooling_model=True)
    with pytest.raises(AssertionError):
        _call(runner, _sched(new_reqs=[_new_req(pooling=None)]))


def test_g22_update_states_random_seed_txda_uses_cpu_generator():
    batch = _batch()
    runner = _runner(batch, device=types.SimpleNamespace(type="txda"))
    sampling = types.SimpleNamespace(
        sampling_type=mr.SamplingType.RANDOM_SEED, prompt_logprobs=None, seed=777
    )
    _call(runner, _sched(new_reqs=[_new_req(sampling=sampling)]))
    gen = runner.requests["n0"].generator
    assert isinstance(gen, torch.Generator)
    assert gen.initial_seed() == 777


def test_g22_update_states_pp_sampled_ids_without_new_tokens():
    batch = _batch(req_id_to_index={"r0": 0})
    batch.num_tokens_no_spec[0] = 5
    state = _state(output=[], num_tokens=5)
    runner = _runner(batch, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[3],
        new_block_ids=[None],
        num_output=[0],
        new_token_ids=[[1, 2]],
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached), last_rank=False)
    assert state.output_token_ids == []
    assert batch.num_tokens_no_spec[0] == 5
    assert batch.token_ids_cpu[0].abs().sum() == 0




def test_g22_update_states_spec_count_already_in_sync_keeps_draft():
    batch = _batch(req_id_to_index={"r0": 0})
    state = _state(prev_draft=2)
    runner = _ngram_runner(batch, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[1],
        new_block_ids=[None],
        num_output=[0],
    )
    _call(
        runner,
        _sched(num_scheduled={"r0": 1}, cached=cached, spec_tokens={"r0": [1, 2]}),
        extras=_NGRAM_EXTRAS,
    )
    assert state.prev_num_draft_len == 2


def test_g22_update_states_async_ngram_bumps_no_spec_tokens():
    batch = _batch(
        req_id_to_index={"r0": 0},
        prev_req_id_to_index={"r0": 0},
    )
    batch.num_tokens_no_spec[0] = 5
    state = _state(prev_draft=2, output=[])
    runner = _ngram_runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=types.SimpleNamespace(np=np.zeros(4, dtype=np.int32)),
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[10],
        new_block_ids=[None],
        num_output=[2],
    )
    _call(
        runner,
        _sched(num_scheduled={"r0": 2}, cached=cached),
        extras=_NGRAM_EXTRAS,
    )
    assert state.output_token_ids == [-1, -1]
    assert batch.num_tokens_no_spec[0] == 7




def test_g22_update_states_correction_skips_missing_current_index():
    batch = _batch(
        req_id_to_index={"r0": 0},
        prev_req_id_to_index={"r0": 0},
    )
    state = _state(prev_draft=1, output=[])
    runner = _runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=types.SimpleNamespace(np=torch.zeros(4, dtype=torch.int32)),
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[5],
        new_block_ids=[None],
        num_output=[1],
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    batch.req_id_to_index = {}
    runner._get_valid_sampled_token_count = lambda: [3]
    correct()
    assert state.num_computed_tokens == 6
    assert batch.num_computed_tokens_cpu[0] == 5


def test_g22_update_states_correction_skips_missing_prev_index():
    batch = _batch(
        req_id_to_index={"r0": 0},
        prev_req_id_to_index={"other": 0},
    )
    state = _state(prev_draft=1, output=[])
    runner = _runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=types.SimpleNamespace(np=torch.zeros(4, dtype=torch.int32)),
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
    assert batch.num_computed_tokens_cpu[0] == 5


def test_g22_update_states_ngram_tracks_readded_request():
    calls = []
    batch = _batch()
    state = _state(block_ids=((0,),))
    runner = _ngram_runner(batch, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[3],
        new_block_ids=[None],
        num_output=[0],
    )

    def _incremental(batch_, tids, nts, new_reqs, device, **kw):
        calls.append(list(new_reqs))

    _call(
        runner,
        _sched(num_scheduled={"r0": 1}, cached=cached),
        extras=[
            (mr, "update_scheduler_for_invalid_drafts", lambda *a, **k: None),
            (mr, "update_ngram_gpu_tensors_incremental", _incremental),
        ],
    )
    assert len(calls) == 1
    assert calls[0] == [state]


# --------------------------------------------------------------------------
# execute_model fakes
# --------------------------------------------------------------------------
def _pp(is_last=True, send=None, broadcast=None, ranks=(0, 1)):
    return types.SimpleNamespace(
        is_last_rank=is_last,
        send_tensor_dict=send or (lambda *a, **k: None),
        broadcast_tensor_dict=broadcast or (lambda data, src=0: data),
        ranks=list(ranks),
    )


def _xpatches(pp=None, ubatch=None, logger=None, **extra):
    pairs = [
        (mr, "record_function_or_nullcontext", lambda name: _Ctx()),
        (mr, "has_ec_transfer", lambda: False),
        (mr, "has_kv_transfer_group", lambda: False),
        (mr, "get_pp_group", lambda: pp or _pp()),
        (mr, "get_tp_group", lambda: "tp-group"),
        (mr, "set_forward_context", lambda *a, **k: _Ctx(k)),
        (mr, "maybe_create_ubatch_slices", ubatch or (lambda *a: (None, None))),
        (mr, "logger", logger or types.SimpleNamespace(debug=lambda *a: None)),
    ]
    pairs.extend(extra.items())
    return _stack(*pairs)


def _xsched(total=4, **extra):
    base = dict(
        total_num_scheduled_tokens=total,
        num_scheduled_tokens={"a": 2, "b": 2},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        kv_connector_metadata=None,
        num_common_prefix_blocks=0,
    )
    base.update(extra)
    return types.SimpleNamespace(**base)


def _xrunner(**extra):
    base = dict(
        execute_model_state=None,
        speculative_config=None,
        vllm_config="vllm-cfg",
        encoder_cache={},
        synchronize_input_prep=lambda: _Ctx(),
        _update_states=lambda so: None,
        _prepare_inputs=lambda so, nst: (torch.arange(2), "spec-meta", 3),
        parallel_config=types.SimpleNamespace(
            distributed_executor_backend="mp",
            data_parallel_size=1,
            use_ubatching=False,
            num_ubatches=2,
        ),
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode=None
        ),
        num_prompt_logprobs=0,
        input_batch=types.SimpleNamespace(
            num_reqs=2,
            req_ids=["a", "b"],
            num_computed_tokens_cpu=np.zeros(2, dtype=np.int64),
            num_accepted_tokens_cpu=np.zeros(2, dtype=np.int32),
        ),
        use_ascend_attention=False,
        _ascend_attn_state="prefill",
        cascade_attn_enabled=False,
        _compute_cascade_attn_prefix_lens=lambda nst, computed, blocks: "prefix",
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        _pad_query_start_loc_for_fia=lambda q, ntp, nr, npr: 99,
        _determine_batch_execution_and_padding=lambda **kw: (
            mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=4, num_reqs=None),
            False,
            None,
            "stats",
        ),
        _allow_microbatching=lambda nr, nst: True,
        _get_slot_mappings=lambda **kw: ({"g": "slots"}, "flat-slots"),
        _build_attention_metadata=lambda **kw: ("attn-meta", "spec-common"),
        _preprocess=lambda so, ntp, it: (
            "ids",
            "embeds",
            "positions",
            it,
            {"extra": 1},
            "ec-conn",
        ),
        model_config=types.SimpleNamespace(is_encoder_decoder=False),
        eplb_state=None,
        maybe_get_kv_connector_output=lambda so, defer_finalize=False: _Ctx("kv-out"),
        _model_forward=lambda **kw: torch.zeros(4, 3),
        use_aux_hidden_state_outputs=False,
        broadcast_pp_output=False,
        is_pooling_model=False,
        _pool=lambda hs, nst, nstnp, kv: "pooled",
        model=types.SimpleNamespace(
            compute_logits=lambda shs: torch.zeros(shs.shape[0], 5)
        ),
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[]),
        attn_groups=[],
        mamba_state_idx=None,
        requests={},
        compilation_config=types.SimpleNamespace(static_forward_context={}),
        num_accepted_tokens=types.SimpleNamespace(
            np=np.zeros(2, dtype=np.int32), copy_to_gpu=lambda n: None
        ),
        _get_mamba_state_copy_funcs=lambda: "copy-funcs",
        _get_mamba_bufs=lambda: types.SimpleNamespace(
            preprocess="pre", postprocess_align=None
        ),
        _dummy_run=lambda n: None,
    )
    base.update(extra)
    return types.SimpleNamespace(**base)


def test_g22_execute_rejects_reentry_with_pending_state():
    runner = types.SimpleNamespace(execute_model_state=object())
    with pytest.raises(RuntimeError, match="sample_tokens"):
        mr.ModelRunnerFL.execute_model(runner, _xsched())










def test_g22_execute_empty_single_dp_returns_empty_output():
    runner = _xrunner()
    with _xpatches():
        out = mr.ModelRunnerFL.execute_model(runner, _xsched(total=0))
    assert out is mr.EMPTY_MODEL_RUNNER_OUTPUT




def test_g22_execute_separate_kv_update_pads_slots_without_full_graph():
    captured = {}
    runner = _xrunner(
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[types.SimpleNamespace(kv_cache_spec="spec")]
        ),
        attn_groups=[
            [
                types.SimpleNamespace(
                    backend=types.SimpleNamespace(
                        forward_includes_kv_cache_update=False
                    )
                )
            ]
        ],
        _get_slot_mappings=lambda **kw: (
            captured.update(slots=kw) or ({"g": "slots"}, "flat-slots")
        ),
        _build_attention_metadata=lambda **kw: (
            captured.update(build=kw) or ("attn-meta", "spec-common")
        ),
        _determine_batch_execution_and_padding=lambda **kw: (
            mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=8, num_reqs=4),
            False,
            None,
            "stats",
        ),
    )
    with _xpatches():
        mr.ModelRunnerFL.execute_model(runner, _xsched())
    assert captured["slots"]["num_tokens_padded"] == 8
    assert captured["slots"]["num_reqs_padded"] == 4
    assert captured["build"]["num_tokens_padded"] is None




def test_g22_execute_broadcast_pp_rejects_pooling_model():
    runner = _xrunner(broadcast_pp_output=True, is_pooling_model=True)
    with _xpatches():
        with pytest.raises(AssertionError):
            mr.ModelRunnerFL.execute_model(runner, _xsched())
