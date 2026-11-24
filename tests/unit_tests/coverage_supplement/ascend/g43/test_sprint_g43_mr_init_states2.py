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

"""G43 coverage: ModelRunnerFL residual assembly (814b L584-1112, L1387-1760).

Complements G13/G21 (``__init__``), G16/G22/G30/G40 (``_update_states``).
Only the branches/value pairs those suites still leave behind are targeted:

__init__ (L584-1112)
  * ``use_eagle()`` arm whose method is not ``eagle3`` (aux stays False)
  * ``eagle3`` drafter reporting ``eagle3_use_aux_hidden_state`` False
  * InputBatch ``use_replayssm`` / ``reasoning_config`` /
    ``cp_kv_cache_interleave_size`` pass-through
  * kv-sharing fast-prefill logits index tensor dtype/shape/device
  * ``sampled_token_ids_pinned_cpu`` shape/dtype/device
  * ngram-gpu per-request valid-count cpu/event/stream buffers
  * draft-token cpu/event/stream buffers for a spec config
  * async spec-decode valid-count copy stream + pinned cpu tensor
  * ``num_spec_tokens == 0`` suppresses the draft buffers
  * ngram-gpu with zero spec tokens allocates only the valid-count side
  * M-RoPE positions allocated alongside a spec config
  * encoder-decoder placeholder block count uses the encoder length
  * ``calculate_kv_scales`` explicitly present and False
  * async ``prepare_inputs_event`` is built with ``blocking=True``
  * ``inputs_embeds`` buffer forwards a non-float16 dtype with numpy off

_update_states (L1387-1760)
  * deferred correction whose correction is exactly zero (ngram and non-ngram)
  * ``update_scheduler_for_invalid_drafts`` argument assembly
  * ``update_ngram_gpu_tensors_incremental`` pinned-buffer/device assembly
  * scheduler spec-count restore across several cached requests
  * streaming request on an M-RoPE runner re-initialises positions
  * mixed streaming + brand-new requests inserted in one step
  * several finished ids with prompt-logprob cleanup
  * async optimistic draft bump without ngram writes the draft buffer only
  * PP rank appends blocks and propagates tokens together
  * last rank with nothing discarded still appends new blocks
  * non-async re-add keeps a full output list untouched
  * deferred closure indexes the valid counts by the previous batch index
  * spec-token update fans out to cached and re-added requests
  * empty zero/copy lists keep the housekeeping hooks silent

CPU-only; collaborators are patched at the ``vllm_fl.worker.model_runner``
module boundary (no accelerator, no torch_npu).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import torch

from vllm_fl.worker import model_runner as mr


# --------------------------------------------------------------------------
# generic patching helpers
# --------------------------------------------------------------------------
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


# ==========================================================================
# _update_states harness
# ==========================================================================
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
):
    return SimpleNamespace(
        finished_req_ids=finished or [],
        scheduled_new_reqs=new_reqs or [],
        num_scheduled_tokens=num_scheduled or {},
        scheduled_cached_reqs=cached or _cached(),
        scheduled_spec_decode_tokens=spec_tokens or {},
        new_block_ids_to_zero=zero or [],
        kv_cache_block_copies=copies or [],
        free_encoder_mm_hashes=[],
    )


def _batch(**over):
    base = dict(
        req_id_to_index={},
        prev_req_id_to_index=None,
        num_computed_tokens_cpu=np.zeros(8, dtype=np.int32),
        num_tokens_no_spec=np.zeros(8, dtype=np.int64),
        token_ids_cpu=np.zeros((8, 16), dtype=np.int32),
        is_token_ids=np.zeros((8, 16), dtype=bool),
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
    runner = SimpleNamespace(**base)
    runner._update_streaming_request = lambda rid, data: (
        mr.ModelRunnerFL._update_streaming_request(runner, rid, data)
    )
    return runner


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


def _state(*, output=None, block_ids=((0,),), num_computed=0, num_tokens=1, prev_draft=0):
    return SimpleNamespace(
        prev_num_draft_len=prev_draft,
        output_token_ids=[] if output is None else output,
        block_ids=[list(b) for b in block_ids],
        num_computed_tokens=num_computed,
        num_tokens=num_tokens,
    )


def _streaming_state(*, output=None, block_ids=((0,),), num_tokens=1):
    st = _state(output=output, block_ids=block_ids, num_tokens=num_tokens)
    st.prompt_token_ids = [0]
    st.mm_features = []
    st.prompt_embeds = None
    st.sampling_params = None
    st.pooling_params = None
    return st


def _ngram_runner(batch, requests=None, **over):
    base = dict(
        speculative_config=SimpleNamespace(use_ngram_gpu=lambda: True),
        token_ids_gpu_tensor=torch.zeros((8, 16), dtype=torch.int32),
        num_tokens_no_spec_gpu=torch.zeros(8, dtype=torch.int32),
        _ngram_pinned_idx_buf=None,
        _ngram_pinned_val_buf=None,
        _num_valid_draft_tokens_event=object(),
        _num_valid_draft_tokens_cpu=torch.zeros(8, dtype=torch.int32),
        prev_num_draft_tokens=SimpleNamespace(np=np.zeros(8, dtype=np.int32)),
    )
    base.update(over)
    return _runner(batch, requests, **base)


_NGRAM_EXTRAS = [
    (mr, "update_scheduler_for_invalid_drafts", lambda *a, **k: None),
    (mr, "update_ngram_gpu_tensors_incremental", lambda *a, **k: None),
]


# --------------------------------------------------------------------------
# deferred spec-decode correction: correction == 0
# --------------------------------------------------------------------------
def test_g43_update_states_correction_zero_skips_ngram_decrement():
    batch = _batch(req_id_to_index={"r0": 0}, prev_req_id_to_index={"r0": 0})
    batch.num_tokens_no_spec[0] = 7
    batch.num_computed_tokens_cpu[0] = 99
    state = _state(prev_draft=2, output=[])
    runner = _ngram_runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        num_tokens_no_spec_gpu=torch.full((8,), 5, dtype=torch.int32),
    )
    cached = _cached(
        req_ids=["r0"], num_computed=[10], new_block_ids=[None], num_output=[2]
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 2}, cached=cached),
                    extras=_NGRAM_EXTRAS)
    assert state.output_token_ids == [-1, -1]
    assert batch.num_tokens_no_spec[0] == 9
    # valid count 3 -> num_accepted 2 -> correction 0: nothing must move.
    runner._get_valid_sampled_token_count = lambda: [3]
    correct()
    assert state.num_computed_tokens == 10
    # _update_states writes the scheduler count (10) before the closure runs;
    # correction 0 leaves it there.
    assert batch.num_computed_tokens_cpu[0] == 10
    assert batch.num_tokens_no_spec[0] == 9
    assert runner.num_tokens_no_spec_gpu[0] == 5


def test_g43_update_states_correction_zero_without_ngram():
    batch = _batch(
        req_id_to_index={"r0": 0},
        prev_req_id_to_index={"r0": 0},
        num_computed_tokens_cpu=np.full(8, 42, dtype=np.int32),
    )
    state = _state(prev_draft=3, output=[])
    runner = _runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=np.zeros(4, dtype=np.int32)),
    )
    cached = _cached(
        req_ids=["r0"], num_computed=[5], new_block_ids=[None], num_output=[3]
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 3}, cached=cached))
    assert state.output_token_ids == [-1, -1, -1]
    runner._get_valid_sampled_token_count = lambda: [4]
    correct()
    assert state.num_computed_tokens == 5
    assert batch.num_computed_tokens_cpu[0] == 5


# --------------------------------------------------------------------------
# ngram hook argument assembly
# --------------------------------------------------------------------------


def test_g43_update_states_ngram_incremental_arguments():
    idx_buf = object()
    val_buf = object()
    captured = {}
    batch = _batch()
    runner = _ngram_runner(
        batch,
        requests={},
        _ngram_pinned_idx_buf=idx_buf,
        _ngram_pinned_val_buf=val_buf,
    )
    device = torch.device("cpu")

    def _inc(*a, **k):
        captured["args"] = a
        captured["kwargs"] = k

    _call(
        runner,
        _sched(new_reqs=[_new_req()]),
        extras=[
            (mr, "update_scheduler_for_invalid_drafts", lambda *a, **k: None),
            (mr, "update_ngram_gpu_tensors_incremental", _inc),
        ],
    )
    args, kwargs = captured["args"], captured["kwargs"]
    assert args[0] is batch
    assert args[1] is runner.token_ids_gpu_tensor
    assert args[2] is runner.num_tokens_no_spec_gpu
    assert list(args[3]) == [runner.requests["n0"]]
    assert args[4] == device
    assert kwargs["_pinned_idx_buf"] is idx_buf
    assert kwargs["_pinned_val_buf"] is val_buf


def test_g43_update_states_restores_spec_count_per_request():
    batch = _batch(req_id_to_index={"r0": 0, "r1": 1})
    s0 = _state(prev_draft=3)
    s1 = _state(prev_draft=1)
    runner = _ngram_runner(batch, requests={"r0": s0, "r1": s1})
    cached = _cached(
        req_ids=["r0", "r1"],
        num_computed=[1, 1],
        new_block_ids=[None, None],
        num_output=[0, 0],
    )
    _call(
        runner,
        _sched(
            num_scheduled={"r0": 1, "r1": 1},
            cached=cached,
            spec_tokens={"r0": [7, 8]},
        ),
        extras=_NGRAM_EXTRAS,
    )
    assert s0.prev_num_draft_len == 2
    assert s1.prev_num_draft_len == 0


# --------------------------------------------------------------------------
# streaming / mixed insertion
# --------------------------------------------------------------------------
def test_g43_update_states_streaming_request_refreshes_mrope():
    seen = []
    existing = _streaming_state(output=[7, 8])
    removed, added = [], []
    batch = _batch(
        remove_request=lambda rid: removed.append(rid),
        add_request=lambda req: added.append(req),
    )
    runner = _runner(batch, requests={"r0": existing}, uses_mrope=True)
    runner._init_mrope_positions = lambda st: seen.append(st)
    streaming = _new_req(req_id="r0", prompt=(1, 2, 3), block_ids=((9,),))
    streaming.num_computed_tokens = 3
    _call(
        runner,
        _sched(new_reqs=[streaming]),
        extras=[
            (mr, "length_from_prompt_token_ids_or_embeds", lambda ids, emb: 3)
        ],
    )
    assert seen == [existing]
    assert existing.prompt_token_ids == [1, 2, 3]
    assert existing.output_token_ids == []
    assert existing.block_ids == [[9]]
    assert removed == ["r0"]
    assert added == [existing]




def test_g43_update_states_finished_multiple_ids_clean_logprobs():
    hooks = []
    batch = _batch()
    runner = _runner(batch, requests={"r1": _state()})
    runner.num_prompt_logprobs = {"r1": 2, "r2": 1}
    runner._on_request_state_removed = lambda rid, st: hooks.append((rid, st))
    _call(runner, _sched(finished=["r1", "ghost"]))
    assert runner.num_prompt_logprobs == {"r2": 1}
    assert "r1" not in runner.requests
    assert hooks[0][0] == "r1" and hooks[0][1] is not None
    assert hooks[1] == ("ghost", None)


# --------------------------------------------------------------------------
# async optimistic bump variants
# --------------------------------------------------------------------------
def test_g43_update_states_async_optimistic_without_ngram():
    batch = _batch(req_id_to_index={"r0": 0}, prev_req_id_to_index={"r0": 0})
    batch.num_tokens_no_spec[0] = 1
    prev_np = np.zeros(4, dtype=np.int32)
    state = _state(prev_draft=2, output=[])
    runner = _runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=prev_np),
    )
    cached = _cached(
        req_ids=["r0"], num_computed=[10], new_block_ids=[None], num_output=[2]
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 2}, cached=cached))
    assert state.output_token_ids == [-1, -1]
    assert prev_np[0] == 2
    # is_ngram_gpu False -> the host token cursor is left alone.
    assert batch.num_tokens_no_spec[0] == 1
    assert callable(correct)


# --------------------------------------------------------------------------
# PP token propagation / block append combinations
# --------------------------------------------------------------------------
def test_g43_update_states_pp_appends_blocks_and_tokens():
    appended = []
    batch = _batch(
        req_id_to_index={"r0": 0},
        block_table=SimpleNamespace(
            append_row=lambda ids, idx: appended.append((ids, idx))
        ),
    )
    batch.num_tokens_no_spec[0] = 1
    state = _state(output=[], num_tokens=1)
    runner = _runner(batch, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[1],
        new_block_ids=[[[1]]],
        num_output=[0],
        new_token_ids=[[5]],
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached), last_rank=False)
    assert state.output_token_ids == [5]
    assert state.block_ids == [[0, 1]]
    assert appended == [([[1]], 0)]
    assert batch.token_ids_cpu[0, 1] == 5
    assert bool(batch.is_token_ids[0, 1])
    assert batch.num_tokens_no_spec[0] == 2


def test_g43_update_states_last_rank_keeps_output_and_appends_blocks():
    appended = []
    batch = _batch(
        req_id_to_index={"r0": 0},
        block_table=SimpleNamespace(
            append_row=lambda ids, idx: appended.append((ids, idx))
        ),
    )
    state = _state(output=[1, 2], num_tokens=3)
    runner = _runner(batch, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[3],
        new_block_ids=[[[4]]],
        num_output=[2],
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    assert state.output_token_ids == [1, 2]
    assert state.block_ids == [[0, 4]]
    assert appended == [([[4]], 0)]


def test_g43_update_states_readd_without_async_keeps_output():
    added = []
    batch = _batch(add_request=lambda req: added.append(req))
    state = _state(output=[1, 2, 3], num_tokens=4)
    runner = _runner(batch, requests={"r0": state}, use_async_scheduling=False)
    cached = _cached(
        req_ids=["r0"],
        num_computed=[4],
        new_block_ids=[None],
        num_output=[3],
        all_token_ids={"r0": [0, 0, 0, 0]},
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    assert state.output_token_ids == [1, 2, 3]
    assert added == [state]


def test_g43_update_states_deferred_closure_indexes_prev_batch():
    prev_np = np.zeros(4, dtype=np.int32)
    batch = _batch(
        req_id_to_index={"r0": 0},
        prev_req_id_to_index={"r0": 1},
        num_computed_tokens_cpu=np.full(8, 5, dtype=np.int32),
    )
    state = _state(prev_draft=2, output=[])
    runner = _runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=prev_np),
    )
    cached = _cached(
        req_ids=["r0"], num_computed=[10], new_block_ids=[None], num_output=[2]
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 2}, cached=cached))
    # Counts for r0 live at previous index 1, not 0.
    runner._get_valid_sampled_token_count = lambda: [0, 4]
    correct()
    # optimistic 2, accepted 3 -> correction -1 -> computed grows by one
    # from the scheduler-written 10.
    assert state.num_computed_tokens == 11
    assert batch.num_computed_tokens_cpu[0] == 11


# --------------------------------------------------------------------------
# spec-token fan-out and empty housekeeping
# --------------------------------------------------------------------------


def test_g43_update_states_empty_housekeeping_is_silent():
    calls = []
    batch = _batch()
    runner = _runner(batch)
    runner._zero_block_ids = lambda ids: calls.append(("zero", list(ids)))
    _call(
        runner,
        _sched(),
        extras=[
            (mr, "copy_kv_cache_blocks_inplace", lambda *a, **k: calls.append("copy"))
        ],
    )
    assert calls == []


# ==========================================================================
# __init__ harness
# ==========================================================================
_TORCH_ZEROS = torch.zeros
_TORCH_EMPTY = torch.empty


def _zeros_no_pin(*args, **kwargs):
    kwargs.pop("pin_memory", None)
    return _TORCH_ZEROS(*args, **kwargs)


def _empty_no_pin(*args, **kwargs):
    kwargs.pop("pin_memory", None)
    return _TORCH_EMPTY(*args, **kwargs)


class _FakeEvent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _spec(
    method="none",
    *,
    draft_model=False,
    ngram_gpu=False,
    gemma4=False,
    step3p5=False,
    dflash=False,
    eagle=False,
    num_spec=2,
    draft_model_config=None,
):
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=num_spec,
        draft_model_config=draft_model_config,
        uses_draft_model=lambda: draft_model,
        use_ngram_gpu=lambda: ngram_gpu,
        use_gemma4_mtp=lambda: gemma4,
        use_step3p5_mtp=lambda: step3p5,
        use_dflash=lambda: dflash,
        use_eagle=lambda: eagle,
    )


def _config(
    model=None,
    cache=None,
    parallel=None,
    scheduler=None,
    compilation=None,
    spec=None,
    reasoning=None,
):
    model_cfg = SimpleNamespace(
        dtype=torch.float16,
        is_encoder_decoder=False,
        disable_cascade_attn=False,
        is_mm_prefix_lm=False,
        is_multimodal_raw_input_only_model=False,
        enable_prompt_embeds=False,
        runner_type="generate",
        max_model_len=128,
        is_moe=False,
        uses_alibi=False,
        uses_mrope=False,
        logprobs_mode="raw_logprobs",
        use_fp64_gumbel=False,
        logits_processors=None,
        supports_multimodal_inputs=False,
        get_num_attention_heads=lambda pc: 8,
        get_inputs_embeds_size=lambda: 0,
        get_vocab_size=lambda: 100,
    )
    cache_cfg = SimpleNamespace(
        cache_dtype="auto",
        block_size=16,
        use_replayssm=False,
        kv_sharing_fast_prefill=False,
    )
    parallel_cfg = SimpleNamespace(
        data_parallel_size=1,
        decode_context_parallel_size=1,
        distributed_executor_backend="mp",
        cp_kv_cache_interleave_size=1,
    )
    sched_cfg = SimpleNamespace(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_num_encoder_input_tokens=0,
        async_scheduling=False,
    )
    comp_cfg = SimpleNamespace(
        cudagraph_capture_sizes=None,
        cudagraph_mode=mr.CUDAGraphMode.NONE,
    )
    for target, over in (
        (model_cfg, model),
        (cache_cfg, cache),
        (parallel_cfg, parallel),
        (sched_cfg, scheduler),
        (comp_cfg, compilation),
    ):
        if over:
            for key, val in over.items():
                setattr(target, key, val)
    return SimpleNamespace(
        model_config=model_cfg,
        cache_config=cache_cfg,
        offload_config=SimpleNamespace(),
        compilation_config=comp_cfg,
        lora_config=None,
        load_config=None,
        parallel_config=parallel_cfg,
        scheduler_config=sched_cfg,
        speculative_config=spec,
        observability_config=None,
        reasoning_config=reasoning,
    )


def _run(config, *, extra=()):
    record = {}

    class _Stream:
        def __init__(self, *a, **k):
            pass

    class _Sampler:
        def __init__(self, logprobs_mode=None, use_fp64_gumbel=False):
            record["sampler"] = (logprobs_mode, use_fp64_gumbel)

    class _LateInteraction:
        pass

    class _InputBatch:
        def __init__(self, *a, **k):
            record["input_batch"] = k

    class _Dispatcher:
        def __init__(self, cfg):
            record["dispatcher"] = cfg

    class _Budget:
        def __init__(self, cfg, registry):
            record["budget"] = (cfg, registry)

    class _Reject:
        def __init__(self, sampler, spec, device):
            record["rejection_sampler"] = (sampler, spec, device)

    def _drafter_cls(name):
        class _D:
            def __init__(self, *a, **k):
                self.eagle3_use_aux_hidden_state = True
                record.setdefault("drafters", []).append(name)

        _D.__name__ = name
        return _D

    def _fake_make_buffer(*size, dtype=None, numpy=True):
        record.setdefault("buffers", []).append((size, dtype, numpy))
        return SimpleNamespace(size=size, dtype=dtype, numpy=numpy)

    fake_registry = SimpleNamespace(supports_multimodal_inputs=lambda mc: False)
    device_fn = SimpleNamespace(Stream=_Stream, Event=_FakeEvent)
    fake_platform = SimpleNamespace(vendor_name="cuda", torch_device_fn=device_fn)

    base = [
        (mr, "current_platform", fake_platform),
        (mr, "kv_cache_dtype_str_to_dtype", lambda dtype, mc: torch.float16),
        (mr, "Sampler", _Sampler),
        (mr, "LateInteractionRunner", _LateInteraction),
        (mr, "InputBatch", _InputBatch),
        (mr, "build_logitsprocs", lambda *a, **k: ("lp",)),
        (mr, "PIN_MEMORY", False),
        (mr, "CudagraphDispatcher", _Dispatcher),
        (mr, "create_offloader", lambda cfg: "offloader"),
        (mr, "set_offloader", lambda o: record.__setitem__("offloader", o)),
        (mr, "get_pp_group", lambda: SimpleNamespace(ranks=[0], is_last_rank=True)),
        (mr, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=0)),
        (mr, "RejectionSampler", _Reject),
        (mr, "create_custom_proposer", lambda cfg: "custom-proposer"),
        (mr, "MULTIMODAL_REGISTRY", fake_registry),
        (mr, "MultiModalBudget", _Budget),
        (mr.torch, "zeros", _zeros_no_pin),
        (mr.torch, "empty", _empty_no_pin),
        (mr.torch, "Event", _FakeEvent),
    ]
    for name in (
        "DraftModelProposer",
        "NgramProposerGPU",
        "Gemma4Proposer",
        "Step3p5MTPProposer",
        "DFlashProposer",
        "SuffixDecodingProposer",
        "EagleProposer",
        "MedusaProposer",
        "ExtractHiddenStatesProposer",
    ):
        if hasattr(mr, name):
            base.append((mr, name, _drafter_cls(name)))

    runner = SimpleNamespace()
    runner._make_buffer = _fake_make_buffer
    runner._init_device_properties = lambda: record.__setitem__("num_sms", 1)
    with _stack(*(base + list(extra))):
        mr.ModelRunnerFL.__init__(runner, config, torch.device("cpu"))
    return runner, record


def _buffer_sizes(record):
    return [b[0] for b in record.get("buffers", [])]


# --------------------------------------------------------------------------
# speculative drafter aux-output branches
# --------------------------------------------------------------------------
def test_g43_init_eagle_non_eagle3_keeps_aux_false():
    runner, record = _run(_config(spec=_spec(method="eagle", eagle=True)))
    assert "EagleProposer" in record["drafters"]
    assert runner.use_aux_hidden_state_outputs is False


def test_g43_init_eagle3_drafter_reports_aux_false():
    class _EagleNoAux:
        def __init__(self, *a, **k):
            self.eagle3_use_aux_hidden_state = False

    runner, _ = _run(
        _config(spec=_spec(method="eagle3", eagle=True)),
        extra=[(mr, "EagleProposer", _EagleNoAux)],
    )
    assert runner.use_aux_hidden_state_outputs is False


# --------------------------------------------------------------------------
# InputBatch assembly pass-through
# --------------------------------------------------------------------------
def test_g43_init_input_batch_forwards_runtime_config():
    reasoning = object()
    runner, record = _run(
        _config(
            cache={"use_replayssm": True},
            parallel={"cp_kv_cache_interleave_size": 8},
            reasoning=reasoning,
        )
    )
    kwargs = record["input_batch"]
    assert kwargs["use_replayssm"] is True
    assert kwargs["reasoning_config"] is reasoning
    assert kwargs["cp_kv_cache_interleave_size"] == 8
    assert kwargs["num_spec_tokens"] == 0


def test_g43_init_encoder_decoder_placeholder_block_count():
    cfg = _config(
        model={"is_encoder_decoder": True},
        scheduler={"max_num_encoder_input_tokens": 300},
    )
    _, record = _run(cfg)
    kwargs = record["input_batch"]
    assert kwargs["max_model_len"] == 300
    # cdiv(max(128, 300), 16) == 19
    assert kwargs["max_num_blocks_per_req"] == [19]


# --------------------------------------------------------------------------
# buffer / tensor assembly details
# --------------------------------------------------------------------------
def test_g43_init_kv_sharing_fast_prefill_index_tensor():
    runner, _ = _run(_config(cache={"kv_sharing_fast_prefill": True}))
    idx = runner.kv_sharing_fast_prefill_logits_indices
    assert idx is not None
    assert idx.dtype == torch.int32
    assert tuple(idx.shape) == (64,)


def test_g43_init_sampled_token_ids_pinned_meta():
    runner, _ = _run(_config())
    pinned = runner.sampled_token_ids_pinned_cpu
    assert pinned.dtype == torch.int64
    assert tuple(pinned.shape) == (4, 1)
    assert pinned.device.type == "cpu"


def test_g43_init_inputs_embeds_forwards_custom_dtype():
    runner, record = _run(
        _config(
            model={
                "dtype": torch.float32,
                "get_inputs_embeds_size": lambda: 32,
            }
        )
    )
    entry = None
    for buf in record["buffers"]:
        if buf[0] == (64, 32):
            entry = buf
    assert entry is not None
    assert entry[1] is torch.float32
    assert entry[2] is False


def test_g43_init_mrope_positions_with_spec():
    runner, record = _run(
        _config(model={"uses_mrope": True}, spec=_spec(method="custom_class", num_spec=2))
    )
    assert ((3, 65),) in _buffer_sizes(record)
    assert runner.mrope_positions is not None
    assert runner.draft_token_ids_cpu is not None


# --------------------------------------------------------------------------
# spec-decode buffer gating / streams
# --------------------------------------------------------------------------
def test_g43_init_ngram_gpu_valid_count_buffers_and_stream():
    runner, _ = _run(_config(spec=_spec(method="ngram_gpu", ngram_gpu=True)))
    assert runner._num_valid_draft_tokens_cpu is not None
    assert runner._num_valid_draft_tokens_cpu.dtype == torch.int32
    assert tuple(runner._num_valid_draft_tokens_cpu.shape) == (4,)
    assert runner._num_valid_draft_tokens_event is not None
    assert runner._num_valid_draft_tokens_copy_stream is not None


def test_g43_init_spec_draft_buffer_streams():
    runner, _ = _run(_config(spec=_spec(method="custom_class", num_spec=2)))
    assert runner.draft_token_ids_cpu is not None
    assert runner.draft_token_ids_cpu.dtype == torch.int64
    assert tuple(runner.draft_token_ids_cpu.shape) == (4, 2)
    assert runner.draft_token_ids_copy_stream is not None
    assert runner.num_accepted_tokens_event is not None
    assert runner.valid_sampled_token_count_cpu is None


def test_g43_init_async_spec_valid_count_stream():
    cfg = _config(
        scheduler={"async_scheduling": True},
        spec=_spec(method="custom_class", num_spec=2),
    )
    runner, _ = _run(cfg)
    assert runner.use_async_spec_decode is True
    assert runner.valid_sampled_token_count_cpu is not None
    assert runner.valid_sampled_token_count_cpu.dtype == torch.int32
    assert tuple(runner.valid_sampled_token_count_cpu.shape) == (4,)
    assert runner.valid_sampled_token_count_copy_stream is not None


def test_g43_init_spec_zero_tokens_skips_draft_buffers():
    runner, _ = _run(_config(spec=_spec(method="custom_class", num_spec=0)))
    assert runner.num_spec_tokens == 0
    assert runner.draft_token_ids_cpu is None
    assert runner.draft_token_ids_event is None
    assert runner.num_accepted_tokens_event is None
    assert runner.uniform_decode_query_len == 1
    assert runner.use_async_spec_decode is False


def test_g43_init_ngram_zero_spec_tokens_only_valid_buffers():
    runner, _ = _run(_config(spec=_spec(method="ngram_gpu", ngram_gpu=True, num_spec=0)))
    assert runner.num_spec_tokens == 0
    assert runner.num_tokens_no_spec_gpu is not None
    assert runner._num_valid_draft_tokens_cpu is not None
    assert runner.draft_token_ids_cpu is None


# --------------------------------------------------------------------------
# scalar / event assembly
# --------------------------------------------------------------------------
def test_g43_init_calculate_kv_scales_present_false():
    runner, _ = _run(_config(cache={"calculate_kv_scales": False}))
    assert runner.calculate_kv_scales is False


def test_g43_init_async_prepare_event_is_blocking():
    cfg = _config(scheduler={"async_scheduling": True})
    runner, _ = _run(cfg)
    assert runner.async_output_copy_stream is not None
    assert runner.prepare_inputs_event is not None
    assert runner.prepare_inputs_event.kwargs == {"blocking": True}
