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

"""G40 coverage: ModelRunnerFL residual branches (814b model_runner.py).

Complements G16/G22/G30 (_update_states), G13/G21 (__init__) and G7/G8/G10/G11
(small helpers) with the branches those suites leave behind:

_update_states (L1387-1760)
  * finished id that is re-scheduled as a brand new request in the same step
  * multi-KV-group block append (zip over several block-id groups)
  * resumed preemption that replaces multi-group block ids wholesale
  * ngram spec-count restore for a request absent from the spec-token map
  * deferred correction with a non-positive ngram correction (skip decrement)
  * deferred correction when a valid sample count of 0 is reported
  * async output recovery for a re-added (resumed) ngram request
  * ``use_async_spec_decode`` resets the per-req draft buffer up front
  * async optimistic draft bump with no previous batch map
  * prompt_logprobs == 0 is stored verbatim

_update_states_after_model_execute (L1762-1808)
  * mamba "align" fused GPU postprocess arm
  * non-align copy-to-pinned-CPU arm

__init__ (L584-1112) residual assembly
  * platform object without a ``vendor_name`` attribute (getattr default)
  * ngram-gpu pinned/GPU buffer dtypes and shapes
  * DCP rank propagated from the group for a non-zero rank
  * async-scheduling-off leaves stream/event/prepare inputs unset

small methods
  * _get_positions with a tensor index and M-RoPE disabled (only arm left)
  * _may_reorder_batch groups present but threshold None (no reorder call)
  * _init_kv_zero_meta wires KVBlockZeroer arguments
  * _get_mamba_bufs postprocess-align gating + align assertion + caching
  * jit_warmup_registry lazy import path and setter
  * _make_buffer forwards device/with_numpy
  * _get_cumsum_and_arange builds offsets and honours cumsum_dtype

Everything is CPU-only. Collaborators are patched at the
``vllm_fl.worker.model_runner`` module boundary.
"""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
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


_MISSING = object()


@contextlib.contextmanager
def _fake_modules(mapping):
    saved = {name: sys.modules.get(name, _MISSING) for name in mapping}
    for name, mod in mapping.items():
        sys.modules[name] = mod
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


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
        # Real InputBatch keeps these host-side as NumPy buffers.
        num_computed_tokens_cpu=np.zeros(8, dtype=np.int32),
        num_tokens_no_spec=np.zeros(8, dtype=np.int32),
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
    lora_request=None,
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
        lora_request=lora_request,
    )


def _state(*, output=None, block_ids=((0,),), num_computed=0, num_tokens=1, prev_draft=0):
    return SimpleNamespace(
        prev_num_draft_len=prev_draft,
        output_token_ids=[] if output is None else output,
        block_ids=[list(b) for b in block_ids],
        num_computed_tokens=num_computed,
        num_tokens=num_tokens,
    )


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
# _update_states targets
# --------------------------------------------------------------------------
def test_g40_update_states_finished_then_readded_same_id():
    removed, added = [], []
    batch = _batch(
        remove_request=lambda rid: removed.append(rid),
        add_request=lambda req: added.append(req),
    )
    runner = _runner(batch, requests={})
    _call(runner, _sched(finished=["r1"], new_reqs=[_new_req(req_id="r1")]))
    assert removed == ["r1"]
    assert "r1" in runner.requests
    assert runner.requests["r1"].output_token_ids == []
    assert added == [runner.requests["r1"]]


def test_g40_update_states_multi_group_block_append():
    appended = []
    batch = _batch(
        req_id_to_index={"r0": 0},
        block_table=SimpleNamespace(
            append_row=lambda ids, idx: appended.append((ids, idx))
        ),
    )
    state = _state(block_ids=((0, 1), (5,)))
    runner = _runner(batch, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[2],
        new_block_ids=[[[2], [6]]],
        num_output=[0],
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    assert state.block_ids == [[0, 1, 2], [5, 6]]
    assert appended == [([[2], [6]], 0)]


def test_g40_update_states_resumed_multi_group_replaces_blocks():
    added = []
    batch = _batch(add_request=lambda req: added.append(req))
    state = _state(block_ids=((0, 1), (5,)))
    runner = _runner(batch, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[0],
        new_block_ids=[[[7], [8]]],
        resumed=["r0"],
        num_output=[0],
    )
    _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached))
    assert state.block_ids == [[7], [8]]
    assert added == [state]


def test_g40_update_states_ngram_restore_missing_spec_key():
    batch = _batch(req_id_to_index={"r0": 0})
    state = _state(prev_draft=3)
    runner = _ngram_runner(batch, requests={"r0": state})
    cached = _cached(
        req_ids=["r0"],
        num_computed=[1],
        new_block_ids=[None],
        num_output=[0],
    )
    # A different request owns the scheduled spec tokens; r0 must fall back to
    # a scheduler draft count of zero via ``original_num_spec_per_req.get``.
    _call(
        runner,
        _sched(num_scheduled={"r0": 1}, cached=cached, spec_tokens={"other": [1]}),
        extras=_NGRAM_EXTRAS,
    )
    assert state.prev_num_draft_len == 0


def test_g40_update_states_correction_negative_ngram_skips_decrement():
    batch = _batch(req_id_to_index={"r0": 0}, prev_req_id_to_index={"r0": 0})
    batch.num_tokens_no_spec[0] = 5
    state = _state(prev_draft=2, output=[])
    runner = _ngram_runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        num_tokens_no_spec_gpu=torch.full((8,), 5, dtype=torch.int32),
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[10],
        new_block_ids=[None],
        num_output=[2],
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 2}, cached=cached),
                    extras=_NGRAM_EXTRAS)
    assert state.output_token_ids == [-1, -1]
    assert batch.num_tokens_no_spec[0] == 7
    # More accepted than drafted -> correction is negative -> ngram buffers
    # must not be decremented.
    runner._get_valid_sampled_token_count = lambda: [5]
    correct()
    assert state.num_computed_tokens == 12
    assert batch.num_computed_tokens_cpu[0] == 12
    assert batch.num_tokens_no_spec[0] == 7
    assert runner.num_tokens_no_spec_gpu[0] == 5


def test_g40_update_states_correction_zero_valid_count():
    batch = _batch(req_id_to_index={"r0": 0}, prev_req_id_to_index={"r0": 0})
    batch.num_tokens_no_spec[0] = 6
    state = _state(prev_draft=1, output=[])
    runner = _ngram_runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        num_tokens_no_spec_gpu=torch.full((8,), 6, dtype=torch.int32),
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[10],
        new_block_ids=[None],
        num_output=[1],
    )
    correct = _call(runner, _sched(num_scheduled={"r0": 1}, cached=cached),
                    extras=_NGRAM_EXTRAS)
    # valid count 0 -> num_accepted = -1 -> correction = optimistic + 1.
    runner._get_valid_sampled_token_count = lambda: [0]
    correct()
    assert state.num_computed_tokens == 8
    assert batch.num_computed_tokens_cpu[0] == 8
    assert batch.num_tokens_no_spec[0] == 5
    assert runner.num_tokens_no_spec_gpu[0] == 4


def test_g40_update_states_async_recover_ngram_readd():
    added, incremental = [], []
    batch = _batch(add_request=lambda req: added.append(req))
    state = _state(output=[], num_tokens=1)
    runner = _ngram_runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
    )

    def _inc(batch_, tids, nts, new_reqs, device, **kw):
        incremental.append(list(new_reqs))

    cached = _cached(
        req_ids=["r0"],
        num_computed=[3],
        new_block_ids=[None],
        num_output=[2],
        all_token_ids={"r0": [9, 8, 7, 6]},
    )
    _call(
        runner,
        _sched(num_scheduled={"r0": 2}, cached=cached),
        extras=[
            (mr, "update_scheduler_for_invalid_drafts", lambda *a, **k: None),
            (mr, "update_ngram_gpu_tensors_incremental", _inc),
        ],
    )
    assert state.output_token_ids == [7, 6]
    assert added == [state]
    assert incremental == [[state]]


def test_g40_update_states_async_spec_decode_resets_draft_buffer():
    batch = _batch()
    runner = _runner(
        batch,
        use_async_spec_decode=True,
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=np.ones(4, dtype=np.int32)),
    )
    result = _call(runner, _sched())
    assert result is None
    assert runner.prev_num_draft_tokens.np.tolist() == [0, 0, 0, 0]


def test_g40_update_states_ngram_optimistic_without_prev_map():
    batch = _batch(req_id_to_index={"r0": 0})
    batch.num_tokens_no_spec[0] = 1
    state = _state(prev_draft=2, output=[])
    runner = _ngram_runner(
        batch,
        requests={"r0": state},
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=np.zeros(4, dtype=np.int32)),
    )
    cached = _cached(
        req_ids=["r0"],
        num_computed=[10],
        new_block_ids=[None],
        num_output=[2],
    )
    _call(runner, _sched(num_scheduled={"r0": 2}, cached=cached),
          extras=_NGRAM_EXTRAS)
    assert state.output_token_ids == [-1, -1]
    assert batch.num_tokens_no_spec[0] == 3
    # No previous batch map -> the per-req draft buffer is left untouched.
    assert runner.prev_num_draft_tokens.np.tolist() == [0, 0, 0, 0]


def test_g40_update_states_prompt_logprobs_zero_stored():
    batch = _batch()
    runner = _runner(batch)
    sampling = SimpleNamespace(sampling_type=None, prompt_logprobs=0, seed=None)
    _call(runner, _sched(new_reqs=[_new_req(sampling=sampling)]))
    assert runner.num_prompt_logprobs["n0"] == 0


# --------------------------------------------------------------------------
# _update_states_after_model_execute targets
# --------------------------------------------------------------------------
def _after_exec_runner(mode, *, spec=None, hybrid=True):
    events = []

    class _Event:
        def record(self):
            events.append("record")

    runner = SimpleNamespace(
        speculative_config=object() if spec is None else spec,
        model_config=SimpleNamespace(is_hybrid=hybrid),
        cache_config=SimpleNamespace(mamba_cache_mode=mode),
        num_accepted_tokens=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
        input_batch=SimpleNamespace(
            num_accepted_tokens_cpu_tensor=torch.zeros(8, dtype=torch.int32)
        ),
        kv_cache_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(static_forward_context={"fc": 1}),
        _get_mamba_bufs=lambda: "bufs",
        _get_mamba_state_copy_funcs=lambda: "cf",
        num_accepted_tokens_event=_Event(),
    )
    return runner, events


def test_g40_after_model_execute_align_gpu_postprocess():
    runner, events = _after_exec_runner("align")
    calls = []
    fake_mu = SimpleNamespace(
        postprocess_mamba_align_gpu=lambda **kw: calls.append(kw)
    )
    out = torch.tensor([[1, 2, 3], [-1, -1, -1]], dtype=torch.int64)
    with _stack((mr, "mamba_utils", fake_mu)):
        mr.ModelRunnerFL._update_states_after_model_execute(
            runner, out, SimpleNamespace()
        )
    assert runner.num_accepted_tokens.gpu[:2].tolist() == [3, 0]
    assert calls and calls[0]["bufs"] == "bufs"
    assert calls[0]["num_reqs"] == 2
    assert calls[0]["mamba_state_copy_funcs"] == "cf"
    assert calls[0]["forward_context"] == {"fc": 1}
    assert events == ["record"]


def test_g40_after_model_execute_non_align_copies_to_cpu():
    runner, events = _after_exec_runner("eager")
    out = torch.tensor([[1, 2, 3], [-1, -1, -1]], dtype=torch.int64)
    mr.ModelRunnerFL._update_states_after_model_execute(runner, out, SimpleNamespace())
    assert runner.input_batch.num_accepted_tokens_cpu_tensor[:2].tolist() == [3, 0]
    assert events == ["record"]


# ==========================================================================
# __init__ harness (adapted from G13/G21)
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
        reasoning_config=None,
    )


def _run(config, *, vendor="cuda", no_vendor=False, extra=()):
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
    if no_vendor:
        fake_platform = SimpleNamespace(torch_device_fn=device_fn)
    else:
        fake_platform = SimpleNamespace(vendor_name=vendor, torch_device_fn=device_fn)

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
        (mr, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=1)),
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


# --------------------------------------------------------------------------
# __init__ targets
# --------------------------------------------------------------------------
def test_g40_init_platform_without_vendor_name_attr():
    runner, _ = _run(_config(), no_vendor=True)
    assert runner.use_ascend_attention is False
    assert runner.use_ascend_hybrid_cache is False
    assert runner.graph_runtime is None


def test_g40_init_ngram_gpu_buffer_dtypes_and_shapes():
    runner, _ = _run(_config(spec=_spec(method="ngram_gpu", ngram_gpu=True)))
    assert runner.num_tokens_no_spec_gpu.shape == (4,)
    assert runner.num_tokens_no_spec_gpu.dtype == torch.int32
    assert runner.token_ids_gpu_tensor.shape == (4, 128)
    assert runner.token_ids_gpu_tensor.dtype == torch.int32
    assert runner._ngram_pinned_idx_buf.dtype == torch.int64
    assert runner._ngram_pinned_val_buf.dtype == torch.int32


def test_g40_init_dcp_rank_from_group():
    runner, _ = _run(
        _config(parallel={"decode_context_parallel_size": 2}),
        extra=[
            (mr, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=3))
        ],
    )
    assert runner.dcp_world_size == 2
    assert runner.dcp_rank == 3


def test_g40_init_async_off_leaves_stream_and_event_unset():
    runner, _ = _run(_config())
    assert runner.use_async_scheduling is False
    assert runner.async_output_copy_stream is None
    assert runner.prepare_inputs_event is None


# --------------------------------------------------------------------------
# small-method targets
# --------------------------------------------------------------------------
def test_g40_get_positions_index_without_mrope():
    positions = torch.arange(10, dtype=torch.int64)
    runner = SimpleNamespace(uses_mrope=False, positions=positions)
    idx = torch.tensor([1, 3, 5])
    out = mr.ModelRunnerFL._get_positions(runner, idx)
    assert out.tolist() == [1, 3, 5]


def test_g40_may_reorder_batch_skips_without_threshold():
    calls = []
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        reorder_batch_threshold=None,
        input_batch="ib",
    )
    with _stack(
        (
            mr,
            "reorder_batch_to_split_decodes_and_prefills",
            lambda *a, **k: calls.append((a, k)),
        )
    ):
        mr.ModelRunnerFL._may_reorder_batch(runner, "so")
        assert calls == []
        runner.reorder_batch_threshold = 2
        mr.ModelRunnerFL._may_reorder_batch(runner, "so")
    assert calls and calls[0][1]["decode_threshold"] == 2


def test_g40_init_kv_zero_meta_builds_zeroer():
    captured = {}

    class _Zeroer:
        def __init__(
            self,
            device,
            attn_groups_iter,
            kernel_block_sizes,
            static_forward_context,
            num_blocks,
            runner_only_attn_layers,
        ):
            captured.update(
                device=device,
                attn_groups_iter=attn_groups_iter,
                kernel_block_sizes=kernel_block_sizes,
                static_forward_context=static_forward_context,
                num_blocks=num_blocks,
                runner_only_attn_layers=runner_only_attn_layers,
            )

    marker = iter([1, 2])
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _kv_cache_spec_attn_group_iterator=lambda: marker,
        _kernel_block_sizes=[16],
        compilation_config=SimpleNamespace(static_forward_context={"fc": 1}),
        kv_cache_config=SimpleNamespace(num_blocks=10),
        runner_only_attn_layers={"l0"},
    )
    with _stack((mr, "KVBlockZeroer", _Zeroer)):
        mr.ModelRunnerFL._init_kv_zero_meta(runner)
    assert isinstance(runner._kv_block_zeroer, _Zeroer)
    assert captured["kernel_block_sizes"] == [16]
    assert captured["num_blocks"] == 10
    assert captured["runner_only_attn_layers"] == {"l0"}
    assert captured["static_forward_context"] == {"fc": 1}
    assert captured["attn_groups_iter"] is marker


def test_g40_get_mamba_bufs_postprocess_gating_and_cache():
    created = []
    fake_mu = SimpleNamespace(
        MambaBuffers=SimpleNamespace(
            create=lambda **kw: created.append(kw) or "bufs"
        )
    )

    def _mk(spec, hybrid):
        return SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode="align"),
            _mamba_bufs=None,
            max_num_reqs=4,
            kv_cache_config="kvcfg",
            _get_mamba_state_copy_funcs=lambda: "cf",
            _make_buffer=lambda *a, **k: "mb",
            device=torch.device("cpu"),
            speculative_config=spec,
            model_config=SimpleNamespace(is_hybrid=hybrid),
        )

    with _stack((mr, "mamba_utils", fake_mu)):
        first = _mk(object(), True)
        assert mr.ModelRunnerFL._get_mamba_bufs(first) == "bufs"
        # Cached on the second call.
        assert mr.ModelRunnerFL._get_mamba_bufs(first) == "bufs"
        assert len(created) == 1
        assert created[0]["with_postprocess_align"] is True
        assert created[0]["copy_funcs"] == "cf"
        assert created[0]["make_buffer"] is not None

        second = _mk(None, True)
        mr.ModelRunnerFL._get_mamba_bufs(second)
        assert created[1]["with_postprocess_align"] is False

        bad = _mk(object(), True)
        bad.cache_config.mamba_cache_mode = "eager"
        with pytest.raises(AssertionError):
            mr.ModelRunnerFL._get_mamba_bufs(bad)


def test_g40_jit_warmup_registry_lazy_import_and_setter():
    leaf = types.ModuleType("vllm.model_executor.warmup.jit_warmup")
    leaf.JitWarmupRegistry = lambda cfg: SimpleNamespace(cfg=cfg)
    parent = types.ModuleType("vllm.model_executor.warmup")
    parent.jit_warmup = leaf
    runner = SimpleNamespace(vllm_config="cfg")
    with _fake_modules(
        {
            "vllm.model_executor.warmup": parent,
            "vllm.model_executor.warmup.jit_warmup": leaf,
        }
    ):
        reg = mr.ModelRunnerFL.jit_warmup_registry.fget(runner)
    assert reg.cfg == "cfg"
    assert runner.__dict__["_jit_warmup_registry"] is reg
    # Cached: no second import needed.
    assert mr.ModelRunnerFL.jit_warmup_registry.fget(runner) is reg
    new = object()
    mr.ModelRunnerFL.jit_warmup_registry.fset(runner, new)
    assert mr.ModelRunnerFL.jit_warmup_registry.fget(runner) is new


def test_g40_make_buffer_forwards_device_and_numpy():
    captured = {}

    class _Buf:
        def __init__(self, *size, dtype=None, device=None, with_numpy=True):
            captured.update(
                size=size, dtype=dtype, device=device, with_numpy=with_numpy
            )

    runner = SimpleNamespace(device=torch.device("cpu"))
    with _stack((mr, "CpuGpuBuffer", _Buf)):
        buf = mr.ModelRunnerFL._make_buffer(
            runner, 4, dtype=torch.int32, numpy=False
        )
    assert isinstance(buf, _Buf)
    assert captured["size"] == (4,)
    assert captured["dtype"] == torch.int32
    assert captured["device"] == torch.device("cpu")
    assert captured["with_numpy"] is False


def test_g40_get_cumsum_and_arange_builds_offsets():
    runner = SimpleNamespace(arange_np=np.arange(20, dtype=np.int64))
    out = np.empty(20, dtype=np.int64)
    cu = mr.ModelRunnerFL._get_cumsum_and_arange(
        runner, np.array([2, 5, 3]), out
    )
    assert cu.tolist() == [2, 7, 10]
    assert out[:10].tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    cu2 = mr.ModelRunnerFL._get_cumsum_and_arange(
        runner, np.array([1, 1]), out, cumsum_dtype=np.int32
    )
    assert cu2.dtype == np.int32
