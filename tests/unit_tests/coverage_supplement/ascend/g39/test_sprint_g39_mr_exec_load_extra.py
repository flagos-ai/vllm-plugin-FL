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

"""G39 coverage: residual deep branches of ModelRunnerFL (814b).

Targets the three heavy methods, adding scenarios that are distinct from
g7/g10/g13/g14/g18/g22/g24/g34:

  * ``execute_model``   (L4405-4784): microbatching opt-out, FULL+ubatch
    attention slice selection, non-FULL ascend skip of FIA padding,
    encoder-inputs on a decoder-only model, mamba-align deferred-correction
    single-shot consumption and model-kwargs forwarding.
  * ``sample_tokens``   (L4802-5060): draft-model / extract-hidden GPU
    token paths, padded-drafter disabled (post-bookkeeping) for EAGLE and
    ngram, ngram no-fit-no-event zeroing, and async EP-fault/nan plumbing.
  * ``load_model``      (L5527-5747): graph-runtime forwarding, MUSA
    dynamo/warmup branch, synchronous EPLB (no async loop), dummy-weight
    MoE skip, drafter without a loader, drafter EPLB with no MoE, EAGLE3
    aux layer resolution, breakable wrapping of the drafter, and drafter
    comm-buffer preparation.

Everything is CPU-only; collaborators are patched at the
``vllm_fl.worker.model_runner`` module boundary.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

import vllm_fl.worker.model_runner as mr
from vllm_fl.worker.model_runner import ModelRunnerFL


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
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


@contextlib.contextmanager
def _fake_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class _Ctx:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


# ===========================================================================
# execute_model
# ===========================================================================
def _pp(is_last=True, send=None, broadcast=None, ranks=(0, 1)):
    return types.SimpleNamespace(
        is_last_rank=is_last,
        send_tensor_dict=send or (lambda *a, **k: None),
        broadcast_tensor_dict=broadcast or (lambda data, src=0: data),
        ranks=list(ranks),
    )


def _patches(pp=None, ubatch=None, **extra):
    pairs = [
        (mr, "record_function_or_nullcontext", lambda name: _Ctx()),
        (mr, "has_ec_transfer", lambda: False),
        (mr, "has_kv_transfer_group", lambda: False),
        (mr, "get_pp_group", lambda: pp or _pp()),
        (mr, "get_tp_group", lambda: "tp-group"),
        (mr, "set_forward_context", lambda *a, **k: _Ctx(k)),
        (mr, "maybe_create_ubatch_slices", ubatch or (lambda *a: (None, None))),
        (mr, "logger", types.SimpleNamespace(debug=lambda *a, **k: None)),
    ]
    pairs.extend((mr, name, value) for name, value in extra.items())
    return _stack(*pairs)


def _sched(total=4, **extra):
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


def _runner(**extra):
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
        maybe_get_kv_connector_output=lambda so, defer_finalize=False: _Ctx(
            "kv-out"
        ),
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


@pytest.fixture
def ascend_attn_module():
    name = "vllm_fl.dispatch.backends.vendor.ascend.impl.attention"
    mod = types.ModuleType(name)

    class AscendAttentionState:
        DecodeOnly = "decode"

    mod.AscendAttentionState = AscendAttentionState
    saved = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        yield AscendAttentionState
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


def test_exec_allow_microbatching_opt_out_forwarded():
    captured = {}
    runner = _runner(
        _allow_microbatching=lambda nr, nst: False,
        _determine_batch_execution_and_padding=lambda **kw: (
            captured.update(determine=kw) or mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=4, num_reqs=None),
            False,
            None,
            "stats",
        ),
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert captured["determine"]["allow_microbatching"] is False
    assert captured["determine"]["num_encoder_reqs"] == 0


def test_exec_full_graph_ubatch_uses_padded_attn_slices():
    captured = {}
    runner = _runner(
        _determine_batch_execution_and_padding=lambda **kw: (
            mr.CUDAGraphMode.FULL,
            types.SimpleNamespace(num_tokens=8, num_reqs=4),
            True,
            "dp-tokens",
            "stats",
        ),
        _build_attention_metadata=lambda **kw: (
            captured.update(build=kw) or ("attn-meta", "spec-common")
        ),
        _get_slot_mappings=lambda **kw: (
            captured.update(slots=kw) or ({"g": "slots"}, "flat-slots")
        ),
    )
    seen = {}

    def _ctx(*a, **k):
        seen.update(k)
        return _Ctx()

    with _patches(
        ubatch=lambda should, nst, ntp, nrp, nub: ("us", "usp"),
        set_forward_context=_ctx,
    ):
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert captured["build"]["ubatch_slices"] == "usp"
    assert captured["slots"]["num_tokens_padded"] == 8
    assert seen["num_tokens_across_dp"] == "dp-tokens"
    assert seen["ubatch_slices"] == "usp"


def test_exec_ascend_attention_non_full_skips_fia_padding(ascend_attn_module):
    captured = {}
    pad_calls = []
    runner = _runner(
        use_ascend_attention=True,
        _ascend_attn_state="prefill",
        _pad_query_start_loc_for_fia=lambda *a: pad_calls.append(a) or 99,
        _determine_batch_execution_and_padding=lambda **kw: (
            captured.update(determine=kw) or mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=4, num_reqs=None),
            False,
            None,
            "stats",
        ),
        _get_slot_mappings=lambda **kw: (
            captured.update(slots=kw) or ({"g": "slots"}, "flat-slots")
        ),
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert pad_calls == []
    assert captured["determine"]["disable_full"] is True
    assert captured["slots"]["num_reqs_padded"] == 2


def test_exec_encoder_inputs_on_decoder_model_keeps_compiled():
    captured = {}
    runner = _runner(model_config=types.SimpleNamespace(is_encoder_decoder=False))
    with _patches(
        set_forward_context=lambda *a, **k: captured.update(k) or _Ctx()
    ):
        mr.ModelRunnerFL.execute_model(
            runner, _sched(scheduled_encoder_inputs={"req": 1})
        )

    assert captured["skip_compiled"] is False
    assert captured["num_tokens"] == 4


def test_exec_mamba_align_consumes_deferred_correction_once():
    order = []
    runner = _runner(
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode="align"
        ),
        _update_states=lambda so: (lambda: order.append("deferred")),
    )
    pre_calls = []
    with _patches(), _Patch(
        mr.mamba_utils, "preprocess_mamba", lambda *a, **k: pre_calls.append(a)
    ):
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert order == ["deferred"]
    assert len(pre_calls) == 1


def test_exec_model_kwargs_and_preprocess_inputs_forwarded():
    captured = {}
    intermediate = object()
    runner = _runner(
        _preprocess=lambda so, ntp, it: (
            "ids",
            "embeds",
            "positions",
            it,
            {"extra": 7},
            "ec-conn",
        ),
        _model_forward=lambda **kw: captured.update(kw) or torch.zeros(4, 3),
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(runner, _sched(), intermediate_tensors=intermediate)

    assert captured["input_ids"] == "ids"
    assert captured["inputs_embeds"] == "embeds"
    assert captured["positions"] == "positions"
    assert captured["intermediate_tensors"] is intermediate
    assert captured["extra"] == 7


# ===========================================================================
# sample_tokens
# ===========================================================================
class _FakeMRO:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.only = False

    @classmethod
    def with_kv_conn_output_only(cls, kv_connector_output):
        obj = cls(kv_connector_output=kv_connector_output)
        obj.only = True
        return obj


class _FakeAsync:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sampled_token_ids_cpu = kwargs["sampled_token_ids"]
        self.async_copy_ready_event = object()


class _FakeDrafter:
    def __init__(self):
        self.prepare_calls = []
        self.dummy_calls = []

    def prepare_next_token_ids_padded(self, *args):
        self.prepare_calls.append(args)
        return torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.int64)

    def dummy_run(self, num_tokens=1):
        self.dummy_calls.append(num_tokens)


class _FakeNgram:
    def __init__(self):
        self.update_calls = []

    def update_token_ids_ngram(self, *args):
        self.update_calls.append(args)
        return (
            torch.zeros(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int64),
            None,
        )


class _InputBatch:
    def __init__(self, req_ids=("a",)):
        self.req_ids = list(req_ids)
        self.sampling_metadata = object()
        self.vocab_size = 32
        self.prev_sampled_token_ids = object()
        self.async_calls = []

    def set_async_sampled_token_ids(self, ids, event):
        self.async_calls.append((ids, event))


def _spec(**flags):
    base = dict(
        use_eagle=lambda: False,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: False,
        use_ngram_gpu=lambda: False,
        use_dflash=lambda: False,
        disable_padded_drafter_batch=False,
    )
    base.update(flags)
    return types.SimpleNamespace(**base)


@contextlib.contextmanager
def _sample_env(pp=None):
    calls = types.SimpleNamespace(grammar=[], advance=[])
    pp = pp or types.SimpleNamespace(is_last_rank=True, world_size=1, rank=0)
    with _stack(
        (mr, "ModelRunnerOutput", _FakeMRO),
        (mr, "AsyncGPUModelRunnerOutput", _FakeAsync),
        (mr, "apply_grammar_bitmask", lambda *a: calls.grammar.append(a)),
        (mr, "record_function_or_nullcontext", lambda *a, **k: contextlib.nullcontext()),
        (mr, "get_pp_group", lambda: pp),
        (mr, "advance_io_step", lambda: calls.advance.append(True)),
        (mr, "EagleProposer", _FakeDrafter),
        (mr, "NgramProposerGPU", _FakeNgram),
    ):
        yield calls


def _sample_runner(
    *,
    spec=None,
    fits=True,
    async_sched=False,
    event=None,
    drafter=None,
    dp=1,
    num_nans=None,
    check_ep_fault=False,
):
    ib = _InputBatch()
    sampled = torch.tensor([[3], [7]], dtype=torch.int64)
    sampler = types.SimpleNamespace(sampled_token_ids=sampled, logprobs_tensors=None)
    scheduler = types.SimpleNamespace(total_num_scheduled_tokens=2)
    logits = torch.zeros(2, 8)
    state = (
        scheduler,
        logits,
        object(),
        object(),
        torch.zeros(2, 4),
        torch.zeros(2, 4),
        None,
        None,
        "cgstats",
        {"slot": torch.zeros(2)},
    )
    calls = types.SimpleNamespace(
        sample=[], update_states=[], receive=[], broadcast=[], propose=[],
        copy_draft=[], bookkeep=[], finalize=[], eplb=[], copy_valid=[],
    )
    runner = types.SimpleNamespace(
        execute_model_state=state,
        kv_connector_output="kv-in",
        use_async_scheduling=async_sched,
        broadcast_pp_output=False,
        input_batch=ib,
        speculative_config=spec,
        num_spec_tokens=2,
        device=torch.device("cpu"),
        requests={"a": object()},
        discard_request_mask=types.SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        token_ids_gpu_tensor=torch.zeros(2, dtype=torch.int32),
        num_tokens_no_spec_gpu=torch.zeros(2, dtype=torch.int32),
        valid_sampled_token_count_event=event,
        parallel_config=types.SimpleNamespace(data_parallel_size=dp),
        supports_mm_inputs=False,
        check_ep_fault=check_ep_fault,
        _draft_token_ids=None,
        _draft_probs=object(),
        _draft_prob_req_ids=object(),
        _draft_token_req_ids=object(),
        valid_sampled_token_count_gpu=object(),
        drafter=drafter,
        calls=calls,
    )
    runner._sample = lambda lg, md: calls.sample.append((lg, md)) or sampler
    runner._update_states_after_model_execute = (
        lambda ids, so: calls.update_states.append((ids, so))
    )
    runner._pp_receive_prev_sampled_token_ids_to_input_batch = (
        lambda: calls.receive.append(True)
    )
    runner._pp_broadcast_prev_sampled_token_ids = (
        lambda ids: calls.broadcast.append(ids)
    )
    runner._input_fits_in_drafter = lambda md: fits
    runner.propose_draft_token_ids = (
        lambda *a, **k: calls.propose.append(a) or "draft"
    )
    runner._copy_draft_token_ids_to_cpu = (
        lambda so, zeros_only=False: calls.copy_draft.append(zeros_only)
    )
    runner._copy_valid_sampled_token_count = (
        lambda n, c: calls.copy_valid.append((n, c))
    )
    runner._bookkeeping_sync = lambda *a, **k: calls.bookkeep.append(a) or (
        {}, num_nans, None, [4, 5], {}, {"a": 0}, {"a": 0}, [1],
    )
    runner.finalize_kv_connector = lambda: calls.finalize.append(True)
    runner.eplb_step = lambda: calls.eplb.append(True)
    runner._get_or_create_async_output_copy_stream = lambda: "copy-stream"
    runner.sample_tokens = ModelRunnerFL.sample_tokens.__get__(runner, ModelRunnerFL)
    return runner


def test_sample_draft_model_uses_gpu_tokens():
    runner = _sample_runner(
        spec=_spec(uses_draft_model=lambda: True), drafter=_FakeDrafter(), fits=True
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert len(runner.calls.propose) == 1
    assert runner.calls.finalize == [True]
    assert runner.calls.copy_draft == [False]


def test_sample_extract_hidden_states_uses_gpu_tokens():
    runner = _sample_runner(
        spec=_spec(uses_extract_hidden_states=lambda: True),
        drafter=_FakeDrafter(),
        fits=True,
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert len(runner.calls.propose) == 1
    assert runner.calls.finalize == [True]


def test_sample_eagle_padded_disabled_runs_after_bookkeeping():
    runner = _sample_runner(
        spec=_spec(use_eagle=lambda: True, disable_padded_drafter_batch=True),
        drafter=_FakeDrafter(),
        fits=True,
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert len(runner.calls.propose) == 1
    assert runner.calls.propose[0][1] == [4, 5]
    assert runner.calls.finalize == [True]


def test_sample_ngram_padded_disabled_runs_after_bookkeeping():
    runner = _sample_runner(
        spec=_spec(use_ngram_gpu=lambda: True, disable_padded_drafter_batch=True),
        drafter=_FakeNgram(),
        fits=True,
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert len(runner.calls.propose) == 1
    assert runner.calls.propose[0][1] == [4, 5]


def test_sample_ngram_no_fit_no_event_zeroes_draft():
    drafter = _FakeNgram()
    runner = _sample_runner(
        spec=_spec(use_ngram_gpu=lambda: True), drafter=drafter, fits=False, event=None
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert drafter.update_calls == []
    assert runner._draft_token_ids.shape == (1, 2)
    assert runner._draft_probs is None
    assert runner.calls.copy_draft[-1] is True
    assert runner.calls.propose == []


def test_sample_async_output_forwards_ep_fault_and_nans():
    runner = _sample_runner(
        async_sched=True, num_nans="nans-device", check_ep_fault=True
    )
    with _sample_env():
        out = runner.sample_tokens(None)

    assert isinstance(out, _FakeAsync)
    assert out.kwargs["check_ep_fault"] is True
    assert out.kwargs["num_nans"] == "nans-device"
    assert len(runner.input_batch.async_calls) == 1


# ===========================================================================
# load_model
# ===========================================================================
class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def get_eagle3_default_aux_hidden_state_layers(self):
        return (0,)

    def set_aux_hidden_state_layers(self, layers):
        self.aux_layers = layers


class _Loader:
    def __init__(self, calls, model):
        self.calls = calls
        self.model = model

    def load_model(self, *, vllm_config, model_config):
        self.calls.append(("load_model", vllm_config, model_config))
        return self.model


class _Profiler:
    def __init__(self, consumed_memory=2048):
        self.consumed_memory = consumed_memory

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Offloader:
    def __init__(self, calls):
        self.calls = calls

    def post_init(self):
        self.calls.append("post_init")


class _Recorder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _EplbState:
    def __init__(self, parallel_config, device):
        self.parallel_config = parallel_config
        self.device = device
        self.added = []
        self.is_async = True
        self.started = False

    def add_model(self, model, config):
        self.added.append((model, config))

    def start_async_loop(self):
        self.started = True


class _SyncEplbState(_EplbState):
    def __init__(self, parallel_config, device):
        super().__init__(parallel_config, device)
        self.is_async = False


class _Counter:
    def __init__(self):
        self.stock_torch_compile_count = 0


def _load_runner(
    model,
    *,
    spec_config=None,
    drafter=None,
    enable_eplb=False,
    mode=None,
    cudagraph_mode=None,
    use_ubatching=False,
    lora_config=None,
    graph_runtime=None,
    use_aux=False,
):
    from vllm.config import CompilationMode, CUDAGraphMode

    if mode is None:
        mode = CompilationMode.NONE
    if cudagraph_mode is None:
        cudagraph_mode = CUDAGraphMode.NONE
    compilation = types.SimpleNamespace(
        mode=mode,
        cudagraph_mode=cudagraph_mode,
        init_backend=lambda cfg: "backend",
    )
    model_config = types.SimpleNamespace(
        model="t/m",
        enforce_eager=True,
        quantization=None,
        multimodal_config=None,
        revision="r",
    )
    vllm_config = types.SimpleNamespace(
        compilation_config=compilation, speculative_config=spec_config
    )
    calls = types.SimpleNamespace(
        loader=[], comm=[], hooks=[], warmup=[], io=[], offload=[], lora=[]
    )
    runner = types.SimpleNamespace(
        model_config=model_config,
        load_config=types.SimpleNamespace(load_format="safetensors"),
        vllm_config=vllm_config,
        parallel_config=types.SimpleNamespace(
            enable_eplb=enable_eplb,
            enable_elastic_ep=False,
            use_ubatching=use_ubatching,
        ),
        lora_config=lora_config,
        device=torch.device("cpu"),
        model=model,
        _moe_model=None,
        compilation_config=compilation,
        graph_runtime=graph_runtime,
        use_aux_hidden_state_outputs=use_aux,
        speculative_config=spec_config,
        jit_warmup_registry=types.SimpleNamespace(
            activate=lambda: contextlib.nullcontext()
        ),
        eplb_state=None,
        calls=calls,
    )
    runner._setup_eagle3_aux_hidden_state_outputs = (
        ModelRunnerFL._setup_eagle3_aux_hidden_state_outputs.__get__(
            runner, ModelRunnerFL
        )
    )
    runner.get_model = lambda: runner.model
    runner.load_lora_model = lambda m, cfg, dev: calls.lora.append(m) or model
    runner.load_model = ModelRunnerFL.load_model.__get__(runner, ModelRunnerFL)
    if drafter is not None:
        runner.drafter = drafter
    return runner


@contextlib.contextmanager
def _load_env(
    runner,
    loader,
    *,
    moe_fn=None,
    eplb_cls=_EplbState,
    brk=False,
    wrapper_cls=None,
    platform=None,
    eagle3=False,
):
    calls = runner.calls
    moe_fn = moe_fn or (lambda m: None)
    wrapper_cls = wrapper_cls or _Recorder
    pairs = [
        (mr, "init_io_dump_from_env", lambda *a, **k: calls.io.append(a)),
        (mr, "get_model_loader", lambda cfg: loader),
        (mr, "DeviceMemoryProfiler", _Profiler()),
        (mr, "warmup_cached_ops", lambda *a, **k: calls.warmup.append(True)),
        (mr, "_prepare_comm_buffer", lambda m: calls.comm.append(m)),
        (mr, "register_io_module_hooks", lambda m: calls.hooks.append(m)),
        (mr, "supports_multimodal_pruning", lambda m: False),
        (mr, "supports_eagle3", lambda m: eagle3),
        (mr, "get_offloader", lambda: _Offloader(calls.offload)),
        (mr, "get_mixture_of_experts_model", moe_fn),
        (mr, "EplbState", eplb_cls),
        (mr, "format_gib", lambda x: "0.0"),
        (mr, "is_breakable_cudagraph_enabled", lambda: brk),
        (mr, "BreakableCUDAGraphWrapper", wrapper_cls),
        (mr, "UBatchWrapper", wrapper_cls),
        (mr, "resolve_obj_by_qualname", lambda q: wrapper_cls),
        (mr, "compilation_counter", _Counter()),
    ]
    if platform is not None:
        pairs.append((mr, "current_platform", platform))
    with _stack(*pairs):
        yield calls


def test_load_graph_runtime_forwarded_to_wrapper():
    from vllm.config import CUDAGraphMode

    model = _Model()
    runtime = object()
    runner = _load_runner(
        model, cudagraph_mode=CUDAGraphMode.FULL, graph_runtime=runtime
    )
    loader = _Loader(runner.calls.loader, model)
    platform = types.SimpleNamespace(
        device_type="cpu", get_static_graph_wrapper_cls=lambda: "W"
    )
    with _load_env(runner, loader, platform=platform):
        runner.load_model()

    assert isinstance(runner.model, _Recorder)
    assert runner.model.kwargs["runtime_mode"] == CUDAGraphMode.FULL
    assert runner.model.kwargs["graph_runtime"] is runtime


def test_load_musa_configures_dynamo_and_warms_cached_ops():
    from vllm.config import CompilationMode

    model = _Model()
    runner = _load_runner(model, mode=CompilationMode.VLLM_COMPILE)
    loader = _Loader(runner.calls.loader, model)
    dispatch_calls = []
    ignore = set()
    dynamo_config = types.SimpleNamespace(ignore_logger_methods=ignore)
    platform = types.SimpleNamespace(device_type="musa")
    with _fake_module(
        "vllm_fl.dispatch",
        warmup_cached_ops=lambda: dispatch_calls.append(True),
    ), _fake_module("torch._dynamo", config=dynamo_config):
        with _load_env(runner, loader, platform=platform):
            runner.load_model()

    assert dispatch_calls == [True]
    assert logging.Logger.debug in ignore
    assert len(runner.calls.warmup) == 1
    assert runner.calls.offload == ["post_init"]


def test_load_sync_eplb_does_not_start_async_loop():
    model = _Model()
    runner = _load_runner(model, enable_eplb=True)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(
        runner, loader, moe_fn=lambda m: ("moe", m), eplb_cls=_SyncEplbState
    ):
        runner.load_model()

    assert runner._moe_model == ("moe", model)
    assert runner.eplb_state.added == [(("moe", model), runner.model_config)]
    assert runner.eplb_state.started is False


def test_load_dummy_weights_with_moe_skips_eplb_registration():
    model = _Model()
    runner = _load_runner(model, enable_eplb=True)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=lambda m: ("moe", m)):
        runner.load_model(load_dummy_weights=True)

    assert runner.load_config.load_format == "dummy"
    assert runner._moe_model == ("moe", model)
    assert runner.eplb_state.added == []
    assert runner.eplb_state.started is False
    assert runner.calls.comm == []
    assert runner.calls.hooks == []


def test_load_drafter_without_load_model_method_is_skipped():
    model = _Model()
    drafter_model = _Model()
    drafter = types.SimpleNamespace(model=drafter_model)
    runner = _load_runner(model, drafter=drafter)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader):
        runner.load_model()

    assert runner.model is model
    assert runner._moe_model is None
    assert runner.calls.comm == [model, drafter_model]


def test_load_drafter_eplb_without_moe_not_registered():
    model = _Model()
    drafter_model = _Model()
    set_calls = []
    drafter = types.SimpleNamespace(
        model=drafter_model,
        load_model=lambda m: None,
        set_eplb_state=lambda s: set_calls.append(s),
    )
    spec_config = types.SimpleNamespace(
        draft_model_config=types.SimpleNamespace(model="draft/m")
    )
    runner = _load_runner(
        model, spec_config=spec_config, drafter=drafter, enable_eplb=True
    )
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=lambda m: None):
        runner.load_model()

    assert runner.eplb_state.added == []
    assert set_calls == []


def test_load_aux_hidden_state_outputs_resolves_default_layers():
    model = _Model()
    runner = _load_runner(model, use_aux=True)
    runner._get_eagle3_aux_layers_from_config = lambda: None
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, eagle3=True):
        runner.load_model()

    assert model.aux_layers == (0,)


def test_load_breakable_wraps_drafter_model():
    from vllm.config import CUDAGraphMode

    model = _Model()
    drafter_model = _Model()
    drafter = types.SimpleNamespace(model=drafter_model, load_model=lambda m: None)
    runner = _load_runner(
        model, cudagraph_mode=CUDAGraphMode.FULL, drafter=drafter
    )
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, brk=True):
        runner.load_model()

    assert isinstance(runner.model, _Recorder)
    assert runner.model.args[0] is model
    assert isinstance(drafter.model, _Recorder)
    assert drafter.model.args[0] is drafter_model
    assert runner.calls.offload == ["post_init"]


def test_load_prepares_drafter_comm_buffer_and_main_hooks():
    model = _Model()
    drafter_model = _Model()
    drafter = types.SimpleNamespace(model=drafter_model, load_model=lambda m: None)
    runner = _load_runner(model, drafter=drafter)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader):
        runner.load_model()

    assert runner.calls.comm == [model, drafter_model]
    assert runner.calls.hooks == [model]
