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

"""G42 coverage: residual branches of ModelRunnerFL (814b).

Targets the three heavy methods with scenarios distinct from the
g7/g10/g13/g14/g18/g22/g24/g31/g34/g39 sweeps:

  * ``execute_model`` (L4405-4784): re-entry guard, KV-transfer metadata
    assertion, EC-producer short circuit, empty schedule with KV transfer,
    unpadded slot selection when neither FULL graphs nor separate KV update
    apply, Ascend decode-only non-FULL padding opt-out, mamba-align without
    deferred corrections / staging, and EPLB prepare_forward plumbing.
  * ``sample_tokens`` (L4802-5060): state-less sync/async returns, grammar
    plus async output, multimodal async output, GPU-token no-fit without an
    event, ngram no-fit without an event, padded-disabled drafters with and
    without DP dummy-run, spec-less bookkeeping and async PP broadcast skip.
  * ``load_model`` (L5527-5747): drafter EPLB guard assertions (elastic EP,
    missing speculative config, missing draft config, missing set_eplb_state),
    cudagraph-mode assertion, EAGLE3 aux support error, config-sourced aux
    layers, NPU dispatch warmup without MUSA, ubatching wrapper modes, static
    graph wrapper without a runtime, multimodal-pruning short circuit,
    out-of-memory re-raise and LoRA wrapping.

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

from vllm.config import CUDAGraphMode, CompilationMode

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


def test_exec_rejects_reentry_with_pending_state():
    runner = _runner(execute_model_state=object())
    with _patches():
        with pytest.raises(RuntimeError, match="sample_tokens"):
            mr.ModelRunnerFL.execute_model(runner, _sched())


def test_exec_kv_transfer_requires_metadata():
    runner = _runner()
    with _patches(
        has_kv_transfer_group=lambda: True,
        get_kv_transfer_group=lambda: types.SimpleNamespace(
            handle_preemptions=lambda m: None
        ),
    ):
        with pytest.raises(AssertionError):
            mr.ModelRunnerFL.execute_model(
                runner, _sched(kv_connector_metadata=None)
            )


def test_exec_ec_producer_returns_empty_encoder_output():
    exc_calls = []
    runner = _runner(
        maybe_get_ec_connector_output=lambda so, encoder_cache=None: _Ctx("ec-ctx"),
        _execute_mm_encoder=lambda so: exc_calls.append(so),
    )
    with _patches(
        has_ec_transfer=lambda: True,
        get_ec_transfer=lambda: types.SimpleNamespace(is_consumer=False),
        make_empty_encoder_model_runner_output=lambda so: "empty-enc",
    ):
        out = mr.ModelRunnerFL.execute_model(runner, _sched(total=0))

    assert out == "empty-enc"
    assert len(exc_calls) == 1
    assert runner.execute_model_state is None


def test_exec_empty_external_launcher_with_kv_transfer_no_forward():
    dummies = []
    runner = _runner(
        parallel_config=types.SimpleNamespace(
            distributed_executor_backend="external_launcher",
            data_parallel_size=2,
            use_ubatching=False,
            num_ubatches=2,
        ),
        _dummy_run=lambda n: dummies.append(n),
        kv_connector_no_forward=lambda so, cfg: "kv-nf",
    )
    with _patches(
        has_kv_transfer_group=lambda: True,
        get_kv_transfer_group=lambda: types.SimpleNamespace(
            handle_preemptions=lambda m: None
        ),
    ):
        out = mr.ModelRunnerFL.execute_model(
            runner, _sched(total=0, kv_connector_metadata="meta")
        )

    assert out == "kv-nf"
    assert dummies == [1]


def test_exec_empty_single_dp_returns_empty_output():
    runner = _runner()
    with _patches():
        out = mr.ModelRunnerFL.execute_model(runner, _sched(total=0))

    assert out is mr.EMPTY_MODEL_RUNNER_OUTPUT


def test_exec_unpadded_slots_when_no_full_graph_or_separate_kv():
    captured = {}
    runner = _runner(
        _determine_batch_execution_and_padding=lambda **kw: (
            mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=8, num_reqs=4),
            False,
            None,
            "stats",
        ),
        _get_slot_mappings=lambda **kw: (
            captured.update(slots=kw) or ({"g": "slots"}, "flat-slots")
        ),
        _build_attention_metadata=lambda **kw: (
            captured.update(build=kw) or ("attn-meta", "spec-common")
        ),
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(runner, _sched(total=4))

    assert captured["slots"]["num_tokens_padded"] == 4
    assert captured["slots"]["num_reqs_padded"] == 2
    assert captured["build"]["num_tokens_padded"] is None
    assert captured["build"]["num_reqs_padded"] is None


def test_exec_ascend_decode_only_non_full_skips_fia_padding(ascend_attn_module):
    captured = {}
    pad_calls = []
    runner = _runner(
        use_ascend_attention=True,
        _ascend_attn_state=ascend_attn_module.DecodeOnly,
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
    assert captured["determine"]["disable_full"] is False
    assert captured["slots"]["num_reqs_padded"] == 2


def test_exec_mamba_align_without_deferred_or_staging():
    pre_calls = []
    stage_calls = []
    runner = _runner(
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode="align"
        ),
        _update_states=lambda so: None,
        _get_mamba_bufs=lambda: types.SimpleNamespace(
            preprocess="pre", postprocess_align=None
        ),
    )
    with _patches(), _Patch(
        mr.mamba_utils, "preprocess_mamba", lambda *a, **k: pre_calls.append(a)
    ), _Patch(
        mr.mamba_utils,
        "stage_postprocess_inputs_to_gpu",
        lambda *a: stage_calls.append(a),
    ):
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert len(pre_calls) == 1
    assert stage_calls == []


def test_exec_eplb_prepare_forward_forwards_unpadded_tokens():
    captured = {}
    runner = _runner(
        eplb_state=types.SimpleNamespace(
            prepare_forward=lambda *a: captured.update(args=a)
        ),
    )
    with _patches(ubatch=lambda should, nst, ntp, nrp, nub: ("us", "usp")):
        mr.ModelRunnerFL.execute_model(runner, _sched(total=4))

    assert captured["args"][1] == 4
    assert captured["args"][2] == "usp"


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


def test_sample_without_state_sync_returns_kv_only():
    runner = _sample_runner()
    runner.execute_model_state = None
    with _sample_env():
        out = runner.sample_tokens(None)

    assert isinstance(out, _FakeMRO)
    assert out.only is True
    assert out.kwargs["kv_connector_output"] == "kv-in"
    assert runner.kv_connector_output is None
    assert runner.calls.receive == []


def test_sample_without_state_async_last_rank_skips_receive():
    runner = _sample_runner(async_sched=True)
    runner.execute_model_state = None
    pp = types.SimpleNamespace(is_last_rank=True, world_size=2, rank=1)
    with _sample_env(pp=pp):
        out = runner.sample_tokens(None)

    assert out.only is True
    assert runner.calls.receive == []


def test_sample_grammar_applied_before_async_output():
    runner = _sample_runner(async_sched=True)
    grammar = object()
    with _sample_env() as env:
        out = runner.sample_tokens(grammar)

    assert isinstance(out, _FakeAsync)
    assert len(env.grammar) == 1
    assert env.grammar[0][1] is grammar


def test_sample_multimodal_ec_output_async():
    runner = _sample_runner(async_sched=True)
    state = list(runner.execute_model_state)
    state[7] = "ec-out"
    runner.execute_model_state = tuple(state)
    runner.supports_mm_inputs = True
    with _sample_env():
        out = runner.sample_tokens(None)

    assert isinstance(out, _FakeAsync)
    assert out.kwargs["model_runner_output"].kwargs["ec_connector_output"] == "ec-out"


def test_sample_eagle_no_fit_no_event_single_dp_zeroes():
    drafter = _FakeDrafter()
    runner = _sample_runner(
        spec=_spec(use_eagle=lambda: True), drafter=drafter, fits=False, event=None, dp=1
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert drafter.prepare_calls == []
    assert drafter.dummy_calls == []
    assert runner._draft_token_ids.shape == (1, 2)
    assert runner.calls.copy_draft[-1] is True
    assert runner.calls.propose == []


def test_sample_ngram_no_fit_no_event_zeroes():
    drafter = _FakeNgram()
    runner = _sample_runner(
        spec=_spec(use_ngram_gpu=lambda: True),
        drafter=drafter,
        fits=False,
        event=None,
        dp=2,
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert drafter.update_calls == []
    assert runner._draft_token_ids.shape == (1, 2)
    assert runner.calls.copy_draft[-1] is True


def test_sample_padded_disabled_eagle_no_fit_single_dp_no_dummy():
    drafter = _FakeDrafter()
    runner = _sample_runner(
        spec=_spec(use_eagle=lambda: True, disable_padded_drafter_batch=True),
        drafter=drafter,
        fits=False,
        dp=1,
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert drafter.dummy_calls == []
    assert runner._draft_token_ids.shape == (1, 2)
    assert runner.calls.copy_draft[-1] is True


def test_sample_padded_disabled_uses_valid_ids_when_fits():
    runner = _sample_runner(
        spec=_spec(use_eagle=lambda: True, disable_padded_drafter_batch=True),
        drafter=_FakeDrafter(),
        fits=True,
        dp=2,
    )
    with _sample_env():
        runner.sample_tokens(None)

    assert len(runner.calls.propose) == 1
    assert runner.calls.propose[0][1] == [4, 5]
    assert runner.calls.finalize == [True]


def test_sample_without_spec_skips_draft_and_finalize():
    runner = _sample_runner(spec=None, dp=1)
    with _sample_env():
        out = runner.sample_tokens(None)

    assert isinstance(out, _FakeMRO)
    assert runner.calls.propose == []
    assert runner.calls.finalize == []
    assert runner.calls.copy_draft == []


def test_sample_async_skips_broadcast_with_broadcast_pp_output():
    runner = _sample_runner(async_sched=True)
    runner.broadcast_pp_output = True
    pp = types.SimpleNamespace(is_last_rank=True, world_size=2, rank=1)
    with _sample_env(pp=pp):
        runner.sample_tokens(None)

    assert runner.calls.broadcast == []


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

    def compile(self, **kwargs):
        self.compile_kwargs = kwargs
        return self


class _Loader:
    def __init__(self, calls, model, raise_oom=False):
        self.calls = calls
        self.model = model
        self.raise_oom = raise_oom

    def load_model(self, *, vllm_config, model_config):
        self.calls.append(("load_model", vllm_config, model_config))
        if self.raise_oom:
            raise torch.OutOfMemoryError("oom")
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


class _Counter:
    def __init__(self):
        self.stock_torch_compile_count = 0


def _load_runner(
    model,
    *,
    spec_config=None,
    drafter=None,
    enable_eplb=False,
    enable_elastic_ep=False,
    mode=None,
    cudagraph_mode=None,
    use_ubatching=False,
    lora_config=None,
    graph_runtime=None,
    use_aux=False,
):
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
            enable_elastic_ep=enable_elastic_ep,
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
    counter=None,
):
    calls = runner.calls
    moe_fn = moe_fn or (lambda m: None)
    wrapper_cls = wrapper_cls or _Recorder
    counter = counter or _Counter()
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
        (mr, "compilation_counter", counter),
    ]
    if platform is not None:
        pairs.append((mr, "current_platform", platform))
    with _stack(*pairs):
        yield calls


def test_load_drafter_eplb_rejects_elastic_ep():
    model = _Model()
    drafter_model = _Model()
    drafter = types.SimpleNamespace(model=drafter_model, load_model=lambda m: None)
    runner = _load_runner(
        model, drafter=drafter, enable_eplb=True, enable_elastic_ep=True
    )
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=lambda m: ("moe", m)):
        with pytest.raises(AssertionError, match="Elastic EP"):
            runner.load_model()


def test_load_drafter_eplb_requires_speculative_config():
    model = _Model()
    drafter_model = _Model()
    drafter = types.SimpleNamespace(model=drafter_model, load_model=lambda m: None)
    runner = _load_runner(model, drafter=drafter, enable_eplb=True, spec_config=None)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=lambda m: ("moe", m)):
        with pytest.raises(AssertionError):
            runner.load_model()


def test_load_drafter_eplb_requires_draft_model_config():
    model = _Model()
    drafter_model = _Model()
    drafter = types.SimpleNamespace(model=drafter_model, load_model=lambda m: None)
    spec_config = types.SimpleNamespace(draft_model_config=None)
    runner = _load_runner(
        model, drafter=drafter, enable_eplb=True, spec_config=spec_config
    )
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=lambda m: ("moe", m)):
        with pytest.raises(AssertionError):
            runner.load_model()


def test_load_drafter_eplb_requires_set_eplb_state():
    model = _Model()
    drafter_model = _Model()
    drafter = types.SimpleNamespace(model=drafter_model, load_model=lambda m: None)
    spec_config = types.SimpleNamespace(
        draft_model_config=types.SimpleNamespace(model="draft/m")
    )
    runner = _load_runner(
        model, drafter=drafter, enable_eplb=True, spec_config=spec_config
    )
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=lambda m: ("moe", m)):
        with pytest.raises(AssertionError):
            runner.load_model()


def test_load_asserts_cudagraph_mode_present():
    model = _Model()
    runner = _load_runner(model)
    runner.compilation_config.cudagraph_mode = None
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader):
        with pytest.raises(AssertionError):
            runner.load_model()


def test_load_eagle3_aux_requires_support():
    model = _Model()
    runner = _load_runner(model, use_aux=True)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, eagle3=False):
        with pytest.raises(RuntimeError, match="EAGLE3"):
            runner.load_model()


def test_load_eagle3_aux_uses_config_layers():
    model = _Model()
    runner = _load_runner(model, use_aux=True)
    runner._get_eagle3_aux_layers_from_config = lambda: (2, 5)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, eagle3=True):
        runner.load_model()

    assert model.aux_layers == (2, 5)


def test_load_npu_dispatch_warmup_without_musa():
    model = _Model()
    runner = _load_runner(model, mode=CompilationMode.VLLM_COMPILE)
    loader = _Loader(runner.calls.loader, model)
    dispatch_calls = []
    platform = types.SimpleNamespace(device_type="npu")
    with _fake_module(
        "vllm_fl.dispatch",
        warmup_cached_ops=lambda: dispatch_calls.append(True),
    ):
        with _load_env(runner, loader, platform=platform):
            runner.load_model()

    assert dispatch_calls == [True]
    assert runner.calls.offload == ["post_init"]


@pytest.mark.parametrize(
    "cudagraph_mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE]
)
def test_load_ubatch_wrapper_modes(cudagraph_mode):
    model = _Model()
    runner = _load_runner(
        model, cudagraph_mode=cudagraph_mode, use_ubatching=True
    )
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader):
        runner.load_model()

    assert isinstance(runner.model, _Recorder)
    assert runner.model.args[0] is model
    assert runner.model.args[2] == cudagraph_mode
    assert runner.calls.offload == ["post_init"]


def test_load_static_graph_wrapper_without_runtime_kwargs():
    model = _Model()
    runner = _load_runner(model, cudagraph_mode=CUDAGraphMode.FULL)
    loader = _Loader(runner.calls.loader, model)
    platform = types.SimpleNamespace(
        device_type="cpu", get_static_graph_wrapper_cls=lambda: "W"
    )
    with _load_env(runner, loader, platform=platform):
        runner.load_model()

    assert isinstance(runner.model, _Recorder)
    assert runner.model.args[0] is model
    assert runner.model.kwargs["runtime_mode"] == CUDAGraphMode.FULL
    assert "graph_runtime" not in runner.model.kwargs


def test_load_multimodal_pruning_short_circuits_and_video_flag():
    model = _Model()
    model.requires_sequential_video_encoding = True
    runner = _load_runner(model)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader), _Patch(
        mr, "supports_multimodal_pruning", lambda m: True
    ):
        runner.load_model()

    assert runner.is_multimodal_pruning_enabled is False
    assert runner.requires_sequential_video_encoding is True


def test_load_out_of_memory_reraises():
    model = _Model()
    runner = _load_runner(model)
    loader = _Loader(runner.calls.loader, model, raise_oom=True)
    with _load_env(runner, loader):
        with pytest.raises(torch.OutOfMemoryError):
            runner.load_model()


def test_load_lora_wraps_model():
    model = _Model()
    runner = _load_runner(model, lora_config=object())
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader):
        runner.load_model()

    assert runner.calls.lora == [model]
    assert runner.model is model
