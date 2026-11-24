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

"""G24: robust re-covers of the g15/g18 failing branches (814b).

Targets ``ModelRunnerFL._dummy_run`` (6111-6555), ``sample_tokens``
(4802-5060) and ``load_model`` (5527-5747).  Only the local module under
test is imported; every collaborator is patched at the
``vllm_fl.worker.model_runner`` namespace boundary and lightweight fake
modules are installed in ``sys.modules`` for the runtime imports performed
inside the method bodies (``vllm_fl.platform`` / ``vllm_fl.dispatch`` /
ascend attention).

Design rules that make these cases stable on the 910C runner:
  * explicit cudagraph mode is always aligned with what the fake
    ``_determine_batch_execution_and_padding`` reports, so the internal
    mode-mismatch assertion can never fire on a passing case;
  * no dependency on real ``vllm.config`` enums, real platform probing or
    real sampling metadata (all replaced by fakes with the exact attribute
    names the source touches);
  * assertions only inspect values that our own fakes produce.
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


_MISSING = object()


# ---------------------------------------------------------------------------
# patch helpers
# ---------------------------------------------------------------------------
class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name, _MISSING)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        if self.saved is _MISSING:
            try:
                delattr(self.module, self.name)
            except AttributeError:
                pass
        else:
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
    previous = sys.modules.get(name, _MISSING)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class _GraphMode:
    def __init__(self, name, *, full=False, valid=True):
        self.name = name
        self._full = full
        self._valid = valid

    def has_full_cudagraphs(self):
        return self._full

    def is_valid_runtime_mode(self):
        return self._valid

    def __repr__(self):
        return f"Mode({self.name})"


GRAPH_MODE = SimpleNamespace(
    NONE=_GraphMode("NONE"),
    PIECEWISE=_GraphMode("PIECEWISE"),
    FULL=_GraphMode("FULL", full=True),
)

COMPILATION_MODE = SimpleNamespace(
    NONE="cm-none",
    VLLM_COMPILE="cm-compile",
    STOCK_TORCH_COMPILE="cm-stock",
)


# ---------------------------------------------------------------------------
# _dummy_run scaffolding
# ---------------------------------------------------------------------------
class _Buf:
    def __init__(self, arr):
        self.np = arr
        self.copied = 0

    def copy_to_gpu(self):
        self.copied += 1


class _GpuBox:
    def __init__(self, tensor):
        self.gpu = tensor


class _Model:
    def __init__(self, rec, max_tokens, aux=False):
        self.rec = rec
        self.max_tokens = max_tokens
        self.aux = aux
        self.empty_calls = 0

    def __call__(self, *args, **kwargs):
        self.rec["model"].append(kwargs)
        hidden = torch.zeros(self.max_tokens, 4)
        if self.aux:
            return (hidden, "AUX")
        return hidden

    def make_empty_intermediate_tensors(self, **kwargs):
        self.empty_calls += 1
        return "EMPTY_IT"


class _Drafter:
    def __init__(self, rec=None):
        self.rec = rec if rec is not None else {}
        self.calls = []

    def dummy_run(self, num_tokens, use_cudagraphs=None, is_graph_capturing=None,
                  slot_mappings=None):
        self.calls.append((num_tokens, use_cudagraphs, is_graph_capturing))


class _SpecConf:
    def __init__(self, eagle=True, draft=False, extract=False,
                 enforce_eager=False, method="eagle"):
        self._eagle = eagle
        self._draft = draft
        self._extract = extract
        self.enforce_eager = enforce_eager
        self.method = method

    def use_eagle(self):
        return self._eagle

    def uses_draft_model(self):
        return self._draft

    def uses_extract_hidden_states(self):
        return self._extract


def _make_runner(**over):
    max_tokens = over.get("max_num_tokens", 64)
    rec = {
        "eplb": [], "commit": [], "dcp": [], "fwd": [], "attn": [],
        "padfia": [], "gather": [], "nvtx": [], "model": [], "determine": [],
    }
    model_config = SimpleNamespace(
        multimodal_config=over.get("multimodal_config", None),
        is_encoder_decoder=over.get("is_encoder_decoder", False),
        use_mla=over.get("use_mla", False),
        dtype=torch.float32,
    )
    parallel_config = SimpleNamespace(
        cp_kv_cache_interleave_size=1,
        num_ubatches=1,
        data_parallel_size=1,
    )
    runner = SimpleNamespace(
        max_num_tokens=max_tokens,
        uniform_decode_query_len=over.get("uniform_decode_query_len", 4),
        dcp_world_size=over.get("dcp_world_size", 1),
        device=torch.device("cpu"),
        parallel_config=parallel_config,
        model_config=model_config,
        vllm_config=SimpleNamespace(
            model_config=model_config, parallel_config=parallel_config
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=over.get("max_num_seqs", 8)),
        use_ascend_attention=over.get("use_ascend_attention", False),
        supports_mm_inputs=over.get("supports_mm_inputs", False),
        enable_prompt_embeds=over.get("enable_prompt_embeds", False),
        uses_mrope=over.get("uses_mrope", False),
        use_aux_hidden_state_outputs=over.get("use_aux_hidden_state_outputs", False),
        speculative_config=over.get("speculative_config", None),
        lora_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_specialize_lora=over.get("cudagraph_specialize_lora", False)
        ),
        intermediate_tensors=over.get("intermediate_tensors", None),
        query_pos=_Buf(np.zeros(max_tokens, dtype=np.int32)),
        query_start_loc=_Buf(np.zeros(max_tokens + 2, dtype=np.int32)),
        optimistic_seq_lens_cpu=torch.zeros(max_tokens, dtype=torch.int64),
        seq_lens=torch.zeros(max_tokens, dtype=torch.int64),
        positions=torch.zeros(max_tokens, dtype=torch.int64),
        mrope_positions=_GpuBox(torch.zeros(3, max_tokens, dtype=torch.int64)),
        input_ids=_GpuBox(torch.zeros(max_tokens, dtype=torch.int64)),
        inputs_embeds=_GpuBox(torch.zeros(max_tokens, 4)),
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(
                commit_block_table=lambda n: rec["commit"].append(n)
            )
        ),
    )

    runner._rec = rec
    runner._mode = over.get("mode", GRAPH_MODE.NONE)
    runner._batch_num_tokens = over.get("batch_num_tokens", None)
    runner._batch_num_reqs = over.get("batch_num_reqs", None)
    runner._should_ubatch = over.get("should_ubatch", False)
    runner._num_tokens_across_dp = over.get("num_tokens_across_dp", None)
    runner._ubatch_slices = over.get("ubatch_slices", (None, None))
    runner._dcp_len = over.get("dcp_len", 0)
    runner._slot_group = over.get("slot_group", None)
    runner._slot_map = over.get("slot_map", None)
    runner._is_first_rank = over.get("is_first_rank", True)

    runner.synchronize_input_prep = lambda: contextlib.nullcontext()
    runner.maybe_dummy_run_with_lora = lambda *a, **k: contextlib.nullcontext()
    runner.maybe_randomize_inputs = lambda *a, **k: contextlib.nullcontext()
    runner._init_model_kwargs = lambda: {}
    runner._prepare_mm_inputs = lambda n: (
        torch.zeros(n, dtype=torch.int64),
        torch.zeros(n, 4),
    )
    runner._dummy_mm_kwargs = lambda n: {"dummy_mm": n}
    runner._get_cumsum_and_arange = lambda nst, qp: np.arange(1, len(nst) + 1)
    runner._pad_query_start_loc_for_fia = lambda *a, **k: (
        rec["padfia"].append(a) or 5
    )
    runner._build_attention_metadata = lambda **kw: (
        rec["attn"].append(kw) or (SimpleNamespace(), None)
    )
    runner.sync_and_gather_intermediate_tensors = lambda n, a, b: (
        rec["gather"].append(n) or "GATHERED"
    )
    runner._register_layerwise_nvtx_hooks = lambda: rec["nvtx"].append(1)
    runner.eplb_step = lambda is_dummy=False, is_profile=False: rec["eplb"].append(
        (is_dummy, is_profile)
    )
    runner.model = _Model(rec, max_tokens, aux=runner.use_aux_hidden_state_outputs)
    runner.drafter = over.get("drafter", _Drafter(rec))

    def _get_slot_mappings(**kwargs):
        return (runner._slot_group, runner._slot_map)

    def _determine(num_tokens=None, num_reqs=None, **kwargs):
        rec["determine"].append(kwargs)
        padded = (
            runner._batch_num_tokens
            if runner._batch_num_tokens is not None
            else num_tokens
        )
        return (
            runner._mode,
            SimpleNamespace(num_tokens=padded, num_reqs=runner._batch_num_reqs),
            runner._should_ubatch,
            runner._num_tokens_across_dp,
            None,
        )

    runner._get_slot_mappings = _get_slot_mappings
    runner._determine_batch_execution_and_padding = _determine
    return runner


def _run(runner, **call_kwargs):
    with _stack(
        (mr, "CUDAGraphMode", GRAPH_MODE),
        (mr, "get_dcp_dummy_context_len", lambda *a, **k: runner._dcp_len),
        (mr, "maybe_create_ubatch_slices", lambda *a, **k: runner._ubatch_slices),
        (
            mr,
            "prepare_dcp_dummy_context_metadata",
            lambda *a, **k: runner._rec["dcp"].append(a),
        ),
        (
            mr,
            "set_forward_context",
            lambda *a, **k: (
                runner._rec["fwd"].append(k) or contextlib.nullcontext()
            ),
        ),
        (
            mr,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=runner._is_first_rank),
        ),
        (mr, "EagleProposer", _Drafter),
        (mr, "DFlashProposer", _Drafter),
        (mr, "DraftModelProposer", _Drafter),
        (mr, "ExtractHiddenStatesProposer", _Drafter),
        (mr, "Gemma4Proposer", _Drafter),
    ):
        return mr.ModelRunnerFL._dummy_run(runner, **call_kwargs)


@contextlib.contextmanager
def _ascend_attention_module(record):
    name = "vllm_fl.dispatch.backends.vendor.ascend.impl.attention"
    module = types.ModuleType(name)
    module.get_ascend_dummy_attention_state = lambda *a, **k: (
        record.append((a, k)) or "ASCEND_STATE"
    )
    previous = sys.modules.get(name, _MISSING)
    sys.modules[name] = module
    try:
        yield record
    finally:
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


# --- _dummy_run branch coverage -------------------------------------------
def test_g24_dummy_spec_capture_aligned_piecewise_mode():
    drafter = _Drafter()
    runner = _make_runner(
        mode=GRAPH_MODE.PIECEWISE,
        speculative_config=_SpecConf(eagle=True, enforce_eager=False),
        drafter=drafter,
    )
    _run(runner, num_tokens=8,
         cudagraph_runtime_mode=GRAPH_MODE.PIECEWISE,
         is_graph_capturing=True)
    assert len(drafter.calls) == 1
    assert drafter.calls[0][1] is True


def test_g24_dummy_spec_specialize_lora_disables_graphs():
    drafter = _Drafter()
    runner = _make_runner(
        mode=GRAPH_MODE.PIECEWISE,
        speculative_config=_SpecConf(method="eagle", enforce_eager=False),
        drafter=drafter,
        cudagraph_specialize_lora=True,
    )
    _run(runner, num_tokens=8,
         cudagraph_runtime_mode=GRAPH_MODE.PIECEWISE,
         is_graph_capturing=False, num_active_loras=2)
    assert drafter.calls and drafter.calls[0][1] is False


def test_g24_dummy_spec_enforce_eager_disables_graphs():
    drafter = _Drafter()
    runner = _make_runner(
        mode=GRAPH_MODE.PIECEWISE,
        speculative_config=_SpecConf(eagle=True, enforce_eager=True),
        drafter=drafter,
    )
    _run(runner, num_tokens=8,
         cudagraph_runtime_mode=GRAPH_MODE.PIECEWISE,
         is_graph_capturing=True)
    assert drafter.calls and drafter.calls[0][1] is False


def test_g24_dummy_ascend_full_forwards_state_and_pads_fia():
    state = []
    runner = _make_runner(
        use_ascend_attention=True,
        mode=GRAPH_MODE.FULL,
        speculative_config=_SpecConf(eagle=False, method="eagle"),
    )
    with _ascend_attention_module(state):
        _run(runner, num_tokens=8, cudagraph_runtime_mode=GRAPH_MODE.FULL)
    assert len(state) == 1
    assert len(runner._rec["padfia"]) == 1
    attn = runner._rec["attn"][0]
    assert attn["ascend_attn_state"] == "ASCEND_STATE"
    assert attn["for_cudagraph_capture"] is True


def test_g24_dummy_mixed_batch_with_dcp_seq_lens():
    runner = _make_runner(max_num_seqs=8, dcp_len=2)
    _run(runner, num_tokens=8, force_attention=True, create_mixed_batch=True)
    assert len(runner._rec["attn"]) == 1
    # num_decode_tokens=4, num_prefill_tokens=4, dcp=2 -> [3]*4 + [6]
    assert int(runner.optimistic_seq_lens_cpu[0]) == 3
    assert int(runner.optimistic_seq_lens_cpu[4]) == 6


def test_g24_dummy_mixed_batch_without_dcp_seq_lens():
    runner = _make_runner(max_num_seqs=8, dcp_len=0)
    _run(runner, num_tokens=8, force_attention=True, create_mixed_batch=True)
    assert int(runner.optimistic_seq_lens_cpu[0]) == 1
    assert int(runner.optimistic_seq_lens_cpu[4]) == 5


def test_g24_dummy_ubatch_adjusts_padded_and_across_dp():
    across_dp = np.zeros(1, dtype=np.int32)
    runner = _make_runner(
        should_ubatch=True,
        ubatch_slices=(None, [SimpleNamespace(num_tokens=6)]),
        num_tokens_across_dp=across_dp,
    )
    hidden, _ = _run(runner, num_tokens=8)
    assert int(across_dp[0]) == 6
    assert hidden.shape == (64, 4)


def test_g24_dummy_non_first_rank_creates_intermediate():
    runner = _make_runner(is_first_rank=False)
    _run(runner, num_tokens=8)
    assert runner.model.empty_calls == 1
    assert runner._rec["gather"] == [8]


def test_g24_dummy_non_first_rank_reuses_intermediate():
    runner = _make_runner(is_first_rank=False, intermediate_tensors="PRE")
    _run(runner, num_tokens=8)
    assert runner.model.empty_calls == 0
    assert runner._rec["gather"] == [8]


def test_g24_dummy_prompt_embeds_branch():
    runner = _make_runner(enable_prompt_embeds=True)
    _run(runner, num_tokens=8)
    call = runner._rec["model"][0]
    assert call["input_ids"] is None
    assert call["inputs_embeds"] is not None


def test_g24_dummy_mrope_positions():
    runner = _make_runner(uses_mrope=True)
    _run(runner, num_tokens=8)
    assert runner._rec["model"][0]["positions"].dim() == 2


def test_g24_dummy_mm_inputs_branch():
    runner = _make_runner(supports_mm_inputs=True)
    _run(runner, num_tokens=8)
    call = runner._rec["model"][0]
    assert call.get("dummy_mm") == 8
    assert call["inputs_embeds"] is not None


def test_g24_dummy_explicit_batch_num_reqs_commits_padded():
    runner = _make_runner(batch_num_reqs=3)
    _run(runner, num_tokens=8, force_attention=True)
    assert runner._rec["commit"] == [3]
    assert runner._rec["attn"][0]["num_reqs"] == 3


def test_g24_dummy_force_attention_clears_slot_mappings():
    slot_a = torch.zeros(8, dtype=torch.int64)
    runner = _make_runner(slot_group={"g": slot_a}, slot_map="S")
    _run(runner, num_tokens=8, force_attention=True)
    assert bool(torch.all(slot_a == -1))
    assert len(runner._rec["dcp"]) >= 1


def test_g24_dummy_baseline_eplb_and_nvtx():
    runner = _make_runner()
    hidden, logits = _run(runner, num_tokens=8)
    assert hidden.shape == (64, 4)
    assert logits.shape == (8, 4)
    assert runner._rec["eplb"] == [(True, False)]
    assert runner._rec["nvtx"] == [1]
    assert runner._rec["commit"] == []


def test_g24_dummy_mm_encoder_only_returns_empty():
    runner = _make_runner(
        multimodal_config=SimpleNamespace(mm_encoder_only=True)
    )
    hidden, logits = _run(runner, num_tokens=8)
    assert torch.is_tensor(hidden) and hidden.numel() == 0
    assert torch.is_tensor(logits) and logits.numel() == 0


def test_g24_dummy_invalid_runtime_mode_asserts():
    runner = _make_runner()
    bad = _GraphMode("BAD", valid=False)
    with pytest.raises(AssertionError):
        _run(runner, num_tokens=8, cudagraph_runtime_mode=bad)


def test_g24_dummy_runtime_mode_mismatch_asserts():
    runner = _make_runner(mode=GRAPH_MODE.NONE)
    with pytest.raises(AssertionError):
        _run(runner, num_tokens=8, cudagraph_runtime_mode=GRAPH_MODE.FULL)


# ---------------------------------------------------------------------------
# _dummy_sampler_run scaffolding (robust: fake vllm_fl.platform + fake modes)
# ---------------------------------------------------------------------------
class _Sampler:
    def __init__(self, mode):
        self.logprobs_mode = mode
        self.calls = 0

    def __call__(self, logits, sampling_metadata):
        self.calls += 1
        return "SAMPLER_OUT"


class _BoomSampler:
    def __init__(self, mode):
        self.logprobs_mode = mode

    def __call__(self, logits, sampling_metadata):
        raise RuntimeError("CUDA out of memory")


def _sampler_runner(mode):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)
        ),
        model=SimpleNamespace(
            compute_logits=lambda h: torch.zeros(h.shape[0], 5)
        ),
        device=torch.device("cpu"),
        sampler=_Sampler(mode),
        speculative_config=None,
    )


@contextlib.contextmanager
def _sampler_env(processed=(), extra=()):
    pairs = [
        (mr, "SamplingMetadata", lambda **kw: SimpleNamespace(**kw)),
        (mr, "LogitsProcessors", lambda: "lp"),
        (mr, "replace", lambda obj, **kw: obj),
        (mr, "PROCESSED_LOGPROBS_MODES", set(processed)),
    ]
    pairs.extend(extra)
    with _fake_module(
        "vllm_fl.platform",
        PlatformFL=SimpleNamespace(vendor_name="ascend", device_type="npu"),
    ):
        with _stack(*pairs):
            yield


def test_g24_sampler_processed_mode_single_call():
    runner = _sampler_runner("mode-processed")
    with _sampler_env(processed={"mode-processed"}):
        out = mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert out == "SAMPLER_OUT"
    assert runner.sampler.calls == 1


def test_g24_sampler_non_processed_warms_forward_native():
    runner = _sampler_runner("mode-other")
    with _sampler_env(processed={"mode-processed"}):
        out = mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert out == "SAMPLER_OUT"
    assert runner.sampler.calls == 2


def test_g24_sampler_oom_reraises_friendly():
    runner = _sampler_runner("mode-processed")
    runner.sampler = _BoomSampler("mode-processed")
    with _sampler_env(processed={"mode-processed"}):
        with pytest.raises(RuntimeError) as exc:
            mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert "max_num_seqs" in str(exc.value)


def test_g24_sampler_other_runtime_error_reraises():
    runner = _sampler_runner("mode-processed")

    def _boom(**kw):
        raise RuntimeError("boom")

    _boom.logprobs_mode = "mode-processed"
    runner.sampler = _boom
    with _sampler_env(processed={"mode-processed"}):
        with pytest.raises(RuntimeError, match="boom"):
            mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))


def test_g24_sampler_spec_rejection_runs_mixed_and_greedy():
    runner = _sampler_runner("mode-processed")
    runner.speculative_config = SimpleNamespace(
        rejection_sample_method="standard", draft_sample_method="probabilistic"
    )
    calls = []

    class _Rejection:
        def __call__(self, *a, **k):
            calls.append(a)

    runner.rejection_sampler = _Rejection()
    extra = [
        (
            mr,
            "SpecDecodeMetadata",
            SimpleNamespace(make_dummy=lambda ids, dev: "META"),
        ),
        (mr.torch, "accelerator", SimpleNamespace(synchronize=lambda: None)),
    ]
    with _sampler_env(processed={"mode-processed"}, extra=extra):
        out = mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert out == "SAMPLER_OUT"
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# sample_tokens scaffolding (focused re-covers of fragile drafter branches)
# ---------------------------------------------------------------------------
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


class _FakeEagle:
    def __init__(self):
        self.prepare_calls = []
        self.dummy_calls = []

    def prepare_next_token_ids_padded(self, *args):
        self.prepare_calls.append(args)
        return torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.int64)

    def dummy_run(self, num_tokens=1):
        self.dummy_calls.append(num_tokens)


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
    return SimpleNamespace(**base)


@contextlib.contextmanager
def _sample_env(pp=None):
    calls = SimpleNamespace(broadcast=[])
    pp = pp or SimpleNamespace(is_last_rank=True, world_size=1, rank=0)
    with _stack(
        (mr, "ModelRunnerOutput", _FakeMRO),
        (mr, "AsyncGPUModelRunnerOutput", _FakeAsync),
        (mr, "apply_grammar_bitmask", lambda *a: None),
        (mr, "record_function_or_nullcontext",
         lambda *a, **k: contextlib.nullcontext()),
        (mr, "get_pp_group", lambda: pp),
        (mr, "advance_io_step", lambda: None),
        (mr, "EagleProposer", _FakeEagle),
        (mr, "DFlashProposer", _FakeEagle),
        (mr, "DraftModelProposer", _FakeEagle),
        (mr, "ExtractHiddenStatesProposer", _FakeEagle),
        (mr, "Gemma4Proposer", _FakeEagle),
        (mr, "NgramProposerGPU", _FakeEagle),
    ):
        yield calls


def _sample_runner(*, spec=None, fits=True, async_sched=False, event=None,
                   drafter=None, dp=1):
    ib = _InputBatch()
    sampler = SimpleNamespace(
        sampled_token_ids=torch.tensor([[3], [7]], dtype=torch.int64),
        logprobs_tensors=None,
    )
    scheduler = SimpleNamespace(total_num_scheduled_tokens=2)
    state = (
        scheduler,
        torch.zeros(2, 8),
        object(),
        object(),
        torch.zeros(2, 4),
        torch.zeros(2, 4),
        None,
        None,
        "cgstats",
        {"slot": torch.zeros(2)},
    )
    calls = SimpleNamespace(
        sample=[], update_states=[], broadcast=[], propose=[], copy_draft=[],
        bookkeep=[], finalize=[], eplb=[], copy_valid=[],
    )
    runner = SimpleNamespace(
        execute_model_state=state,
        kv_connector_output="kv-in",
        use_async_scheduling=async_sched,
        broadcast_pp_output=False,
        input_batch=ib,
        speculative_config=spec,
        num_spec_tokens=2,
        device=torch.device("cpu"),
        requests={"a": object()},
        discard_request_mask=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        token_ids_gpu_tensor=torch.zeros(2, dtype=torch.int32),
        num_tokens_no_spec_gpu=torch.zeros(2, dtype=torch.int32),
        valid_sampled_token_count_event=event,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        supports_mm_inputs=False,
        check_ep_fault=False,
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
        lambda: calls.broadcast.append("receive")
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
        0, None, None, [4, 5], None, {"a": 0}, {"a": 0}, [1],
    )
    runner.finalize_kv_connector = lambda: calls.finalize.append(True)
    runner.eplb_step = lambda: calls.eplb.append(True)
    runner._get_or_create_async_output_copy_stream = lambda: "copy-stream"
    runner.sample_tokens = mr.ModelRunnerFL.sample_tokens.__get__(runner, mr.ModelRunnerFL)
    return runner, calls


def test_g24_sample_zero_draft_fallback_without_event():
    drafter = _FakeEagle()
    runner, calls = _sample_runner(
        spec=_spec(use_eagle=lambda: True), fits=False, drafter=drafter
    )
    with _sample_env():
        runner.sample_tokens(None)
    assert drafter.prepare_calls == []
    assert drafter.dummy_calls == []
    assert runner._draft_token_ids.shape == (1, 2)
    assert runner._draft_probs is None
    assert calls.copy_draft[-1] is True


def test_g24_sample_draft_after_bookkeeping_dp_dummy():
    drafter = _FakeEagle()
    runner, calls = _sample_runner(
        spec=_spec(use_eagle=lambda: True, disable_padded_drafter_batch=True),
        fits=False,
        drafter=drafter,
        dp=2,
    )
    with _sample_env():
        runner.sample_tokens(None)
    assert calls.propose == []
    assert drafter.dummy_calls == [1]
    assert runner._draft_token_ids.shape == (1, 2)


def test_g24_sample_async_broadcast_respects_pp_config():
    runner, calls = _sample_runner(async_sched=True)
    pp = SimpleNamespace(is_last_rank=True, world_size=2, rank=1)
    with _sample_env(pp=pp):
        runner.sample_tokens(None)
    assert len(calls.broadcast) == 1

    runner2, calls2 = _sample_runner(async_sched=True)
    runner2.broadcast_pp_output = True
    with _sample_env(pp=pp):
        runner2.sample_tokens(None)
    assert calls2.broadcast == []


# ---------------------------------------------------------------------------
# load_model scaffolding (fake enums + fake platform, no real vllm.config)
# ---------------------------------------------------------------------------
class _Loader:
    def __init__(self, calls, model):
        self.calls = calls
        self.model = model

    def load_model(self, *, vllm_config, model_config):
        self.calls.append(("load_model", vllm_config, model_config))
        return self.model


class _Profiler:
    def __init__(self, consumed_memory=4096):
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


def _load_runner(model, *, spec_config=None, drafter=_MISSING, enable_eplb=False,
                 mode="cm-none", cudagraph_mode=None, use_ubatching=False,
                 lora_config=None):
    compilation = SimpleNamespace(
        mode=mode,
        cudagraph_mode=(
            cudagraph_mode if cudagraph_mode is not None else GRAPH_MODE.NONE
        ),
        init_backend=lambda cfg: "backend",
    )
    model_config = SimpleNamespace(
        model="t/m", enforce_eager=True, quantization=None,
        multimodal_config=None, revision="r",
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation, speculative_config=spec_config
    )
    calls = SimpleNamespace(
        loader=[], comm=[], hooks=[], warmup=[], io=[], offload=[], lora=[]
    )
    runner = SimpleNamespace(
        model_config=model_config,
        load_config=SimpleNamespace(load_format="safetensors"),
        vllm_config=vllm_config,
        parallel_config=SimpleNamespace(
            enable_eplb=enable_eplb,
            enable_elastic_ep=False,
            use_ubatching=use_ubatching,
        ),
        lora_config=lora_config,
        device=torch.device("cpu"),
        model=model,
        _moe_model=None,
        compilation_config=compilation,
        graph_runtime=None,
        use_aux_hidden_state_outputs=False,
        speculative_config=spec_config,
        jit_warmup_registry=SimpleNamespace(
            activate=lambda: contextlib.nullcontext()
        ),
        eplb_state=None,
        calls=calls,
    )
    runner._setup_eagle3_aux_hidden_state_outputs = (
        mr.ModelRunnerFL._setup_eagle3_aux_hidden_state_outputs.__get__(
            runner, mr.ModelRunnerFL
        )
    )
    runner.get_model = lambda: runner.model
    runner.load_lora_model = lambda m, cfg, dev: calls.lora.append(m) or m
    runner.load_model = mr.ModelRunnerFL.load_model.__get__(runner, mr.ModelRunnerFL)
    if drafter is not _MISSING:
        runner.drafter = drafter
    return runner


@contextlib.contextmanager
def _load_env(runner, loader, *, moe_fn=None, platform=None):
    calls = runner.calls
    moe_fn = moe_fn or (lambda m: None)
    if platform is None:
        platform = SimpleNamespace(
            device_type="cpu",
            get_static_graph_wrapper_cls=lambda: "W",
        )
    pairs = [
        (mr, "init_io_dump_from_env", lambda *a, **k: calls.io.append(a)),
        (mr, "get_model_loader", lambda cfg: loader),
        (mr, "DeviceMemoryProfiler", _Profiler()),
        (mr, "warmup_cached_ops", lambda *a, **k: calls.warmup.append(True)),
        (mr, "_prepare_comm_buffer", lambda m: calls.comm.append(m)),
        (mr, "register_io_module_hooks", lambda m: calls.hooks.append(m)),
        (mr, "supports_multimodal_pruning", lambda m: False),
        (mr, "get_offloader", lambda: _Offloader(calls.offload)),
        (mr, "get_mixture_of_experts_model", moe_fn),
        (mr, "EplbState", _EplbState),
        (mr, "format_gib", lambda x: "0.0"),
        (mr, "is_breakable_cudagraph_enabled", lambda: False),
        (mr, "BreakableCUDAGraphWrapper", _Recorder),
        (mr, "UBatchWrapper", _Recorder),
        (mr, "resolve_obj_by_qualname", lambda q: _Recorder),
        (mr, "compilation_counter", _Counter()),
        (mr, "current_platform", platform),
        (mr, "CUDAGraphMode", GRAPH_MODE),
        (mr, "CompilationMode", COMPILATION_MODE),
    ]
    with _stack(*pairs):
        yield calls


class _CompileModel:
    def __init__(self, calls):
        self.calls = calls

    def compile(self, **kwargs):
        self.calls.append(kwargs)
        return self


def test_g24_load_stock_torch_compile_early_return():
    compile_calls = []
    model = _CompileModel(compile_calls)
    runner = _load_runner(model, mode=COMPILATION_MODE.STOCK_TORCH_COMPILE)
    loader = _Loader(runner.calls.loader, model)
    env_calls = []
    with _fake_module(
        "vllm.env_override",
        _apply_constrain_to_fx_strides_patch=lambda: env_calls.append(True),
    ):
        with _load_env(runner, loader):
            runner.load_model()
    assert env_calls == [True]
    assert compile_calls == [{"fullgraph": True, "backend": "backend"}]
    assert runner.calls.offload == []


def test_g24_load_npu_dispatch_warmup():
    model = SimpleNamespace()
    runner = _load_runner(model, mode=COMPILATION_MODE.VLLM_COMPILE)
    loader = _Loader(runner.calls.loader, model)
    dispatch_calls = []
    platform = SimpleNamespace(
        device_type="npu", get_static_graph_wrapper_cls=lambda: "W"
    )
    with _fake_module(
        "vllm_fl.dispatch",
        warmup_cached_ops=lambda *a, **k: dispatch_calls.append(True),
    ):
        with _load_env(runner, loader, platform=platform):
            runner.load_model()
    assert dispatch_calls == [True]
    assert len(runner.calls.warmup) == 1
    assert runner.calls.offload == ["post_init"]


def test_g24_load_ubatch_wrapper_modes():
    def run_case(cudagraph_mode):
        model = SimpleNamespace()
        runner = _load_runner(
            model,
            mode=COMPILATION_MODE.VLLM_COMPILE,
            cudagraph_mode=cudagraph_mode,
            use_ubatching=True,
        )
        loader = _Loader(runner.calls.loader, model)
        with _load_env(runner, loader):
            runner.load_model()
        return runner, model

    runner, model = run_case(GRAPH_MODE.FULL)
    assert isinstance(runner.model, _Recorder)
    assert runner.model.args[0] is model
    assert runner.model.args[2] is GRAPH_MODE.FULL

    runner, model = run_case(GRAPH_MODE.NONE)
    assert isinstance(runner.model, _Recorder)
    assert runner.model.args[2] is GRAPH_MODE.NONE


def test_g24_load_drafter_eplb_registers_moe():
    model = SimpleNamespace()
    drafter_model = SimpleNamespace()
    drafter = SimpleNamespace(
        model=drafter_model,
        load_model=lambda m: None,
        set_eplb_state=lambda state: setattr(drafter, "state", state),
    )
    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(model="draft/m")
    )

    def moe_fn(m):
        return ("moe", m) if m is drafter_model else None

    runner = _load_runner(
        model, spec_config=spec_config, drafter=drafter, enable_eplb=True
    )
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=moe_fn):
        runner.load_model()
    assert runner.eplb_state is not None
    assert runner.eplb_state.added == [
        (("moe", drafter_model), spec_config.draft_model_config)
    ]
    assert drafter.state is runner.eplb_state
    assert runner._moe_model is None
    assert runner.eplb_state.started is False
