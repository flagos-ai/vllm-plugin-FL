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

"""G15 coverage: ModelRunnerFL._dummy_run (814b 6111-6555) + _dummy_sampler_run.

The runner is a SimpleNamespace; every collaborator that _dummy_run touches
(batch dispatch, slot mappings, attention metadata, forward context, PP group,
mm helpers, drafter) is patched at the ``model_runner`` boundary.  CPU only,
no NPU / no network.
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


# ---------------------------------------------------------------------------
# Fake CUDAGraphMode (avoids depending on the real enum members / helpers)
# ---------------------------------------------------------------------------
class _Mode:
    def __init__(self, name, valid=True):
        self.name = name
        self._valid = valid

    def is_valid_runtime_mode(self):
        return self._valid

    def __repr__(self):
        return f"Mode({self.name})"


_MODE_NONE = _Mode("NONE")
_MODE_PIECEWISE = _Mode("PIECEWISE")
_MODE_FULL = _Mode("FULL")
CUDAGraphMode = SimpleNamespace(
    NONE=_MODE_NONE, PIECEWISE=_MODE_PIECEWISE, FULL=_MODE_FULL
)


# ---------------------------------------------------------------------------
# Runner scaffolding
# ---------------------------------------------------------------------------
class _Buf:
    """Mimic the .np buffer wrapper used for query_pos / query_start_loc."""

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
    runner._mode = over.get("mode", _MODE_NONE)
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
        (mr, "CUDAGraphMode", CUDAGraphMode),
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
def _fake_ascend_module(record):
    name = "vllm_fl.dispatch.backends.vendor.ascend.impl.attention"
    mod = types.ModuleType(name)
    mod.get_ascend_dummy_attention_state = lambda *a, **k: (
        record.append((a, k)) or "ASCEND_STATE"
    )
    saved = sys.modules.get(name, _MISSING)
    sys.modules[name] = mod
    try:
        yield record
    finally:
        if saved is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


# ---------------------------------------------------------------------------
# _dummy_run
# ---------------------------------------------------------------------------
def test_dummy_run_mm_encoder_only_returns_empty():
    runner = _make_runner(
        multimodal_config=SimpleNamespace(mm_encoder_only=True)
    )
    hidden, logits = _run(runner, num_tokens=8)
    assert torch.is_tensor(hidden) and hidden.numel() == 0
    assert torch.is_tensor(logits) and logits.numel() == 0
    assert runner._rec["determine"] == []


def test_dummy_run_baseline_eager_runs_eplb_and_nvtx():
    runner = _make_runner()
    hidden, logits = _run(runner, num_tokens=8)
    assert hidden.shape == (64, 4)
    assert logits.shape == (8, 4)
    assert runner._rec["eplb"] == [(True, False)]
    assert runner._rec["nvtx"] == [1]
    assert runner._rec["commit"] == []  # build_attention is False
    assert runner._rec["attn"] == []
    assert runner._rec["dcp"] == []


def test_dummy_run_skip_eplb():
    runner = _make_runner()
    _run(runner, num_tokens=8, skip_eplb=True, is_profile=True)
    assert runner._rec["eplb"] == []


def test_dummy_run_force_attention_fills_slot_mappings():
    slot_a = torch.zeros(8, dtype=torch.int64)
    runner = _make_runner(slot_group={"g": slot_a}, slot_map="S")
    _run(runner, num_tokens=8, force_attention=True)
    assert torch.all(slot_a == -1)
    assert len(runner._rec["dcp"]) == 2
    assert runner._rec["commit"] == [8]
    attn = runner._rec["attn"][0]
    assert attn["num_tokens_padded"] is None  # pad_attn False (NONE mode)
    assert attn["use_spec_decode"] is False


def test_dummy_run_full_runtime_pads_and_dcp_context():
    runner = _make_runner(
        mode=_MODE_FULL,
        dcp_len=3,
        max_num_seqs=4,
        batch_num_reqs=4,
    )
    _run(runner, num_tokens=8, cudagraph_runtime_mode=_MODE_FULL)
    attn = runner._rec["attn"][0]
    assert attn["num_tokens_padded"] is not None
    assert attn["num_reqs"] == 4
    assert attn["for_cudagraph_capture"] is True


def test_dummy_run_profile_seq_lens_override():
    runner = _make_runner()
    _run(runner, num_tokens=8, force_attention=True, profile_seq_lens=11)
    attn = runner._rec["attn"][0]
    assert len(runner._rec["attn"]) == 1
    assert attn["max_query_len"] == 8


def test_dummy_run_create_mixed_batch_without_dcp():
    runner = _make_runner(max_num_seqs=8, dcp_len=0)
    _run(runner, num_tokens=8, force_attention=True, create_mixed_batch=True)
    assert len(runner._rec["attn"]) == 1
    assert len(runner._rec["dcp"]) == 2


def test_dummy_run_create_mixed_batch_with_dcp():
    runner = _make_runner(max_num_seqs=8, dcp_len=2)
    _run(runner, num_tokens=8, force_attention=True, create_mixed_batch=True)
    assert len(runner._rec["attn"]) == 1


def test_dummy_run_uniform_decode_no_remainder():
    runner = _make_runner(uniform_decode_query_len=4, max_num_seqs=8)
    _run(runner, num_tokens=8, uniform_decode=True, force_attention=True)
    attn = runner._rec["attn"][0]
    assert attn["max_query_len"] == 4


def test_dummy_run_uniform_decode_with_remainder():
    runner = _make_runner(uniform_decode_query_len=4, max_num_seqs=8)
    hidden, logits = _run(runner, num_tokens=10, uniform_decode=True)
    assert hidden.shape == (64, 4)
    assert logits.shape == (3, 4)


def test_dummy_run_invalid_runtime_mode_asserts():
    runner = _make_runner()
    bad = _Mode("BAD", valid=False)
    with pytest.raises(AssertionError):
        _run(runner, num_tokens=8, cudagraph_runtime_mode=bad)


def test_dummy_run_runtime_mode_mismatch_asserts():
    runner = _make_runner(mode=_MODE_NONE)
    with pytest.raises(AssertionError):
        _run(runner, num_tokens=8, cudagraph_runtime_mode=_MODE_FULL)


def test_dummy_run_mm_inputs_branch():
    runner = _make_runner(supports_mm_inputs=True)
    _run(runner, num_tokens=8)
    call = runner._rec["model"][0]
    assert call.get("dummy_mm") == 8
    assert call["inputs_embeds"] is not None


def test_dummy_run_prompt_embeds_branch():
    runner = _make_runner(enable_prompt_embeds=True)
    _run(runner, num_tokens=8)
    call = runner._rec["model"][0]
    assert call["input_ids"] is None
    assert call["inputs_embeds"] is not None


def test_dummy_run_uses_mrope_positions():
    runner = _make_runner(uses_mrope=True)
    _run(runner, num_tokens=8)
    call = runner._rec["model"][0]
    assert call["positions"].dim() == 2


def test_dummy_run_non_first_rank_makes_intermediate_tensors():
    runner = _make_runner(is_first_rank=False)
    _run(runner, num_tokens=8)
    assert runner.model.empty_calls == 1
    assert runner._rec["gather"] == [8]


def test_dummy_run_non_first_rank_reuses_intermediate_tensors():
    runner = _make_runner(is_first_rank=False, intermediate_tensors="PRE")
    _run(runner, num_tokens=8)
    assert runner.model.empty_calls == 0
    assert runner._rec["gather"] == [8]


def test_dummy_run_aux_hidden_state_outputs_unpacked():
    runner = _make_runner(use_aux_hidden_state_outputs=True)
    hidden, logits = _run(runner, num_tokens=8)
    assert hidden.shape == (64, 4)
    assert logits.shape == (8, 4)


def test_dummy_run_spec_decode_captures_drafter_graphs():
    drafter = _Drafter()
    runner = _make_runner(
        speculative_config=_SpecConf(eagle=True, enforce_eager=False),
        drafter=drafter,
        # _dummy_run asserts the requested runtime mode equals the mode the
        # dispatcher reports, so the fake dispatcher must return PIECEWISE.
        mode=_MODE_PIECEWISE,
    )
    _run(
        runner,
        num_tokens=8,
        cudagraph_runtime_mode=_MODE_PIECEWISE,
        is_graph_capturing=True,
    )
    assert len(drafter.calls) == 1
    assert drafter.calls[0][1] is True  # use_cudagraphs


def test_dummy_run_spec_decode_specialize_lora_disables_graphs():
    drafter = _Drafter()
    runner = _make_runner(
        speculative_config=_SpecConf(method="eagle", enforce_eager=False),
        drafter=drafter,
        cudagraph_specialize_lora=True,
        mode=_MODE_PIECEWISE,
    )
    _run(
        runner,
        num_tokens=8,
        cudagraph_runtime_mode=_MODE_PIECEWISE,
        is_graph_capturing=False,
        num_active_loras=1,
    )
    assert drafter.calls[0][1] is False


def test_dummy_run_ascend_attention_full_pads_fia():
    state = []
    runner = _make_runner(
        use_ascend_attention=True,
        mode=_MODE_FULL,
        speculative_config=_SpecConf(
            eagle=False, draft=False, extract=False, method="eagle"
        ),
    )
    with _fake_ascend_module(state):
        _run(runner, num_tokens=8, cudagraph_runtime_mode=_MODE_FULL)
    assert len(state) == 1
    assert len(runner._rec["padfia"]) == 1
    attn = runner._rec["attn"][0]
    assert attn["ascend_attn_state"] == "ASCEND_STATE"


def test_dummy_run_ubatch_slices_adjust_padded_tokens():
    across_dp = np.zeros(1, dtype=np.int32)
    runner = _make_runner(
        should_ubatch=True,
        ubatch_slices=(None, [SimpleNamespace(num_tokens=8)]),
        num_tokens_across_dp=across_dp,
    )
    hidden, logits = _run(runner, num_tokens=8)
    assert across_dp[0] == 8
    assert hidden.shape == (64, 4)


def test_dummy_run_explicit_batch_num_reqs():
    runner = _make_runner(batch_num_reqs=3)
    _run(runner, num_tokens=8, force_attention=True)
    attn = runner._rec["attn"][0]
    assert attn["num_reqs"] == 3
    assert runner._rec["commit"] == [3]


# ---------------------------------------------------------------------------
# _dummy_sampler_run
# ---------------------------------------------------------------------------
class _Sampler:
    def __init__(self, mode):
        self.logprobs_mode = mode
        self.calls = 0

    def __call__(self, logits, sampling_metadata):
        self.calls += 1
        return "SAMPLER_OUT"


class _BoomSampler:
    logprobs_mode = None

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


# _dummy_sampler_run imports ``vllm_fl.platform`` lazily to read
# ``PlatformFL.vendor_name`` (kunlunxin workaround).  Pulling the real module
# in a CPU-only test run drags in device probing / native extensions, so stub
# it with the 910c vendor.  PROCESSED_LOGPROBS_MODES is pinned to a token we
# control so both processed / non-processed branches are exercised on purpose.
_PLATFORM_MODULE = "vllm_fl.platform"
_PROCESSED_MODE = "__g15_processed__"


@contextlib.contextmanager
def _fake_platform_module(vendor="ascend"):
    mod = types.ModuleType(_PLATFORM_MODULE)
    mod.PlatformFL = SimpleNamespace(
        vendor_name=vendor, device_type="npu", device_name="npu"
    )
    saved = sys.modules.get(_PLATFORM_MODULE, _MISSING)
    sys.modules[_PLATFORM_MODULE] = mod
    try:
        yield mod
    finally:
        if saved is _MISSING:
            sys.modules.pop(_PLATFORM_MODULE, None)
        else:
            sys.modules[_PLATFORM_MODULE] = saved


@contextlib.contextmanager
def _sampler_patches(extra=(), processed=(_PROCESSED_MODE,)):
    pairs = [
        (mr, "SamplingMetadata", lambda **kw: SimpleNamespace(**kw)),
        (mr, "LogitsProcessors", lambda: "lp"),
        (mr, "replace", lambda obj, **kw: obj),
        (mr, "PROCESSED_LOGPROBS_MODES", frozenset(processed)),
    ]
    pairs.extend(extra)
    with _fake_platform_module(), _stack(*pairs):
        yield


def test_dummy_sampler_run_mm_encoder_only_returns_empty():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        )
    )
    out = mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert torch.is_tensor(out) and out.numel() == 0


def test_dummy_sampler_run_processed_mode_single_call():
    mode = _PROCESSED_MODE
    runner = _sampler_runner(mode)
    with _sampler_patches():
        out = mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert out == "SAMPLER_OUT"
    assert runner.sampler.calls == 1


def test_dummy_sampler_run_non_processed_mode_warms_forward_native():
    runner = _sampler_runner("__g15_not_processed__")
    with _sampler_patches():
        out = mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert out == "SAMPLER_OUT"
    assert runner.sampler.calls == 2  # greedy + forward_native warmup


def test_dummy_sampler_run_out_of_memory_reraises_friendly():
    mode = _PROCESSED_MODE
    runner = _sampler_runner(mode)
    runner.sampler = _BoomSampler(mode)
    with _sampler_patches():
        with pytest.raises(RuntimeError) as exc:
            mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert "max_num_seqs" in str(exc.value)


def test_dummy_sampler_run_other_runtime_error_reraises():
    mode = _PROCESSED_MODE
    runner = _sampler_runner(mode)

    def _boom(**kw):
        raise RuntimeError("boom")

    runner.sampler = _boom
    runner.sampler.logprobs_mode = mode
    with _sampler_patches():
        with pytest.raises(RuntimeError, match="boom"):
            mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))


def test_dummy_sampler_run_spec_decode_warms_rejection_sampler():
    mode = _PROCESSED_MODE
    runner = _sampler_runner(mode)
    runner.speculative_config = SimpleNamespace(
        rejection_sample_method="standard", draft_sample_method="probabilistic"
    )
    calls = []

    class _Rejection:
        def __call__(self, *a, **k):
            calls.append(a)

    runner.rejection_sampler = _Rejection()
    with _sampler_patches(
        extra=[
            (
                mr,
                "SpecDecodeMetadata",
                SimpleNamespace(make_dummy=lambda ids, dev: "META"),
            ),
            (mr.torch, "accelerator", SimpleNamespace(synchronize=lambda: None)),
        ]
    ):
        out = mr.ModelRunnerFL._dummy_sampler_run(runner, torch.zeros(3, 4))
    assert out == "SAMPLER_OUT"
    assert len(calls) == 2
