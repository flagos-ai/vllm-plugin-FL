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

"""G14 coverage: ModelRunnerFL.execute_model main body (814b).

Complements the G10 early-exit tests by driving the full forward path of
``execute_model``: state caching, cascade attention, padding, ascend query
padding, mamba-alignment preprocessing, EPLB, encoder-input eager mode,
aux hidden states, pooling, PP broadcast and the spec-decode deferred KV
connector finalization.

Everything is CPU-only and device-free: the runner is a SimpleNamespace,
and every module-level collaborator is stubbed at the
``vllm_fl.worker.model_runner`` boundary.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr


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
    """Stand-in for vllm.sequence.IntermediateTensors."""

    def __init__(self, tensors=None):
        self.tensors = tensors or {}

    def __getitem__(self, key):
        return self


class _TensorsHolder(_Intermediate):
    """Hidden-state holder that supports logits indexing and .tensors."""

    def __getitem__(self, key):
        return self


def _pp(is_last=True, send=None, broadcast=None, ranks=(0, 1)):
    return types.SimpleNamespace(
        is_last_rank=is_last,
        send_tensor_dict=send or (lambda *a, **k: None),
        broadcast_tensor_dict=broadcast or (lambda data, src=0: data),
        ranks=list(ranks),
    )


def _patches(pp=None, ubatch=None, logger=None, **extra):
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


# --------------------------------------------------------------------------
# Full happy path
# --------------------------------------------------------------------------
def test_main_happy_path_caches_state():
    calls = []

    def _forward(**kw):
        calls.append(("forward", kw.get("input_ids")))
        return torch.zeros(4, 3)

    runner = _runner(
        _model_forward=_forward,
        model=types.SimpleNamespace(
            compute_logits=lambda shs: (
                calls.append(("logits", tuple(shs.shape)))
                or torch.zeros(shs.shape[0], 5)
            )
        ),
        maybe_get_kv_connector_output=lambda so, defer_finalize=False: (
            calls.append(("kv", defer_finalize)) or _Ctx("kv-out")
        ),
    )
    sched = _sched()
    with _patches():
        out = mr.ModelRunnerFL.execute_model(runner, sched)

    assert out is None
    state = runner.execute_model_state
    assert isinstance(state, mr.ExecuteModelState)
    assert state.scheduler_output is sched
    assert state.spec_decode_metadata == "spec-meta"
    assert state.spec_decode_common_attn_metadata == "spec-common"
    assert state.ec_connector_output == "ec-conn"
    assert state.cudagraph_stats == "stats"
    assert state.slot_mappings == "flat-slots"
    assert runner.kv_connector_output == "kv-out"
    assert ("forward", "ids") in calls
    assert ("logits", (2, 3)) in calls
    assert ("kv", False) in calls


def test_main_deferred_corrections_run_after_forward():
    order = []

    runner = _runner(
        _update_states=lambda so: (lambda: order.append("deferred")),
        _model_forward=lambda **kw: (
            order.append("forward") or torch.zeros(4, 3)
        ),
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert order == ["forward", "deferred"]


def test_main_kv_sharing_fast_prefill_rejects_prompt_logprobs():
    runner = _runner(
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=True, mamba_cache_mode=None
        ),
        num_prompt_logprobs=1,
    )
    with _patches():
        with pytest.raises(AssertionError, match="kv-sharing-fast-prefill"):
            mr.ModelRunnerFL.execute_model(runner, _sched())


# --------------------------------------------------------------------------
# Cascade attention
# --------------------------------------------------------------------------
def test_main_cascade_attention_computes_prefix_lens():
    captured = {}
    runner = _runner(
        cascade_attn_enabled=True,
        _compute_cascade_attn_prefix_lens=lambda nst, computed, blocks: (
            captured.update(nst=nst, computed=computed, blocks=blocks)
            or "prefix-lens"
        ),
        _determine_batch_execution_and_padding=lambda **kw: (
            captured.update(determine=kw)
            or mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=4, num_reqs=None),
            False,
            None,
            "stats",
        ),
        _build_attention_metadata=lambda **kw: (
            captured.update(build=kw) or ("attn-meta", "spec-common")
        ),
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(runner, _sched(num_common_prefix_blocks=1))

    assert captured["blocks"] == 1
    assert captured["determine"]["use_cascade_attn"] is True
    assert captured["build"]["cascade_attn_prefix_lens"] == "prefix-lens"


def test_main_cascade_attention_disabled_by_ubatching():
    calls = []
    runner = _runner(
        cascade_attn_enabled=True,
        parallel_config=types.SimpleNamespace(
            distributed_executor_backend="mp",
            data_parallel_size=1,
            use_ubatching=True,
            num_ubatches=2,
        ),
        _compute_cascade_attn_prefix_lens=lambda *a: calls.append(a) or "prefix",
    )
    captured = {}
    runner._determine_batch_execution_and_padding = lambda **kw: (
        captured.update(kw) or mr.CUDAGraphMode.NONE,
        types.SimpleNamespace(num_tokens=4, num_reqs=None),
        False,
        None,
        "stats",
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert calls == []
    assert captured["use_cascade_attn"] is False


# --------------------------------------------------------------------------
# Ascend attention / padded graphs
# --------------------------------------------------------------------------
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


@pytest.mark.parametrize("decode_only", [True, False])
def test_main_ascend_attention_sets_disable_full(ascend_attn_module, decode_only):
    captured = {}
    AscendAttentionState = ascend_attn_module
    state = (
        AscendAttentionState.DecodeOnly if decode_only else "prefill"
    )
    runner = _runner(
        use_ascend_attention=True,
        _ascend_attn_state=state,
        _determine_batch_execution_and_padding=lambda **kw: (
            captured.update(determine=kw) or mr.CUDAGraphMode.FULL,
            types.SimpleNamespace(num_tokens=4, num_reqs=4),
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

    assert captured["determine"]["disable_full"] is (not decode_only)
    assert captured["slots"]["num_reqs_padded"] == 99


def test_main_full_graph_pads_slot_mappings_for_separate_kv_update():
    captured = {}
    runner = _runner(
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
        _determine_batch_execution_and_padding=lambda **kw: (
            mr.CUDAGraphMode.FULL,
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
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert captured["slots"]["num_tokens_padded"] == 8
    assert captured["slots"]["num_reqs_padded"] == 4
    assert captured["build"]["num_tokens_padded"] == 8
    assert captured["build"]["num_reqs_padded"] == 4


def test_main_ubatch_slices_flow_to_context():
    captured = {}
    runner = _runner(
        _determine_batch_execution_and_padding=lambda **kw: (
            mr.CUDAGraphMode.NONE,
            types.SimpleNamespace(num_tokens=4, num_reqs=None),
            True,
            None,
            "stats",
        ),
        eplb_state=types.SimpleNamespace(
            prepare_forward=lambda *a: captured.update(eplb=a)
        ),
    )
    with _patches(
        ubatch=lambda should, nst, ntp, nrp, nub: ("us", "usp"),
    ):
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert captured["eplb"][2] == "usp"


# --------------------------------------------------------------------------
# Mamba alignment
# --------------------------------------------------------------------------
def test_main_mamba_align_without_fused_postprocess():
    order = []
    runner = _runner(
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode="align"
        ),
        _update_states=lambda so: (lambda: order.append("deferred")),
        num_accepted_tokens=types.SimpleNamespace(
            np=np.zeros(2, dtype=np.int32),
            copy_to_gpu=lambda n: order.append(("copy", n)),
        ),
    )
    pre_calls = []
    with _patches(), _Patch(
        mr.mamba_utils, "preprocess_mamba", lambda *a, **k: pre_calls.append(a)
    ), _Patch(
        mr.mamba_utils,
        "stage_postprocess_inputs_to_gpu",
        lambda *a: (_ for _ in ()).throw(AssertionError("must not stage")),
    ):
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert len(pre_calls) == 1
    assert order == ["deferred", ("copy", 2)]


def test_main_mamba_align_stages_fused_postprocess():
    staged = []
    buf = types.SimpleNamespace(preprocess="pre", postprocess_align="fused")
    runner = _runner(
        cache_config=types.SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode="align"
        ),
        _get_mamba_bufs=lambda: buf,
    )
    with _patches(), _Patch(
        mr.mamba_utils, "preprocess_mamba", lambda *a, **k: None
    ), _Patch(
        mr.mamba_utils,
        "stage_postprocess_inputs_to_gpu",
        lambda *a: staged.append(a),
    ):
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert len(staged) == 1
    assert staged[0][0] == "fused"
    assert staged[0][3] == 2


# --------------------------------------------------------------------------
# EPLB + encoder-input eager mode
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Hidden-state handling
# --------------------------------------------------------------------------
def test_main_aux_hidden_state_outputs():
    hidden = torch.ones(4, 3)
    aux = [torch.zeros(4, 3)]
    runner = _runner(
        use_aux_hidden_state_outputs=True,
        _model_forward=lambda **kw: (hidden, aux),
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(runner, _sched())

    state = runner.execute_model_state
    assert state.hidden_states is hidden
    assert state.aux_hidden_states is aux


# --------------------------------------------------------------------------
# Pipeline-parallel / pooling returns
# --------------------------------------------------------------------------
def test_main_non_last_rank_returns_intermediate_tensors():
    intermediate = _Intermediate({"x": torch.zeros(1)})
    runner = _runner(_model_forward=lambda **kw: intermediate)
    with _patches(pp=_pp(is_last=False)), _Patch(
        mr, "IntermediateTensors", _Intermediate
    ):
        out = mr.ModelRunnerFL.execute_model(runner, _sched())

    assert out is intermediate
    assert runner.execute_model_state is None
    assert runner.kv_connector_output == "kv-out"


def test_main_pooling_model_returns_pooled():
    calls = []
    runner = _runner(
        is_pooling_model=True,
        _pool=lambda hs, nst, nstnp, kv: (
            calls.append((nst, tuple(nstnp), kv)) or "pooled-out"
        ),
    )
    with _patches():
        out = mr.ModelRunnerFL.execute_model(runner, _sched())

    assert out == "pooled-out"
    assert calls[0][0] == 4
    assert calls[0][2] == "kv-out"
    assert runner.execute_model_state is None


def test_main_broadcast_pp_last_rank_uses_broadcasted_logits():
    calls = []
    runner = _runner(
        broadcast_pp_output=True,
        model=types.SimpleNamespace(
            compute_logits=lambda shs: torch.zeros(shs.shape[0], 5)
        ),
    )
    with _patches(
        pp=_pp(
            is_last=True,
            broadcast=lambda data, src=0: (
                calls.append(("broadcast", data.keys(), src))
                or {"logits": "bcast-logits"}
            ),
        )
    ):
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert runner.execute_model_state.logits == "bcast-logits"
    assert calls[0][1] is not None
    assert calls[0][2] == 1


def test_main_broadcast_pp_non_last_rank_sends_hidden_states():
    sent = []
    residual = {"residual": torch.zeros(2, 2)}
    runner = _runner(
        broadcast_pp_output=True,
        _model_forward=lambda **kw: _TensorsHolder(residual),
    )
    with _patches(
        pp=_pp(
            is_last=False,
            send=lambda *a, **k: sent.append((a, k)),
            broadcast=lambda data, src=0: {"logits": "bcast-from-last"},
        )
    ):
        mr.ModelRunnerFL.execute_model(runner, _sched())

    assert sent
    assert sent[0][0][0] is residual
    assert sent[0][1]["all_gather_group"] == "tp-group"
    assert runner.execute_model_state.logits == "bcast-from-last"


# --------------------------------------------------------------------------
# Spec decode / ngram / kv transfer
# --------------------------------------------------------------------------
def test_main_spec_decode_defers_kv_finalize():
    captured = {}
    runner = _runner(
        speculative_config=types.SimpleNamespace(
            use_ngram_gpu=lambda: False
        ),
        maybe_get_kv_connector_output=lambda so, defer_finalize=False: (
            captured.update(defer=defer_finalize) or _Ctx("kv-out")
        ),
        _build_attention_metadata=lambda **kw: (
            captured.update(build=kw) or ("attn-meta", "spec-common")
        ),
    )
    with _patches():
        mr.ModelRunnerFL.execute_model(
            runner, _sched(scheduled_spec_decode_tokens={"a": [1]})
        )

    assert captured["defer"] is True
    assert captured["build"]["use_spec_decode"] is True






# --------------------------------------------------------------------------
# Empty schedule with external launcher DP
# --------------------------------------------------------------------------
def test_main_empty_schedule_external_launcher_dummy_run():
    dummies = []
    runner = _runner(
        parallel_config=types.SimpleNamespace(
            distributed_executor_backend="external_launcher",
            data_parallel_size=2,
            use_ubatching=False,
            num_ubatches=2,
        ),
        _dummy_run=lambda n: dummies.append(n),
    )
    with _patches():
        out = mr.ModelRunnerFL.execute_model(
            runner, _sched(total=0)
        )

    assert out is mr.EMPTY_MODEL_RUNNER_OUTPUT
    assert dummies == [1]
