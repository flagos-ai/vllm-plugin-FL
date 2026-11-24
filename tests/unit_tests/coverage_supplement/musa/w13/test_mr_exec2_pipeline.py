# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the ModelRunnerFL.execute_model pipeline.

Every runner is built with ``object.__new__`` and hand-set attributes; no GPU,
weights, network or distributed runtime is required.  The module-level helpers
used by ``execute_model`` are replaced with recorders so assertions can check
the real control flow of the source.
"""

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr

ModelRunnerFL = mr.ModelRunnerFL


class _IT:
    def __init__(self, tensors=None):
        self.tensors = {} if tensors is None else tensors


class _HS(_IT):
    def __getitem__(self, idx):
        return torch.zeros(len(idx), 4)


class _AttnState:
    DecodeOnly = 1
    Other = 2


def _sched(**kw):
    base = dict(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={"r0": 2, "r1": 2},
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        kv_connector_metadata=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _patch_base(monkeypatch):
    rec = SimpleNamespace(
        set_fwd=[],
        ubatch=[],
        det=[],
        slots=[],
        build=[],
        preempt=[],
        pad_qsl=[],
        cascade=[],
        preproc_mamba=[],
        stage=[],
        bcast=[],
        send=[],
        mm=[],
    )

    monkeypatch.setattr(
        mr, "record_function_or_nullcontext", lambda name: contextlib.nullcontext()
    )

    def _set_fwd(*a, **k):
        rec.set_fwd.append(k)
        return contextlib.nullcontext()

    monkeypatch.setattr(mr, "set_forward_context", _set_fwd)

    def _ubatch(*a):
        rec.ubatch.append(a)
        return ("UB", a), ("UBP", a)

    monkeypatch.setattr(mr, "maybe_create_ubatch_slices", _ubatch)
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=True, ranks=[0], rank=0, world_size=1
        ),
    )
    monkeypatch.setattr(mr, "get_tp_group", lambda: "TP")
    monkeypatch.setattr(
        mr, "make_empty_encoder_model_runner_output", lambda so: "ECEMPTY"
    )
    monkeypatch.setattr(mr, "IntermediateTensors", _IT)
    return rec


def _exec_runner(monkeypatch, rec):
    r = object.__new__(ModelRunnerFL)
    r.execute_model_state = None
    r.kv_connector_output = None
    r.speculative_config = None
    r.cache_config = SimpleNamespace(
        kv_sharing_fast_prefill=False, mamba_cache_mode=None
    )
    r.num_prompt_logprobs = 0
    r.parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        data_parallel_size=1,
        use_ubatching=False,
        num_ubatches=2,
    )
    r.vllm_config = SimpleNamespace()
    r.encoder_cache = {}
    r.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        num_computed_tokens_cpu=np.zeros(2, dtype=np.int32),
        num_accepted_tokens_cpu=np.zeros(2, dtype=np.int32),
    )
    r.model_config = SimpleNamespace(is_encoder_decoder=False)
    r.use_ascend_attention = False
    r._ascend_attn_state = None
    r.cascade_attn_enabled = False
    r.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    r.attn_groups = {}
    r.mamba_state_idx = {}
    r.requests = {"r0": SimpleNamespace(), "r1": SimpleNamespace()}
    r.compilation_config = SimpleNamespace(static_forward_context={})
    r.device = "cpu"
    r.eplb_state = None
    r.use_aux_hidden_state_outputs = False
    r.broadcast_pp_output = False
    r.is_pooling_model = False
    r.model = SimpleNamespace(compute_logits=lambda hs: "LOGITS")
    r.query_start_loc = SimpleNamespace(np=np.arange(8))
    r.num_accepted_tokens = SimpleNamespace(
        np=np.zeros(4, dtype=np.int32), copy_to_gpu=lambda n: None
    )

    r.synchronize_input_prep = lambda: contextlib.nullcontext()
    r._update_states = lambda so: None
    r._prepare_inputs = lambda so, arr: (torch.tensor([0, 2]), "SPECMETA", 1)
    r._allow_microbatching = lambda *a: True
    r._determine_batch_execution_and_padding = lambda **kw: (
        rec.det.append(kw)
        or (
            mr.CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=4, num_reqs=2),
            False,
            None,
            "CGSTATS",
        )
    )
    r._get_slot_mappings = lambda **kw: rec.slots.append(kw) or (
        {"g": 1},
        "SLOTS",
    )
    r._build_attention_metadata = lambda **kw: rec.build.append(kw) or (
        "ATTN",
        "SPECCAM",
    )
    r._preprocess = lambda so, n, it: (
        torch.arange(n),
        None,
        torch.zeros(n, dtype=torch.long),
        "IT",
        {},
        "EC",
    )
    r._model_forward = lambda **kw: torch.zeros(4, 4)
    r._get_mamba_bufs = lambda: SimpleNamespace(
        preprocess="MPRE", postprocess_align=None
    )
    r._get_mamba_state_copy_funcs = lambda: ("CPU", "GPU")
    r._pad_query_start_loc_for_fia = lambda *a: (
        rec.pad_qsl.append(a) or 3
    )
    r._compute_cascade_attn_prefix_lens = lambda *a: (
        rec.cascade.append(a) or "PREFIX"
    )
    r._dummy_run = lambda n: rec.det.append(("dummy", n))
    r._pool = lambda *a: "POOL"
    r.maybe_get_ec_connector_output = lambda *a, **k: contextlib.nullcontext()
    r._execute_mm_encoder = lambda so: rec.mm.append(so)
    r.maybe_get_kv_connector_output = lambda *a, **k: contextlib.nullcontext(
        "KVOUT"
    )
    return r


def test_execute_model_full_success(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)

    out = runner.execute_model(_sched())

    assert out is None
    state = runner.execute_model_state
    assert isinstance(state, mr.ExecuteModelState)
    assert state.logits == "LOGITS"
    assert state.spec_decode_metadata == "SPECMETA"
    assert state.spec_decode_common_attn_metadata == "SPECCAM"
    assert state.cudagraph_stats == "CGSTATS"
    assert state.slot_mappings == "SLOTS"
    assert state.ec_connector_output == "EC"
    assert runner.kv_connector_output == "KVOUT"
    assert rec.det[0]["num_tokens"] == 4
    assert rec.det[0]["num_reqs"] == 2
    assert rec.det[0]["use_cascade_attn"] is False
    assert rec.slots[0]["num_tokens_unpadded"] == 4
    assert rec.build[0]["use_spec_decode"] is False
    assert rec.set_fwd[0]["num_tokens"] == 4
    assert rec.set_fwd[0]["skip_compiled"] is False


def test_execute_model_ngram_copy_and_kv_preempt(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.speculative_config = SimpleNamespace(use_ngram_gpu=lambda: True)

    seen = {}

    def _replace(obj, **kw):
        seen["orig"] = obj
        data = dict(vars(obj))
        data.update(kw)
        return SimpleNamespace(**data)

    monkeypatch.setattr(mr, "replace", _replace)
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
    transfer = SimpleNamespace(
        handle_preemptions=lambda md: rec.preempt.append(md)
    )
    monkeypatch.setattr(mr, "get_kv_transfer_group", lambda: transfer)

    sched = _sched(kv_connector_metadata="META")
    out = runner.execute_model(sched)

    assert out is None
    assert rec.preempt == ["META"]
    assert seen["orig"] is sched
    assert runner.execute_model_state.scheduler_output is not sched
    assert (
        runner.execute_model_state.scheduler_output.kv_connector_metadata
        == "META"
    )


def test_execute_model_ec_transfer_non_consumer_returns_empty(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: True)
    monkeypatch.setattr(
        mr, "get_ec_transfer", lambda: SimpleNamespace(is_consumer=False)
    )

    out = runner.execute_model(_sched())

    assert out == "ECEMPTY"
    assert runner.execute_model_state is None
    assert len(rec.mm) == 1


def test_execute_model_ec_transfer_consumer_continues(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: True)
    monkeypatch.setattr(
        mr, "get_ec_transfer", lambda: SimpleNamespace(is_consumer=True)
    )

    out = runner.execute_model(_sched())

    assert out is None
    assert isinstance(runner.execute_model_state, mr.ExecuteModelState)


def test_execute_model_kv_sharing_requires_no_prompt_logprobs(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.cache_config.kv_sharing_fast_prefill = True
    runner.num_prompt_logprobs = 2

    with pytest.raises(AssertionError):
        runner.execute_model(_sched())


def test_execute_model_ascend_full_pads_query_start_loc(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    stub = SimpleNamespace(AscendAttentionState=_AttnState)
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.attention",
        stub,
    )
    runner.use_ascend_attention = True
    runner._ascend_attn_state = _AttnState.Other

    def _det(**kw):
        rec.det.append(kw)
        return (
            mr.CUDAGraphMode.FULL,
            SimpleNamespace(num_tokens=4, num_reqs=4),
            False,
            None,
            "CGSTATS",
        )

    runner._determine_batch_execution_and_padding = _det

    out = runner.execute_model(_sched())

    assert out is None
    assert rec.det[0]["disable_full"] is True
    assert len(rec.pad_qsl) == 1
    assert rec.pad_qsl[0][2] == 2
    assert rec.pad_qsl[0][3] == 4
    assert rec.slots[0]["num_tokens_padded"] == 4


def test_execute_model_ascend_decode_only_keeps_full(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.attention",
        SimpleNamespace(AscendAttentionState=_AttnState),
    )
    runner.use_ascend_attention = True
    runner._ascend_attn_state = _AttnState.DecodeOnly

    out = runner.execute_model(_sched())

    assert out is None
    assert rec.det[0]["disable_full"] is False


def test_execute_model_cascade_prefix_when_not_ubatching(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.cascade_attn_enabled = True

    out = runner.execute_model(_sched())

    assert out is None
    assert len(rec.cascade) == 1
    assert rec.det[0]["use_cascade_attn"] is True


def test_execute_model_cascade_skipped_when_ubatching(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.cascade_attn_enabled = True
    runner.parallel_config.use_ubatching = True

    out = runner.execute_model(_sched())

    assert out is None
    assert rec.cascade == []
    assert rec.det[0]["use_cascade_attn"] is False


def test_execute_model_mamba_align_defers_and_stages(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.cache_config.mamba_cache_mode = "align"
    runner.input_batch.num_accepted_tokens_cpu = np.array([3, 4], dtype=np.int32)
    runner._get_mamba_bufs = lambda: SimpleNamespace(
        preprocess="MPRE", postprocess_align="POST"
    )
    order = []

    def _preprocess_mamba(*a, **k):
        rec.preproc_mamba.append((a, k))
        order.append("mamba")

    def _stage(*a):
        rec.stage.append(a)
        order.append("stage")

    monkeypatch.setattr(mr.mamba_utils, "preprocess_mamba", _preprocess_mamba)
    monkeypatch.setattr(
        mr.mamba_utils, "stage_postprocess_inputs_to_gpu", _stage
    )

    def _deferred():
        order.append(("deferred", runner.execute_model_state is not None))

    runner._update_states = lambda so: _deferred

    out = runner.execute_model(_sched())

    assert out is None
    assert len(rec.preproc_mamba) == 1
    assert len(rec.stage) == 1
    assert runner.num_accepted_tokens.np.tolist() == [3, 4, 0, 0]
    assert order[0] == ("deferred", False)
    assert order[1] == "mamba"
    assert order[2] == "stage"
    assert order.count(("deferred", False)) == 1


def test_execute_model_deferred_corrections_after_state_set(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    flags = {}

    def _deferred():
        flags["state"] = runner.execute_model_state is not None

    runner._update_states = lambda so: _deferred

    out = runner.execute_model(_sched())

    assert out is None
    assert flags["state"] is True


def test_execute_model_separate_kv_update_and_spec_decode(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    runner.attn_groups = {
        0: [
            SimpleNamespace(
                backend=SimpleNamespace(forward_includes_kv_cache_update=False)
            )
        ]
    }

    out = runner.execute_model(_sched(scheduled_spec_decode_tokens={"r0": [1]}))

    assert out is None
    assert rec.build[0]["use_spec_decode"] is True
    # Padded dimensions flow into slot mappings when KV updates are separate.
    assert rec.slots[0]["num_tokens_padded"] == 4
    assert rec.slots[0]["num_reqs_padded"] == 2


def test_execute_model_broadcast_pp_last_rank_uses_broadcast_logits(
    monkeypatch,
):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.broadcast_pp_output = True
    broadcast_logits = torch.ones(2, 4)
    bc = []

    def _bcast(data, src):
        bc.append((data, src))
        return {"logits": broadcast_logits}

    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=True,
            ranks=[0, 1],
            rank=1,
            world_size=2,
            broadcast_tensor_dict=_bcast,
        ),
    )
    runner.model = SimpleNamespace(
        compute_logits=lambda hs: torch.zeros(2, 4)
    )

    out = runner.execute_model(_sched())

    assert out is None
    assert len(bc) == 1
    assert bc[0][1] == 1
    assert set(bc[0][0]) == {"logits"}
    assert runner.execute_model_state.logits is broadcast_logits


def test_execute_model_broadcast_pp_non_last_rank_sends_dict(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.broadcast_pp_output = True
    runner._model_forward = lambda **kw: _HS({})
    broadcast_logits = torch.ones(2, 4)
    bc = []

    def _send(tensors, **k):
        rec.send.append((tensors, k))

    def _bcast(data, src):
        bc.append((data, src))
        return {"logits": broadcast_logits}

    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=False,
            ranks=[0, 1],
            rank=0,
            world_size=2,
            send_tensor_dict=_send,
            broadcast_tensor_dict=_bcast,
        ),
    )

    out = runner.execute_model(_sched())

    assert out is None
    assert len(rec.send) == 1
    assert rec.send[0][1]["all_gather_group"] == "TP"
    assert rec.send[0][1]["all_gather_tensors"] == {"residual": True}
    assert bc[0][0] == {}
    assert runner.execute_model_state.logits is broadcast_logits


def test_execute_model_pooling_returns_before_state(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.is_pooling_model = True
    runner._pool = lambda *a: "POOL"

    out = runner.execute_model(_sched())

    assert out == "POOL"
    assert runner.execute_model_state is None
    assert runner.kv_connector_output is None


def test_execute_model_non_last_rank_returns_intermediate(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    inter = _IT()
    runner._model_forward = lambda **kw: inter
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False, ranks=[0, 1], rank=0, world_size=2),
    )

    out = runner.execute_model(_sched())

    assert out is inter
    assert runner.execute_model_state is None
    assert runner.kv_connector_output == "KVOUT"


def test_execute_model_aux_hidden_eplb_and_encoder_input(monkeypatch):
    rec = _patch_base(monkeypatch)
    runner = _exec_runner(monkeypatch, rec)
    runner.use_aux_hidden_state_outputs = True
    aux = [torch.zeros(4, 2)]
    runner._model_forward = lambda **kw: (torch.zeros(4, 4), aux)
    prep = []
    runner.eplb_state = SimpleNamespace(
        prepare_forward=lambda *a: prep.append(a)
    )
    runner.model_config = SimpleNamespace(is_encoder_decoder=True)

    out = runner.execute_model(_sched(scheduled_encoder_inputs={"r0": 1}))

    assert out is None
    state = runner.execute_model_state
    assert state.aux_hidden_states is aux
    assert len(prep) == 1
    assert prep[0][1] == 4
    assert rec.set_fwd[0]["skip_compiled"] is True
