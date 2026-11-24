# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for ModelRunnerFL.sample_tokens spec-decode dispatch.

The runner is created with ``object.__new__``; the speculative config, drafter
and sampler are mocked so the real branch structure of ``sample_tokens`` can be
exercised without a GPU.
"""

import contextlib
from types import SimpleNamespace

import torch

import vllm_fl.worker.model_runner as mr

ModelRunnerFL = mr.ModelRunnerFL


class _FakeOut:
    def __init__(self, **kw):
        self.kw = kw


class _FakeAsyncOut:
    def __init__(self, **kw):
        self.kw = kw
        self.sampled_token_ids_cpu = "SAMPLED_CPU"
        self.async_copy_ready_event = "READY"


class _Eagle:
    def __init__(self):
        self.calls = []

    def prepare_next_token_ids_padded(self, *a):
        self.calls.append(("prepared", a))
        return "NEXT_IDS", "VALID_COUNT"

    def dummy_run(self, num_tokens=1):
        self.calls.append(("dummy", num_tokens))


class _NgramGPU:
    def __init__(self):
        self.calls = []

    def update_token_ids_ngram(self, *a):
        self.calls.append(("update", a))
        return "NGRAM_NEXT", "NGRAM_COUNT", "NGRAM_VALID"


def _sched():
    return SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={"r0": 2, "r1": 2},
    )


def _spec(**kw):
    base = dict(
        method="eagle",
        use_eagle=lambda: False,
        use_dflash=lambda: False,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: False,
        use_ngram_gpu=lambda: False,
        disable_padded_drafter_batch=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Rec:
    def __init__(self):
        self.after = []
        self.propose = []
        self.copy_draft = []
        self.copy_valid = []
        self.finalize = []
        self.eplb = []
        self.async_set = []
        self.io_step = []
        self.grammar = []
        self.ppb = []
        self.dummy = []


def _state(rec):
    return mr.ExecuteModelState(
        _sched(),
        "LOGITS",
        "SPECMETA",
        "SPECCAM",
        torch.zeros(4, 4),
        torch.zeros(2, 4),
        None,
        "EC",
        "CGSTATS",
        "SLOTS",
    )


def _patch_mr(monkeypatch):
    monkeypatch.setattr(
        mr, "record_function_or_nullcontext", lambda name: contextlib.nullcontext()
    )
    monkeypatch.setattr(mr, "apply_grammar_bitmask", lambda *a: None)
    monkeypatch.setattr(mr, "advance_io_step", lambda: None)
    monkeypatch.setattr(mr, "ModelRunnerOutput", _FakeOut)
    monkeypatch.setattr(mr, "AsyncGPUModelRunnerOutput", _FakeAsyncOut)
    monkeypatch.setattr(mr, "EagleProposer", _Eagle)
    monkeypatch.setattr(mr, "DFlashProposer", _Eagle)
    monkeypatch.setattr(mr, "DraftModelProposer", _Eagle)
    monkeypatch.setattr(mr, "ExtractHiddenStatesProposer", _Eagle)
    monkeypatch.setattr(mr, "Gemma4Proposer", _Eagle)
    monkeypatch.setattr(mr, "NgramProposerGPU", _NgramGPU)
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=True, world_size=1, rank=0, device_group="G"
        ),
    )


def _runner(monkeypatch, rec, async_=False, spec=None):
    r = object.__new__(ModelRunnerFL)
    r.execute_model_state = _state(rec)
    r.kv_connector_output = "KVOUT"
    r.use_async_scheduling = async_
    r.broadcast_pp_output = False
    r.speculative_config = spec
    r.num_spec_tokens = 2
    r.device = "cpu"
    r.supports_mm_inputs = False
    r.check_ep_fault = True
    r.parallel_config = SimpleNamespace(data_parallel_size=1)
    r.valid_sampled_token_count_event = None
    r.discard_request_mask = SimpleNamespace(gpu="DISCARD")
    r.requests = {"r0": SimpleNamespace()}
    r.use_aux_hidden_state_outputs = False
    r._draft_token_ids = "PREV"
    r._draft_probs = "P"
    r._draft_prob_req_ids = ["x"]
    r._draft_token_req_ids = ["x"]
    r.valid_sampled_token_count_gpu = "V"
    r.input_batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        sampling_metadata=SimpleNamespace(output_token_ids=None),
        vocab_size=32,
        prev_sampled_token_ids="PREV",
        set_async_sampled_token_ids=lambda ids, ev: rec.async_set.append((ids, ev)),
    )
    r.drafter = _Eagle()
    r._sample = lambda logits, smeta: SimpleNamespace(
        sampled_token_ids=torch.tensor([[5], [6]], dtype=torch.int32),
        logprobs_tensors=None,
    )
    r._update_states_after_model_execute = lambda ids, so: rec.after.append(ids)
    r._bookkeeping_sync = lambda *a: (
        "NANS",
        "NANDEV",
        "LP",
        [[5], [6]],
        {"r0": 1},
        ["r0", "r1"],
        {"r0": 0, "r1": 1},
        [],
    )
    r.finalize_kv_connector = lambda: rec.finalize.append(True)
    r.eplb_step = lambda: rec.eplb.append(True)
    r._get_or_create_async_output_copy_stream = lambda: "STREAM"
    r._pp_broadcast_prev_sampled_token_ids = lambda ids: rec.ppb.append(ids)
    r.propose_draft_token_ids = lambda *a: rec.propose.append(a) or "DRAFTIDS"
    r._copy_draft_token_ids_to_cpu = lambda so, **kw: rec.copy_draft.append(
        (so, kw)
    )
    r._copy_valid_sampled_token_count = lambda n, c: rec.copy_valid.append(
        (n, c)
    )
    r._input_fits_in_drafter = lambda cam: True
    return r


def test_sample_tokens_sync_spec_none(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    io = []
    monkeypatch.setattr(mr, "advance_io_step", lambda: io.append(True))
    runner = _runner(monkeypatch, rec, spec=None)

    out = runner.sample_tokens(None)

    assert isinstance(out, _FakeOut)
    assert rec.after
    assert rec.propose == []
    assert rec.finalize == []
    assert io == [True]
    assert out.kw["req_ids"] == ["r0", "r1"]
    assert out.kw["kv_connector_output"] == "KVOUT"
    assert out.kw["num_nans_in_logits"] == "NANS"
    assert out.kw["cudagraph_stats"] == "CGSTATS"
    assert out.kw["ec_connector_output"] is None
    assert runner.kv_connector_output is None
    assert runner.execute_model_state is None
    assert runner._draft_token_ids is None
    assert runner._draft_probs is None
    assert runner._draft_prob_req_ids is None
    assert runner._draft_token_req_ids is None
    assert runner.valid_sampled_token_count_gpu is None
    assert runner.input_batch.prev_sampled_token_ids is None


def test_sample_tokens_mm_passes_ec_connector_output(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    runner = _runner(monkeypatch, rec, spec=None)
    runner.supports_mm_inputs = True

    out = runner.sample_tokens(None)

    assert out.kw["ec_connector_output"] == "EC"


def test_sample_tokens_grammar_applied(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    calls = []
    monkeypatch.setattr(
        mr, "apply_grammar_bitmask", lambda *a: calls.append(a)
    )
    runner = _runner(monkeypatch, rec, spec=None)

    runner.sample_tokens("GRAMMAR")

    assert len(calls) == 1
    assert calls[0][1] == "GRAMMAR"
    assert calls[0][2] is runner.input_batch


def test_sample_tokens_async_broadcasts_and_builds_async_output(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=True, world_size=2, rank=1, device_group="G"
        ),
    )
    runner = _runner(monkeypatch, rec, async_=True, spec=None)

    out = runner.sample_tokens(None)

    assert isinstance(out, _FakeAsyncOut)
    assert len(rec.ppb) == 1
    assert rec.ppb[0] is out.kw["sampled_token_ids"]
    assert out.kw["vocab_size"] == 32
    assert out.kw["check_ep_fault"] is True
    assert out.kw["num_nans"] == "NANDEV"
    assert out.kw["invalid_req_indices"] == []
    assert out.kw["async_output_copy_stream"] == "STREAM"
    assert rec.async_set == [("SAMPLED_CPU", "READY")]


def test_sample_tokens_async_non_last_rank_skips_broadcast(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=False, world_size=2, rank=0, device_group="G"
        ),
    )
    runner = _runner(monkeypatch, rec, async_=True, spec=None)

    runner.sample_tokens(None)

    assert rec.ppb == []


def test_sample_tokens_use_gpu_toks_fits_proposes(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    spec = _spec(use_eagle=lambda: True, disable_padded_drafter_batch=False)
    runner = _runner(monkeypatch, rec, spec=spec)
    runner._input_fits_in_drafter = lambda cam: True

    runner.sample_tokens(None)

    assert len(rec.propose) == 1
    args = rec.propose[0]
    assert hasattr(args[0], "total_num_scheduled_tokens")
    assert args[4] is not None and args[8] == "SLOTS"
    assert torch.is_tensor(args[1]) and args[1].tolist() == [[5], [6]]
    assert rec.finalize == [True]
    assert rec.copy_draft and rec.copy_draft[0][1] == {}


def test_sample_tokens_use_gpu_toks_not_fits_event_and_dp(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    spec = _spec(use_eagle=lambda: True, disable_padded_drafter_batch=False)
    runner = _runner(monkeypatch, rec, spec=spec)
    runner._input_fits_in_drafter = lambda cam: False
    runner.valid_sampled_token_count_event = "EV"
    runner.parallel_config.data_parallel_size = 2

    runner.sample_tokens(None)

    assert rec.copy_valid == [("NEXT_IDS", "VALID_COUNT")]
    assert ("dummy", 1) in runner.drafter.calls
    assert rec.propose == []
    assert rec.copy_draft[-1][1] == {"zeros_only": True}
    assert torch.is_tensor(runner._draft_token_ids)
    assert runner._draft_token_ids.shape == (2, 2)
    assert torch.count_nonzero(runner._draft_token_ids) == 0


def test_sample_tokens_use_gpu_toks_not_fits_no_event_no_dp(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    spec = _spec(use_eagle=lambda: True, disable_padded_drafter_batch=False)
    runner = _runner(monkeypatch, rec, spec=spec)
    runner._input_fits_in_drafter = lambda cam: False

    runner.sample_tokens(None)

    assert rec.copy_valid == []
    assert runner.drafter.calls == []
    assert rec.copy_draft[-1][1] == {"zeros_only": True}


def test_sample_tokens_ngram_gpu_fits_proposes(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    spec = _spec(
        method="ngram_gpu",
        use_ngram_gpu=lambda: True,
        disable_padded_drafter_batch=False,
    )
    runner = _runner(monkeypatch, rec, spec=spec)
    runner.drafter = _NgramGPU()
    runner._input_fits_in_drafter = lambda cam: True

    runner.sample_tokens(None)

    assert len(rec.propose) == 1
    assert torch.is_tensor(rec.propose[0][1])


def test_sample_tokens_ngram_gpu_not_fits_updates_counts(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    spec = _spec(
        method="ngram_gpu",
        use_ngram_gpu=lambda: True,
        disable_padded_drafter_batch=False,
    )
    runner = _runner(monkeypatch, rec, spec=spec)
    runner.drafter = _NgramGPU()
    runner._input_fits_in_drafter = lambda cam: False
    runner.valid_sampled_token_count_event = "EV"
    runner.token_ids_gpu_tensor = "TOK_GPU"
    runner.num_tokens_no_spec_gpu = "NUM_GPU"

    runner.sample_tokens(None)

    assert rec.copy_valid == [("NGRAM_NEXT", "NGRAM_COUNT")]
    assert runner.drafter.calls[0][0] == "update"
    assert rec.propose == []
    assert rec.copy_draft[-1][1] == {"zeros_only": True}


def test_sample_tokens_draft_after_bookkeeping_proposes_cpu_ids(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    spec = _spec(
        method="ngram",
        use_ngram_gpu=lambda: False,
        disable_padded_drafter_batch=True,
    )
    runner = _runner(monkeypatch, rec, spec=spec)
    runner._input_fits_in_drafter = lambda cam: True
    booked = {"done": False}

    def _book(*a):
        booked["done"] = True
        return (
            "NANS",
            "NANDEV",
            "LP",
            [[5], [6]],
            {},
            ["r0", "r1"],
            {"r0": 0, "r1": 1},
            [],
        )

    runner._bookkeeping_sync = _book
    seen = {}

    def _propose(*a):
        seen["booked"] = booked["done"]
        rec.propose.append(a)
        return "DRAFT"

    runner.propose_draft_token_ids = _propose

    runner.sample_tokens(None)

    assert seen["booked"] is True
    assert rec.propose[0][1] == [[5], [6]]
    assert rec.finalize == [True]


def test_sample_tokens_draft_after_bookkeeping_dp_dummy_run(monkeypatch):
    rec = _Rec()
    _patch_mr(monkeypatch)
    spec = _spec(
        method="draft_model",
        uses_draft_model=lambda: True,
        use_ngram_gpu=lambda: False,
        disable_padded_drafter_batch=True,
    )
    runner = _runner(monkeypatch, rec, spec=spec)
    runner._input_fits_in_drafter = lambda cam: False
    runner.parallel_config.data_parallel_size = 2

    runner.sample_tokens(None)

    assert ("dummy", 1) in runner.drafter.calls
    assert rec.propose == []
    assert rec.finalize == [True]
