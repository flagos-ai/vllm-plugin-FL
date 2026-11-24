"""Residual coverage for vllm_fl.worker.model_runner._update_states (w19, local CPU).

Targets the branches not already exercised by w3b: new-request construction,
pooling/random-seed setup, PP token propagation, preemption resume, async spec
optimistic extension, deferred correction closure and ngram incremental copy.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.worker import model_runner as mr


RANDOM_SEED = getattr(mr.SamplingType, "RANDOM_SEED")


@pytest.fixture(autouse=True)
def _cpu_pinned(monkeypatch):
    monkeypatch.setattr(mr, "PIN_MEMORY", False, raising=False)


class _Sink:
    def __init__(self):
        self.sets = []

    def __setitem__(self, key, value):
        self.sets.append((key, value))

    def __getitem__(self, key):
        return 0

    def __setattr__(self, key, value):
        if key == "sets":
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)


class FakeBlockTable:
    def __init__(self):
        self.rows = []

    def append_row(self, block_ids, idx):
        self.rows.append((block_ids, idx))


class FakeInputBatch:
    def __init__(self, req_id_to_index=None, req_ids=None, prev_req_id_to_index=None,
                 vocab_size=128, num_prompt_tokens=None):
        self.req_id_to_index = dict(req_id_to_index or {})
        self.prev_req_id_to_index = prev_req_id_to_index
        self.req_ids = list(req_ids or [])
        self.vocab_size = vocab_size
        self.num_prompt_tokens = dict(num_prompt_tokens or {})
        self.num_computed_tokens_cpu = {}
        self.num_tokens_no_spec = {}
        self.token_ids_cpu = _Sink()
        self.is_token_ids = _Sink()
        self.block_table = FakeBlockTable()
        self.added = []
        self.removed = []
        self.spec_calls = []
        self.condensed = 0
        self.refreshed = 0
        self.num_accepted_tokens_cpu_tensor = None

    def remove_request(self, rid):
        self.removed.append(rid)

    def condense(self):
        self.condensed += 1

    def refresh_metadata(self):
        self.refreshed += 1

    def add_request(self, req):
        self.added.append(req)

    def update_req_spec_token_ids(self, req, spec):
        self.spec_calls.append((req, spec))


def _state(**kw):
    d = dict(
        output_token_ids=[],
        block_ids=[[0]],
        num_computed_tokens=0,
        prev_num_draft_len=0,
        num_tokens=0,
        mm_features=None,
        prompt_token_ids=None,
        prompt_embeds=None,
        num_prompt_tokens=0,
    )
    d.update(kw)
    return SimpleNamespace(**d)


class FakeCRS:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.output_token_ids = list(kw.get("output_token_ids", []))
        self.num_prompt_tokens = 0
        self.prev_num_draft_len = 0
        self.num_tokens = 0
        self.recorded = dict(kw)


def _cached(req_ids=(), num_computed=(), new_block_ids=(), num_output=(),
            resumed=(), new_token_ids=(), all_token_ids=None):
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_computed_tokens=list(num_computed),
        new_block_ids=list(new_block_ids),
        num_output_tokens=list(num_output),
        resumed_req_ids=set(resumed),
        new_token_ids=list(new_token_ids),
        all_token_ids=dict(all_token_ids or {}),
    )


def _sched(cached=None, **kw):
    return SimpleNamespace(
        finished_req_ids=kw.get("finished_req_ids", []),
        new_block_ids_to_zero=kw.get("new_block_ids_to_zero", []),
        kv_cache_block_copies=kw.get("kv_cache_block_copies"),
        num_scheduled_tokens=kw.get("num_scheduled_tokens", {}),
        scheduled_cached_reqs=cached or _cached(),
        scheduled_new_reqs=kw.get("scheduled_new_reqs", []),
        scheduled_spec_decode_tokens=kw.get("scheduled_spec_decode_tokens", {}),
        free_encoder_mm_hashes=kw.get("free_encoder_mm_hashes", []),
    )


class _NpRecorder:
    def __init__(self):
        self.filled = None
        self.items = {}

    def fill(self, v):
        self.filled = v

    def __setitem__(self, key, value):
        self.items[key] = value


def _runner(**over):
    inp = over.pop("input_batch", None) or FakeInputBatch()
    rec = {"finished": [], "registered": [], "mrope": [], "ngram": []}
    lir = SimpleNamespace(
        on_requests_finished=lambda ids: rec["finished"].append(list(ids)),
        register_request=lambda rid, p: rec["registered"].append((rid, p)),
    )
    base = dict(
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=lir,
        input_batch=inp,
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        device=torch.device("cpu"),
        encoder_cache={},
        is_pooling_model=False,
        uses_mrope=False,
        kv_caches=[],
        kv_cache_config=SimpleNamespace(num_blocks=4),
    )
    base.update(over)
    r = object.__new__(mr.ModelRunnerFL)
    r.__dict__.update(base)
    r._may_reorder_batch = lambda so: rec.setdefault("reorder", []).append(so)
    rec["runner"] = r
    return r, inp, rec


def _new_req(**kw):
    d = dict(
        req_id="r",
        prompt_token_ids=[1, 2],
        prompt_embeds=None,
        prompt_is_token_ids=True,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=[[0]],
        num_computed_tokens=0,
        lora_request=None,
    )
    d.update(kw)
    return SimpleNamespace(**d)


# ------------------------------------------------------------------ new reqs
def test_update_states_new_request_greedy_mrope_and_logprobs(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(mr, "CachedRequestState", FakeCRS)
    r, inp, rec = _runner(uses_mrope=True)
    r._init_mrope_positions = lambda state: rec["mrope"].append(state)
    sp = SimpleNamespace(sampling_type="greedy", seed=None, prompt_logprobs=3)
    sched = _sched(
        num_scheduled_tokens={"r": 2},
        scheduled_new_reqs=[_new_req(sampling_params=sp)],
    )
    assert r._update_states(sched) is None
    state = r.requests["r"]
    assert isinstance(state, FakeCRS)
    assert state.recorded["prompt_token_ids"] == [1, 2]
    assert state.generator is None
    assert r.num_prompt_logprobs == {"r": 3}
    assert rec["mrope"] == [state]
    assert inp.added == [state]
    assert rec["registered"] == [("r", None)]
    assert inp.condensed == 1 and inp.refreshed == 1


def test_update_states_new_request_random_seed_txda_and_all_logprobs(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(mr, "CachedRequestState", FakeCRS)
    r, inp, _ = _runner(device=SimpleNamespace(type="txda"), speculative_config=None)
    sp = SimpleNamespace(
        sampling_type=RANDOM_SEED, seed=1234, prompt_logprobs=-1
    )
    r._update_states(
        _sched(
            num_scheduled_tokens={"r": 1},
            scheduled_new_reqs=[
                _new_req(prompt_token_ids=None, prompt_embeds=torch.zeros(2, 3),
                         sampling_params=sp)
            ],
        )
    )
    state = r.requests["r"]
    assert state.generator is not None
    assert state.generator.initial_seed() == 1234
    assert r.num_prompt_logprobs == {"r": inp.vocab_size}


def test_update_states_new_request_pooling_applies_pooler_updates(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(mr, "CachedRequestState", FakeCRS)
    applied = []
    pooler = SimpleNamespace(
        get_pooling_updates=lambda task: SimpleNamespace(apply=applied.append)
    )
    pp = SimpleNamespace(task="embed", extra_kwargs=None)
    r, inp, rec = _runner(is_pooling_model=True)
    r.get_model = lambda: SimpleNamespace(pooler=pooler)
    r._update_states(
        _sched(num_scheduled_tokens={"r": 1},
               scheduled_new_reqs=[_new_req(pooling_params=pp)])
    )
    assert applied == [pp]
    assert rec["registered"] == [("r", pp)]
    assert inp.added and inp.added[0].pooling_params is pp


def test_update_states_new_request_ngram_incremental_copy(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(mr, "CachedRequestState", FakeCRS)
    calls = []
    monkeypatch.setattr(
        mr, "update_scheduler_for_invalid_drafts",
        lambda *a, **k: calls.append(("invalid", a)),
    )
    monkeypatch.setattr(
        mr, "update_ngram_gpu_tensors_incremental",
        lambda batch, tok, nts, new_reqs, device, **kw: calls.append(
            ("incremental", list(new_reqs))
        ),
    )
    spec = SimpleNamespace(use_ngram_gpu=lambda: True)
    r, inp, _ = _runner(
        speculative_config=spec,
        use_async_spec_decode=True,
        prev_num_draft_tokens=SimpleNamespace(np=_NpRecorder()),
        _num_valid_draft_tokens_event="ev",
        _num_valid_draft_tokens_cpu="cpu",
        token_ids_gpu_tensor="tok",
        num_tokens_no_spec_gpu="nts",
        _ngram_pinned_idx_buf=None,
        _ngram_pinned_val_buf=None,
    )
    sp = SimpleNamespace(sampling_type="greedy", seed=None, prompt_logprobs=None)
    r._update_states(
        _sched(num_scheduled_tokens={"r": 1},
               scheduled_new_reqs=[_new_req(sampling_params=sp)])
    )
    state = r.requests["r"]
    assert r.prev_num_draft_tokens.np.filled == 0
    assert [c[0] for c in calls] == ["invalid", "incremental"]
    assert calls[1][1] == [state]


# ------------------------------------------------- persistent (last-rank) update
def test_update_states_last_rank_trims_tokens_and_extends_blocks(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    st = _state(output_token_ids=[1, 2, 3], block_ids=[[1]])
    inp = FakeInputBatch(req_id_to_index={"r": 0}, req_ids=["r"],
                         num_prompt_tokens={0: 4})
    inp.num_tokens_no_spec[0] = 6
    r, _, _ = _runner(requests={"r": st}, input_batch=inp)
    r._update_states(
        _sched(
            cached=_cached(
                req_ids=["r"],
                num_computed=[7],
                new_block_ids=[[[8, 9]]],
                num_output=[2],
            ),
            num_scheduled_tokens={"r": 1},
        )
    )
    assert st.output_token_ids == [1, 2]
    assert st.num_computed_tokens == 7
    assert st.block_ids == [[1, 8, 9]]
    assert inp.num_computed_tokens_cpu[0] == 7
    assert inp.block_table.rows == [([[8, 9]], 0)]
    assert inp.num_tokens_no_spec[0] == 6
    assert inp.spec_calls and inp.spec_calls[0][0] is st


def test_update_states_last_rank_skips_extend_when_no_new_blocks(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    st = _state(output_token_ids=[1], block_ids=[[3]])
    inp = FakeInputBatch(req_id_to_index={"r": 0}, req_ids=["r"],
                         num_prompt_tokens={0: 1})
    r, _, _ = _runner(requests={"r": st}, input_batch=inp)
    r._update_states(
        _sched(
            cached=_cached(req_ids=["r"], num_computed=[1], new_block_ids=[None],
                           num_output=[1]),
            num_scheduled_tokens={"r": 1},
        )
    )
    assert st.block_ids == [[3]]
    assert inp.block_table.rows == []


# -------------------------------------------------------- pipeline parallel writes
def test_update_states_not_last_rank_appends_and_extends_tokens(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    st1 = _state(output_token_ids=[1], block_ids=[[0]], num_tokens=2)
    st2 = _state(output_token_ids=[2], block_ids=[[0]], num_tokens=2)
    inp = FakeInputBatch(
        req_id_to_index={"a": 0, "b": 1}, req_ids=["a", "b"],
        num_prompt_tokens={0: 2, 1: 2},
    )
    inp.num_tokens_no_spec.update({0: 2, 1: 2})
    r, _, _ = _runner(requests={"a": st1, "b": st2}, input_batch=inp)
    r._update_states(
        _sched(
            cached=_cached(
                req_ids=["a", "b"],
                num_computed=[3, 3],
                new_block_ids=[[], []],
                num_output=[2, 3],
                new_token_ids=[[5], [6, 7]],
            ),
            num_scheduled_tokens={"a": 1, "b": 1},
        )
    )
    assert st1.output_token_ids == [1, 5]
    assert st2.output_token_ids == [2, 6, 7]
    assert inp.num_tokens_no_spec[0] == 4
    assert inp.num_tokens_no_spec[1] == 5
    assert inp.token_ids_cpu.sets == [
        ((0, slice(2, 4)), [5]),
        ((1, slice(2, 5)), [6, 7]),
    ]
    assert inp.is_token_ids.sets == [
        ((0, slice(2, 4)), True),
        ((1, slice(2, 5)), True),
    ]


def test_update_states_not_last_rank_empty_new_token_ids(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    st = _state(output_token_ids=[1], block_ids=[[0]], num_tokens=2)
    inp = FakeInputBatch(req_id_to_index={"a": 0}, req_ids=["a"],
                         num_prompt_tokens={0: 2})
    inp.num_tokens_no_spec[0] = 2
    r, _, _ = _runner(requests={"a": st}, input_batch=inp)
    r._update_states(
        _sched(
            cached=_cached(req_ids=["a"], num_computed=[4], new_block_ids=[[]],
                           num_output=[1]),
            num_scheduled_tokens={"a": 1},
        )
    )
    assert st.output_token_ids == [1]
    assert inp.num_tokens_no_spec[0] == 4
    assert inp.token_ids_cpu.sets == []
    assert inp.is_token_ids.sets == [((0, slice(2, 4)), True)]


# ------------------------------------------------------------ resume / requeue
def test_update_states_resumed_from_preemption_replaces_blocks(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    st = _state(output_token_ids=[1], block_ids=[[9]])
    inp = FakeInputBatch()
    r, _, _ = _runner(requests={"r": st}, input_batch=inp)
    r._update_states(
        _sched(
            cached=_cached(req_ids=["r"], num_computed=[5],
                           new_block_ids=[[[11, 12]]], num_output=[1],
                           resumed={"r"}),
            num_scheduled_tokens={"r": 1},
        )
    )
    assert st.block_ids == [[11, 12]]
    assert inp.added == [st]


def test_update_states_requeued_async_scheduling_recovers_outputs(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    st = _state(output_token_ids=[], block_ids=[[0]])
    inp = FakeInputBatch()
    r, _, _ = _runner(requests={"r": st}, input_batch=inp,
                      use_async_scheduling=True)
    r._update_states(
        _sched(
            cached=_cached(req_ids=["r"], num_computed=[6], new_block_ids=[[]],
                           num_output=[2], all_token_ids={"r": [1, 2, 3, 4]}),
            num_scheduled_tokens={"r": 1},
        )
    )
    assert st.output_token_ids == [3, 4]
    assert inp.added == [st]


# ---------------------------------------------------- async spec decode + ngram
def _async_spec_runner(monkeypatch, state, inp, req_index=True):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    calls = []
    monkeypatch.setattr(
        mr, "update_scheduler_for_invalid_drafts", lambda *a, **k: calls.append("invalid")
    )
    monkeypatch.setattr(
        mr, "update_ngram_gpu_tensors_incremental",
        lambda *a, **k: calls.append("incremental"),
    )
    spec = SimpleNamespace(use_ngram_gpu=lambda: True)
    r, _, _ = _runner(
        requests={"r": state},
        input_batch=inp,
        speculative_config=spec,
        use_async_scheduling=True,
        prev_num_draft_tokens=SimpleNamespace(np=_NpRecorder()),
        _num_valid_draft_tokens_event="ev",
        _num_valid_draft_tokens_cpu="cpu",
        token_ids_gpu_tensor="tok",
        num_tokens_no_spec_gpu={0: 9},
        _ngram_pinned_idx_buf=None,
        _ngram_pinned_val_buf=None,
    )
    sched = _sched(
        cached=_cached(
            req_ids=["r"], num_computed=[10], new_block_ids=[[]],
            num_output=[2], all_token_ids={"r": [4, 5]},
        ),
        num_scheduled_tokens={"r": 1},
        scheduled_spec_decode_tokens={"r": [77]},
    )
    corr = r._update_states(sched)
    return r, inp, corr, calls


def test_update_states_async_spec_optimistic_then_corrected(monkeypatch):
    st = _state(output_token_ids=[], block_ids=[[0]], prev_num_draft_len=2)
    inp = FakeInputBatch(req_id_to_index={"r": 0}, req_ids=["r"],
                         num_prompt_tokens={0: 4})
    inp.prev_req_id_to_index = {"r": 0}
    inp.num_tokens_no_spec[0] = 5
    r, inp, corr, calls = _async_spec_runner(monkeypatch, st, inp)
    assert st.output_token_ids == [-1, -1]
    assert st.prev_num_draft_len == 1
    assert r.prev_num_draft_tokens.np.items == {0: 2}
    assert inp.num_tokens_no_spec[0] == 7
    assert calls == ["invalid", "incremental"]
    assert callable(corr)

    r._get_valid_sampled_token_count = lambda: [2]
    corr()
    assert st.num_computed_tokens == 10 - 1
    assert inp.num_computed_tokens_cpu[0] == 10 - 1
    assert inp.num_tokens_no_spec[0] == 6
    assert r.num_tokens_no_spec_gpu[0] == 8


def test_update_states_async_spec_resumed_sets_zero(monkeypatch):
    st = _state(output_token_ids=[], block_ids=[[0]], prev_num_draft_len=3)
    inp = FakeInputBatch()
    r, inp, corr, _ = _async_spec_runner(monkeypatch, st, inp)
    assert st.prev_num_draft_len == 0
    assert st.output_token_ids == [4, 5]
    assert inp.added == [st]


def test_update_states_deferred_correction_guards(monkeypatch):
    st = _state(output_token_ids=[], block_ids=[[0]], prev_num_draft_len=2)
    inp = FakeInputBatch(req_id_to_index={"r": 0}, req_ids=["r"],
                         num_prompt_tokens={0: 4})
    inp.prev_req_id_to_index = {"r": 0}
    inp.num_tokens_no_spec[0] = 5
    r, inp, corr, _ = _async_spec_runner(monkeypatch, st, inp)
    assert callable(corr)

    r._get_valid_sampled_token_count = lambda: []
    corr()

    r._get_valid_sampled_token_count = lambda: [2]
    inp.prev_req_id_to_index = {}
    corr()

    inp.prev_req_id_to_index = {"other": 0}
    corr()

    st.num_computed_tokens = 10
    inp.prev_req_id_to_index = {"r": 0}
    del inp.req_id_to_index["r"]
    corr()
    assert st.num_computed_tokens == 9
    # The persistent update in _update_states already wrote the optimistic
    # count; the closure must not touch it when the request is no longer in
    # req_id_to_index (cur_req_index is None).
    assert inp.num_computed_tokens_cpu == {0: 10}
