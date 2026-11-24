"""Coverage for vllm_fl.worker.model_runner state/buffer helpers (w3b, local CPU)."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.worker import model_runner as mr


def make_runner(**attrs):
    runner = object.__new__(mr.ModelRunnerFL)
    runner.__dict__.update(attrs)
    return runner


def buf(size, dtype=np.int64, value=0, gpu_dtype=torch.int32):
    return SimpleNamespace(
        np=np.full(size, value, dtype=dtype),
        gpu=torch.full((size,), value, dtype=gpu_dtype),
        copy_to_gpu=lambda *a, **k: None,
    )


class FakeInputBatch:
    def __init__(self, **kw):
        self.req_id_to_index = kw.get("req_id_to_index", {})
        self.prev_req_id_to_index = kw.get("prev_req_id_to_index")
        self.req_ids = kw.get("req_ids", [])
        self.removed = []
        self.condensed = 0
        self.refreshed = 0
        self.added = []
        for k, v in kw.items():
            setattr(self, k, v)

    def remove_request(self, rid):
        self.removed.append(rid)

    def condense(self):
        self.condensed += 1

    def refresh_metadata(self):
        self.refreshed += 1

    def add_request(self, req):
        self.added.append(req)

    def update_req_spec_token_ids(self, req, spec):
        pass


@pytest.fixture(autouse=True)
def _cpu_pinned(monkeypatch):
    monkeypatch.setattr(mr, "PIN_MEMORY", False, raising=False)


# --------------------------------------------------------- jit_warmup_registry
def test_jit_registry_setter_and_cached_getter():
    r = make_runner(vllm_config=object())
    sentinel = object()
    r.jit_warmup_registry = sentinel
    assert r.__dict__["_jit_warmup_registry"] is sentinel
    assert r.jit_warmup_registry is sentinel


def test_jit_registry_lazy_create(monkeypatch):
    created = {}

    class FakeRegistry:
        def __init__(self, cfg):
            created["cfg"] = cfg

    fake_mod = types.ModuleType("vllm.model_executor.warmup.jit_warmup")
    fake_mod.JitWarmupRegistry = FakeRegistry
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.warmup.jit_warmup", fake_mod
    )
    cfg = object()
    r = make_runner(vllm_config=cfg)
    reg = r.jit_warmup_registry
    assert isinstance(reg, FakeRegistry)
    assert created["cfg"] is cfg
    assert r.jit_warmup_registry is reg


# -------------------------------------------------------------- _make_buffer
def test_make_buffer_forwards_args(monkeypatch):
    captured = {}

    def fake_cgb(*size, dtype, device, with_numpy):
        captured.update(size=size, dtype=dtype, device=device, with_numpy=with_numpy)
        return "buf"

    monkeypatch.setattr(mr, "CpuGpuBuffer", fake_cgb)
    r = make_runner(device=torch.device("cpu"))
    out = r._make_buffer(4, 3, dtype=torch.int32, numpy=False)
    assert out == "buf"
    assert captured == {
        "size": (4, 3),
        "dtype": torch.int32,
        "device": torch.device("cpu"),
        "with_numpy": False,
    }


# ------------------------------------------- _get_mamba_state_copy_funcs
def test_mamba_copy_funcs_cached():
    r = make_runner(_mamba_state_copy_funcs={"x": 1})
    assert r._get_mamba_state_copy_funcs() == {"x": 1}


def test_mamba_copy_funcs_new_api(monkeypatch):
    monkeypatch.setattr(
        mr.mamba_utils,
        "get_mamba_groups",
        lambda cfg: [SimpleNamespace(mamba_type="m1")],
    )

    def validate(groups, funcs):
        raise AttributeError("absent in older vllm")

    monkeypatch.setattr(mr.mamba_utils, "validate_mamba_state_copy_funcs", validate)
    model = SimpleNamespace(
        get_mamba_state_copy_funcs=lambda types_: {"m1": "fn"}
    )
    r = make_runner(model=model, kv_cache_config=object())
    assert r._get_mamba_state_copy_funcs() == {"m1": "fn"}
    assert r._mamba_state_copy_funcs == {"m1": "fn"}


def test_mamba_copy_funcs_legacy(monkeypatch):
    monkeypatch.setattr(
        mr.mamba_utils,
        "get_mamba_groups",
        lambda cfg: [SimpleNamespace(mamba_type="m1"), SimpleNamespace(mamba_type="m2")],
    )
    monkeypatch.setattr(
        mr.mamba_utils, "validate_mamba_state_copy_funcs", lambda *a: None
    )

    class Legacy:
        def get_mamba_state_copy_func(self):
            return ("tuple",)

    r = make_runner(model=Legacy(), kv_cache_config=object())
    assert r._get_mamba_state_copy_funcs() == {
        "m1": ("tuple",),
        "m2": ("tuple",),
    }


# ------------------------------------------------------------ _get_mamba_bufs
def test_mamba_bufs_requires_align_mode():
    r = make_runner(cache_config=SimpleNamespace(mamba_cache_mode="other"))
    with pytest.raises(AssertionError):
        r._get_mamba_bufs()


def test_mamba_bufs_creates_once(monkeypatch):
    seen = []

    def create(**kw):
        seen.append(kw)
        return "BUFS"

    monkeypatch.setattr(
        mr.mamba_utils, "MambaBuffers", SimpleNamespace(create=create)
    )
    r = make_runner(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        _mamba_bufs=None,
        max_num_reqs=4,
        kv_cache_config=object(),
        device=torch.device("cpu"),
        speculative_config=None,
        model_config=SimpleNamespace(is_hybrid=False),
    )
    r._get_mamba_state_copy_funcs = lambda: {"f": 1}
    r._make_buffer = "mk"
    assert r._get_mamba_bufs() == "BUFS"
    assert r._get_mamba_bufs() == "BUFS"
    assert len(seen) == 1
    assert seen[0]["with_postprocess_align"] is False


def test_mamba_bufs_align_with_spec_hybrid(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        mr.mamba_utils,
        "MambaBuffers",
        SimpleNamespace(create=lambda **kw: seen.update(kw) or "B"),
    )
    r = make_runner(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        _mamba_bufs=None,
        max_num_reqs=2,
        kv_cache_config=object(),
        device=torch.device("cpu"),
        speculative_config=object(),
        model_config=SimpleNamespace(is_hybrid=True),
    )
    r._get_mamba_state_copy_funcs = lambda: {}
    r._make_buffer = None
    assert r._get_mamba_bufs() == "B"
    assert seen["with_postprocess_align"] is True


# --------------------------------------------------------- _init_model_kwargs
def test_init_model_kwargs_not_pooling():
    r = make_runner(is_pooling_model=False)
    assert r._init_model_kwargs() == {}


def test_init_model_kwargs_without_token_types():
    inp = SimpleNamespace(
        num_reqs=1,
        get_pooling_params=lambda: [SimpleNamespace(extra_kwargs=None)],
    )
    r = make_runner(is_pooling_model=True, input_batch=inp)
    assert r._init_model_kwargs() == {}


def test_init_model_kwargs_builds_compressed_ids():
    inp = SimpleNamespace(
        num_reqs=1,
        get_pooling_params=lambda: [
            SimpleNamespace(
                extra_kwargs={"compressed_token_type_ids": 1}
            )
        ],
    )
    r = make_runner(
        is_pooling_model=True,
        input_batch=inp,
        optimistic_seq_lens_cpu=torch.tensor([2, 3]),
        device=torch.device("cpu"),
    )
    out = r._init_model_kwargs()
    assert out["token_type_ids"].tolist() == [0, 1]


# --------------------------------------------------------- _may_reorder_batch
def test_may_reorder_batch_no_groups(monkeypatch):
    called = []
    monkeypatch.setattr(
        mr, "reorder_batch_to_split_decodes_and_prefills", lambda *a, **k: called.append(1)
    )
    r = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        reorder_batch_threshold=4,
    )
    r._may_reorder_batch(SimpleNamespace())
    assert called == []


def test_may_reorder_batch_threshold_none(monkeypatch):
    called = []
    monkeypatch.setattr(
        mr, "reorder_batch_to_split_decodes_and_prefills", lambda *a, **k: called.append(1)
    )
    r = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[1]),
        reorder_batch_threshold=None,
    )
    r._may_reorder_batch(SimpleNamespace())
    assert called == []


def test_may_reorder_batch_delegates(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        mr,
        "reorder_batch_to_split_decodes_and_prefills",
        lambda batch, out, decode_threshold: seen.update(
            batch=batch, out=out, thr=decode_threshold
        ),
    )
    inp = SimpleNamespace()
    so = SimpleNamespace()
    r = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[1]),
        reorder_batch_threshold=8,
        input_batch=inp,
    )
    r._may_reorder_batch(so)
    assert seen == {"batch": inp, "out": so, "thr": 8}


# ------------------------------------------------------- kv zeroer / device
def test_init_kv_zero_meta(monkeypatch):
    captured = {}

    class FakeZeroer:
        def __init__(self, *a, **kw):
            captured["args"] = a
            captured["kw"] = kw

    monkeypatch.setattr(mr, "KVBlockZeroer", FakeZeroer)
    r = make_runner(
        device=torch.device("cpu"),
        _kv_cache_spec_attn_group_iterator=lambda: "iter",
        _kernel_block_sizes=[16],
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_cache_config=SimpleNamespace(num_blocks=9),
        runner_only_attn_layers=["l"],
    )
    r._init_kv_zero_meta()
    assert isinstance(r._kv_block_zeroer, FakeZeroer)
    assert captured["kw"]["num_blocks"] == 9


def test_zero_block_ids_guards_and_forwards():
    r = make_runner()
    r._zero_block_ids([1, 2])  # no zeroer -> no-op
    seen = []
    r._kv_block_zeroer = SimpleNamespace(zero_block_ids=seen.append)
    r._zero_block_ids([3, 4])
    assert seen == [[3, 4]]


def test_init_device_properties(monkeypatch):
    monkeypatch.setattr(mr, "num_compute_units", lambda idx: 11)
    r = make_runner(device=SimpleNamespace(index=5))
    r._init_device_properties()
    assert r.num_sms == 11


def test_sync_device(monkeypatch):
    calls = []
    monkeypatch.setattr(mr, "_accelerator_synchronize", lambda: calls.append(1))
    make_runner()._sync_device()
    assert calls == [1]


# --------------------------------------------------- misc cache/stream hooks
def test_on_request_state_removed_returns_none():
    assert make_runner()._on_request_state_removed("r", None) is None


def test_process_encoder_cache_pops_hashes():
    r = make_runner(encoder_cache={"a": 1, "b": 2})
    r._process_encoder_cache_scheduler_output(
        SimpleNamespace(free_encoder_mm_hashes=["a", "missing"])
    )
    assert r.encoder_cache == {"b": 2}


def test_async_output_copy_stream_existing():
    stream = object()
    r = make_runner(async_output_copy_stream=stream)
    assert r._get_or_create_async_output_copy_stream() is stream


def test_async_output_copy_stream_creates(monkeypatch):
    stream = object()
    fake_platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(Stream=lambda: stream)
    )
    monkeypatch.setattr(mr, "current_platform", fake_platform)
    r = make_runner(async_output_copy_stream=None)
    assert r._get_or_create_async_output_copy_stream() is stream
    assert r.async_output_copy_stream is stream


def test_cache_encoder_output_and_lookup():
    saved = []
    r = make_runner(
        encoder_cache={},
        maybe_save_ec_to_connector=lambda cache, h: saved.append(h),
    )
    t = torch.zeros(2)
    r._cache_encoder_output("h", t, None, [])
    assert r.encoder_cache["h"] is t
    assert saved == ["h"]
    assert r._get_encoder_output_from_cache("h") is t
    assert r._get_encoder_output_from_cache("nope") is None


# ------------------------------------------------------------- _execute_mm_encoder
def test_execute_mm_encoder_empty():
    r = make_runner()
    assert r._execute_mm_encoder(SimpleNamespace(scheduled_encoder_inputs={})) == []


def test_execute_mm_encoder_prompt_embeds_only(monkeypatch):
    monkeypatch.setattr(
        mr,
        "async_tensor_h2d",
        lambda t, device=None: torch.as_tensor(t),
    )
    embeds = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    item = SimpleNamespace(data=embeds)
    r = make_runner(device=torch.device("cpu"))
    r._batch_mm_inputs_from_scheduler = lambda so: (
        ["h"],
        [("prompt_embeds", {"embedding": item})],
        [("r", object())],
    )
    r._cache_encoder_output = lambda h, out, m, f: r.encoder_cache.__setitem__(h, out)
    r.encoder_cache = {}
    so = SimpleNamespace(ec_manager_metadata=None, free_encoder_mm_hashes=[])
    assert r._execute_mm_encoder(so) == []
    assert "h" in r.encoder_cache


# ------------------------------------------------------ _update_states basics
def _sched(**kw):
    cached_reqs = SimpleNamespace(
        resumed_req_ids=kw.get("resumed_req_ids", set()),
        req_ids=kw.get("cached_req_ids", []),
        num_computed_tokens=kw.get("num_computed_tokens", []),
        new_block_ids=kw.get("cached_new_block_ids", []),
        num_output_tokens=kw.get("num_output_tokens", []),
    )
    return SimpleNamespace(
        finished_req_ids=kw.get("finished_req_ids", []),
        new_block_ids_to_zero=kw.get("new_block_ids_to_zero", []),
        kv_cache_block_copies=kw.get("kv_cache_block_copies"),
        num_scheduled_tokens=kw.get("num_scheduled_tokens", {}),
        scheduled_cached_reqs=cached_reqs,
        scheduled_new_reqs=kw.get("scheduled_new_reqs", []),
        scheduled_spec_decode_tokens=kw.get("scheduled_spec_decode_tokens", {}),
        free_encoder_mm_hashes=kw.get("free_encoder_mm_hashes", []),
    )


def _update_state_runner(monkeypatch, **over):
    monkeypatch.setattr(
        mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    records = []

    class LIR:
        def on_requests_finished(self, ids):
            records.append(("finished", list(ids)))

        def register_request(self, rid, params):
            records.append(("register", rid))

    base = dict(
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=LIR(),
        input_batch=FakeInputBatch(),
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        device=torch.device("cpu"),
        encoder_cache={},
    )
    base.update(over)
    r = make_runner(**base)
    r._may_reorder_batch = lambda so: records.append(("reorder", None))
    return r, records


def test_update_states_empty_no_new_requests(monkeypatch):
    r, records = _update_state_runner(monkeypatch)
    assert r._update_states(_sched()) is None
    assert r.input_batch.condensed == 1
    assert r.input_batch.refreshed == 1
    assert ("finished", []) in records
    assert ("reorder", None) in records


def test_update_states_finished_requests(monkeypatch):
    r, records = _update_state_runner(
        monkeypatch,
        requests={"a": SimpleNamespace()},
        num_prompt_logprobs={"a": 3},
    )
    r._update_states(_sched(finished_req_ids=["a"]))
    assert r.requests == {}
    assert r.num_prompt_logprobs == {}
    assert r.input_batch.removed == ["a"]
    assert ("finished", ["a"]) in records


def test_update_states_zeroes_and_copies_blocks(monkeypatch):
    copied = []
    monkeypatch.setattr(
        mr,
        "copy_kv_cache_blocks_inplace",
        lambda caches, n, copies: copied.append((n, copies)),
    )
    r, _ = _update_state_runner(
        monkeypatch,
        kv_caches=["c"],
        kv_cache_config=SimpleNamespace(num_blocks=8),
    )
    zeroed = []
    r._zero_block_ids = zeroed.append
    r._update_states(
        _sched(
            new_block_ids_to_zero=[5, 6],
            kv_cache_block_copies=[("a", "b")],
        )
    )
    assert zeroed == [[5, 6]]
    assert copied == [(8, [("a", "b")])]


# ------------------------------------- _update_states_after_model_execute
def test_after_model_execute_early_returns():
    r = make_runner(speculative_config=None)
    r._update_states_after_model_execute(torch.zeros(1, 1), SimpleNamespace())
    r2 = make_runner(
        speculative_config=object(), model_config=SimpleNamespace(is_hybrid=False)
    )
    r2._update_states_after_model_execute(torch.zeros(1, 1), SimpleNamespace())


def test_after_model_execute_align_path(monkeypatch):
    called = []
    monkeypatch.setattr(
        mr.mamba_utils,
        "postprocess_mamba_align_gpu",
        lambda **kw: called.append(kw["num_reqs"]),
        raising=False,
    )
    event = SimpleNamespace(record=lambda: None)
    r = make_runner(
        speculative_config=object(),
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        num_accepted_tokens=SimpleNamespace(
            gpu=torch.zeros(3, dtype=torch.int64)
        ),
        input_batch=SimpleNamespace(num_accepted_tokens_cpu_tensor=torch.zeros(3)),
        kv_cache_config=object(),
        compilation_config=SimpleNamespace(static_forward_context={}),
        num_accepted_tokens_event=event,
    )
    r._get_mamba_bufs = lambda: "bufs"
    r._get_mamba_state_copy_funcs = lambda: {}
    out = torch.tensor([[1, 2, -1], [3, -1, -1]])
    r._update_states_after_model_execute(out, SimpleNamespace())
    assert called == [2]
    assert r.num_accepted_tokens.gpu[:2].tolist() == [2, 1]


def test_after_model_execute_copies_cpu(monkeypatch):
    event = SimpleNamespace(record=lambda: None)
    cpu = torch.zeros((3,), dtype=torch.int64)
    r = make_runner(
        speculative_config=object(),
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(mamba_cache_mode="other"),
        num_accepted_tokens=SimpleNamespace(
            gpu=torch.tensor([2, 1, 1], dtype=torch.int64)
        ),
        input_batch=SimpleNamespace(num_accepted_tokens_cpu_tensor=torch.zeros(3, dtype=torch.int64)),
        num_accepted_tokens_event=event,
    )
    r._update_states_after_model_execute(
        torch.tensor([[0, 0], [0, -1]]), SimpleNamespace()
    )
    assert r.input_batch.num_accepted_tokens_cpu_tensor[:2].tolist() == [2, 1]


# --------------------------------------------------- _update_streaming_request
def test_update_streaming_request(monkeypatch):
    monkeypatch.setattr(
        mr, "length_from_prompt_token_ids_or_embeds", lambda ids, e: len(ids)
    )
    registered = []
    state = SimpleNamespace(output_token_ids=[9, 9])
    inp = FakeInputBatch()
    new_data = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        mm_features=["f"],
        prompt_embeds=None,
        sampling_params="sp",
        pooling_params="pp",
        block_ids=[[0]],
        num_computed_tokens=3,
    )
    r = make_runner(
        input_batch=inp,
        requests={"r": state},
        uses_mrope=False,
        late_interaction_runner=SimpleNamespace(
            register_request=lambda rid, p: registered.append((rid, p))
        ),
    )
    out = r._update_streaming_request("r", new_data)
    assert out is state
    assert state.prompt_token_ids == [1, 2, 3]
    assert state.output_token_ids == []
    assert state.num_prompt_tokens == 3
    assert inp.removed == ["r"]
    assert registered == [("r", "pp")]


# ----------------------------------------------------- _init_mrope_positions
class _MropeModel:
    def __init__(self, ret):
        self.ret = ret
        self.calls = []

    def get_mrope_input_positions(self, tokens, feats):
        self.calls.append((tokens, feats))
        return self.ret


def test_init_mrope_positions_from_token_ids(monkeypatch):
    monkeypatch.setattr(mr, "supports_mrope", lambda m: True)
    pos = (torch.arange(6).reshape(3, 2), 0)
    model = _MropeModel(pos)
    img = SimpleNamespace(modality="image")
    pe = SimpleNamespace(modality="prompt_embeds")
    r = make_runner(model=model)
    r.get_model = lambda: model
    state = SimpleNamespace(
        mm_features=[img, pe], prompt_token_ids=[7, 8], prompt_embeds=None
    )
    r._init_mrope_positions(state)
    assert model.calls == [([7, 8], [img])]
    assert state.mrope_positions is pos[0]


def test_init_mrope_positions_from_embeds(monkeypatch):
    monkeypatch.setattr(mr, "supports_mrope", lambda m: True)
    model = _MropeModel((torch.zeros(3, 2), 1))
    r = make_runner(model=model)
    r.get_model = lambda: model
    state = SimpleNamespace(
        mm_features=[], prompt_token_ids=None, prompt_embeds=torch.zeros(3, 4)
    )
    r._init_mrope_positions(state)
    assert model.calls[0][0] == [0, 1, 2]


def test_init_mrope_positions_requires_input(monkeypatch):
    monkeypatch.setattr(mr, "supports_mrope", lambda m: True)
    r = make_runner()
    r.get_model = lambda: SimpleNamespace()
    state = SimpleNamespace(mm_features=[], prompt_token_ids=None, prompt_embeds=None)
    with pytest.raises(ValueError):
        r._init_mrope_positions(state)


# --------------------------------------------------------- _prepare_input_ids
def test_prepare_input_ids_normal_path():
    inp = SimpleNamespace(prev_sampled_token_ids=None)
    calls = []
    r = make_runner(
        _pp_recv_work=None,
        input_batch=inp,
        input_ids=SimpleNamespace(copy_to_gpu=lambda n: calls.append(n)),
        enable_prompt_embeds=False,
    )
    r._prepare_input_ids(SimpleNamespace(), 2, 5, np.array([2, 5]))
    assert calls == [5]


def test_prepare_input_ids_async_common_case():
    prev_sampled = torch.tensor([[7], [8]], dtype=torch.int32)
    inp = SimpleNamespace(
        prev_sampled_token_ids=prev_sampled,
        req_ids=["a"],
    )
    gpu = torch.zeros(4, dtype=torch.int32)
    r = make_runner(
        _pp_recv_work=None,
        input_batch=inp,
        prev_positions=buf(4, np.int64),
        input_ids=SimpleNamespace(
            gpu=gpu, copy_to_gpu=lambda n: None
        ),
        enable_prompt_embeds=False,
        prev_num_spec_tokens=1,
        _draft_token_ids=None,
    )
    r.prev_positions.np[:1] = np.array([0])
    r._prepare_input_ids(
        SimpleNamespace(scheduled_spec_decode_tokens={}), 1, 2, np.array([1, 2])
    )
    assert gpu[0].item() == 7


def test_prepare_input_ids_async_no_common_requests():
    prev_sampled = torch.tensor([[7]], dtype=torch.int32)
    inp = SimpleNamespace(prev_sampled_token_ids=prev_sampled, req_ids=["a"])
    copies = []
    r = make_runner(
        _pp_recv_work=None,
        input_batch=inp,
        prev_positions=buf(4, np.int64),
        input_ids=SimpleNamespace(
            gpu=torch.zeros(4), copy_to_gpu=lambda n: copies.append(n)
        ),
        enable_prompt_embeds=False,
        prev_num_spec_tokens=1,
        _draft_token_ids=None,
    )
    r.prev_positions.np[:1] = np.array([-1])
    r._prepare_input_ids(
        SimpleNamespace(scheduled_spec_decode_tokens={}), 1, 2, np.array([2])
    )
    assert copies == [2]
