"""Residual coverage for post-forward/streaming/mrope/mm helpers (w19, local CPU).

Covers the branches of `_update_states_after_model_execute`,
`_update_streaming_request`, `_init_mrope_positions`, `_extract_mm_kwargs`,
`_dummy_mm_kwargs` left untouched by w3b.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.worker import model_runner as mr


@pytest.fixture(autouse=True)
def _cpu_pinned(monkeypatch):
    monkeypatch.setattr(mr, "PIN_MEMORY", False, raising=False)


def make_runner(**attrs):
    r = object.__new__(mr.ModelRunnerFL)
    r.__dict__.update(attrs)
    return r


class _MropeModel:
    def __init__(self, ret):
        self.ret = ret
        self.calls = []

    def get_mrope_input_positions(self, tokens, feats):
        self.calls.append((tokens, feats))
        return self.ret


# ------------------------------------------- _update_states_after_model_execute
def test_after_model_execute_align_updates_only_num_reqs(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mr.mamba_utils,
        "postprocess_mamba_align_gpu",
        lambda **kw: seen.append(kw["num_reqs"]),
        raising=False,
    )
    recorded = []
    r = make_runner(
        speculative_config=object(),
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        num_accepted_tokens=SimpleNamespace(
            gpu=torch.zeros(5, dtype=torch.int64)
        ),
        input_batch=SimpleNamespace(num_accepted_tokens_cpu_tensor=torch.zeros(5)),
        kv_cache_config=object(),
        compilation_config=SimpleNamespace(static_forward_context={}),
        num_accepted_tokens_event=SimpleNamespace(record=lambda: recorded.append(1)),
    )
    r._get_mamba_bufs = lambda: "bufs"
    r._get_mamba_state_copy_funcs = lambda: {}
    out = torch.tensor([[1, 2, 3], [4, -1, -1]])
    r._update_states_after_model_execute(out, SimpleNamespace())
    assert seen == [2]
    assert recorded == [1]
    assert r.num_accepted_tokens.gpu[:2].tolist() == [3, 1]
    assert r.num_accepted_tokens.gpu[2:].tolist() == [0, 0, 0]


def test_after_model_execute_non_align_copies_slice(monkeypatch):
    recorded = []
    r = make_runner(
        speculative_config=object(),
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(mamba_cache_mode="other"),
        num_accepted_tokens=SimpleNamespace(
            gpu=torch.tensor([2, 0, 3], dtype=torch.int64)
        ),
        input_batch=SimpleNamespace(
            num_accepted_tokens_cpu_tensor=torch.full((4,), 9, dtype=torch.int64)
        ),
        num_accepted_tokens_event=SimpleNamespace(record=lambda: recorded.append(1)),
    )
    r._update_states_after_model_execute(
        torch.tensor([[5, 0], [5, -1], [5, 5]]), SimpleNamespace()
    )
    assert recorded == [1]
    assert r.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [2, 1, 2, 9]


def test_after_model_execute_returns_without_spec_hybrid(monkeypatch):
    monkeypatch.setattr(
        mr.mamba_utils,
        "postprocess_mamba_align_gpu",
        lambda **kw: pytest.fail("must not run"),
        raising=False,
    )
    # No model execution happens when speculative decoding or hybrid mode is off.
    r = make_runner(speculative_config=None,
                    model_config=SimpleNamespace(is_hybrid=True))
    r2 = make_runner(speculative_config=object(),
                     model_config=SimpleNamespace(is_hybrid=False))
    r._update_states_after_model_execute(torch.tensor([[5]]), SimpleNamespace())
    r2._update_states_after_model_execute(torch.tensor([[5]]), SimpleNamespace())


# --------------------------------------------------- _update_streaming_request
def test_update_streaming_request_uses_mrope(monkeypatch):
    monkeypatch.setattr(
        mr, "length_from_prompt_token_ids_or_embeds", lambda ids, e: len(ids)
    )
    mrope = []
    removed = []
    registered = []
    state = SimpleNamespace(
        output_token_ids=[9, 9],
        prompt_token_ids=[0],
        mm_features=None,
        prompt_embeds=None,
        sampling_params=None,
        pooling_params=None,
        block_ids=None,
        num_computed_tokens=0,
        num_prompt_tokens=0,
    )
    inp = SimpleNamespace(remove_request=removed.append)
    r = make_runner(
        input_batch=inp,
        requests={"r": state},
        uses_mrope=True,
        late_interaction_runner=SimpleNamespace(
            register_request=lambda rid, p: registered.append((rid, p))
        ),
    )
    r._init_mrope_positions = lambda s: mrope.append(s)
    new_data = SimpleNamespace(
        prompt_token_ids=[1, 2, 3, 4],
        mm_features=["f"],
        prompt_embeds=None,
        sampling_params="sp",
        pooling_params="pp",
        block_ids=[[3]],
        num_computed_tokens=4,
    )
    out = r._update_streaming_request("r", new_data)
    assert out is state
    assert removed == ["r"]
    assert state.prompt_token_ids == [1, 2, 3, 4]
    assert state.mm_features == ["f"]
    assert state.sampling_params == "sp"
    assert state.block_ids == [[3]]
    assert state.num_prompt_tokens == 4
    assert state.output_token_ids == []
    assert mrope == [state]
    assert registered == [("r", "pp")]


# ----------------------------------------------------- _init_mrope_positions
def test_init_mrope_positions_requires_support(monkeypatch):
    monkeypatch.setattr(mr, "supports_mrope", lambda m: False)
    r = make_runner()
    r.get_model = lambda: SimpleNamespace()
    state = SimpleNamespace(mm_features=[], prompt_token_ids=[1], prompt_embeds=None)
    with pytest.raises(AssertionError):
        r._init_mrope_positions(state)


def test_init_mrope_positions_filters_prompt_embeds_feature(monkeypatch):
    monkeypatch.setattr(mr, "supports_mrope", lambda m: True)
    model = _MropeModel((torch.zeros(2, 2), 5))
    r = make_runner()
    r.get_model = lambda: model
    img = SimpleNamespace(modality="image")
    pe = SimpleNamespace(modality="prompt_embeds")
    state = SimpleNamespace(
        mm_features=[img, pe],
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(2, 4),
    )
    r._init_mrope_positions(state)
    assert model.calls == [([0, 1], [img])]
    assert state.mrope_position_delta == 5


# ------------------------------------------------------- _extract_mm_kwargs
def test_extract_mm_kwargs_empty_scheduled_new_reqs(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mr,
        "group_and_batch_mm_kwargs",
        lambda items, device, pin_memory: seen.append(list(items)) or iter(()),
    )
    r = make_runner(is_multimodal_raw_input_only_model=True,
                    device=torch.device("cpu"))
    assert r._extract_mm_kwargs(SimpleNamespace(scheduled_new_reqs=[])) == {}
    assert seen == [[]]


def test_extract_mm_kwargs_skips_empty_data_and_merges(monkeypatch):
    captured = []

    def fake_group(items, device, pin_memory):
        captured.append(list(items))
        return iter([("image", 1, {"a": 1}), ("video", 1, {"b": 2})])

    monkeypatch.setattr(mr, "group_and_batch_mm_kwargs", fake_group)
    good = SimpleNamespace(modality="image", data={"x": 1})
    empty = SimpleNamespace(modality="video", data=None)
    r = make_runner(is_multimodal_raw_input_only_model=True,
                    device=torch.device("cpu"))
    out = r._extract_mm_kwargs(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(mm_features=[good, empty]),
                SimpleNamespace(mm_features=[good]),
            ]
        )
    )
    assert out == {"a": 1, "b": 2}
    assert captured == [[("image", {"x": 1}), ("image", {"x": 1})]]


# --------------------------------------------------------- _dummy_mm_kwargs
def test_dummy_mm_kwargs_requires_budget():
    r = make_runner(is_multimodal_raw_input_only_model=True, mm_budget=None)
    with pytest.raises(AssertionError):
        r._dummy_mm_kwargs(2)
