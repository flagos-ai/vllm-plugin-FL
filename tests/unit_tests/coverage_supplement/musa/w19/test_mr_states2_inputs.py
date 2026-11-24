"""Residual coverage for input-id/encoder helper branches (w19, local CPU).

Targets `_get_cumsum_and_arange` default dtype, `_compute_prev_positions`
subsetting, the async/prefetch/scatter branches of `_prepare_input_ids`, and the
cudagraph `max_source_positions` override in `_get_encoder_seq_lens`.
"""

from types import SimpleNamespace

import numpy as np
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


class _Buf:
    def __init__(self, np_arr, gpu=None):
        self.np = np_arr
        self.gpu = gpu if gpu is not None else torch.zeros(len(np_arr), dtype=torch.int32)
        self.copies = []

    def copy_to_gpu(self, n=None):
        self.copies.append(n)


# ---------------------------------------------------- _get_cumsum_and_arange
def test_get_cumsum_and_arange_default_dtype_is_int64():
    r = make_runner(arange_np=np.arange(10))
    out = np.empty(10, dtype=np.int64)
    cu = r._get_cumsum_and_arange(np.array([3, 2, 4]), out)
    assert cu.dtype == np.int64
    assert cu.tolist() == [3, 5, 9]
    assert out[:9].tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]


# --------------------------------------------------- _compute_prev_positions
def test_compute_prev_positions_updates_only_prefix():
    prev = _Buf(np.full(4, 9, dtype=np.int64))
    inp = SimpleNamespace(
        prev_req_id_to_index={"a": 2, "b": 0},
        req_ids=["a", "b", "c"],
    )
    r = make_runner(input_batch=inp, prev_positions=prev)
    r._compute_prev_positions(2)
    assert prev.np.tolist() == [2, 0, 9, 9]


def test_compute_prev_positions_missing_key_fills_minus_one():
    prev = _Buf(np.zeros(2, dtype=np.int64))
    inp = SimpleNamespace(
        prev_req_id_to_index={"a": 1}, req_ids=["a", "b"]
    )
    r = make_runner(input_batch=inp, prev_positions=prev)
    r._compute_prev_positions(2)
    assert prev.np.tolist() == [1, -1]


# ----------------------------------------------------- _prepare_input_ids
class _Work:
    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


def _input_ids(size=8):
    return _Buf(np.zeros(size, dtype=np.int64), torch.zeros(size, dtype=torch.int32))


def test_prepare_input_ids_syncs_pp_recv_work():
    work = _Work()
    ids = _input_ids()
    r = make_runner(
        _pp_recv_work=work,
        input_batch=SimpleNamespace(prev_sampled_token_ids=None),
        input_ids=ids,
        enable_prompt_embeds=False,
    )
    r._prepare_input_ids(SimpleNamespace(), 2, 5, np.array([2, 3]))
    assert work.waits == 1
    assert r._pp_recv_work is None
    assert ids.copies == [5]


def test_prepare_input_ids_normal_path_copies_prompt_embeds():
    ids = _input_ids()
    embeds = _Buf(np.zeros(8, dtype=np.int64))
    is_ids = _Buf(np.zeros(8, dtype=np.int64))
    r = make_runner(
        _pp_recv_work=None,
        input_batch=SimpleNamespace(prev_sampled_token_ids=None),
        input_ids=ids,
        inputs_embeds=embeds,
        is_token_ids=is_ids,
        enable_prompt_embeds=True,
    )
    r._prepare_input_ids(SimpleNamespace(), 2, 4, np.array([2, 2]))
    assert ids.copies == [4]
    assert embeds.copies == [4]
    assert is_ids.copies == [4]


def test_prepare_input_ids_async_scatter_after_reorder():
    ids = _input_ids(4)
    inp = SimpleNamespace(
        prev_sampled_token_ids=torch.tensor([[7], [8]], dtype=torch.int32),
        req_ids=["a", "b"],
    )
    prev_positions = SimpleNamespace(np=np.array([0, -1]))
    r = make_runner(
        _pp_recv_work=None,
        input_batch=inp,
        input_ids=ids,
        prev_positions=prev_positions,
        prev_num_spec_tokens=1,
        _draft_token_ids=None,
        enable_prompt_embeds=False,
        device=torch.device("cpu"),
    )
    r._prepare_input_ids(
        SimpleNamespace(scheduled_spec_decode_tokens={}),
        2,
        5,
        np.array([2, 5]),
    )
    assert ids.copies == [5]
    assert ids.gpu.tolist() == [0, 7, 0, 0]


def test_prepare_input_ids_async_scatter_sampled_and_draft():
    ids = _input_ids(4)
    inp = SimpleNamespace(
        prev_sampled_token_ids=torch.tensor([[5]], dtype=torch.int32),
        req_ids=["a"],
    )
    prev_positions = SimpleNamespace(np=np.array([0]))
    r = make_runner(
        _pp_recv_work=None,
        input_batch=inp,
        input_ids=ids,
        prev_positions=prev_positions,
        prev_num_spec_tokens=1,
        _draft_token_ids=torch.tensor([[7, 8]], dtype=torch.int32),
        enable_prompt_embeds=False,
        device=torch.device("cpu"),
    )
    r._prepare_input_ids(
        SimpleNamespace(scheduled_spec_decode_tokens={"a": [9, 9]}),
        1,
        3,
        np.array([3]),
    )
    assert ids.copies == []
    assert ids.gpu.tolist() == [5, 7, 8, 0]


def test_prepare_input_ids_async_refreshes_prompt_embeds_flags():
    ids = _input_ids(4)
    is_ids = _Buf(np.zeros(4, dtype=np.int64))
    embeds = _Buf(np.zeros(4, dtype=np.int64))
    inp = SimpleNamespace(
        prev_sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        req_ids=["a"],
    )
    prev_positions = SimpleNamespace(np=np.array([-1]))
    r = make_runner(
        _pp_recv_work=None,
        input_batch=inp,
        input_ids=ids,
        inputs_embeds=embeds,
        is_token_ids=is_ids,
        prev_positions=prev_positions,
        prev_num_spec_tokens=1,
        _draft_token_ids=None,
        enable_prompt_embeds=True,
    )
    r._prepare_input_ids(
        SimpleNamespace(scheduled_spec_decode_tokens={}), 1, 3, np.array([3])
    )
    assert is_ids.copies == [3]
    assert ids.copies == [3]
    assert embeds.copies == [3]


# ----------------------------------------------------- _get_encoder_seq_lens
def _cross_spec():
    return object.__new__(mr.CrossAttentionSpec)


def _encoder_runner(requests, lens, hf_config, max_encoder_len=99):
    inp = SimpleNamespace(req_id_to_index={rid: i for i, rid in enumerate(requests)})
    return make_runner(
        encoder_seq_lens=lens,
        requests=requests,
        input_batch=inp,
        max_encoder_len=max_encoder_len,
        model_config=SimpleNamespace(hf_config=hf_config),
    )


def test_get_encoder_seq_lens_cudagraph_uses_max_source_positions():
    lens = _Buf(np.zeros(4, dtype=np.int32))
    r = _encoder_runner({}, lens, SimpleNamespace(max_source_positions=64))
    gpu, cpu = r._get_encoder_seq_lens({}, _cross_spec(), 2, for_cudagraph_capture=True)
    assert cpu.tolist() == [64, 64]
    assert gpu.tolist()[:2] == [0, 0]
    assert lens.copies == [2]


def test_get_encoder_seq_lens_sums_all_features():
    lens = _Buf(np.zeros(4, dtype=np.int32))
    feats = [
        SimpleNamespace(mm_position=SimpleNamespace(length=2)),
        SimpleNamespace(mm_position=SimpleNamespace(length=3)),
    ]
    requests = {"a": SimpleNamespace(mm_features=feats)}
    r = _encoder_runner(requests, lens, SimpleNamespace())
    _, cpu = r._get_encoder_seq_lens({"a": 1}, _cross_spec(), 2)
    assert cpu.tolist()[:2] == [5, 0]
    assert lens.copies == [2]
