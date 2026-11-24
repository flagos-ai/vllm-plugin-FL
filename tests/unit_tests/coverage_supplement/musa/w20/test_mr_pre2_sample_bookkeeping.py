# SPDX-License-Identifier: Apache-2.0
"""Extra branch coverage for ModelRunnerFL sampling/bookkeeping.

Complements ``test_mr_sample_bookkeeping.py`` (w3c) which covers the plain
sample, the spec-decode refresh path, sync bookkeeping, spec-decode output
parsing, async deferral and async NaN counting on device. Added here:

* ``_sample`` async path where ``_draft_token_req_ids`` is unset
* ``_sample`` CPU-generator filter dropping non-CPU generators
* ``_bookkeeping_sync`` sync NaN path (real ``_get_nans_in_logits``)
* ``_bookkeeping_sync`` async NaN path with ``logits is None``
* ``_bookkeeping_sync`` CPU vs accelerator generator rewind selection
* ``_bookkeeping_sync`` missing generator object
* ``_bookkeeping_sync`` routed-experts D2H copies
* ``_bookkeeping_sync`` single-token logprob ``tolists``
* ``_bookkeeping_sync`` async reuse of ``prev_sampled_token_ids``
* ``_bookkeeping_sync`` async discarded request yields no token
* ``_bookkeeping_sync`` max-model-length guard
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr

ModelRunnerFL = mr.ModelRunnerFL


def _runner():
    return object.__new__(ModelRunnerFL)


# ---------------------------------------------------------------------------
# _sample extras
# ---------------------------------------------------------------------------


def test_sample_async_without_draft_req_ids_skips_refresh():
    runner = _runner()
    meta = SimpleNamespace(generators={})
    touched = []
    calls = []
    runner.input_batch = SimpleNamespace(
        sampling_metadata=meta,
        update_async_output_token_ids=lambda: None,
        update_async_spec_token_ids=lambda ids: touched.append(ids),
    )
    runner.use_async_scheduling = True
    runner._draft_token_req_ids = None
    runner._get_draft_token_ids_cpu = lambda: calls.append(True) or ([], [])
    runner._get_spec_decode_draft_probs = lambda spec: "P"
    runner.rejection_sampler = lambda spec, probs, logits, meta: "REJ"
    assert runner._sample(None, SimpleNamespace()) == "REJ"
    # No request-id mapping is known, so the draft token refresh is skipped.
    assert calls == []
    assert touched == []


def test_sample_only_caches_cpu_generator_state():
    runner = _runner()
    cpu = SimpleNamespace(device=SimpleNamespace(type="cpu"), get_state=lambda: "C")
    accel = SimpleNamespace(device=SimpleNamespace(type="cuda"), get_state=lambda: "X")
    meta = SimpleNamespace(generators={0: cpu, 1: accel})
    runner.input_batch = SimpleNamespace(
        sampling_metadata=meta, update_async_output_token_ids=lambda: None
    )
    runner.sampler = lambda logits, sampling_metadata: "OUT"
    assert runner._sample(None, None) == "OUT"
    assert runner._cpu_generator_states == {0: "C"}


# ---------------------------------------------------------------------------
# _bookkeeping_sync harness
# ---------------------------------------------------------------------------


def _book_runner(use_async, sample_shape):
    runner = _runner()
    runner.use_async_scheduling = use_async
    runner.routed_experts_initialized = False
    runner._cpu_generator_states = {}
    runner.max_model_len = 128
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        generators={},
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        vocab_size=16,
        num_tokens_no_spec=np.array([5, 5], dtype=np.int32),
        token_ids_cpu=np.full((2, 8), -9, dtype=np.int32),
        is_token_ids=np.zeros((2, 8), dtype=bool),
        prev_sampled_token_ids=None,
        sampling_metadata=SimpleNamespace(output_token_ids=None),
    )
    runner.discard_request_mask = SimpleNamespace(np=np.zeros(2, dtype=bool))
    runner.requests = {
        "r0": SimpleNamespace(output_token_ids=[]),
        "r1": SimpleNamespace(output_token_ids=[]),
    }
    runner._to_list = lambda t: t.tolist()
    runner._get_prompt_logprobs_dict = lambda hs, d: {"r0": "PLP"}
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.zeros(*sample_shape, dtype=torch.int32),
        logprobs_tensors=None,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 2, "r1": 2}
    )
    return runner, sampler_output, scheduler_output


def _no_nans(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False, raising=False)
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: contextlib.nullcontext())


# ---------------------------------------------------------------------------
# NaN accounting
# ---------------------------------------------------------------------------


def test_bookkeeping_sync_computes_nans_on_host(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", True, raising=False)
    monkeypatch.setattr(mr.envs, "VLLM_RAISE_ON_LOGIT_NANS", False, raising=False)
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: contextlib.nullcontext())
    monkeypatch.setattr(
        mr, "count_nans_per_row", lambda logits: torch.tensor([2, 0]), raising=False
    )
    monkeypatch.setattr(
        mr,
        "nans_to_dict",
        lambda counts, mapping: {"r0": int(counts[0])},
        raising=False,
    )
    runner, sampler_output, sched = _book_runner(False, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    out = runner._bookkeeping_sync(sched, sampler_output, torch.zeros(4, 1), torch.zeros(4, 3), 4)
    assert out[0] == {"r0": 2}
    assert out[1] is None


def test_bookkeeping_async_nan_no_logits_keeps_none(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", True, raising=False)

    def _boom(_logits):
        raise AssertionError("count_nans_per_row must not run when logits is None")

    monkeypatch.setattr(mr, "count_nans_per_row", _boom, raising=False)
    runner, sampler_output, sched = _book_runner(True, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert out[0] == {}
    assert out[1] is None


# ---------------------------------------------------------------------------
# generator rewind selection
# ---------------------------------------------------------------------------


def test_bookkeeping_rewinds_accelerator_generator_offset(monkeypatch):
    _no_nans(monkeypatch)
    runner, sampler_output, sched = _book_runner(False, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False]))
    offsets = []

    # Mirror the source contract: gen.set_offset(gen.get_offset() - 4),
    # i.e. set_offset takes exactly one positional argument.
    def get_offset():
        return 8

    def set_offset(offset):
        offsets.append(offset)

    runner.input_batch.generators = {
        0: SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            get_offset=get_offset,
            set_offset=set_offset,
        )
    }
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert offsets == [4]
    assert out[3] == [[], [8]]


def test_bookkeeping_missing_generator_is_ignored(monkeypatch):
    _no_nans(monkeypatch)
    runner, sampler_output, sched = _book_runner(False, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False]))
    runner.input_batch.generators = {}
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert out[3] == [[], [8]]


# ---------------------------------------------------------------------------
# routed experts D2H copy
# ---------------------------------------------------------------------------


def test_bookkeeping_sync_copies_routed_experts(monkeypatch):
    _no_nans(monkeypatch)
    runner, sampler_output, sched = _book_runner(False, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    runner.routed_experts_initialized = True
    runner.routed_experts_capturer = SimpleNamespace(
        get_device_buffer=lambda: torch.arange(8, dtype=torch.int64).reshape(4, 2)
    )
    runner.routed_experts_cpu = torch.zeros(4, 2, dtype=torch.int64)
    runner.routed_experts_slot_mapping_device = torch.tensor(
        [10, 20, 30, 40], dtype=torch.int64
    )
    runner.routed_experts_slot_mapping_cpu = torch.zeros(4, dtype=torch.int64)
    runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert runner.routed_experts_cpu.tolist() == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]
    assert runner.routed_experts_slot_mapping_cpu.tolist() == [10, 20, 30, 40]


# ---------------------------------------------------------------------------
# single-token logprobs
# ---------------------------------------------------------------------------


def test_bookkeeping_sync_single_token_logprobs_tolists(monkeypatch):
    _no_nans(monkeypatch)
    runner, sampler_output, sched = _book_runner(False, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    sampler_output.logprobs_tensors = SimpleNamespace(
        tolists=lambda: [["a", "b"], ["c", "d"]]
    )
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert out[2] == [["a", "b"], ["c", "d"]]


# ---------------------------------------------------------------------------
# async reuse/deferred paths
# ---------------------------------------------------------------------------


def test_bookkeeping_async_reuses_prev_sampled_token_ids(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False, raising=False)
    runner, sampler_output, sched = _book_runner(True, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    prev = torch.tensor([[3], [4]], dtype=torch.int32)
    runner.input_batch.prev_sampled_token_ids = prev
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    # The cached tensor is kept as-is rather than being replaced this step.
    assert runner.input_batch.prev_sampled_token_ids is prev
    assert out[3] == []


def test_bookkeeping_async_discards_request_token(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False, raising=False)
    runner, sampler_output, sched = _book_runner(True, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False]))
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert out[7] == [0]
    # Discarded request writes nothing, kept request records a pending -1.
    assert runner.input_batch.token_ids_cpu[0, 5] == -9
    assert runner.input_batch.token_ids_cpu[1, 5] == -1
    assert runner.input_batch.num_tokens_no_spec.tolist() == [5, 6]


def test_bookkeeping_rejects_token_past_max_model_len(monkeypatch):
    _no_nans(monkeypatch)
    runner, sampler_output, sched = _book_runner(False, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    runner.max_model_len = 3
    with pytest.raises(AssertionError, match="max model length"):
        runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
