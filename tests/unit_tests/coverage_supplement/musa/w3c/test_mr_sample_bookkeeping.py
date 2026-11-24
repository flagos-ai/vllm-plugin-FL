# SPDX-License-Identifier: Apache-2.0
"""CPU-only branch coverage for ModelRunnerFL sampling and bookkeeping."""

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
# _sample
# ---------------------------------------------------------------------------


def test_sample_plain_returns_sampler_output():
    runner = _runner()
    meta = SimpleNamespace(generators={})
    updates = []
    runner.input_batch = SimpleNamespace(
        sampling_metadata=meta,
        update_async_output_token_ids=lambda: updates.append(True),
    )
    runner.sampler = lambda logits, sampling_metadata: ("SAMPLED", logits)
    out = runner._sample("LOGITS", None)
    assert out == ("SAMPLED", "LOGITS")
    assert runner._cpu_generator_states == {}
    assert updates == [True]


def test_sample_caches_cpu_generator_state():
    runner = _runner()
    gen = SimpleNamespace(device=SimpleNamespace(type="cpu"), get_state=lambda: "STATE")
    meta = SimpleNamespace(generators={3: gen})
    runner.input_batch = SimpleNamespace(
        sampling_metadata=meta, update_async_output_token_ids=lambda: None
    )
    runner.sampler = lambda logits, sampling_metadata: "OUT"
    runner._sample(None, None)
    assert runner._cpu_generator_states == {3: "STATE"}


def test_sample_spec_decode_async_updates_async_spec_tokens():
    runner = _runner()
    meta = SimpleNamespace(generators={})
    recorded = {}
    runner.input_batch = SimpleNamespace(
        sampling_metadata=meta,
        update_async_output_token_ids=lambda: None,
        update_async_spec_token_ids=lambda ids: recorded.update(ids=ids),
    )
    runner.use_async_scheduling = True
    runner._draft_token_req_ids = ["r0"]
    runner._get_draft_token_ids_cpu = lambda: ("DRAFT_CPU", ["r0"])
    runner._get_spec_decode_draft_probs = lambda spec: "PROBS"
    runner.rejection_sampler = lambda spec, probs, logits, meta: ("REJECTED", probs)
    out = runner._sample("L", SimpleNamespace())
    assert out == ("REJECTED", "PROBS")
    assert recorded["ids"] == "DRAFT_CPU"


def test_sample_spec_decode_without_async_skips_draft_refresh():
    runner = _runner()
    meta = SimpleNamespace(generators={})
    runner.input_batch = SimpleNamespace(
        sampling_metadata=meta, update_async_output_token_ids=lambda: None
    )
    runner.use_async_scheduling = False
    runner._draft_token_req_ids = ["r0"]
    runner._get_spec_decode_draft_probs = lambda spec: "P"
    runner.rejection_sampler = lambda spec, probs, logits, meta: "REJ"
    assert runner._sample(None, SimpleNamespace()) == "REJ"


# ---------------------------------------------------------------------------
# _bookkeeping_sync
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
        token_ids_cpu=np.zeros((2, 8), dtype=np.int32),
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


def test_bookkeeping_sync_single_token(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False, raising=False)
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: contextlib.nullcontext())
    runner, sampler_output, sched = _book_runner(False, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    out = runner._bookkeeping_sync(
        sched, sampler_output, None, torch.zeros(4, 3), 4
    )
    (
        nan_counts,
        nan_dev,
        logprobs,
        valid,
        prompt_lp,
        req_ids,
        req_map,
        invalid,
    ) = out
    assert nan_counts == {} and nan_dev is None and logprobs is None
    assert valid == [[7], [8]]
    assert prompt_lp == {"r0": "PLP"}
    assert req_ids == ["r0", "r1"] and req_map == {"r0": 0, "r1": 1}
    assert invalid == []
    assert runner.input_batch.token_ids_cpu[0, 5] == 7
    assert runner.input_batch.token_ids_cpu[1, 5] == 8
    assert runner.input_batch.is_token_ids[0, 5]
    assert runner.input_batch.num_tokens_no_spec.tolist() == [6, 6]
    assert runner.requests["r0"].output_token_ids == [7]


def test_bookkeeping_sync_spec_decode_uses_parse_output(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False, raising=False)
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: contextlib.nullcontext())
    seen = {}

    def parse_output(sampled_ids, vocab_size, discard, logprobs_tensors=None):
        seen["vocab"] = vocab_size
        seen["discard"] = discard
        return ([[1, 2, 3], []], ["L0", "L1"])

    monkeypatch.setattr(
        mr, "RejectionSampler", SimpleNamespace(parse_output=staticmethod(parse_output))
    )
    runner, sampler_output, sched = _book_runner(False, (2, 3))
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert out[2] == ["L0", "L1"]
    assert out[3] == [[1, 2, 3], []]
    assert seen["vocab"] == 16
    assert runner.input_batch.num_tokens_no_spec.tolist() == [8, 5]
    assert runner.requests["r0"].output_token_ids == [1, 2, 3]


def test_bookkeeping_sync_discard_restores_generator(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False, raising=False)
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: contextlib.nullcontext())
    runner, sampler_output, sched = _book_runner(False, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False], dtype=bool))
    restored = []
    runner.input_batch.generators = {
        0: SimpleNamespace(
            device=SimpleNamespace(type="cpu"),
            set_state=lambda s: restored.append(s),
        )
    }
    runner._cpu_generator_states = {0: "SAVED"}
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert out[3] == [[], [8]]
    assert restored == ["SAVED"]
    assert runner.input_batch.num_tokens_no_spec.tolist() == [5, 6]


def test_bookkeeping_sync_async_deferred(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False, raising=False)
    runner, sampler_output, sched = _book_runner(True, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    out = runner._bookkeeping_sync(sched, sampler_output, None, torch.zeros(4, 3), 4)
    assert out[3] == []
    assert out[7] == []
    assert runner.input_batch.prev_sampled_token_ids is sampler_output.sampled_token_ids
    assert runner.input_batch.prev_req_id_to_index == {"r0": 0, "r1": 1}
    assert runner.input_batch.token_ids_cpu[0, 5] == -1
    assert runner.input_batch.num_tokens_no_spec.tolist() == [6, 6]
    assert runner.requests["r1"].output_token_ids == [-1]


def test_bookkeeping_sync_async_nan_count_on_device(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", True, raising=False)
    monkeypatch.setattr(mr, "count_nans_per_row", lambda logits: "NANROWS", raising=False)
    runner, sampler_output, sched = _book_runner(True, (2, 1))
    sampler_output.sampled_token_ids = torch.tensor([[7], [8]], dtype=torch.int32)
    out = runner._bookkeeping_sync(sched, sampler_output, "LOGITS", torch.zeros(4, 3), 4)
    assert out[1] == "NANROWS"
    assert out[0] == {}


# ---------------------------------------------------------------------------
# sample_tokens empty path
# ---------------------------------------------------------------------------


def test_sample_tokens_without_execute_state(monkeypatch):
    runner = _runner()
    runner.execute_model_state = None
    runner.use_async_scheduling = False
    runner.kv_connector_output = None
    out = runner.sample_tokens(None)
    assert isinstance(out, mr.ModelRunnerOutput)
    assert runner.kv_connector_output is None


# ---------------------------------------------------------------------------
# _input_fits_in_drafter
# ---------------------------------------------------------------------------


def test_input_fits_in_drafter_none_metadata():
    runner = _runner()
    assert runner._input_fits_in_drafter(None) is False


@pytest.mark.parametrize(
    "max_seq_len,use_dflash,expected",
    [(7, False, True), (9, False, False), (8, True, False)],
)
def test_input_fits_in_drafter_boundaries(max_seq_len, use_dflash, expected):
    runner = _runner()
    runner.speculative_config = SimpleNamespace(use_dflash=lambda: use_dflash)
    runner.num_spec_tokens = 2
    runner.effective_drafter_max_model_len = 10
    meta = SimpleNamespace(max_seq_len=max_seq_len)
    assert runner._input_fits_in_drafter(meta) is expected


# ---------------------------------------------------------------------------
# _copy_draft_token_ids_to_cpu early exits
# ---------------------------------------------------------------------------


def test_copy_draft_token_ids_async_skip_keeps_req_ids_unset():
    runner = _runner()
    runner._draft_token_ids = torch.zeros(3, 2, dtype=torch.int32)
    runner._draft_token_req_ids = None
    runner.use_async_scheduling = True
    runner.input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(output_token_ids=None)
    )
    sched = SimpleNamespace(has_structured_output_requests=False)
    runner._copy_draft_token_ids_to_cpu(sched)
    assert runner.prev_num_spec_tokens == 2
    assert runner._draft_token_req_ids is None


def test_copy_draft_token_ids_non_tensor_after_req_ids(monkeypatch):
    runner = _runner()
    runner._draft_token_ids = [[1, 2]]
    runner._draft_token_req_ids = None
    runner.use_async_scheduling = False
    runner.input_batch = SimpleNamespace(req_ids=["r0"])
    sched = SimpleNamespace(has_structured_output_requests=False)
    runner._copy_draft_token_ids_to_cpu(sched)
    # req ids copied, then function returns because payload is not a tensor.
    assert runner._draft_token_req_ids == ["r0"]


# ---------------------------------------------------------------------------
# _pool
# ---------------------------------------------------------------------------


def _pool_runner(pooler_output):
    runner = _runner()
    pm = SimpleNamespace(
        prompt_lens=[5, 6],
        pooling_params=["p0", "p1"],
        build_pooling_cursor=lambda *a, **k: None,
    )
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        pooling_params=["p0", "p1"],
        get_pooling_metadata=lambda: pm,
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
    )
    runner.optimistic_seq_lens_cpu = torch.tensor([5, 2])
    runner.query_start_loc = SimpleNamespace(gpu=torch.zeros(3))
    runner.model = SimpleNamespace(pooler=lambda **kwargs: pooler_output)
    runner.late_interaction_runner = SimpleNamespace(
        postprocess_pooler_output=lambda **kwargs: pooler_output
    )
    runner._sync_device = lambda: None
    return runner


def test_pool_without_raw_output_returns_none_slots():
    runner = _pool_runner(None)
    out = runner._pool(torch.zeros(4, 2), 4, np.array([2, 2]), None)
    assert out.pooler_output == [None, None]


def test_pool_no_finished_requests_returns_none_slots():
    runner = _pool_runner("RAW")
    runner.optimistic_seq_lens_cpu = torch.tensor([1, 2])
    out = runner._pool(torch.zeros(4, 2), 4, np.array([2, 2]), None)
    assert out.pooler_output == [None, None]


def test_pool_non_cuda_copies_finished_on_cpu(monkeypatch):
    runner = _pool_runner("RAW")
    monkeypatch.setattr(
        mr, "current_platform", SimpleNamespace(is_cuda_alike=lambda: False), raising=False
    )
    seen = {}
    monkeypatch.setattr(
        mr,
        "_copy_pooler_output_to_cpu",
        lambda raw_pooler_output, finished_mask: seen.update(
            raw=raw_pooler_output, mask=finished_mask
        )
        or ["C0", "C1"],
        raising=False,
    )
    out = runner._pool(torch.zeros(4, 2), 4, np.array([2, 2]), None)
    assert out.pooler_output == ["C0", "C1"]
    assert [bool(x) for x in seen["mask"]] == [True, False]


# ---------------------------------------------------------------------------
# _pp broadcast/receive chunked-prefill skips
# ---------------------------------------------------------------------------


def test_pp_broadcast_skips_for_chunked_prefill(monkeypatch):
    runner = _runner()
    runner.input_batch = SimpleNamespace(num_reqs=2)
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, True], dtype=bool))
    monkeypatch.setattr(
        mr,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_last_rank=True, has_last_rank=True, rank=1, device_group=object()
        ),
        raising=False,
    )
    called = []
    monkeypatch.setattr(
        mr.torch.distributed,
        "broadcast",
        lambda *a, **k: called.append("bcast"),
        raising=False,
    )
    runner._pp_broadcast_prev_sampled_token_ids(torch.zeros(2, 1, dtype=torch.int32))
    assert called == []
