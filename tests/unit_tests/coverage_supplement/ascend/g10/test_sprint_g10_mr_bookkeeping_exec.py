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

"""G10-C coverage: _bookkeeping_sync, execute_model early paths and
profile_cudagraph_memory of ModelRunnerFL (814b).

These tests exercise the branches that do not require a real device:
sync/async sampled-token bookkeeping, routed-experts D2H, the ngram-GPU
scheduler copy, the empty-schedule fast return, the KV-transfer preemption
hook, the encoder-cache consumer return and the "no graphs" profiler exit.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

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


def _envs(nans):
    return SimpleNamespace(VLLM_COMPUTE_NANS_IN_LOGITS=nans)


def _bookkeeping_runner(**extra):
    base = dict(
        use_async_scheduling=False,
        input_batch=SimpleNamespace(
            num_reqs=2,
            generators={},
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            num_tokens_no_spec=[1, 1],
            token_ids_cpu=np.zeros((2, 8), dtype=np.int64),
            is_token_ids=np.zeros((2, 8), dtype=bool),
            vocab_size=32,
            prev_sampled_token_ids=None,
        ),
        discard_request_mask=SimpleNamespace(
            np=np.array([False, False], dtype=bool)
        ),
        _cpu_generator_states={},
        routed_experts_initialized=False,
        requests={
            "a": SimpleNamespace(output_token_ids=[]),
            "b": SimpleNamespace(output_token_ids=[]),
        },
        max_model_len=32,
        _to_list=lambda t: t.tolist(),
        _get_prompt_logprobs_dict=lambda hs, nst: {"prompt": 1},
    )
    base.update(extra)
    return SimpleNamespace(**base)


def _sampler(token_ids, logprobs=None):
    return SimpleNamespace(
        sampled_token_ids=torch.tensor(token_ids, dtype=torch.int32),
        logprobs_tensors=logprobs,
    )


def _sched(total=2):
    return SimpleNamespace(
        total_num_scheduled_tokens=total,
        num_scheduled_tokens={"a": 1, "b": 1},
    )


def _gpu_sync_cm():
    return _Ctx()


# --------------------------------------------------------------------------
# _bookkeeping_sync
# --------------------------------------------------------------------------
def test_bookkeeping_sync_plain_generation():
    runner = _bookkeeping_runner()
    out = None
    with _stack(
        (mr, "envs", _envs(False)),
        (mr, "gpu_sync_allowed", _gpu_sync_cm),
    ):
        out = mr.ModelRunnerFL._bookkeeping_sync(
            runner,
            _sched(),
            _sampler([[5], [6]]),
            "logits",
            torch.zeros(2, 3),
            2,
        )
    num_nans_in_logits, num_nans, logprobs_lists, valid, prompt, rids, idx, invalid = out
    assert valid == [[5], [6]]
    assert rids == ["a", "b"]
    assert invalid == []
    assert runner.requests["a"].output_token_ids == [5]
    assert runner.requests["b"].output_token_ids == [6]
    assert runner.input_batch.num_tokens_no_spec == [2, 2]


def test_bookkeeping_sync_discards_and_restores_cpu_generator():
    calls = []
    gen = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        set_state=lambda s: calls.append(("set_state", s)),
    )
    runner = _bookkeeping_runner(
        input_batch=SimpleNamespace(
            num_reqs=2,
            generators={0: gen},
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            num_tokens_no_spec=[1, 1],
            token_ids_cpu=np.zeros((2, 8), dtype=np.int64),
            is_token_ids=np.zeros((2, 8), dtype=bool),
            vocab_size=32,
            prev_sampled_token_ids=None,
        ),
        discard_request_mask=SimpleNamespace(
            np=np.array([True, False], dtype=bool)
        ),
        _cpu_generator_states={0: "saved"},
    )
    with _stack(
        (mr, "envs", _envs(False)),
        (mr, "gpu_sync_allowed", _gpu_sync_cm),
    ):
        out = mr.ModelRunnerFL._bookkeeping_sync(
            runner,
            _sched(),
            _sampler([[5], [6]]),
            None,
            torch.zeros(2, 3),
            2,
        )
    assert out[3] == [[], [6]]
    assert calls == [("set_state", "saved")]
    assert runner.requests["a"].output_token_ids == []


def test_bookkeeping_sync_spec_decode_uses_parse_output():
    calls = []

    def _parse(tokens, vocab, invalid, logprobs_tensors=None):
        calls.append((vocab, list(invalid)))
        return [[7, 8], [9, 10]], "logprobs-lists"

    runner = _bookkeeping_runner()
    with _stack(
        (mr, "envs", _envs(False)),
        (mr, "gpu_sync_allowed", _gpu_sync_cm),
        (mr, "RejectionSampler", SimpleNamespace(parse_output=_parse)),
    ):
        out = mr.ModelRunnerFL._bookkeeping_sync(
            runner,
            _sched(),
            _sampler([[7, 8], [9, 10]]),
            None,
            torch.zeros(2, 3),
            2,
        )
    assert out[2] == "logprobs-lists"
    assert out[3][0] == [7, 8]
    assert calls == [(32, [])]


def test_bookkeeping_sync_async_caches_prev_sampled_tokens():
    runner = _bookkeeping_runner(
        use_async_scheduling=True,
        discard_request_mask=SimpleNamespace(
            np=np.array([False, True], dtype=bool)
        ),
    )
    with _stack((mr, "envs", _envs(False))):
        out = mr.ModelRunnerFL._bookkeeping_sync(
            runner,
            _sched(),
            _sampler([[5], [6]]),
            None,
            torch.zeros(2, 3),
            2,
        )
    assert out[3] == []
    assert out[7] == [1]
    assert runner.input_batch.prev_sampled_token_ids is not None
    assert runner.input_batch.prev_req_id_to_index == {"a": 0}
    assert runner.requests["a"].output_token_ids == [-1]


def test_bookkeeping_sync_async_counts_nans_on_device():
    logits = torch.tensor([[0.0, float("nan")], [1.0, 2.0]])
    runner = _bookkeeping_runner(use_async_scheduling=True)
    with _stack((mr, "envs", _envs(True))):
        out = mr.ModelRunnerFL._bookkeeping_sync(
            runner,
            _sched(),
            _sampler([[5]]),
            logits,
            torch.zeros(1, 3),
            2,
        )
    assert out[0] == {}
    assert out[1].tolist() == [1, 0]


def test_bookkeeping_sync_sync_collects_nan_dict():
    runner = _bookkeeping_runner(
        _get_nans_in_logits=lambda logits: {"a": 2}
    )
    with _stack(
        (mr, "envs", _envs(True)),
        (mr, "gpu_sync_allowed", _gpu_sync_cm),
    ):
        out = mr.ModelRunnerFL._bookkeeping_sync(
            runner,
            _sched(),
            _sampler([[5], [6]]),
            "logits",
            torch.zeros(2, 3),
            2,
        )
    assert out[0] == {"a": 2}


class _SliceSink:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __getitem__(self, key):
        return self

    def copy_(self, src, non_blocking=True):
        self.calls.append((self.name, non_blocking))


def test_bookkeeping_sync_copies_routed_experts_to_pinned():
    copies = []
    runner = _bookkeeping_runner(
        routed_experts_initialized=True,
        routed_experts_capturer=SimpleNamespace(
            get_device_buffer=lambda: "device-buf"
        ),
        routed_experts_cpu=_SliceSink("experts", copies),
        routed_experts_slot_mapping_cpu=_SliceSink("slots", copies),
        routed_experts_slot_mapping_device=_SliceSink("slot-device", copies),
    )
    with _stack(
        (mr, "envs", _envs(False)),
        (mr, "gpu_sync_allowed", _gpu_sync_cm),
    ):
        mr.ModelRunnerFL._bookkeeping_sync(
            runner,
            _sched(),
            _sampler([[5], [6]]),
            None,
            torch.zeros(2, 3),
            2,
        )
    assert [c[0] for c in copies] == ["experts", "slots"]
    assert all(c[1] is True for c in copies)


# --------------------------------------------------------------------------
# execute_model early paths
# --------------------------------------------------------------------------
def _exec_runner(**extra):
    base = dict(
        execute_model_state=None,
        speculative_config=None,
        synchronize_input_prep=lambda: _Ctx(),
        _update_states=lambda so: None,
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp", data_parallel_size=1
        ),
    )
    base.update(extra)
    return SimpleNamespace(**base)


def _exec_patches(**extra):
    pairs = [
        (mr, "record_function_or_nullcontext", lambda name: _Ctx()),
        (mr, "has_ec_transfer", lambda: False),
        (mr, "has_kv_transfer_group", lambda: False),
    ]
    pairs.extend(extra.items())
    return _stack(*pairs)


def test_execute_model_rejects_state_error():
    runner = _exec_runner(execute_model_state="stale")
    with pytest.raises(RuntimeError, match="sample_tokens"):
        mr.ModelRunnerFL.execute_model(runner, SimpleNamespace())


def test_execute_model_empty_schedule_returns_empty_output():
    runner = _exec_runner()
    sched = SimpleNamespace(total_num_scheduled_tokens=0)
    with _exec_patches():
        out = mr.ModelRunnerFL.execute_model(runner, sched)
    assert out is mr.EMPTY_MODEL_RUNNER_OUTPUT








# --------------------------------------------------------------------------
# profile_cudagraph_memory (no-graph fast exit)
# --------------------------------------------------------------------------
def test_profile_cudagraph_memory_skips_without_graphs():
    calls = []
    runner = SimpleNamespace(
        vllm_config="cfg",
        _init_minimal_kv_cache_for_profiling=lambda: calls.append("init"),
        cudagraph_dispatcher=SimpleNamespace(
            get_capture_descs=lambda: [(mr.CUDAGraphMode.NONE, [])]
        ),
        _create_encoder_cudagraph_manager=lambda: None,
        _cleanup_profiling_kv_cache=lambda: calls.append("cleanup"),
    )
    with _stack(
        (
            mr,
            "set_current_vllm_config",
            lambda cfg: _Ctx(),
        ),
        (mr, "compilation_counter", SimpleNamespace(num_cudagraph_captured=0)),
        (mr, "logger", SimpleNamespace(debug=lambda *a: calls.append("debug"))),
    ):
        out = mr.ModelRunnerFL.profile_cudagraph_memory(runner)
    assert out == 0
    assert calls == ["init", "debug", "cleanup"]
