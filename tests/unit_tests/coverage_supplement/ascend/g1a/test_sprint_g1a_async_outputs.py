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

"""Coverage for the async runner-output wrappers (814b lines 373-471, 519-557).

All paths run on CPU tensors: the platform is replaced with an inert
stream/event stand-in, so no NPU allocation happens. The rejection-sampler
and EP-fault entry points are module-level symbols replaced per test.
"""

import pytest
import torch
from unittest.mock import patch

from vllm_fl.worker import model_runner as mr


class _Stream:
    def wait_stream(self, other):
        return None


class _Event:
    def __init__(self, *args, **kwargs):
        self.recorded = False
        self.synchronized = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        self.synchronized = True


class _Platform:
    class torch_device_fn:
        Stream = staticmethod(lambda *a, **k: _Stream())
        Event = staticmethod(lambda *a, **k: _Event())
        current_stream = staticmethod(lambda *a, **k: _Stream())
        stream = staticmethod(lambda s: __import__("contextlib").nullcontext())


class _AsyncOutput(mr.AsyncGPUModelRunnerOutput):
    def get_kv_connector_output(self):
        return None


class _AsyncPoolingOutput(mr.AsyncGPUPoolingModelRunnerOutput):
    def get_kv_connector_output(self):
        return None


class _Logprobs:
    def __init__(self, lists=None):
        self.lists = lists if lists is not None else [[0.1], []]

    def to_cpu_nonblocking(self):
        return self

    def tolists(self):
        return self.lists


def _make_output(**overrides):
    from types import SimpleNamespace

    values = dict(
        sampled_token_ids=None,
        logprobs=None,
        req_id_to_index={"a": 0, "b": 1, "c": 2},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_single_token_without_logprobs_or_nans():
    sampled = torch.tensor([[7], [8], [9]], dtype=torch.int64)
    output = _make_output()
    with patch.object(mr, "current_platform", _Platform()):
        async_out = _AsyncOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[1],
            async_output_copy_stream=_Stream(),
            vocab_size=100,
        )
        result = async_out.get_output()
    assert result.sampled_token_ids == [[7], [], [9]]
    assert result.logprobs is None
    assert not hasattr(async_out, "_sampled_token_ids")
    assert not hasattr(async_out, "_logprobs_tensors")
    assert async_out.sampled_token_ids_cpu.tolist() == [[7], [8], [9]]


def test_single_token_with_logprobs_and_num_nans():
    sampled = torch.tensor([[1], [2]], dtype=torch.int64)
    num_nans = torch.tensor([1, 0], dtype=torch.int32)
    output = _make_output(req_id_to_index={"a": 0, "b": 1})
    logprobs = _Logprobs([[0.5], [0.25]])
    with patch.object(mr, "current_platform", _Platform()), patch.object(
        mr, "envs", type("E", (), {"VLLM_RAISE_ON_LOGIT_NANS": False})()
    ):
        async_out = _AsyncOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs,
            invalid_req_indices=[],
            async_output_copy_stream=_Stream(),
            vocab_size=100,
            num_nans=num_nans,
        )
        result = async_out.get_output()
    assert result.sampled_token_ids == [[1], [2]]
    assert result.logprobs == [[0.5], [0.25]]
    assert result.num_nans_in_logits == {"a": 1, "b": 0}
    assert async_out._num_nans_cpu.tolist() == [1, 0]
    assert not hasattr(async_out, "_num_nans")


def test_spec_tokens_use_rejection_sampler_parse_output():
    sampled = torch.tensor([[5, 6, 7], [8, -1, -1]], dtype=torch.int64)
    output = _make_output(req_id_to_index={"a": 0, "b": 1})

    class _RS:
        @staticmethod
        def parse_output(ids, vocab_size, invalid, logprobs_tensors=None):
            return ([[5, 6, 7], []], [[0.1], []])

    with patch.object(mr, "current_platform", _Platform()), patch.object(
        mr, "RejectionSampler", _RS
    ):
        async_out = _AsyncOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=_Logprobs(),
            invalid_req_indices=[1],
            async_output_copy_stream=_Stream(),
            vocab_size=100,
        )
        result = async_out.get_output()
    assert result.sampled_token_ids == [[5, 6, 7], []]
    assert result.logprobs == [[0.1], []]


def test_num_nans_raise_hook_runs_when_enabled():
    sampled = torch.tensor([[3]], dtype=torch.int64)
    num_nans = torch.tensor([2], dtype=torch.int32)
    output = _make_output(req_id_to_index={"a": 0})
    seen = []
    with patch.object(mr, "current_platform", _Platform()), patch.object(
        mr, "envs", type("E", (), {"VLLM_RAISE_ON_LOGIT_NANS": True})()
    ), patch.object(mr, "raise_if_nan_logits", lambda d: seen.append(d)):
        async_out = _AsyncOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=_Stream(),
            vocab_size=100,
            num_nans=num_nans,
        )
        result = async_out.get_output()
    assert result.num_nans_in_logits == {"a": 2}
    assert seen == [{"a": 2}]


def test_ep_fault_raises_with_active_mask():
    class _Manager:
        @staticmethod
        def query_fault():
            return torch.tensor([1], dtype=torch.int32)

        @staticmethod
        def query_active_mask():
            return torch.tensor([0, 1], dtype=torch.int32)

    sampled = torch.tensor([[4]], dtype=torch.int64)
    output = _make_output(req_id_to_index={"a": 0})
    with patch.object(mr, "current_platform", _Platform()), patch.object(
        mr, "get_ep_all2all_manager", _Manager
    ):
        async_out = _AsyncOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=_Stream(),
            vocab_size=100,
            check_ep_fault=True,
        )
        assert async_out._has_fault.tolist() == [1]
        with pytest.raises(RuntimeError, match="Fault detected"):
            async_out.get_output()


def test_ep_check_without_fault_returns_output():
    class _Manager:
        @staticmethod
        def query_fault():
            return torch.tensor([0], dtype=torch.int32)

        @staticmethod
        def query_active_mask():
            raise AssertionError("mask queried without a fault")

    sampled = torch.tensor([[6]], dtype=torch.int64)
    output = _make_output(req_id_to_index={"a": 0})
    with patch.object(mr, "current_platform", _Platform()), patch.object(
        mr, "get_ep_all2all_manager", _Manager
    ):
        async_out = _AsyncOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=_Stream(),
            vocab_size=100,
            check_ep_fault=True,
        )
        result = async_out.get_output()
    assert result.sampled_token_ids == [[6]]


def test_pooling_output_tensor_path_records_and_releases():
    raw = torch.tensor([[1.5], [2.5], [3.5]])
    output = _make_output()
    with patch.object(mr, "current_platform", _Platform()):
        async_out = _AsyncPoolingOutput(
            model_runner_output=output,
            raw_pooler_output=raw,
            finished_mask=[False, True, True],
            async_output_copy_stream=_Stream(),
        )
        result = async_out.get_output()
    assert result is output
    assert result.pooler_output[0] is None
    assert result.pooler_output[1].tolist() == [2.5]
    assert result.pooler_output[2].tolist() == [3.5]
    assert not hasattr(async_out, "_raw_pooler_output")
    assert async_out.async_copy_ready_event.synchronized is True


def test_pooling_output_list_path_keeps_finished_rows():
    raw = [torch.tensor([4.0]), None, torch.tensor([6.0])]
    output = _make_output()
    with patch.object(mr, "current_platform", _Platform()):
        async_out = _AsyncPoolingOutput(
            model_runner_output=output,
            raw_pooler_output=raw,
            finished_mask=[True, True, False],
            async_output_copy_stream=_Stream(),
        )
        result = async_out.get_output()
    assert result.pooler_output[0].tolist() == [4.0]
    assert result.pooler_output[1] is None
    assert result.pooler_output[2] is None
