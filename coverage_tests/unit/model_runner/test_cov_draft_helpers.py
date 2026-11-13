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

"""Coverage tests for the speculative draft-token CPU helpers.

``_copy_draft_token_ids_to_cpu`` chooses between the real async copy, the
zeros-only fill, and the async-scheduling skip; ``_get_draft_token_ids_cpu``
/``take_draft_token_ids`` read the staged list or CPU tensor; and
``_get_valid_sampled_token_count`` gates on the copy event and previous
batch. The runner is a namespace holding exactly the fields these helpers
read; the device stream/event layer is a CPU stand-in (input boundary).
"""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm_fl.worker.model_runner import ModelRunnerFL


class _CpuStream:
    """Stream stand-in that records wait_stream calls."""

    def __init__(self):
        self.waited = 0

    def wait_stream(self, other):
        self.waited += 1


class _CpuEvent:
    def __init__(self):
        self.records = 0
        self.syncs = 0

    def record(self):
        self.records += 1

    def synchronize(self):
        self.syncs += 1


class _Platform:
    """CPU platform exposing the stream API used by the copy helpers."""

    class torch_device_fn:
        current_stream = staticmethod(lambda *a, **k: _CpuStream())
        stream = staticmethod(lambda s: nullcontext())


class _Patch:
    """Deterministic set/restore of a module attribute (input boundary)."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _runner(*, async_scheduling=False, draft_ids=None, structured=False):
    stream = _CpuStream()
    event = _CpuEvent()
    runner = SimpleNamespace(
        use_async_scheduling=async_scheduling,
        _draft_token_ids=draft_ids,
        _draft_token_req_ids=None,
        prev_num_spec_tokens=0,
        draft_token_ids_event=event,
        draft_token_ids_copy_stream=stream,
        draft_token_ids_cpu=torch.zeros((2, 2), dtype=torch.int64),
        input_batch=SimpleNamespace(
            req_ids=["a", "b"],
            sampling_metadata=SimpleNamespace(output_token_ids=None),
        ),
    )
    return runner, stream, event


def _sched(structured=False):
    return SimpleNamespace(has_structured_output_requests=structured)


def test_copy_draft_token_ids_copies_to_pinned_cpu_on_sync_schedule():
    """Synchronous scheduling always stages the draft ids on the copy stream."""
    draft = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    runner, stream, event = _runner(draft_ids=draft)
    with _Patch(mr_module, "current_platform", _Platform()):
        ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, _sched())

    assert runner.draft_token_ids_cpu.tolist() == [[1, 2], [3, 4]]
    assert runner.prev_num_spec_tokens == 2
    assert runner._draft_token_req_ids == ["a", "b"]
    # The copy waited on the default stream and recorded completion.
    assert stream.waited == 1
    assert event.records == 1


def test_copy_draft_token_ids_zeros_only_fills_without_copying():
    """zeros_only overwrites the CPU staging buffer and skips the copy."""
    draft = torch.full((2, 2), 7, dtype=torch.int64)
    runner, stream, event = _runner(draft_ids=draft)
    runner.draft_token_ids_cpu.fill_(5)
    with _Patch(mr_module, "current_platform", _Platform()):
        ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, _sched(), zeros_only=True)

    assert runner.draft_token_ids_cpu.tolist() == [[0, 0], [0, 0]]
    assert runner._draft_token_req_ids == ["a", "b"]
    # No stream dependency was created for the zero fill.
    assert stream.waited == 0
    assert event.records == 1


def test_copy_draft_token_ids_async_skips_when_not_needed():
    """Async scheduling skips the copy unless ids are required on the CPU."""
    draft = torch.tensor([[8, 9], [10, 11]], dtype=torch.int64)
    runner, stream, event = _runner(
        async_scheduling=True, draft_ids=draft, structured=False
    )
    with _Patch(mr_module, "current_platform", _Platform()):
        ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, _sched())

    # The draft width is still recorded, but nothing was copied.
    assert runner.prev_num_spec_tokens == 2
    assert runner._draft_token_req_ids is None
    assert runner.draft_token_ids_cpu.eq(0).all()
    assert stream.waited == 0
    assert event.records == 0


def test_copy_draft_token_ids_async_copies_for_structured_output():
    """Structured output requests force the async copy even in async mode."""
    draft = torch.tensor([[8, 9], [10, 11]], dtype=torch.int64)
    runner, stream, event = _runner(
        async_scheduling=True, draft_ids=draft, structured=True
    )
    with _Patch(mr_module, "current_platform", _Platform()):
        ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, _sched(structured=True))

    assert runner.draft_token_ids_cpu.tolist() == [[8, 9], [10, 11]]
    assert runner._draft_token_req_ids == ["a", "b"]
    assert stream.waited == 1
    assert event.records == 1


def test_copy_draft_token_ids_non_tensor_only_sets_request_ids():
    """A list of draft ids bypasses the device staging entirely."""
    runner, stream, event = _runner(draft_ids=[[1], [2]])
    runner.draft_token_ids_event = None
    runner.draft_token_ids_copy_stream = None
    with _Patch(mr_module, "current_platform", _Platform()):
        ModelRunnerFL._copy_draft_token_ids_to_cpu(runner, _sched())

    assert runner._draft_token_req_ids == ["a", "b"]
    assert runner.prev_num_spec_tokens == 0
    assert stream.waited == 0


def test_take_draft_token_ids_returns_none_without_spec_or_request_ids():
    """No spec tokens or no staged request ids yields None."""
    runner = SimpleNamespace(
        num_spec_tokens=0, _draft_token_req_ids=["a"], _draft_token_ids=[[1]]
    )
    assert ModelRunnerFL.take_draft_token_ids(runner) is None

    runner.num_spec_tokens = 2
    runner._draft_token_req_ids = None
    assert ModelRunnerFL.take_draft_token_ids(runner) is None


def test_take_draft_token_ids_returns_staged_list():
    """A list of per-request draft ids is returned with the batch request ids."""
    runner = SimpleNamespace(
        num_spec_tokens=2,
        _draft_token_req_ids=["a"],
        _draft_token_ids=[[5, 6], [7, 8]],
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
    )
    runner._get_draft_token_ids_cpu = ModelRunnerFL._get_draft_token_ids_cpu.__get__(
        runner, ModelRunnerFL
    )
    result = ModelRunnerFL.take_draft_token_ids(runner)
    assert result.req_ids == ["a", "b"]
    assert result.draft_token_ids == [[5, 6], [7, 8]]


def test_take_draft_token_ids_reads_waiting_cpu_tensor():
    """A device tensor copy is synchronized before the CPU rows are read."""
    event = _CpuEvent()
    runner = SimpleNamespace(
        num_spec_tokens=2,
        _draft_token_req_ids=["a", "b"],
        _draft_token_ids=torch.zeros(2, 2, dtype=torch.int64),
        draft_token_ids_event=event,
        draft_token_ids_cpu=torch.tensor([[3, 4], [5, 6]], dtype=torch.int64),
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
    )
    runner._get_draft_token_ids_cpu = ModelRunnerFL._get_draft_token_ids_cpu.__get__(
        runner, ModelRunnerFL
    )
    result = ModelRunnerFL.take_draft_token_ids(runner)
    assert event.syncs == 1
    assert result.req_ids == ["a", "b"]
    assert result.draft_token_ids == [[3, 4], [5, 6]]


def test_get_valid_sampled_token_count_requires_event_and_previous_ids():
    """Missing event or previous sampled ids report an empty count list."""
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(prev_sampled_token_ids=None),
        valid_sampled_token_count_event=None,
        valid_sampled_token_count_cpu=None,
    )
    assert ModelRunnerFL._get_valid_sampled_token_count(runner) == []

    runner.valid_sampled_token_count_event = _CpuEvent()
    assert ModelRunnerFL._get_valid_sampled_token_count(runner) == []


def test_get_valid_sampled_token_count_reads_previous_request_rows():
    """The count list is sliced to the previous batch's request count."""
    event = _CpuEvent()
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            prev_sampled_token_ids=torch.zeros(2, 1, dtype=torch.int32)
        ),
        valid_sampled_token_count_event=event,
        valid_sampled_token_count_cpu=torch.tensor([2, 3, 9], dtype=torch.int32),
    )
    assert ModelRunnerFL._get_valid_sampled_token_count(runner) == [2, 3]
    assert event.syncs == 1
