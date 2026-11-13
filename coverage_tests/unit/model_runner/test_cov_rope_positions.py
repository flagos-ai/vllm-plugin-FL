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

"""Coverage tests for the M-RoPE / XD-RoPE position calculators and the
async output wrappers of vllm_fl/worker/model_runner.py.

``_calc_mrope_positions`` and ``_calc_xdrope_positions`` are pure tensor
plumbing: pre-computed prompt positions are sliced into the runner
buffer and completion positions come from the rotary embedding helper.
They are driven through the unbound method on a namespace runner whose
``mrope_positions``/``xdrope_positions`` is a real (3, N) / (4, N)
buffer, with the completion helper replaced by a deterministic recorder
(the vllm rotary-embedding symbol is an input, CONTRACT R2). The
``AsyncGPUModelRunnerOutput``/``AsyncGPUPoolingModelRunnerOutput``
``get_output`` paths run with a plain CPU stream/event stand-in.
"""

from types import SimpleNamespace

import numpy as np
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm_fl.worker.model_runner import ModelRunnerFL


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


class _MRopeRecorder:
    """Deterministic stand-in for MRotaryEmbedding.get_next_input_..."""

    calls = []

    @staticmethod
    def get_next_input_positions_tensor(
        out, out_offset, mrope_position_delta, context_len, num_new_tokens
    ):
        _MRopeRecorder.calls.append(
            (out_offset, mrope_position_delta, context_len, num_new_tokens)
        )
        values = np.arange(
            mrope_position_delta + context_len,
            mrope_position_delta + context_len + num_new_tokens,
            dtype=out.dtype,
        )
        out[:, out_offset : out_offset + num_new_tokens] = values


class _XDRecorder:
    """Deterministic stand-in for XDRotaryEmbedding.get_next_input_..."""

    calls = []

    @staticmethod
    def get_next_input_positions_tensor(out, out_offset, context_len, num_new_tokens):
        _XDRecorder.calls.append((out_offset, context_len, num_new_tokens))
        values = np.arange(context_len, context_len + num_new_tokens, dtype=out.dtype)
        out[:, out_offset : out_offset + num_new_tokens] = values


class _RopeBuffer:
    """Minimal stand-in for the runner's CpuGpuBuffer: np view of cpu."""

    def __init__(self, dim, length):
        self.cpu = torch.zeros((dim, length), dtype=torch.int64)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu


def _rope_runner(dim, buffer_name):
    """Namespace runner with an m/xd-rope buffer and one request."""
    buf = _RopeBuffer(dim, 33)
    runner = SimpleNamespace(
        requests={},
        input_batch=SimpleNamespace(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            num_computed_tokens_cpu=np.array([2], dtype=np.int32),
        ),
    )
    setattr(runner, buffer_name, buf)
    return runner, buf


def _sched(tokens_per_req):
    return SimpleNamespace(num_scheduled_tokens=tokens_per_req)


def test_calc_mrope_positions_prompt_only_copies_precomputed_slice():
    """Prompt-only steps copy precomputed mrope positions."""
    runner, buf = _rope_runner(3, "mrope_positions")
    req = SimpleNamespace(
        prompt_token_ids=list(range(5)),
        prompt_embeds=None,
        mrope_positions=torch.arange(15, dtype=torch.int64).reshape(3, 5) + 100,
        mrope_position_delta=0,
    )
    runner.requests["r0"] = req
    # computed=2, scheduled=2 -> 2+2 <= prompt_len 5: pure prompt part.
    out = _sched({"r0": 2})

    with _Patch(mr_module, "MRotaryEmbedding", _MRopeRecorder):
        ModelRunnerFL._calc_mrope_positions(runner, out)

    # Prompt positions 2..4 of the request copied to slots 0..2.
    torch.testing.assert_close(
        buf.cpu[:, :2], (torch.arange(15).reshape(3, 5) + 100)[:, 2:4]
    )
    assert _MRopeRecorder.calls == []  # no completion part


def test_calc_mrope_positions_completion_part_uses_delta_recorder():
    """Steps crossing the prompt boundary call the completion helper."""
    runner, buf = _rope_runner(3, "mrope_positions")
    req = SimpleNamespace(
        prompt_token_ids=list(range(3)),
        prompt_embeds=None,
        mrope_positions=torch.arange(9, dtype=torch.int64).reshape(3, 3) + 50,
        mrope_position_delta=4,
    )
    runner.requests["r0"] = req
    # computed=2, scheduled=3 -> prompt part 1 (position 2), completion 2.
    out = _sched({"r0": 3})

    with _Patch(mr_module, "MRotaryEmbedding", _MRopeRecorder):
        ModelRunnerFL._calc_mrope_positions(runner, out)

    # Slot 0 holds prompt position 2; slots 1..2 hold the completion.
    torch.testing.assert_close(
        buf.cpu[:, 0], (torch.arange(9).reshape(3, 3) + 50)[:, 2]
    )
    # Completion: delta 4 + context 3 -> positions 7, 8 in every dim.
    assert buf.cpu[:, 1].tolist() == [7, 7, 7]
    assert buf.cpu[:, 2].tolist() == [8, 8, 8]
    assert _MRopeRecorder.calls == [(1, 4, 3, 2)]


def test_calc_mrope_positions_multiple_requests_append():
    """Multiple requests are laid out consecutively in the buffer."""
    runner, buf = _rope_runner(3, "mrope_positions")
    req0 = SimpleNamespace(
        prompt_token_ids=[0, 1],
        prompt_embeds=None,
        mrope_positions=torch.tensor([[10, 11], [12, 13], [14, 15]]),
        mrope_position_delta=0,
    )
    req1 = SimpleNamespace(
        prompt_token_ids=[0, 1],
        prompt_embeds=None,
        mrope_positions=torch.tensor([[20, 21], [22, 23], [24, 25]]),
        mrope_position_delta=0,
    )
    runner.requests["r0"] = req0
    runner.requests["r1"] = req1
    runner.input_batch.req_ids = ["r0", "r1"]
    runner.input_batch.num_computed_tokens_cpu = np.array([0, 0], dtype=np.int32)
    out = _sched({"r0": 2, "r1": 2})

    with _Patch(mr_module, "MRotaryEmbedding", _MRopeRecorder):
        ModelRunnerFL._calc_mrope_positions(runner, out)

    # Both prompts' positions laid out consecutively.
    assert buf.cpu[:, 0].tolist() == [10, 12, 14]
    assert buf.cpu[:, 1].tolist() == [11, 13, 15]
    assert buf.cpu[:, 2].tolist() == [20, 22, 24]
    assert buf.cpu[:, 3].tolist() == [21, 23, 25]


def test_calc_xdrope_positions_prompt_and_completion():
    """XD-RoPE steps splice the prompt and continue the completion."""
    runner, buf = _rope_runner(4, "xdrope_positions")
    runner.input_batch.num_computed_tokens_cpu = np.array([3], dtype=np.int32)
    req = SimpleNamespace(
        prompt_token_ids=list(range(4)),
        prompt_embeds=None,
        xdrope_positions=torch.arange(16, dtype=torch.int64).reshape(4, 4) + 200,
    )
    runner.requests["r0"] = req
    # computed=3, scheduled=3 -> prompt part 1 (position 3), completion 2.
    out = _sched({"r0": 3})

    with _Patch(mr_module, "XDRotaryEmbedding", _XDRecorder):
        ModelRunnerFL._calc_xdrope_positions(runner, out)

    torch.testing.assert_close(
        buf.cpu[:, 0], (torch.arange(16).reshape(4, 4) + 200)[:, 3]
    )
    # Completion continues from context 4 (no delta in the XD variant).
    assert buf.cpu[:, 1].tolist() == [4, 4, 4, 4]
    assert buf.cpu[:, 2].tolist() == [5, 5, 5, 5]
    assert _XDRecorder.calls == [(1, 4, 2)]


class _CpuStream:
    """Event/stream stand-in: plain CPU synchronization."""

    def wait_stream(self, other):
        pass


class _CpuEvent:
    """torch.Event stand-in; the cpu backend supports no events."""

    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        pass


class _CpuStreamPlatform:
    """CPU platform whose torch_device_fn offers inert stream/Event."""

    class torch_device_fn:
        Stream = staticmethod(lambda *a, **k: _CpuStream())
        current_stream = staticmethod(lambda *a, **k: _CpuStream())
        stream = staticmethod(lambda s: __import__("contextlib").nullcontext())
        Event = staticmethod(_CpuEvent)


class _CpuAsyncGPUOutput(mr_module.AsyncGPUModelRunnerOutput):
    """Concrete form for CPU hosts.

    The CI image's vllm base class declares no abstract members beyond
    ``get_output`` (implemented by the plugin); local vllm 0.33 added
    ``get_kv_connector_output`` as abstract, so this local-only subclass
    supplies the null implementation the production code never calls on
    the paths under test.
    """

    def get_kv_connector_output(self):
        return None


class _CpuAsyncPoolingOutput(mr_module.AsyncGPUPoolingModelRunnerOutput):
    def get_kv_connector_output(self):
        return None


def test_async_gpu_model_runner_output_get_output_decodes_single_token():
    """max_gen_len == 1 path clears invalid rows and returns the output."""
    from vllm.v1.outputs import ModelRunnerOutput

    sampled = torch.tensor([[7], [8], [9]], dtype=torch.int64)
    output = ModelRunnerOutput(
        req_ids=["a", "b", "c"], req_id_to_index={"a": 0, "b": 1, "c": 2}
    )
    saved_platform = mr_module.current_platform
    saved_event = torch.Event
    mr_module.current_platform = _CpuStreamPlatform()
    torch.Event = _CpuEvent
    try:
        async_out = _CpuAsyncGPUOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[1],
            async_output_copy_stream=_CpuStream(),
            vocab_size=100,
        )
        result = async_out.get_output()
    finally:
        mr_module.current_platform = saved_platform
        torch.Event = saved_event
    # Row 1 was invalid: its sampled token list is emptied.
    assert result.sampled_token_ids == [[7], [], [9]]
    assert result.logprobs is None
    # The device tensor reference was released.
    assert not hasattr(async_out, "_sampled_token_ids")


def test_async_gpu_model_runner_output_get_output_parses_spec_tokens():
    """max_gen_len > 1 goes through RejectionSampler.parse_output."""
    # RejectionSampler uses -1 (PLACEHOLDER_TOKEN_ID) to mark rejected
    # draft tokens; the parse filters them and discards invalid rows.
    sampled = torch.tensor([[5, 6, 7], [8, -1, -1]], dtype=torch.int64)
    output = SimpleNamespace(sampled_token_ids=None, logprobs=None)
    saved_platform = mr_module.current_platform
    saved_event = torch.Event
    mr_module.current_platform = _CpuStreamPlatform()
    torch.Event = _CpuEvent
    try:
        async_out = _CpuAsyncGPUOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[1],
            async_output_copy_stream=_CpuStream(),
            vocab_size=100,
        )
        result = async_out.get_output()
    finally:
        mr_module.current_platform = saved_platform
        torch.Event = saved_event
    # Accepted draft tokens survive; the rejected -1 markers are filtered.
    assert result.sampled_token_ids[0] == [5, 6, 7]
    # The invalid row is discarded outright (empty list).
    assert result.sampled_token_ids[1] == []
    assert result.logprobs is None


def test_async_gpu_pooling_output_get_output_returns_pooler_rows():
    """Pooling async outputs copy only finished rows to the CPU output."""
    from vllm.v1.outputs import ModelRunnerOutput

    raw = torch.tensor([[1.5], [2.5], [3.5]])
    inner = ModelRunnerOutput(
        req_ids=["a", "b", "c"], req_id_to_index={"a": 0, "b": 1, "c": 2}
    )
    saved_platform = mr_module.current_platform
    saved_event = torch.Event
    mr_module.current_platform = _CpuStreamPlatform()
    torch.Event = _CpuEvent
    try:
        async_out = _CpuAsyncPoolingOutput(
            model_runner_output=inner,
            raw_pooler_output=raw,
            finished_mask=[False, True, True],
            async_output_copy_stream=_CpuStream(),
        )
        result = async_out.get_output()
    finally:
        mr_module.current_platform = saved_platform
        torch.Event = saved_event
    assert result is inner
    # Rows 1 and 2 finished; row 0 stays None.
    assert result.pooler_output[0] is None
    assert result.pooler_output[1].tolist() == [2.5]
    assert result.pooler_output[2].tolist() == [3.5]
    # The device tensor reference was released.
    assert not hasattr(async_out, "_raw_pooler_output")
