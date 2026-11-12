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

"""Coverage tests for ModelRunnerFL._get_prompt_logprobs_dict
(vllm_fl/worker/model_runner.py): chunked prompt-logprob accumulation.

The requests are namespaces with real ``LogprobsTensors`` CPU buffers
(real slice copies run); the model / sampler and device sync are
recording collaborators (CONTRACT R2). ``PIN_MEMORY`` in
``vllm.v1.outputs`` is patched to False for the buffer allocation — a
device probe boundary; on the CI CUDA image it stays True and pinned
allocation succeeds.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.outputs as v1_outputs
from vllm.v1.outputs import LogprobsTensors

from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


@pytest.fixture(autouse=True)
def _cpu_logprob_buffers(monkeypatch):
    """Allocate logprob CPU buffers without a pinned-memory device."""
    if hasattr(v1_outputs, "PIN_MEMORY"):
        monkeypatch.setattr(v1_outputs, "PIN_MEMORY", False)


class _Request:
    """CPU request state for prompt logprobs."""

    def __init__(self, prompt_token_ids, num_computed_tokens=0):
        self.prompt_token_ids = prompt_token_ids
        self.num_computed_tokens = num_computed_tokens
        self.in_progress_prompt_logprobs_cpu = None


class _Runner:
    """Namespace runner recording collaborator calls."""

    def __init__(self, requests, req_id_to_index, query_start_loc):
        self.num_prompt_logprobs = {}
        self.requests = requests
        self.input_batch = SimpleNamespace(
            req_id_to_index=req_id_to_index,
        )
        self.query_start_loc = SimpleNamespace(np=query_start_loc)
        self.device = torch.device("cpu")
        self.calls = SimpleNamespace(
            compute_logits=[],
            logprobs=[],
            gather=[],
            sync=0,
        )

        class _Model:
            def compute_logits(inner_self, hidden_states):
                self.calls.compute_logits.append(hidden_states)
                # Deterministic per-row logits: row r -> token r.
                n = hidden_states.shape[0]
                logits = torch.eye(16)[:n]
                return logits

        class _Sampler:
            def compute_logprobs(inner_self, logits):
                self.calls.logprobs.append(logits)
                return torch.log_softmax(logits.float(), dim=-1)

            def gather_logprobs(inner_self, logprobs, k, tgt):
                self.calls.gather.append((logprobs, k, tgt))
                n = logprobs.shape[0]
                # Real shapes: token ids / logprobs [n, k + 1], ranks [n].
                token_ids = torch.arange(n * (k + 1), dtype=torch.int32).reshape(
                    n, k + 1
                )
                gathered = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                out_logprobs = torch.zeros(n, k + 1)
                out_logprobs[:, 0] = gathered
                ranks = torch.zeros(n, dtype=torch.int32)
                return token_ids, out_logprobs, ranks, None

        self.model = _Model()
        self.sampler = _Sampler()

    def _sync_device(self):
        self.calls.sync += 1


def _runner_with(req_id, request, scheduled, num_logprobs=1):
    runner = _Runner(
        requests={req_id: request},
        req_id_to_index={req_id: 0},
        query_start_loc=np.array([0, scheduled], dtype=np.int64),
    )
    runner.num_prompt_logprobs[req_id] = num_logprobs
    return runner


def test_prompt_logprobs_empty_when_no_requests_ask_for_them():
    """An empty num_prompt_logprobs map short-circuits to {}."""
    runner = _Runner(requests={}, req_id_to_index={}, query_start_loc=np.zeros(1))
    hidden = torch.zeros(4, 8)

    result = ModelRunnerFL._get_prompt_logprobs_dict(
        runner,
        hidden,
        num_scheduled_tokens={"r0": 4},
    )

    assert result == {}
    assert runner.calls.compute_logits == []


def test_prompt_logprobs_skip_preempted_and_embedding_requests():
    """Preempted requests (no scheduled tokens) and embedding requests
    (no prompt token ids) are skipped without touching the model."""
    preempted = _Request([1, 2, 3])
    embedded = _Request(None)
    runner = _Runner(
        requests={"r0": preempted, "r1": embedded},
        req_id_to_index={"r0": 0, "r1": 1},
        query_start_loc=np.array([0, 2, 4], dtype=np.int64),
    )
    runner.num_prompt_logprobs = {"r0": 1, "r1": 1}

    result = ModelRunnerFL._get_prompt_logprobs_dict(
        runner,
        torch.zeros(4, 8),
        num_scheduled_tokens={"r1": 2},
    )

    assert result == {}
    assert preempted.in_progress_prompt_logprobs_cpu is None
    assert embedded.in_progress_prompt_logprobs_cpu is None
    assert runner.calls.compute_logits == []
    # No sync: nothing was produced.
    assert runner.calls.sync == 0


def test_prompt_logprobs_middle_chunk_defers_return():
    """A middle chunk fills the in-progress buffer but returns nothing:
    the request stays in num_prompt_logprobs for the next step."""
    prompt = [5, 6, 7, 8, 9, 10]
    request = _Request(prompt, num_computed_tokens=0)
    runner = _runner_with("r0", request, scheduled=3)
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)

    result = ModelRunnerFL._get_prompt_logprobs_dict(
        runner,
        hidden,
        num_scheduled_tokens={"r0": 3},
    )

    # Middle chunk: no output yet, request kept, buffer retained.
    assert result == {}
    assert "r0" in runner.num_prompt_logprobs
    buf = request.in_progress_prompt_logprobs_cpu
    assert isinstance(buf, LogprobsTensors)
    # The model saw this chunk's hidden rows.
    assert len(runner.calls.compute_logits) == 1
    torch.testing.assert_close(runner.calls.compute_logits[0], hidden)
    # num_logits = 3 scheduled (<= 5 remaining), rows [0:3] filled.
    assert buf.logprob_token_ids[:3].shape == (3, 2)
    assert runner.calls.sync == 0


def test_prompt_logprobs_final_chunk_returns_and_cleans_up():
    """The final chunk returns the accumulated buffer, removes the
    request from num_prompt_logprobs and syncs the device."""
    prompt = [5, 6, 7, 8]
    # First chunk already ran 3 tokens: 1 prompt logprob remains.
    request = _Request(prompt, num_computed_tokens=3)
    buffer = LogprobsTensors.empty_cpu(3, 2)
    request.in_progress_prompt_logprobs_cpu = buffer
    runner = _runner_with("r0", request, scheduled=4)
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    result = ModelRunnerFL._get_prompt_logprobs_dict(
        runner,
        hidden,
        num_scheduled_tokens={"r0": 4},
    )

    # The request completed prefill and its buffer was returned...
    assert set(result) == {"r0"}
    assert result["r0"] is buffer
    # ...removed from the pending map, and reset for the next step.
    assert runner.num_prompt_logprobs == {}
    assert request.in_progress_prompt_logprobs_cpu is None
    # Device sync ran because something was produced.
    assert runner.calls.sync == 1
    # num_logits = 4 - 4 = 0: the final zero-logit chunk returned the
    # buffer without invoking the model (num_logits <= 0 continues).
    assert runner.calls.compute_logits == []


def test_prompt_logprobs_final_chunk_with_remaining_tokens_computes():
    """The final chunk with remaining tokens computes logits for them."""
    prompt = [5, 6, 7, 8, 9]
    # 1 token computed, 4 scheduled: last chunk of 3 remaining logits.
    request = _Request(prompt, num_computed_tokens=1)
    runner = _runner_with("r0", request, scheduled=4)
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    result = ModelRunnerFL._get_prompt_logprobs_dict(
        runner,
        hidden,
        num_scheduled_tokens={"r0": 4},
    )

    assert set(result) == {"r0"}
    # num_logits = 5 - 2 = 3 rows computed from offset 0.
    assert len(runner.calls.compute_logits) == 1
    torch.testing.assert_close(runner.calls.compute_logits[0], hidden[:3])
    # The gather saw the target tokens prompt[2:5].
    _, _, tgt = runner.calls.gather[0]
    assert tgt.tolist() == prompt[2:5]
    # The chunk slice [1:4] of the fresh in-progress buffer was filled.
    buf = result["r0"]
    assert buf.logprob_token_ids.shape[0] == 4
    assert runner.calls.sync == 1
