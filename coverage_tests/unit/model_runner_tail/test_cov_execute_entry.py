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

"""Coverage tests for the entry branches of ModelRunnerFL.execute_model
(vllm_fl/worker/model_runner.py): the state-machine guard, the empty-
schedule early return (with its external-launcher DP dummy run), the
routed-experts capture-buffer clear, and the ngram-gpu scheduler-output
copy feeding a full recorded forward pipeline.

The scheduler output is a real dataclass instance; the W1b cooperation
methods and forward collaborators are inputs replaced with recorders
(CONTRACT R2); ``get_pp_group`` is patched at the module boundary, an
allowed torch.distributed boundary.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)

import vllm_fl.worker.model_runner as model_runner_module
from ..model_runner._imports import make_scheduler_output
from . import _imports  # noqa: F401  (installs local import shims first)
from vllm_fl.worker.model_runner import ModelRunnerFL


class _Patch:
    """Deterministic set/restore of a module attribute (R2 input boundary)."""

    def __init__(self, module, name, value):
        self.module = module
        self.name = name
        self.value = value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _scheduler_output(total=0):
    """Minimal real SchedulerOutput dataclass instance."""
    return make_scheduler_output(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=total,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _empty_step_runner():
    """Runner namespace for execute_model's empty-schedule early return."""
    calls = SimpleNamespace(update_states=None, dummy_runs=[])

    def _update_states(scheduler_output):
        calls.update_states = scheduler_output
        return None

    runner = SimpleNamespace(
        execute_model_state=None,
        routed_experts_initialized=False,
        speculative_config=None,
        synchronize_input_prep=ModelRunnerFL.synchronize_input_prep.__get__(
            SimpleNamespace(prepare_inputs_event=None),
            ModelRunnerFL,
        ),
        _update_states=_update_states,
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp",
            data_parallel_size=1,
            num_ubatches=2,
            use_ubatching=False,
        ),
        _dummy_run=lambda n, **kw: calls.dummy_runs.append(n),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        encoder_cache={},
    )
    return runner, calls


def _pp_group_singleton():
    return SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        world_size=1,
        rank=0,
    )


def test_execute_model_rejects_call_while_state_pending():
    """Calling execute_model again before sample_tokens raises RuntimeError."""
    runner = SimpleNamespace(execute_model_state=object())
    with pytest.raises(RuntimeError, match="sample_tokens"):
        ModelRunnerFL.execute_model(runner, _scheduler_output())


def test_execute_model_empty_schedule_returns_empty_output():
    """Zero scheduled tokens return the shared EMPTY_MODEL_RUNNER_OUTPUT."""
    runner, calls = _empty_step_runner()
    sched = _scheduler_output(total=0)

    with _Patch(
        model_runner_module,
        "get_pp_group",
        _pp_group_singleton,
    ):
        result = ModelRunnerFL.execute_model(runner, sched)

    assert result is EMPTY_MODEL_RUNNER_OUTPUT
    # _update_states ran with the scheduler output; no dummy run was forced.
    assert calls.update_states is sched
    assert calls.dummy_runs == []


def test_execute_model_empty_schedule_external_launcher_dp_runs_dummy():
    """external_launcher + DP>1 runs one dummy token before the empty return."""
    runner, calls = _empty_step_runner()
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend="external_launcher",
        data_parallel_size=2,
    )

    with _Patch(
        model_runner_module,
        "get_pp_group",
        _pp_group_singleton,
    ):
        result = ModelRunnerFL.execute_model(runner, _scheduler_output(total=0))

    assert result is EMPTY_MODEL_RUNNER_OUTPUT
    assert calls.dummy_runs == [1]


def test_execute_model_clears_routed_experts_capture_buffer():
    """An initialized routed-experts capturer is cleared each step."""
    cleared = []
    capturer = SimpleNamespace(clear_buffer=lambda: cleared.append(True))

    class _SpecConfig:
        def use_ngram_gpu(self):
            return False

    runner, _ = _empty_step_runner()
    runner.routed_experts_initialized = True
    runner.routed_experts_capturer = capturer
    runner.speculative_config = _SpecConfig()

    with _Patch(
        model_runner_module,
        "get_pp_group",
        _pp_group_singleton,
    ):
        result = ModelRunnerFL.execute_model(runner, _scheduler_output(total=0))

    assert result is EMPTY_MODEL_RUNNER_OUTPUT
    assert cleared == [True]


def test_execute_model_ngram_gpu_copies_scheduler_token_dicts():
    """ngram-gpu spec decode replaces the scheduler output with copies.

    The ngram GPU drafter mutates num_scheduled_tokens /
    scheduled_spec_decode_tokens, so execute_model must hand it private
    copies instead of the scheduler's dicts. The full pipeline then runs
    through recording collaborators (as with the empty-schedule test, the
    vllm distributed groups are the only patched module boundary).
    """
    runner, calls = _empty_step_runner()
    runner.speculative_config = SimpleNamespace(
        use_ngram_gpu=lambda: True,
        method="ngram_gpu",
    )
    sched = _scheduler_output(total=3)
    sched.num_scheduled_tokens = {"r0": 1, "r1": 2}
    sched.scheduled_spec_decode_tokens = {"r0": [9]}
    sched.total_num_scheduled_tokens = 3
    sched.scheduled_encoder_inputs = {}
    sched.num_spec_tokens_to_schedule = 0

    received = {}

    def _update_states(scheduler_output):
        received["sched"] = scheduler_output
        return None

    runner._update_states = _update_states
    # Persistent batch state: two requests, their scheduled token counts.
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        sampling_metadata=object(),
    )
    runner.use_ascend_attention = False
    runner.cascade_attn_enabled = False
    runner.common_attention_metadata_graph = None
    # Dispatch result recorder: piecewise-free, unpadded batch descriptor.
    runner._prepare_inputs = lambda so, np_tok: ([2, 4], None)
    runner._determine_batch_execution_and_padding = lambda **kw: (
        CUDAGraphMode.NONE,
        SimpleNamespace(num_tokens=3, num_reqs=2),
        False,
        None,
        None,
    )
    runner.attn_groups = {0: []}
    # No KV cache config on this runner: _get_slot_mappings returns the
    # (None, None) pair and the separate-KV-update scan is trivially empty.
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner.cache_config = SimpleNamespace(
        kv_sharing_fast_prefill=False,
        mamba_cache_mode="none",
    )
    runner._run_common_attention_metadata = lambda num_reqs, mode: None
    # Bind the real slot-mapping builder (covered above); with no KV cache
    # groups it returns the (None, None) pair.
    runner._get_slot_mappings = ModelRunnerFL._get_slot_mappings.__get__(
        runner,
        ModelRunnerFL,
    )
    runner._build_attention_metadata = lambda **kw: (object(), None)
    runner._preprocess = lambda so, tp, nr, nrp, it: (None, None, None, it, {}, None)
    runner.calculate_kv_scales = False
    runner.use_aux_hidden_state_outputs = False
    runner.broadcast_pp_output = False
    runner.is_pooling_model = False
    runner.supports_mm_inputs = False
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            use_sequence_parallel_moe=False,
            is_moe_model=None,
            use_ubatching=False,
            num_ubatches=2,
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            fast_moe_cold_start=False,
        ),
    )
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)

    hidden_states = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    logits = torch.arange(5, dtype=torch.float32).reshape(5, 1)

    class _Model:
        def compute_logits(self, sample_hidden_states):
            received["sample_rows"] = sample_hidden_states.tolist()
            return logits

    runner.model = _Model()
    runner._model_forward = lambda **kw: hidden_states
    # Bind the real KV-connector mixin helpers: with no kv transfer group
    # active they are nullcontexts (the module-level distributed state is
    # untouched by this test).
    runner.maybe_get_kv_connector_output = (
        KVConnectorModelRunnerMixin.maybe_get_kv_connector_output
    )

    with _Patch(
        model_runner_module,
        "get_pp_group",
        _pp_group_singleton,
    ):
        result = ModelRunnerFL.execute_model(runner, sched)

    # execute_model returns None and stores the pending state for
    # sample_tokens (verified in test_cov_sample_tokens.py).
    assert result is None
    # The scheduler output reaching _update_states is a distinct object
    # whose token dicts are copies of the originals.
    assert received["sched"] is not sched
    assert received["sched"].num_scheduled_tokens == {"r0": 1, "r1": 2}
    assert received["sched"].num_scheduled_tokens is not sched.num_scheduled_tokens
    assert received["sched"].scheduled_spec_decode_tokens is not (
        sched.scheduled_spec_decode_tokens
    )
    # Non-copied fields ride along unchanged.
    assert received["sched"].total_num_scheduled_tokens == 3
    # The sampling rows are the logits_indices rows of the forward output.
    assert received["sample_rows"] == [[2.0], [4.0]]
    # Pending state carries the sampled hidden states and metadata.
    state = runner.execute_model_state
    assert state is not None
    assert state.scheduler_output is received["sched"]
    torch.testing.assert_close(state.hidden_states, hidden_states)
    torch.testing.assert_close(state.logits, logits)
