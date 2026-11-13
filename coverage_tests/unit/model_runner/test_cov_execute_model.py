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

"""Coverage tests for ModelRunnerFL.execute_model control flow.

Every collaborator the method calls (state update, input preparation,
attention metadata, forward, logits) is replaced with a recorder on a
namespace runner, so the tests assert the exact plumbing: the scheduling
arguments handed to ``_prepare_inputs``/``_determine_batch_execution_and_padding``,
the padded/unpadded dimensions passed to slot-mapping and metadata
builders, the deferred state corrections ordering, the pooling and
pipeline-parallel early returns, and the mamba-align staging arguments.
No device kernel is needed; ``set_forward_context`` and the record-function
contexts are module boundaries.
"""

from contextlib import ExitStack, contextmanager, nullcontext  # noqa: I001  (shim first)
from types import SimpleNamespace

import numpy as np
import torch

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm.config import CUDAGraphMode
from vllm.sequence import IntermediateTensors

from vllm_fl.worker.model_runner import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerFL


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


class _PatchGroup:
    """Enter/exit a list of context managers as a single manager."""

    def __init__(self, patches):
        self._patches = list(patches)
        self._stack = None

    def __enter__(self):
        self._stack = ExitStack()
        for patch in self._patches:
            self._stack.enter_context(patch)
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


class _Recorder:
    """Callable that stores its calls for later assertions."""

    def __init__(self, result=None, result_factory=None):
        self.calls = []
        self.result = result
        self.result_factory = result_factory

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.result_factory is not None:
            return self.result_factory(*args, **kwargs)
        return self.result


def _ctx_recorder(calls, value=None):
    @contextmanager
    def _ctx(*args, **kwargs):
        calls.append((args, kwargs))
        yield value

    return _ctx


def _sched(num_tokens=3, spec_tokens=None, encoder_inputs=None):
    return SimpleNamespace(
        total_num_scheduled_tokens=num_tokens,
        num_scheduled_tokens={"a": num_tokens},
        scheduled_spec_decode_tokens=spec_tokens or {},
        scheduled_encoder_inputs=encoder_inputs or {},
        num_common_prefix_blocks=[0],
        kv_connector_metadata=SimpleNamespace(tag="kv-meta"),
        has_structured_output_requests=False,
    )


class _Buffer:
    """CpuGpuBuffer stand-in with numpy and device views."""

    def __init__(self, values):
        self.np = np.array(values)
        self.gpu_ops = []

    def copy_to_gpu(self, num_reqs=None):
        self.gpu_ops.append(num_reqs)


def _runner(**overrides):
    calls = SimpleNamespace()
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    logits = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    def _determine(**kwargs):
        calls.determine_kwargs = kwargs
        return (
            CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=4, num_reqs=1, uniform=True, num_active_loras=0),
            False,
            None,
            None,
        )

    calls.deferred_calls = 0

    def _deferred():
        calls.deferred_calls += 1

    runner = SimpleNamespace(
        execute_model_state=None,
        routed_experts_initialized=False,
        speculative_config=None,
        synchronize_input_prep=lambda: nullcontext(),
        _update_states=_Recorder(result=_deferred),
        has_ec_transfer=lambda: False,
        input_batch=SimpleNamespace(
            num_reqs=1,
            req_ids=["a"],
            num_computed_tokens_cpu=torch.zeros(4, dtype=torch.int32),
            pooling_params=[],
            req_id_to_index={"a": 0},
            num_accepted_tokens_cpu=torch.ones(4, dtype=torch.int32),
        ),
        cache_config=SimpleNamespace(
            kv_sharing_fast_prefill=False,
            mamba_cache_mode="all",
        ),
        num_prompt_logprobs=0,
        _prepare_inputs=_Recorder(
            result=(torch.tensor([0]), SimpleNamespace(kind="spec"))
        ),
        use_ascend_attention=False,
        cascade_attn_enabled=False,
        parallel_config=SimpleNamespace(
            use_ubatching=False,
            num_ubatches=1,
            distributed_executor_backend="mp",
            data_parallel_size=1,
        ),
        _determine_batch_execution_and_padding=_determine,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        attn_groups={},
        _run_common_attention_metadata=_Recorder(),
        _get_slot_mappings=_Recorder(result=(None, None)),
        _build_attention_metadata=_Recorder(
            result=({"layer": "attn"}, SimpleNamespace(kind="common"))
        ),
        _preprocess=_Recorder(
            result=(
                torch.tensor([1, 2, 3], dtype=torch.int32),
                None,
                torch.zeros(3, dtype=torch.int64),
                None,
                {"mm_kw": torch.zeros(1)},
                SimpleNamespace(kind="ec"),
            )
        ),
        calculate_kv_scales=False,
        vllm_config=SimpleNamespace(),
        common_attention_metadata_graph=None,
        model_config=SimpleNamespace(is_encoder_decoder=False, dtype=torch.float16),
        _model_forward=_Recorder(result=hidden),
        use_aux_hidden_state_outputs=False,
        broadcast_pp_output=False,
        is_pooling_model=False,
        model=SimpleNamespace(compute_logits=_Recorder(result=logits)),
        device=torch.device("cpu"),
        num_accepted_tokens=_Buffer([0, 0, 0, 0]),
        _pool=_Recorder(result="pooled"),
        _dummy_run=_Recorder(),
        kv_connector_no_forward=_Recorder(result="connector-out"),
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner, calls, hidden, logits


def _patch_forward_env(stated_calls, connector_output="kv-conn-out"):
    """Module-boundary patches shared by the forward-path tests.

    Returns one context manager; ``with (a, *patches)`` would parse as a
    tuple and fail, so callers use it as one item in the with-list.
    """

    def _set_forward_context(*args, **kwargs):
        stated_calls.append((args, kwargs))
        return nullcontext()

    return _PatchGroup(
        [
            _Patch(
                mr_module, "record_function_or_nullcontext", lambda name: nullcontext()
            ),
            _Patch(mr_module, "set_forward_context", _set_forward_context),
            _Patch(mr_module, "maybe_create_ubatch_slices", lambda *a: (None, None)),
        ]
    )


def test_execute_model_rejects_reentrant_call():
    """A pending execute_model state forbids a second execute_model call."""
    runner, _, _, _ = _runner(execute_model_state=object())
    try:
        ModelRunnerFL.execute_model(runner, _sched())
    except RuntimeError as exc:
        assert "sample_tokens() must be called" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_execute_model_no_scheduled_tokens_returns_empty_output():
    """Zero scheduled tokens short-circuits to the shared empty output."""
    runner, _, _, _ = _runner()
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: False),
    ):
        out = ModelRunnerFL.execute_model(runner, _sched(num_tokens=0))

    assert out is EMPTY_MODEL_RUNNER_OUTPUT
    assert runner._dummy_run.calls == []


def test_execute_model_external_launcher_runs_dummy_before_empty_output():
    """external_launcher+DP keeps ranks in sync with a one-token dummy run."""
    runner, _, _, _ = _runner(
        parallel_config=SimpleNamespace(
            use_ubatching=False,
            num_ubatches=1,
            distributed_executor_backend="external_launcher",
            data_parallel_size=2,
        )
    )
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: False),
    ):
        out = ModelRunnerFL.execute_model(runner, _sched(num_tokens=0))

    assert out is EMPTY_MODEL_RUNNER_OUTPUT
    assert runner._dummy_run.calls == [((1,), {})]


def test_execute_model_no_tokens_with_kv_transfer_delegates_to_connector():
    """With an active KV transfer group the empty step defers to the connector."""
    runner, _, _, _ = _runner()
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: True),
        _Patch(
            mr_module,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(handle_preemptions=lambda meta: None),
        ),
    ):
        out = ModelRunnerFL.execute_model(runner, _sched(num_tokens=0))

    assert out == "connector-out"
    assert runner.kv_connector_no_forward.calls[0][0][0].total_num_scheduled_tokens == 0


def test_execute_model_generation_path_plumbs_and_stores_state():
    """The generate path resolves metadata, runs the model, and defers output."""
    runner, calls, hidden, logits = _runner()
    kv = SimpleNamespace(kind="kv")
    preemptions = _Recorder()
    runner.maybe_get_kv_connector_output = _ctx_recorder([], kv)

    sched = _sched(num_tokens=3, spec_tokens={"a": [7]})
    forward_calls = []
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: True),
        _Patch(
            mr_module,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(handle_preemptions=preemptions),
        ),
        _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)),
        _patch_forward_env(forward_calls),
    ):
        out = ModelRunnerFL.execute_model(runner, sched)

    assert out is None
    # Preemptions were forwarded before input preparation.
    assert preemptions.calls[0][0][0] is sched.kv_connector_metadata
    # _prepare_inputs received the per-request token counts.
    prep_call = runner._prepare_inputs.calls[0]
    assert prep_call[0][0] is sched
    np.testing.assert_array_equal(prep_call[0][1], np.array([3], dtype=np.int32))
    # Dispatcher sizing used unpadded tokens/max query len.
    det = calls.determine_kwargs
    assert det["num_tokens"] == 3
    assert det["num_reqs"] == 1
    assert det["max_num_scheduled_tokens"] == 3
    assert det["use_cascade_attn"] is False
    # NONE mode means slot mappings and metadata are unpadded.
    slot_kwargs = runner._get_slot_mappings.calls[0][1]
    assert slot_kwargs["num_tokens_padded"] == 3
    assert slot_kwargs["num_reqs_padded"] == 1
    assert slot_kwargs["num_tokens_unpadded"] == 3
    meta_kwargs = runner._build_attention_metadata.calls[0][1]
    assert meta_kwargs["num_tokens"] == 3
    assert meta_kwargs["num_tokens_padded"] is None
    assert meta_kwargs["num_reqs_padded"] is None
    assert meta_kwargs["use_spec_decode"] is True
    assert meta_kwargs["max_query_len"] == 3
    assert meta_kwargs["cascade_attn_prefix_lens"] is None
    # Preprocess got the padded token count from the batch descriptor.
    pre_kwargs = runner._preprocess.calls[0][0]
    assert pre_kwargs[1] == 4
    # The forward passes through the model kwargs and positions.
    fwd = runner._model_forward.calls[0][1]
    assert fwd["input_ids"].tolist() == [1, 2, 3]
    assert fwd["mm_kw"].tolist() == [0.0]
    # Logits index the sampled hidden rows; the connector output was saved.
    assert runner.model.compute_logits.calls[0][0][0].shape == (1, 3)
    assert runner.kv_connector_output is kv
    state = runner.execute_model_state
    assert state.scheduler_output is sched
    assert state.logits is logits
    assert state.spec_decode_metadata is not None
    assert state.spec_decode_common_attn_metadata is not None
    assert state.hidden_states is hidden
    assert state.slot_mappings is None
    # Deferred corrections run after the forward.
    assert calls.deferred_calls == 1


def test_execute_model_calculate_kv_scales_forces_eager_mode():
    """The first step with KV-scale calculation forces CUDAGraphMode.NONE."""
    runner, _, hidden, _ = _runner(calculate_kv_scales=True)
    forward_calls = []

    def _determine(**kwargs):
        return (
            CUDAGraphMode.PIECEWISE,
            SimpleNamespace(num_tokens=4, num_reqs=1, uniform=True, num_active_loras=0),
            False,
            None,
            None,
        )

    runner._determine_batch_execution_and_padding = _determine
    runner.maybe_get_kv_connector_output = _ctx_recorder([], None)
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: False),
        _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)),
        _patch_forward_env(forward_calls),
    ):
        assert ModelRunnerFL.execute_model(runner, _sched()) is None

    set_kwargs = forward_calls[0][1]
    assert set_kwargs["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
    assert runner.calculate_kv_scales is False


def test_execute_model_pooling_path_returns_pool_output():
    """Pooling models skip logits and return _pool's output."""
    runner, _, hidden, _ = _runner(is_pooling_model=True)
    runner.maybe_get_kv_connector_output = _ctx_recorder([], None)
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: False),
        _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)),
        _patch_forward_env([]),
    ):
        out = ModelRunnerFL.execute_model(runner, _sched())

    assert out == "pooled"
    pool_args = runner._pool.calls[0][0]
    assert pool_args[0] is hidden
    assert pool_args[1] == 3
    np.testing.assert_array_equal(pool_args[2], np.array([3], dtype=np.int32))
    assert runner.execute_model_state is None
    assert runner.model.compute_logits.calls == []


def test_execute_model_non_last_pp_rank_returns_intermediate_tensors():
    """A non-last PP rank returns the hidden IntermediateTensors verbatim."""
    tensors = IntermediateTensors({"x": torch.zeros(2)})
    runner, _, _, _ = _runner()
    runner._model_forward = _Recorder(result=tensors)
    runner.maybe_get_kv_connector_output = _ctx_recorder([], "kv")
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: False),
        _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)),
        _patch_forward_env([]),
    ):
        out = ModelRunnerFL.execute_model(runner, _sched())

    assert out is tensors
    assert runner.kv_connector_output == "kv"
    assert runner.execute_model_state is None


class _BroadcastHidden:
    """IntermediateTensors-shaped stand-in carrying a tensor mapping."""

    def __init__(self):
        self.tensors = {"residual": torch.zeros(2)}

    def __getitem__(self, indices):
        return torch.zeros(len(indices), 3)


def test_execute_model_broadcast_pp_sends_and_reads_broadcast_logits():
    """broadcast_pp_output sends hidden states and adopts the broadcast logits."""
    runner, _, _, _ = _runner(broadcast_pp_output=True)
    hidden = _BroadcastHidden()
    runner._model_forward = _Recorder(result=hidden)
    sent, broadcast, gathered = [], [], []
    broadcast_logits = torch.full((2, 4), 5.0)
    pp = SimpleNamespace(
        is_last_rank=False,
        ranks=[0, 1],
        rank=0,
        send_tensor_dict=lambda *a, **k: sent.append((a, k)),
        broadcast_tensor_dict=lambda data, src: (
            broadcast.append((data, src)) or {"logits": broadcast_logits}
        ),
    )
    runner.maybe_get_kv_connector_output = _ctx_recorder([], None)
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: False),
        _Patch(mr_module, "get_pp_group", lambda: pp),
        _Patch(mr_module, "get_tp_group", lambda: "tp-group"),
        _Patch(mr_module, "is_residual_scattered_for_sp", lambda cfg, n: False),
        _patch_forward_env([]),
    ):
        assert ModelRunnerFL.execute_model(runner, _sched()) is None

    assert len(sent) == 1
    assert sent[0][0][0] is hidden.tensors
    assert sent[0][1]["all_gather_group"] == "tp-group"
    assert sent[0][1]["all_gather_tensors"] == {"residual": True}
    assert len(broadcast) == 1
    assert broadcast[0][1] == 1  # src = len(ranks) - 1
    assert runner.execute_model_state.logits is broadcast_logits
    assert runner.model.compute_logits.calls == []
    assert gathered == []


def test_execute_model_mamba_align_applies_corrections_and_stages_inputs():
    """Mamba align mode applies deferred corrections then stages fused inputs."""
    runner, calls, _, _ = _runner(
        cache_config=SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode="align"
        )
    )
    runner.mamba_state_idx = {"a": 3}
    runner.requests = {"a": object()}
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.model = SimpleNamespace(
        compute_logits=_Recorder(result=torch.zeros(1, 4)),
        get_mamba_state_copy_func=lambda: "copy-func",
    )
    runner.input_batch.num_accepted_tokens_cpu = torch.tensor([2, 3, 4, 5])
    runner._get_mamba_bufs = lambda: SimpleNamespace(
        preprocess="preprocess-buf", postprocess_align="postprocess-buf"
    )
    preprocess_mamba = _Recorder()
    stage_inputs = _Recorder()
    runner.maybe_get_kv_connector_output = _ctx_recorder([], None)
    order = []
    runner._update_states = _Recorder(result=lambda: order.append("correct"))
    with (
        _Patch(mr_module, "has_ec_transfer", lambda: False),
        _Patch(mr_module, "has_kv_transfer_group", lambda: False),
        _Patch(mr_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)),
        _Patch(
            mr_module,
            "mamba_utils",
            SimpleNamespace(
                preprocess_mamba=preprocess_mamba,
                stage_postprocess_inputs_to_gpu=stage_inputs,
            ),
        ),
        _patch_forward_env([]),
    ):
        assert ModelRunnerFL.execute_model(runner, _sched()) is None

    assert order == ["correct"]
    assert runner.num_accepted_tokens.np[:1].tolist() == [2]
    assert runner.num_accepted_tokens.gpu_ops == [1]
    pre_args = preprocess_mamba.calls[0][0]
    assert pre_args[4] is runner.input_batch
    assert pre_args[5] is runner.requests
    assert pre_args[7] == "copy-func"
    assert pre_args[8] == "preprocess-buf"
    stage_args = stage_inputs.calls[0][0]
    assert stage_args[0] == "postprocess-buf"
    assert stage_args[3] == 1
    assert stage_args[4] is runner.requests
    assert stage_args[5] is runner.mamba_state_idx
