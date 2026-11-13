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

"""Coverage tests for the batch-execution and padding dispatcher.

``_determine_batch_execution_and_padding`` is the single decision point for
CUDA-graph mode, batch descriptor, ubatching, data-parallel token counts and
cudagraph metrics. The cudagraph dispatcher and the cross-DP coordinator are
module boundaries replaced with fakes; the rest of the method is exercised
directly so the assertions cover the argument plumbing and validation.
"""

from types import SimpleNamespace  # noqa: I001  (shim import must precede vllm)

import pytest

from . import _imports  # noqa: F401  (installs local import shims first)
from ._imports import mr_module
from vllm.config import CUDAGraphMode

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


class _Indexable:
    """Tensor-like object whose items report an agreed token count."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, idx):
        return SimpleNamespace(item=lambda: self.value)


class _Dispatcher:
    """CudagraphDispatcher stand-in returning queued results."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


def _desc(num_tokens, num_reqs=None, uniform=True, num_active_loras=0):
    return SimpleNamespace(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        uniform=uniform,
        num_active_loras=num_active_loras,
    )


def _runner(
    dispatcher,
    *,
    loras=0,
    enable_sp=False,
    tp_size=1,
    dp_size=1,
    is_encoder_decoder=False,
    cudagraph_metrics=False,
):
    return SimpleNamespace(
        _is_uniform_decode=_uniform_recorder(),
        uniform_decode_query_len=1,
        model_config=SimpleNamespace(is_encoder_decoder=is_encoder_decoder),
        input_batch=SimpleNamespace(
            lora_id_to_lora_request={i: object() for i in range(loras)}
        ),
        _pad_for_sequence_parallelism=lambda n: n,
        cudagraph_dispatcher=dispatcher,
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp)
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tp_size,
                data_parallel_size=dp_size,
            ),
            observability_config=SimpleNamespace(cudagraph_metrics=cudagraph_metrics),
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )


def _uniform_recorder():
    state = {}

    def _record(**kwargs):
        state.update(kwargs)
        return (
            (kwargs["max_num_scheduled_tokens"] == kwargs["uniform_decode_query_len"])
            and (
                kwargs["num_tokens"]
                == kwargs["max_num_scheduled_tokens"] * kwargs["num_reqs"]
            )
            if kwargs["force_uniform_decode"] is None
            else kwargs["force_uniform_decode"]
        )

    _record.state = state
    return _record


def _tokens(n):
    import numpy as np

    return np.ones(n, dtype=np.int32)


def test_is_uniform_decode_matches_identical_query_lengths():
    """Uniform decode requires equal query length and exact token product."""
    assert ModelRunnerFL._is_uniform_decode(1, 1, 4, 4) is True
    assert ModelRunnerFL._is_uniform_decode(2, 1, 8, 4) is False
    assert ModelRunnerFL._is_uniform_decode(2, 1, 9, 4) is False
    assert ModelRunnerFL._is_uniform_decode(3, 1, 3, 4, force_uniform_decode=True)


def test_dispatcher_force_eager_restricts_valid_modes():
    """force_eager narrows dispatch to CUDAGraphMode.NONE."""
    dispatcher = _Dispatcher([(CUDAGraphMode.NONE, _desc(4))])
    runner = _runner(dispatcher)
    with _Patch(mr_module, "coordinate_batch_across_dp", lambda **k: (False, None, 0)):
        mode, desc, ubatch, across_dp, stats = (
            ModelRunnerFL._determine_batch_execution_and_padding(
                runner,
                num_tokens=3,
                num_reqs=3,
                num_scheduled_tokens_np=_tokens(3),
                max_num_scheduled_tokens=1,
                use_cascade_attn=False,
                force_eager=True,
            )
        )

    assert mode == CUDAGraphMode.NONE
    assert desc.num_tokens == 4
    assert ubatch is False and across_dp is None and stats is None
    call = dispatcher.calls[0]
    assert call["valid_modes"] == {CUDAGraphMode.NONE}
    assert call["invalid_modes"] is None
    assert call["has_lora"] is False
    assert call["num_active_loras"] == 0
    # Uniform decode was evaluated with the query length and token product.
    assert runner._is_uniform_decode.state["max_num_scheduled_tokens"] == 1
    assert runner._is_uniform_decode.state["num_tokens"] == 3


def test_dispatcher_forbids_full_on_cascade_and_encoder_batches():
    """Cascade attention and encoder inputs both invalidate FULL graphs."""
    dispatcher = _Dispatcher([(CUDAGraphMode.NONE, _desc(4))])
    runner = _runner(dispatcher, is_encoder_decoder=True)
    with _Patch(mr_module, "coordinate_batch_across_dp", lambda **k: (False, None, 0)):
        ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=4,
            num_reqs=4,
            num_scheduled_tokens_np=_tokens(4),
            max_num_scheduled_tokens=1,
            use_cascade_attn=True,
            num_encoder_reqs=2,
        )

    assert dispatcher.calls[0]["invalid_modes"] == {CUDAGraphMode.FULL}


def test_dispatcher_lora_state_flows_into_dispatch():
    """Active LoRA count and forced flags steer the dispatcher state."""
    dispatcher = _Dispatcher([(CUDAGraphMode.NONE, _desc(4))])
    runner = _runner(dispatcher, loras=2)
    with _Patch(mr_module, "coordinate_batch_across_dp", lambda **k: (False, None, 0)):
        ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=4,
            num_reqs=4,
            num_scheduled_tokens_np=_tokens(4),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )
    assert dispatcher.calls[0]["has_lora"] is True
    assert dispatcher.calls[0]["num_active_loras"] == 2

    dispatcher = _Dispatcher([(CUDAGraphMode.NONE, _desc(4))])
    runner = _runner(dispatcher, loras=2)
    with _Patch(mr_module, "coordinate_batch_across_dp", lambda **k: (False, None, 0)):
        ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=4,
            num_reqs=4,
            num_scheduled_tokens_np=_tokens(4),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            force_has_lora=False,
            force_num_active_loras=3,
        )
    assert dispatcher.calls[0]["has_lora"] is False
    assert dispatcher.calls[0]["num_active_loras"] == 3


def test_dispatcher_sequence_parallelism_requires_multiple():
    """SP rejects a descriptor whose token count is not a TP multiple."""
    dispatcher = _Dispatcher([(CUDAGraphMode.NONE, _desc(5))])
    runner = _runner(dispatcher, enable_sp=True, tp_size=4)
    with (
        _Patch(mr_module, "coordinate_batch_across_dp", lambda **k: (False, None, 0)),
        pytest.raises(AssertionError, match="multiple of tensor parallel size"),
    ):
        ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=5,
            num_reqs=4,
            num_scheduled_tokens_np=_tokens(4),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )


def test_dispatcher_data_parallel_redispatch_uses_synced_size():
    """A DP sync re-dispatches with the agreed token count and mode."""
    dispatcher = _Dispatcher(
        [
            (CUDAGraphMode.NONE, _desc(4)),
            (CUDAGraphMode.PIECEWISE, _desc(5)),
        ]
    )
    runner = _runner(dispatcher, dp_size=2)
    coord_calls = []

    def _coordinate(**kwargs):
        coord_calls.append(kwargs)
        return True, _Indexable(5), CUDAGraphMode.PIECEWISE.value

    with _Patch(mr_module, "coordinate_batch_across_dp", _coordinate):
        mode, desc, ubatch, across_dp, _ = (
            ModelRunnerFL._determine_batch_execution_and_padding(
                runner,
                num_tokens=4,
                num_reqs=4,
                num_scheduled_tokens_np=_tokens(4),
                max_num_scheduled_tokens=1,
                use_cascade_attn=False,
            )
        )

    assert coord_calls[0]["num_tokens_unpadded"] == 4
    assert coord_calls[0]["num_tokens_padded"] == 4
    assert coord_calls[0]["cudagraph_mode"] == CUDAGraphMode.NONE.value
    assert ubatch is True
    assert across_dp is not None
    assert desc.num_tokens == 5
    assert mode == CUDAGraphMode.PIECEWISE
    assert len(dispatcher.calls) == 2
    assert dispatcher.calls[1]["valid_modes"] == {CUDAGraphMode.PIECEWISE}


def test_dispatcher_reports_cudagraph_padding_metrics():
    """cudagraph_metrics builds a CUDAGraphStat from padded minus unpadded."""
    dispatcher = _Dispatcher([(CUDAGraphMode.FULL, _desc(8))])
    runner = _runner(dispatcher, cudagraph_metrics=True)
    with _Patch(mr_module, "coordinate_batch_across_dp", lambda **k: (False, None, 0)):
        _, _, _, _, stats = ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=6,
            num_reqs=4,
            num_scheduled_tokens_np=_tokens(4),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )

    assert stats.num_unpadded_tokens == 6
    assert stats.num_padded_tokens == 8
    assert stats.num_paddings == 2
    assert stats.runtime_mode == str(CUDAGraphMode.FULL)
