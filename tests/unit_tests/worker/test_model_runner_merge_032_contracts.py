"""CPU contracts for graph mode selection in the 0.32 runner."""

from enum import IntEnum
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.unit_tests.worker.test_model_runner_032_contracts import _runner_class


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@pytest.mark.parametrize("dp", [False, True])
@pytest.mark.parametrize("reason", ["none", "explicit", "cascade", "encoder", "eager"])
def test_full_graph_exclusions_survive_dp_redispatch(dp, reason):
    calls = []

    def dispatch(**kwargs):
        calls.append(kwargs)
        if kwargs["valid_modes"] is not None:
            mode = next(iter(kwargs["valid_modes"]))
        elif kwargs["invalid_modes"]:
            mode = Mode.PIECEWISE
        else:
            mode = Mode.FULL
        return mode, SimpleNamespace(num_tokens=kwargs["num_tokens"])

    parallel = SimpleNamespace(data_parallel_size=2 if dp else 1, data_parallel_rank=0)

    def coordinate(**kwargs):
        return False, torch.tensor([8, 8]), kwargs["cudagraph_mode"]

    runner = _runner_class(
        {"_determine_batch_execution_and_padding"},
        CUDAGraphMode=Mode,
        coordinate_batch_across_dp=coordinate,
    )()
    runner.uniform_decode_query_len = 1
    runner._is_uniform_decode = lambda **kwargs: True
    runner.model_config = SimpleNamespace(is_encoder_decoder=reason == "encoder")
    runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
    runner.parallel_config = parallel
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    runner.cudagraph_dispatcher = SimpleNamespace(dispatch=dispatch)
    mode, descriptor, *_ = runner._determine_batch_execution_and_padding(
        4,
        4,
        np.ones(4, dtype=np.int32),
        1,
        use_cascade_attn=reason == "cascade",
        disable_full=reason == "explicit",
        num_encoder_reqs=1 if reason == "encoder" else 0,
        force_eager=reason == "eager",
    )
    expected_invalid = (
        {Mode.FULL} if reason in ("explicit", "cascade", "encoder") else None
    )
    assert len(calls) == (2 if dp else 1)
    assert all(call["invalid_modes"] == expected_invalid for call in calls)
    assert mode is (
        Mode.NONE
        if reason == "eager"
        else Mode.PIECEWISE
        if expected_invalid
        else Mode.FULL
    )
    assert descriptor.num_tokens == (8 if dp else 4)
    if reason == "eager":
        assert all(call["valid_modes"] == {Mode.NONE} for call in calls)
