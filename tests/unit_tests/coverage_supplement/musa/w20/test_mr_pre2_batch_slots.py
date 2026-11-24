# SPDX-License-Identifier: Apache-2.0
"""Extra branch coverage for ModelRunnerFL batch/execution helpers.

Complements ``test_mr_exec_core.py`` (w3c). Added here:

* ``_determine_batch_execution_and_padding``: encoder-output disables FULL cudagraph,
  forced LoRA state overrides, data-parallel coordination returning no token
  vector, and the DP descriptor/token-count assertion.
* ``_get_slot_mappings``: ``kv_cache_config`` present but ``None`` and present but
  empty.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr

ModelRunnerFL = mr.ModelRunnerFL


def _runner():
    return object.__new__(ModelRunnerFL)


# ---------------------------------------------------------------------------
# _determine_batch_execution_and_padding
# ---------------------------------------------------------------------------


class _Dispatcher:
    def __init__(self, tokens=None, num_reqs=None):
        self.calls = []
        self.tokens = tokens
        self.num_reqs = num_reqs

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        tokens = kwargs["num_tokens"] if self.tokens is None else self.tokens
        return (
            mr.CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=tokens, num_reqs=self.num_reqs),
        )


def _batch_runner(dispatch, metrics=False, dp=1, dp_rank=0):
    runner = _runner()
    runner.uniform_decode_query_len = 1
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=dp, data_parallel_rank=dp_rank
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=runner.parallel_config,
        observability_config=SimpleNamespace(cudagraph_metrics=metrics),
    )
    runner.cudagraph_dispatcher = dispatch
    return runner


def test_determine_batch_encoder_output_disables_full():
    dispatch = _Dispatcher()
    runner = _batch_runner(dispatch)
    runner.model_config = SimpleNamespace(is_encoder_decoder=True)
    runner._determine_batch_execution_and_padding(
        num_tokens=4,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([4]),
        max_num_scheduled_tokens=4,
        use_cascade_attn=False,
        num_encoder_reqs=1,
    )
    call = dispatch.calls[0]
    # has_encoder_output forces FULL cudagraph mode off.
    assert call["invalid_modes"] == {mr.CUDAGraphMode.FULL}


def test_determine_batch_forced_lora_state_wins():
    dispatch = _Dispatcher()
    runner = _batch_runner(dispatch)
    runner._determine_batch_execution_and_padding(
        4,
        1,
        np.array([4]),
        4,
        False,
        force_num_active_loras=0,
        force_has_lora=True,
    )
    call = dispatch.calls[0]
    assert call["num_active_loras"] == 0
    assert call["has_lora"] is True


def test_determine_batch_forced_num_active_loras_implies_lora():
    dispatch = _Dispatcher()
    runner = _batch_runner(dispatch)
    runner._determine_batch_execution_and_padding(
        4, 1, np.array([4]), 4, False, force_num_active_loras=3
    )
    call = dispatch.calls[0]
    assert call["num_active_loras"] == 3
    assert call["has_lora"] is True


def test_determine_batch_dp_without_coordination(monkeypatch):
    dispatch = _Dispatcher()
    runner = _batch_runner(dispatch, dp=2, dp_rank=0)
    monkeypatch.setattr(
        mr,
        "coordinate_batch_across_dp",
        lambda **kwargs: (False, None, mr.CUDAGraphMode.NONE.value),
        raising=False,
    )
    mode, _desc, ubatch, dp_tokens, stats = (
        runner._determine_batch_execution_and_padding(
            4, 2, np.array([2, 2]), 2, False
        )
    )
    assert mode == mr.CUDAGraphMode.NONE
    assert ubatch is False
    assert dp_tokens is None
    assert stats is None
    # No DP token vector => no re-dispatch.
    assert len(dispatch.calls) == 1


def test_determine_batch_dp_descriptor_mismatch_asserts(monkeypatch):
    runner = _batch_runner(_Dispatcher(), dp=2, dp_rank=1)
    monkeypatch.setattr(
        mr,
        "coordinate_batch_across_dp",
        lambda **kwargs: (True, torch.tensor([6, 9]), mr.CUDAGraphMode.NONE.value),
        raising=False,
    )
    # DP-synced padded size for rank 1 is 9, but dispatch claims 10.
    runner.cudagraph_dispatcher = SimpleNamespace(
        dispatch=lambda **kwargs: (
            mr.CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=kwargs["num_tokens"] + 1, num_reqs=None),
        )
    )
    with pytest.raises(AssertionError):
        runner._determine_batch_execution_and_padding(
            6, 3, np.array([2, 2, 2]), 2, False
        )


# ---------------------------------------------------------------------------
# _get_slot_mappings early exits
# ---------------------------------------------------------------------------


def test_get_slot_mappings_none_kv_cache_config():
    runner = _runner()
    runner.kv_cache_config = None
    assert runner._get_slot_mappings(2, 1, 2) == (None, None)


def test_get_slot_mappings_empty_kv_cache_groups():
    runner = _runner()
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    assert runner._get_slot_mappings(2, 1, 2) == (None, None)
