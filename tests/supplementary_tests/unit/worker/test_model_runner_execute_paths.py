# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ModelRunnerFL execution-path plumbing.

CPU-only checks (production methods, stub doubles for heavy machinery):

* ``execute_model``: state-machine error, empty-batch early returns, the
  non-last PP rank intermediate-tensor return, and last-rank pooling
  output shape.
* ``sample_tokens``: no-op state path returns the KV-connector-only output.
* ``_bookkeeping_sync``: async-scheduling branch bookkeeping.
* ``_get_nans_in_logits`` + ``AsyncGPUModelRunnerOutput`` / pooler-output
  copy helpers.
* ``maybe_randomize_inputs`` context manager.
* ``_allow_microbatching`` veto logic.
* ``timed_encoder_operation`` timing registry.
* ``_pool`` / ``_copy_pooler_output_to_cpu`` tensor/list handling.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.supplementary]

MR = pytest.importorskip("vllm_fl.worker.model_runner")


# ---------------------------------------------------------------------------
# execute_model
# ---------------------------------------------------------------------------


def _empty_scheduler_output(**overrides):
    """A minimal but complete SchedulerOutput for _update_states plumbing."""
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        SchedulerOutput,
    )

    fields = dict(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[], resumed_req_ids=set(), new_token_ids=[],
            all_token_ids={}, new_block_ids=[], num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    fields.update(overrides)
    return SchedulerOutput(**fields)



def _pp(monkeypatch, is_last_rank=True, is_first_rank=True):
    group = NS(
        is_last_rank=is_last_rank,
        is_first_rank=is_first_rank,
        ranks=[0],
        world_size=1,
    )
    monkeypatch.setattr(MR, "get_pp_group", lambda: group)
    return group


def test_execute_model_rejects_pending_state():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.execute_model_state = object()
    with pytest.raises(RuntimeError, match="sample_tokens"):
        MR.ModelRunnerFL.execute_model(
            instance, NS(total_num_scheduled_tokens=0)
        )


def test_execute_model_empty_batch_returns_empty_output(monkeypatch):
    _pp(monkeypatch)
    monkeypatch.setattr(MR, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(MR, "has_ec_transfer", lambda: False)
    instance = object.__new__(MR.ModelRunnerFL)
    instance.execute_model_state = None
    instance.speculative_config = None
    instance.prepare_inputs_event = None
    instance.parallel_config = NS(
        distributed_executor_backend="mp", data_parallel_size=1
    )
    # _update_states state (nothing scheduled -> pure removal path).
    instance.requests = {}
    instance.num_prompt_logprobs = {}
    instance.late_interaction_runner = Mock()
    instance.input_batch = Mock(
        req_id_to_index={},
        remove_request=Mock(),
        condense=Mock(),
        refresh_metadata=Mock(),
    )
    instance.use_async_scheduling = False
    instance.use_async_spec_decode = False
    instance.kv_cache_config = NS(kv_cache_groups=[])
    instance.reorder_batch_threshold = None
    out = MR.ModelRunnerFL.execute_model(instance, _empty_scheduler_output())
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    assert out is EMPTY_MODEL_RUNNER_OUTPUT


def test_execute_model_empty_batch_with_kv_connector(monkeypatch):
    _pp(monkeypatch)
    monkeypatch.setattr(MR, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(MR, "has_ec_transfer", lambda: False)
    sentinel = object()
    monkeypatch.setattr(
        MR.ModelRunnerFL,
        "kv_connector_no_forward",
        lambda self, so, cfg: sentinel,
    )
    instance = object.__new__(MR.ModelRunnerFL)
    instance.execute_model_state = None
    instance.speculative_config = None
    instance.prepare_inputs_event = None
    instance.parallel_config = NS(
        distributed_executor_backend="mp", data_parallel_size=1
    )
    # _update_states state (nothing scheduled -> pure removal path).
    instance.requests = {}
    instance.num_prompt_logprobs = {}
    instance.late_interaction_runner = Mock()
    instance.input_batch = Mock(
        req_id_to_index={},
        remove_request=Mock(),
        condense=Mock(),
        refresh_metadata=Mock(),
    )
    instance.use_async_scheduling = False
    instance.use_async_spec_decode = False
    instance.kv_cache_config = NS(kv_cache_groups=[])
    instance.reorder_batch_threshold = None
    instance.vllm_config = object()
    kv_group = NS(handle_preemptions=Mock())
    monkeypatch.setattr(MR, "get_kv_transfer_group", lambda: kv_group)
    so = _empty_scheduler_output(kv_connector_metadata="meta")
    out = MR.ModelRunnerFL.execute_model(instance, so)
    assert out is sentinel
    kv_group.handle_preemptions.assert_called_once_with("meta")


def test_execute_model_ec_producer_short_circuits(monkeypatch):
    group = _pp(monkeypatch)
    monkeypatch.setattr(MR, "has_ec_transfer", lambda: True)
    ec = NS(is_consumer=False)
    monkeypatch.setattr(MR, "get_ec_transfer", lambda: ec)
    connector_ctx = nullcontext()
    instance = object.__new__(MR.ModelRunnerFL)
    instance.execute_model_state = None
    instance.prepare_inputs_event = None
    instance.parallel_config = NS(
        distributed_executor_backend="mp", data_parallel_size=1
    )
    # _update_states is exercised through a stub scheduler output that
    # finishes nothing and schedules nothing.
    instance.requests = {}
    instance.num_prompt_logprobs = {}
    instance.late_interaction_runner = Mock()
    instance.input_batch = Mock(
        req_id_to_index={},
        num_reqs=0,
        remove_request=Mock(),
    )
    instance.speculative_config = None
    instance.use_async_scheduling = False
    instance.use_async_spec_decode = False
    instance.kv_cache_config = NS(kv_cache_groups=[])
    instance.reorder_batch_threshold = None
    instance.encoder_cache = {}
    monkeypatch.setattr(
        MR.ModelRunnerFL,
        "maybe_get_ec_connector_output",
        lambda self, scheduler_output, encoder_cache: connector_ctx,
    )
    ran_encoder = []
    monkeypatch.setattr(
        MR.ModelRunnerFL,
        "_execute_mm_encoder",
        lambda self, so: ran_encoder.append(so),
    )
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        SchedulerOutput,
    )

    so = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[], resumed_req_ids=set(), new_token_ids=[],
            all_token_ids={}, new_block_ids=[], num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=5,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    out = MR.ModelRunnerFL.execute_model(instance, so)
    from vllm.v1.outputs import ModelRunnerOutput

    assert isinstance(out, ModelRunnerOutput)
    assert out.sampled_token_ids == []
    assert ran_encoder == [so]


def test_sample_tokens_without_state_returns_kv_conn_only(monkeypatch):
    _pp(monkeypatch, is_last_rank=False, is_first_rank=True)
    monkeypatch.setattr(
        MR.ModelRunnerFL,
        "_pp_receive_prev_sampled_token_ids_to_input_batch",
        lambda self: None,
    )
    instance = object.__new__(MR.ModelRunnerFL)
    instance.execute_model_state = None
    kv_out = NS(is_empty=lambda: False)
    instance.kv_connector_output = kv_out
    instance.use_async_scheduling = True
    instance.broadcast_pp_output = False
    out = MR.ModelRunnerFL.sample_tokens(instance, None)
    # The connector output is passed through and the slot cleared.
    from vllm.v1.outputs import ModelRunnerOutput

    assert isinstance(out, ModelRunnerOutput)
    assert out.kv_connector_output is kv_out
    assert instance.kv_connector_output is None


# ---------------------------------------------------------------------------
# _bookkeeping_sync async branch
# ---------------------------------------------------------------------------


def test_bookkeeping_sync_async_branch(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    monkeypatch.setattr(MR, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(MR.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False)

    # async scheduling: sampled tokens stay on GPU; CPU gets placeholders.
    req_ids = ["a", "b"]
    discard = np.array([True, False])
    instance.use_async_scheduling = True
    instance.input_batch = NS(
        num_reqs=2,
        req_ids=req_ids,
        req_id_to_index={"a": 0, "b": 1},
        prev_sampled_token_ids=None,
        vocab_size=100,
        num_tokens_no_spec=[4, 6],
        token_ids_cpu=np.zeros((2, 16), dtype=np.int64),
        is_token_ids=np.zeros((2, 16), dtype=bool),
        generators={},
        requests={
            "a": NS(output_token_ids=[]),
            "b": NS(output_token_ids=[]),
        },
    )
    instance.discard_request_mask = NS(np=discard)
    instance._cpu_generator_states = {}
    instance.routed_experts_initialized = False
    instance.max_model_len = 16
    instance.requests = instance.input_batch.requests

    sampled = torch.tensor([[9], [7]])
    sampler_output = NS(
        sampled_token_ids=sampled, logprobs_tensors=None
    )
    monkeypatch.setattr(
        MR.ModelRunnerFL,
        "_get_prompt_logprobs_dict",
        lambda self, hidden, scheduled: {},
    )
    hidden = torch.zeros(10, 4)
    (
        num_nans_in_logits,
        num_nans,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_copy,
        req_id_to_index_copy,
        invalid_req_indices,
    ) = MR.ModelRunnerFL._bookkeeping_sync(
        instance,
        NS(total_num_scheduled_tokens=10, num_scheduled_tokens={"a": 5, "b": 5}),
        sampler_output,
        None,
        hidden,
        10,
    )
    assert num_nans_in_logits == {}
    assert num_nans is None
    assert logprobs_lists is None
    assert valid_sampled_token_ids == []
    # discard-masked request 'a' is invalid; 'b' stays valid.
    assert invalid_req_indices == [0]
    # prev_sampled_token_ids cached on the input batch.
    assert instance.input_batch.prev_sampled_token_ids is sampled
    # prev_req_id_to_index only keeps valid rows.
    assert instance.input_batch.prev_req_id_to_index == {"b": 1}
    # The discarded request wrote no tokens; 'b' recorded the async
    # placeholder token (-1) pending the next _prepare_input_ids fill.
    assert instance.requests["a"].output_token_ids == []
    assert instance.requests["b"].output_token_ids == [-1]
    assert req_ids_copy == req_ids


def test_bookkeeping_sync_sync_branch_single_token(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    monkeypatch.setattr(MR, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(MR.envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False)

    instance.use_async_scheduling = False
    instance.input_batch = NS(
        num_reqs=2,
        req_ids=["x", "y"],
        req_id_to_index={"x": 0, "y": 1},
        vocab_size=100,
        num_tokens_no_spec=[3, 5],
        token_ids_cpu=np.zeros((2, 16), dtype=np.int64),
        is_token_ids=np.zeros((2, 16), dtype=bool),
        generators={},
        requests={
            "x": NS(output_token_ids=[]),
            "y": NS(output_token_ids=[]),
        },
    )
    instance.discard_request_mask = NS(np=np.array([False, False]))
    instance._cpu_generator_states = {}
    instance.routed_experts_initialized = False
    instance.max_model_len = 16
    instance.requests = instance.input_batch.requests

    sampled = torch.tensor([[4], [8]])
    sampler_output = NS(sampled_token_ids=sampled, logprobs_tensors=None)
    monkeypatch.setattr(
        MR.ModelRunnerFL,
        "_get_prompt_logprobs_dict",
        lambda self, hidden, scheduled: {},
    )
    pinned = Mock()
    pinned.tolist = lambda: [[4], [8]]
    monkeypatch.setattr(
        MR.ModelRunnerFL, "_to_list", lambda self, t: [[4], [8]]
    )
    (
        _num_nans_in_logits,
        _num_nans,
        logprobs_lists,
        valid_sampled_token_ids,
        _prompt_logprobs_dict,
        _req_ids,
        _req_id_to_index,
        invalid_req_indices,
    ) = MR.ModelRunnerFL._bookkeeping_sync(
        instance,
        NS(total_num_scheduled_tokens=2, num_scheduled_tokens={"x": 1, "y": 1}),
        sampler_output,
        None,
        torch.zeros(2, 4),
        2,
    )
    assert invalid_req_indices == []
    assert valid_sampled_token_ids == [[4], [8]]
    assert logprobs_lists is None
    assert instance.requests["x"].output_token_ids == [4]
    assert instance.requests["y"].output_token_ids == [8]
    assert instance.input_batch.num_tokens_no_spec == [4, 6]


# ---------------------------------------------------------------------------
# NaN diagnostics and async output wrappers
# ---------------------------------------------------------------------------


def test_get_nans_in_logits_maps_requests(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    monkeypatch.setattr(MR, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(MR.envs, "VLLM_RAISE_ON_LOGIT_NANS", False)
    instance.input_batch = NS(req_id_to_index={"r0": 0, "r1": 1})

    logits = torch.tensor(
        [[float("nan"), 1.0], [0.0, float("nan")]]
    )
    out = MR.ModelRunnerFL._get_nans_in_logits(instance, logits)
    assert out == {"r0": 1, "r1": 1}

    # None logits reports zeros.
    assert MR.ModelRunnerFL._get_nans_in_logits(instance, None) == {
        "r0": 0, "r1": 0,
    }


def test_get_nans_in_logits_raises_when_configured(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    monkeypatch.setattr(MR, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(MR.envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
    instance.input_batch = NS(req_id_to_index={"r0": 0})

    logits = torch.tensor([[float("nan"), float("nan")]])

    def fail(d):
        raise RuntimeError("nan logits found")

    monkeypatch.setattr(MR, "raise_if_nan_logits", fail)
    with pytest.raises(RuntimeError, match="nan logits"):
        MR.ModelRunnerFL._get_nans_in_logits(instance, logits)


def test_async_gpu_model_runner_output_max_gen_len_one():
    """get_output with max_gen_len == 1: sampled ids cleared per request."""
    import types

    platform = types.SimpleNamespace()
    # Build a minimal fake torch_device_fn: CPU works for everything we need.
    import vllm.platforms as platforms

    # Real CPU platform for the stream/event plumbing on the CPU venv:
    # the FL platform (cuda/kunlunxin) exposes CUDA streams that don't
    # exist here, so substitute a plain stub namespace.
    fake_stream = types.SimpleNamespace(
        wait_stream=lambda other: None,
        record=lambda: None,
        synchronize=lambda: None,
    )
    device_fn = types.SimpleNamespace(
        Stream=lambda *a, **k: fake_stream,
        Event=lambda blocking=False: types.SimpleNamespace(
            record=lambda: None, synchronize=lambda: None, item=lambda: False
        ),
        current_stream=lambda: fake_stream,
        stream=lambda s: nullcontext(),
    )
    monkey_platform = types.SimpleNamespace(torch_device_fn=device_fn)

    import vllm_fl.worker.model_runner as mr

    original = mr.current_platform
    mr.current_platform = monkey_platform
    try:
        output = MR.ModelRunnerOutput(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1})
        sampled = torch.tensor([[5], [6]])
        async_out = MR.AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[0],
            async_output_copy_stream=fake_stream,
            vocab_size=10,
        )
        assert async_out.get_output() is output
        # Invalid request rows are cleared.
        assert output.sampled_token_ids == [[], [6]]
        assert output.logprobs is None
    finally:
        mr.current_platform = original


def test_copy_pooler_output_to_cpu_tensor_paths():
    pooler = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    # All finished: returns every row on CPU.
    out = MR._copy_pooler_output_to_cpu(pooler, [True, True, True])
    assert len(out) == 3
    assert torch.equal(out[1], pooler[1].cpu())

    # None finished: all rows dropped.
    assert MR._copy_pooler_output_to_cpu(pooler, [False, False, False]) == [
        None, None, None,
    ]

    # Partial: only the finished rows are transferred.
    out = MR._copy_pooler_output_to_cpu(pooler, [True, False, True])
    assert out[0] is not None and out[1] is None and out[2] is not None
    assert torch.equal(out[2], pooler[2].cpu())

    # Batch-size mismatch raises.
    with pytest.raises(ValueError, match="does not match"):
        MR._copy_pooler_output_to_cpu(pooler, [True])


def test_copy_pooler_output_to_cpu_list_paths():
    rows = [torch.ones(2), None, torch.zeros(2)]
    out = MR._copy_pooler_output_to_cpu(rows, [True, True, True])
    assert out[0] is rows[0]
    assert out[1] is None
    assert out[2] is rows[2]

    with pytest.raises(ValueError, match="does not match"):
        MR._copy_pooler_output_to_cpu(rows, [True])


# ---------------------------------------------------------------------------
# maybe_randomize_inputs
# ---------------------------------------------------------------------------


def test_maybe_randomize_inputs_passthrough(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    instance.vllm_config = NS(
        parallel_config=NS(data_parallel_size=1)
    )
    monkeypatch.setattr(MR.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False)
    ran = []
    with MR.ModelRunnerFL.maybe_randomize_inputs(instance, None, None):
        ran.append("inside")
    assert ran == ["inside"]


def test_maybe_randomize_inputs_randomizes_token_ids(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    instance.vllm_config = NS(parallel_config=NS(data_parallel_size=2))
    instance.input_ids = NS(gpu=torch.zeros(8, dtype=torch.int64))
    instance.model_config = NS(get_vocab_size=lambda: 50)

    input_ids = torch.zeros(4, dtype=torch.int64)
    with MR.ModelRunnerFL.maybe_randomize_inputs(
        instance, input_ids, None, randomize_inputs=True
    ):
        inside = input_ids.clone()
    # After the block, ids are zeroed again.
    assert bool((input_ids == 0).all())
    # And during the block they were randomized (nonzero somewhere).
    assert int(inside.abs().sum()) > 0


def test_maybe_randomize_inputs_randomizes_embeds(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    instance.vllm_config = NS(parallel_config=NS(data_parallel_size=2))
    instance.inputs_embeds = NS(gpu=torch.zeros(4, 3))

    embeds = torch.zeros(4, 3)
    with MR.ModelRunnerFL.maybe_randomize_inputs(
        instance, None, embeds, randomize_inputs=True
    ):
        inside = embeds.clone()
    assert bool((embeds == 0).all())
    assert float(inside.abs().sum()) > 0


# ---------------------------------------------------------------------------
# _allow_microbatching
# ---------------------------------------------------------------------------


def test_allow_microbatching_disabled_or_single_request():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.parallel_config = NS(use_ubatching=False)
    instance.input_batch = Mock()
    assert (
        MR.ModelRunnerFL._allow_microbatching(instance, 4, np.array([1, 2, 3, 4]))
        is True
    )

    instance.parallel_config = NS(use_ubatching=True)
    assert (
        MR.ModelRunnerFL._allow_microbatching(instance, 1, np.array([5]))
        is True
    )


def test_allow_microbatching_decode_only_batch_allows():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.parallel_config = NS(use_ubatching=True)
    instance.reorder_batch_threshold = None
    instance.input_batch = Mock()
    # computed == 0 for every request: nothing reads an unwritten block.
    instance.input_batch.num_computed_tokens_cpu = np.array([0, 0])
    assert (
        MR.ModelRunnerFL._allow_microbatching(instance, 2, np.array([1, 1]))
        is True
    )


def test_allow_microbatching_vetoes_shared_block_reads():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.parallel_config = NS(use_ubatching=True)
    instance.reorder_batch_threshold = None

    # Two requests, both mid-prefill (computed > 0) with multi-token queries.
    # The first request's read span covers block 5 which the second request
    # is still writing -> the first request's "whole" blocks (block 5)
    # intersect the written set -> veto.
    table = np.array([[5, 6, 0, 0], [5, 0, 0, 0]])
    block_table = Mock()
    block_table.block_size = 2
    block_table.get_numpy_array = lambda: table
    instance.input_batch = NS(
        num_computed_tokens_cpu=np.array([4, 0]),
        block_table=NS(block_tables=[block_table]),
    )
    # Request 0: computed=4, query=2 -> whole blocks = [0:2) -> blocks 5,6.
    # Request 1: computed=0 -> span starts at 0, covers block 5.
    assert (
        MR.ModelRunnerFL._allow_microbatching(instance, 2, np.array([2, 2]))
        is False
    )


def test_allow_microbatching_allows_disjoint_blocks():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.parallel_config = NS(use_ubatching=True)
    instance.reorder_batch_threshold = None

    table = np.array([[1, 2, 0, 0], [3, 4, 0, 0]])
    block_table = Mock()
    block_table.block_size = 2
    block_table.get_numpy_array = lambda: table
    instance.input_batch = NS(
        num_computed_tokens_cpu=np.array([2, 2]),
        block_table=NS(block_tables=[block_table]),
    )
    assert (
        MR.ModelRunnerFL._allow_microbatching(instance, 2, np.array([2, 2]))
        is True
    )


# ---------------------------------------------------------------------------
# timed_encoder_operation
# ---------------------------------------------------------------------------


def test_timed_encoder_operation_disabled_is_passthrough():
    instance = object.__new__(MR.ModelRunnerFL)
    ran = []
    with MR.ModelRunnerFL.timed_encoder_operation(
        instance, False, [], 0, 0
    ):
        ran.append("inside")
    assert ran == ["inside"]
    # No registry created for a disabled timer.
    assert not hasattr(instance, "encoder_timing_registry")


def test_timed_encoder_operation_records_per_request(monkeypatch):
    import threading

    instance = object.__new__(MR.ModelRunnerFL)
    instance._encoder_timing_lock = threading.Lock()
    instance.encoder_timing_registry = {}
    monkeypatch.setattr(MR.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(MR.time, "perf_counter", lambda: 1.0)

    refs = [("req-a", object()), ("req-b", object())]
    with MR.ModelRunnerFL.timed_encoder_operation(instance, True, refs, 0, 2):
        pass
    assert set(instance.encoder_timing_registry) == {"req-a", "req-b"}
    for stats in instance.encoder_timing_registry.values():
        assert stats.num_encoder_calls == 1
    # Second timing call accumulates.
    monkeypatch.setattr(MR.time, "perf_counter", lambda: 2.0)
    with MR.ModelRunnerFL.timed_encoder_operation(instance, True, refs, 0, 2):
        pass
    assert instance.encoder_timing_registry["req-a"].num_encoder_calls == 2


# ---------------------------------------------------------------------------
# _pool / pooling output plumbing
# ---------------------------------------------------------------------------


def test_pool_returns_none_output_when_no_finished(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    monkeypatch.setattr(
        MR.ModelRunnerFL, "_sync_device", lambda self: None
    )
    instance.input_batch = NS(
        num_reqs=1,
        pooling_params={"p1": NS(task="classify")},
        get_pooling_metadata=lambda: NS(
            build_pooling_cursor=lambda *a, **k: None,
            prompt_lens=torch.tensor([6]),
            pooling_params=[],
        ),
        req_ids=["p1"],
        req_id_to_index={"p1": 0},
    )
    num_scheduled = np.array([6])
    instance.optimistic_seq_lens_cpu = torch.tensor([6])
    instance.query_start_loc = NS(gpu=torch.tensor([0, 6]))
    instance.model = NS(pooler=Mock(return_value=torch.ones(1, 2)))
    instance.late_interaction_runner = NS(
        postprocess_pooler_output=lambda **kw: None
    )
    hidden = torch.zeros(6, 4)

    out = MR.ModelRunnerFL._pool(
        instance, hidden, 6, num_scheduled, None
    )
    # No finished request: pooler output is all-None, sync path.
    assert out.pooler_output == [None]


# ---------------------------------------------------------------------------
# _input_fits_in_drafter / take_draft_token_ids / get_draft_model
# ---------------------------------------------------------------------------


def test_input_fits_in_drafter_requires_metadata():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.speculative_config = NS(use_dflash=lambda: False)
    instance.num_spec_tokens = 2
    instance.effective_drafter_max_model_len = 100
    assert MR.ModelRunnerFL._input_fits_in_drafter(instance, None) is False

    cm = NS(max_seq_len=90)
    assert MR.ModelRunnerFL._input_fits_in_drafter(instance, cm) is True
    cm = NS(max_seq_len=99)
    assert MR.ModelRunnerFL._input_fits_in_drafter(instance, cm) is False


def test_input_fits_in_drafter_dflash_bonus_token():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.speculative_config = NS(use_dflash=lambda: True)
    instance.num_spec_tokens = 2
    instance.effective_drafter_max_model_len = 100
    # DFlash queries one extra (bonus) token beyond num_spec_tokens.
    assert MR.ModelRunnerFL._input_fits_in_drafter(
        instance, NS(max_seq_len=97)
    ) is True
    assert MR.ModelRunnerFL._input_fits_in_drafter(
        instance, NS(max_seq_len=98)
    ) is False


def test_take_draft_token_ids_requires_tokens_and_ids(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    instance.num_spec_tokens = 0
    assert MR.ModelRunnerFL.take_draft_token_ids(instance) is None

    instance.num_spec_tokens = 2
    instance._draft_token_req_ids = None
    assert MR.ModelRunnerFL.take_draft_token_ids(instance) is None

    instance._draft_token_req_ids = ["a"]
    monkeypatch.setattr(
        MR.ModelRunnerFL,
        "_get_draft_token_ids_cpu",
        lambda self: ([[1, 2]], ["a"]),
    )
    from vllm.v1.outputs import DraftTokenIds

    out = MR.ModelRunnerFL.take_draft_token_ids(instance)
    assert isinstance(out, DraftTokenIds)
    assert out.req_ids == ["a"]
    assert out.draft_token_ids == [[1, 2]]


def test_get_draft_model_unwraps_graph_wrappers():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.drafter = None
    assert MR.ModelRunnerFL.get_draft_model(instance) is None

    raw = object()
    wrapped = MR.GraphWrapper.__new__(MR.GraphWrapper)
    wrapped.unwrap = lambda: raw
    instance.drafter = NS(model=wrapped)
    assert MR.ModelRunnerFL.get_draft_model(instance) is raw

    instance.drafter = NS(model="plain-module")
    assert MR.ModelRunnerFL.get_draft_model(instance) == "plain-module"


def test_get_model_unwraps_and_validates():
    instance = object.__new__(MR.ModelRunnerFL)
    # No model attribute: ValueError.
    try:
        MR.ModelRunnerFL.get_model(instance)
        raised = False
    except ValueError:
        raised = True
    assert raised

    raw = object()
    wrapped = MR.UBatchWrapper.__new__(MR.UBatchWrapper)
    wrapped.unwrap = lambda: raw
    instance.model = wrapped
    assert MR.ModelRunnerFL.get_model(instance) is raw

    instance.model = "bare-module"
    assert MR.ModelRunnerFL.get_model(instance) == "bare-module"


def test_get_supported_tasks_dispatches_on_runner_type():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.model_config = NS(runner_type="generate")
    instance.get_model = lambda: "fake-model"
    monkey_mr = MR.ModelRunnerFL.get_supported_generation_tasks
    try:
        MR.ModelRunnerFL.get_supported_generation_tasks = (
            lambda self: ["generate"]
        )
        assert MR.ModelRunnerFL.get_supported_tasks(instance) == ("generate",)

        instance.model_config = NS(runner_type="pooling")
        MR.ModelRunnerFL.get_supported_generation_tasks = monkey_mr
        MR.ModelRunnerFL.get_supported_pooling_tasks = lambda self: ["embed"]
        assert MR.ModelRunnerFL.get_supported_tasks(instance) == ("embed",)
    finally:
        MR.ModelRunnerFL.get_supported_generation_tasks = monkey_mr
        MR.ModelRunnerFL.get_supported_pooling_tasks = (
            type(instance).get_supported_pooling_tasks
        )


# ---------------------------------------------------------------------------
# _to_list pinned buffer transfer
# ---------------------------------------------------------------------------


def test_to_list_uses_pinned_buffer_and_event(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    sampled = torch.tensor([[3], [4]])
    pinned = torch.zeros(4, 1, dtype=torch.int64)
    instance.sampled_token_ids_pinned_cpu = pinned
    event = Mock()
    instance.transfer_event = event
    out = MR.ModelRunnerFL._to_list(instance, sampled)
    assert out == [[3], [4]]
    assert pinned[:2].tolist() == [[3], [4]]
    event.record.assert_called_once_with()
    event.synchronize.assert_called_once_with()


# ---------------------------------------------------------------------------
# get_kv_cache_spec routing
# ---------------------------------------------------------------------------


class TestGetKvCacheSpec:
    def _instance(self):
        instance = object.__new__(MR.ModelRunnerFL)
        instance.shared_kv_cache_layers = {}
        instance.use_ascend_hybrid_cache = False
        instance.vllm_config = NS(compilation_config=NS(static_forward_context={}))
        return instance

    def test_ec_producer_returns_empty(self, monkeypatch):
        instance = self._instance()
        monkeypatch.setattr(MR, "has_ec_transfer", lambda: True)
        monkeypatch.setattr(
            MR, "get_ec_transfer", lambda: NS(is_consumer=False)
        )
        assert MR.ModelRunnerFL.get_kv_cache_spec(instance) == {}

    def test_attention_layers_with_sharing_target_are_skipped(
        self, monkeypatch
    ):
        instance = self._instance()
        monkeypatch.setattr(MR, "has_ec_transfer", lambda: False)
        called = []

        # The sharing continue only runs for real Attention modules
        # (isinstance check in production). Use a real Attention shell.
        class FakeAttention(MR.Attention):
            def __init__(self, target, spec_result):
                self.kv_sharing_target_layer_name = target
                self._spec_result = spec_result
                self._called = False

            def get_kv_cache_spec(self, cfg):
                self._called = True
                return self._spec_result

        attn_a = FakeAttention(None, None)
        attn_b = FakeAttention("attn-a", "never-consulted")
        monkeypatch.setattr(
            MR,
            "get_layers_from_vllm_config",
            lambda cfg, layer_type: {"attn-a": attn_a, "attn-b": attn_b},
        )
        spec = MR.ModelRunnerFL.get_kv_cache_spec(instance)
        # attn-a maps to no KV cache; attn-b is skipped via sharing.
        assert spec == {}
        assert instance.shared_kv_cache_layers == {"attn-b": "attn-a"}
        # The sharing layer's get_kv_cache_spec was never consulted.
        assert attn_b._called is False
        assert attn_a._called is True

    def test_non_attention_spec_recorded_directly(self, monkeypatch):
        instance = self._instance()
        monkeypatch.setattr(MR, "has_ec_transfer", lambda: False)
        mamba_spec = object()
        monkeypatch.setattr(
            MR,
            "get_layers_from_vllm_config",
            lambda cfg, layer_type: {
                "mamba-0": NS(
                    kv_sharing_target_layer_name=None,
                    get_kv_cache_spec=lambda cfg: mamba_spec,
                ),
            },
        )
        spec = MR.ModelRunnerFL.get_kv_cache_spec(instance)
        assert spec == {"mamba-0": mamba_spec}
