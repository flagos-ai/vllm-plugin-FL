# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ModelRunnerFL batch-state plumbing.

CPU-only checks against production methods with a real ``InputBatch`` /
``CachedRequestState`` / ``SchedulerOutput`` built on CPU:

* ``_update_states``: finished-request removal, new/resumed request add,
  block-table append, output-token bookkeeping and the async spec-decode
  deferred-correction closure.
* ``_update_streaming_request``: cached-state refresh for streaming
  sessions.
* ``_calc_spec_decode_metadata``: index math oracles.
* ``_compute_prev_positions`` / ``_get_cumsum_and_arange`` reuse.
* ``_get_slot_mappings``: per-group slot mapping construction and -1 padding.
* ``_get_encoder_seq_lens``: cross-attention path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace as NS
from unittest.mock import Mock

pytestmark = [pytest.mark.supplementary]

MR = pytest.importorskip("vllm_fl.worker.model_runner")

# ---------------------------------------------------------------------------
# Shared CPU harness: real InputBatch on CPU with pin-memory neutralised.
# ---------------------------------------------------------------------------


@pytest.fixture()
def cpu_env(monkeypatch):
    """Patch every pin-memory decision point for CPU-only construction."""
    import vllm.v1.utils as v1u
    import vllm.v1.worker.gpu_input_batch as gib
    import vllm.utils.torch_utils as tt

    # keyword-only default captured at class definition time
    monkeypatch.setitem(
        v1u.CpuGpuBuffer.__init__.__kwdefaults__, "pin_memory", False
    )
    monkeypatch.setattr(v1u, "PIN_MEMORY", False)
    monkeypatch.setattr(gib, "PIN_MEMORY", False)
    monkeypatch.setattr(tt, "is_pin_memory_available", lambda: False)

    from vllm.sampling_params import SamplingParams

    class Env:
        device = torch.device("cpu")

        @staticmethod
        def make_batch(reqs=(), max_num_reqs=8, max_model_len=64,
                       num_spec_tokens=0):
            batch = gib.InputBatch(
                max_num_reqs=max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=64,
                device=Env.device,
                vocab_size=32,
                block_sizes=[16],
                max_num_blocks_per_req=[cdiv(max_model_len, 16)],
                num_spec_tokens=num_spec_tokens,
            )
            for req in reqs:
                batch.add_request(req)
            return batch

        @staticmethod
        def req_state(req_id, prompt_len=10, block_ids=None, computed=0,
                      spec_tokens=None, prev_draft_len=0, output=None):
            return gib.CachedRequestState(
                req_id=req_id,
                prompt_token_ids=list(range(prompt_len)),
                mm_features=[],
                sampling_params=SamplingParams(temperature=0.0),
                generator=None,
                block_ids=(list(block_ids or [0, 1]),),
                num_computed_tokens=computed,
                output_token_ids=list(output or []),
                prev_num_draft_len=prev_draft_len,
            )

    return Env


def cdiv(a, b):
    return -(-a // b)


def _scheduler_output(new_reqs=(), cached_reqs=None, num_scheduled=None,
                      spec_tokens=None, finished=(), encoder_inputs=None,
                      total=None, new_block_ids_to_zero=None,
                      kv_cache_block_copies=None):
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        SchedulerOutput,
    )

    cached = cached_reqs or CachedRequestData(
        req_ids=[], resumed_req_ids=set(), new_token_ids=[],
        all_token_ids={}, new_block_ids=[], num_computed_tokens=[],
        num_output_tokens=[],
    )
    return SchedulerOutput(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled or {},
        total_num_scheduled_tokens=total
        if total is not None
        else sum(num_scheduled.values()) if num_scheduled else 0,
        scheduled_spec_decode_tokens=spec_tokens or {},
        scheduled_encoder_inputs=encoder_inputs or {},
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
        new_block_ids_to_zero=new_block_ids_to_zero,
        kv_cache_block_copies=kv_cache_block_copies,
    )


def _runner_for_update(cpu_env, monkeypatch, **attrs):
    """Bare ModelRunnerFL with the fields _update_states touches."""
    instance = object.__new__(MR.ModelRunnerFL)
    pp_group = NS(is_last_rank=True, is_first_rank=True)
    monkeypatch.setattr(MR, "get_pp_group", lambda: pp_group)
    defaults = dict(
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=Mock(),
        input_batch=None,
        is_pooling_model=False,
        uses_mrope=False,
        enable_prompt_embeds=False,
        kv_caches=[],
        speculative_config=None,
        use_async_scheduling=False,
        use_async_spec_decode=False,
        reorder_batch_threshold=None,
        kv_cache_config=NS(kv_cache_groups=[]),
    )
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(instance, key, value)
    return instance


# ---------------------------------------------------------------------------
# _update_states
# ---------------------------------------------------------------------------


class TestUpdateStates:
    def test_finished_requests_are_removed_from_state_and_batch(
        self, cpu_env, monkeypatch
    ):
        req = cpu_env.req_state("r1", prompt_len=6, computed=6, output=[1, 2])
        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.requests = {"r1": req}
        runner.input_batch = cpu_env.make_batch([req])
        assert runner.input_batch.num_reqs == 1

        out = MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(
                num_scheduled={"r1": 1},
                finished=["r1"],
                total=1,
            ),
        )
        # The correction closure is only produced for async spec decode.
        assert out is None
        assert "r1" not in runner.requests
        assert runner.input_batch.num_reqs == 0
        runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
            {"r1"}
        )

    def test_new_requests_are_added_with_cached_state(
        self, cpu_env, monkeypatch
    ):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData

        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.input_batch = cpu_env.make_batch()
        new_data = NewRequestData(
            req_id="new-1",
            prompt_token_ids=list(range(8)),
            mm_features=[],
            sampling_params=SamplingParams(temperature=0.0),
            pooling_params=None,
            block_ids=([3],),
            num_computed_tokens=0,
            lora_request=None,
        )
        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(new_reqs=[new_data], num_scheduled={"new-1": 8}),
        )
        assert "new-1" in runner.requests
        assert runner.requests["new-1"].prompt_token_ids == list(range(8))
        assert runner.input_batch.num_reqs == 1
        assert runner.input_batch.req_ids == ["new-1"]
        runner.late_interaction_runner.register_request.assert_called_once_with(
            "new-1", None
        )
        # prompt_logprobs tracking stays empty when not requested.
        assert runner.num_prompt_logprobs == {}

    def test_new_request_prompt_logprobs_registration(
        self, cpu_env, monkeypatch
    ):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData

        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.input_batch = cpu_env.make_batch()
        runner.input_batch.vocab_size = 128
        new_data = NewRequestData(
            req_id="lp-1",
            prompt_token_ids=list(range(4)),
            mm_features=[],
            sampling_params=SamplingParams(temperature=0.0, prompt_logprobs=5),
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
            lora_request=None,
        )
        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(new_reqs=[new_data], num_scheduled={"lp-1": 4}),
        )
        assert runner.num_prompt_logprobs == {"lp-1": 5}

        # -1 means "vocab size" (all logits).
        new_data = NewRequestData(
            req_id="lp-2",
            prompt_token_ids=list(range(4)),
            mm_features=[],
            sampling_params=SamplingParams(temperature=0.0, prompt_logprobs=-1),
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
            lora_request=None,
        )
        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(new_reqs=[new_data], num_scheduled={"lp-2": 4}),
        )
        assert runner.num_prompt_logprobs == {"lp-1": 5, "lp-2": 128}

    def test_random_seed_requests_get_a_seeded_generator(
        self, cpu_env, monkeypatch
    ):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData

        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.input_batch = cpu_env.make_batch()
        runner.device = torch.device("cpu")
        new_data = NewRequestData(
            req_id="seed-1",
            prompt_token_ids=[1],
            mm_features=[],
            sampling_params=SamplingParams(seed=1234),
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
            lora_request=None,
        )
        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(new_reqs=[new_data], num_scheduled={"seed-1": 1}),
        )
        gen = runner.requests["seed-1"].generator
        assert gen is not None
        assert gen.initial_seed() == 1234

    def test_cached_requests_update_blocks_and_tokens(
        self, cpu_env, monkeypatch
    ):
        from vllm.v1.core.sched.output import CachedRequestData

        req = cpu_env.req_state("c1", prompt_len=6, computed=4, output=[5, 6, 7])
        runner = _runner_for_update(cpu_env, monkeypatch)
        # Non-last PP rank receives sampled tokens back from the scheduler.
        monkeypatch.setattr(
            MR, "get_pp_group", lambda: NS(is_last_rank=False, is_first_rank=True)
        )
        runner.requests = {"c1": req}
        runner.input_batch = cpu_env.make_batch([req])
        # Extend blocks to verify append semantics.
        req.block_ids = ([0, 1],)

        cached = CachedRequestData(
            req_ids=["c1"],
            resumed_req_ids=set(),
            new_token_ids=[[8, 9]],
            all_token_ids={"c1": [5, 6, 7, 8, 9]},
            new_block_ids=[([2, 3],)],
            num_computed_tokens=[8],
            num_output_tokens=[5],
        )
        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(
                cached_reqs=cached,
                num_scheduled={"c1": 4},
                total=4,
            ),
        )
        # num_computed_tokens refreshed from the scheduler data.
        assert req.num_computed_tokens == 8
        # num_new_tokens = 8 + 2 - (6 + 3) = 1 -> the single-token fast path
        # appends just the last scheduler-supplied token.
        assert req.output_token_ids == [5, 6, 7, 9]
        # New blocks appended to the cached list.
        assert req.block_ids == ([0, 1, 2, 3],)
        # Persistent batch updated for scheduled request.
        assert runner.input_batch.num_computed_tokens_cpu[0] == 8
        # First non-PP-rank write window: the end index is
        # max(start, num_computed + len(new_token_ids)) = max(9, 10).
        assert runner.input_batch.num_tokens_no_spec[0] == 10

    def test_resumed_request_replaces_block_ids_and_rejoins_batch(
        self, cpu_env, monkeypatch
    ):
        from vllm.v1.core.sched.output import CachedRequestData

        req = cpu_env.req_state("p1", prompt_len=8, computed=8, output=[1, 2])
        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.requests = {"p1": req}
        runner.input_batch = cpu_env.make_batch()  # p1 not in batch

        cached = CachedRequestData(
            req_ids=["p1"],
            resumed_req_ids={"p1"},
            new_token_ids=[[]],
            all_token_ids={"p1": [1, 2]},
            new_block_ids=[([9],)],
            num_computed_tokens=[8],
            num_output_tokens=[2],
        )
        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(
                cached_reqs=cached, num_scheduled={"p1": 1}, total=1
            ),
        )
        # Resumed from preemption: blocks replaced wholesale.
        assert req.block_ids == ([9],)
        # Re-added to the persistent batch.
        assert runner.input_batch.req_id_to_index.get("p1") == 0

    def test_output_token_shrink_alignment(
        self, cpu_env, monkeypatch
    ):
        from vllm.v1.core.sched.output import CachedRequestData

        req = cpu_env.req_state("s1", prompt_len=6, computed=6, output=[1, 2, 3, 4])
        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.requests = {"s1": req}
        runner.input_batch = cpu_env.make_batch([req])
        runner.input_batch.num_tokens_no_spec[0] = 8

        cached = CachedRequestData(
            req_ids=["s1"],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={"s1": [1, 2]},
            new_block_ids=[([0],)],
            num_computed_tokens=[6],
            num_output_tokens=[2],  # scheduler says only 2 outputs are valid
        )
        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(
                cached_reqs=cached, num_scheduled={"s1": 1}, total=1
            ),
        )
        # Cached output tokens trimmed back to the scheduler count.
        assert req.output_token_ids == [1, 2]
        # num_tokens_no_spec realigned to prompt + valid outputs.
        assert runner.input_batch.num_tokens_no_spec[0] == 8

    def test_unscheduled_requests_leave_persistent_batch(
        self, cpu_env, monkeypatch
    ):
        a = cpu_env.req_state("a", prompt_len=4, computed=4)
        b = cpu_env.req_state("b", prompt_len=4, computed=4)
        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.requests = {"a": a, "b": b}
        runner.input_batch = cpu_env.make_batch([a, b])

        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(num_scheduled={"a": 1}, total=1),
        )
        # b was not scheduled -> removed from the persistent batch but its
        # cached state is kept for later rescheduling.
        assert runner.input_batch.num_reqs == 1
        assert set(runner.input_batch.req_ids) == {"a"}
        assert "b" in runner.requests

    def test_new_block_ids_to_zero_and_block_copies_plumb_through(
        self, cpu_env, monkeypatch
    ):
        req = cpu_env.req_state("z1", prompt_len=4, computed=4)
        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.requests = {"z1": req}
        runner.input_batch = cpu_env.make_batch([req])
        runner._zero_block_ids = Mock()
        copies = Mock()
        monkeypatch.setattr(MR, "copy_kv_cache_blocks_inplace", copies)
        kv_caches = ["cache-0"]
        runner.kv_caches = kv_caches
        runner.kv_cache_config = NS(num_blocks=10, kv_cache_groups=[])

        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(
                num_scheduled={"z1": 1},
                total=1,
                new_block_ids_to_zero=[7, 8, 9],
                kv_cache_block_copies=[object()],
            ),
        )
        # Zeroing requested: the hook receives the exact block ids.
        runner._zero_block_ids.assert_called_once_with([7, 8, 9])
        # Block copies reach the upstream inplace copier.
        copies.assert_called_once()
        call_args, call_kwargs = copies.call_args
        assert call_args[0] is kv_caches
        assert call_args[1] == 10

    def test_zeroing_not_requested_stays_silent(
        self, cpu_env, monkeypatch
    ):
        req = cpu_env.req_state("z2", prompt_len=4, computed=4)
        runner = _runner_for_update(cpu_env, monkeypatch)
        runner.requests = {"z2": req}
        runner.input_batch = cpu_env.make_batch([req])
        runner._zero_block_ids = Mock()
        copies = Mock()
        monkeypatch.setattr(MR, "copy_kv_cache_blocks_inplace", copies)

        MR.ModelRunnerFL._update_states(
            runner,
            _scheduler_output(num_scheduled={"z2": 1}, total=1),
        )
        runner._zero_block_ids.assert_not_called()
        copies.assert_not_called()


# ---------------------------------------------------------------------------
# _update_streaming_request
# ---------------------------------------------------------------------------


def test_update_streaming_request_refreshes_cached_state(cpu_env_ref=None):
    from vllm.v1.core.sched.output import NewRequestData

    instance = object.__new__(MR.ModelRunnerFL)
    old_state = MR.CachedRequestState(
        req_id="stream-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=1,
        output_token_ids=[7, 8],
    )
    instance.requests = {"stream-1": old_state}
    instance.input_batch = Mock()
    instance.uses_mrope = False
    instance.late_interaction_runner = Mock()

    new_data = NewRequestData(
        req_id="stream-1",
        prompt_token_ids=[1, 2, 3, 7, 8, 9],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([4, 5],),
        num_computed_tokens=5,
        lora_request=None,
    )
    out = MR.ModelRunnerFL._update_streaming_request(instance, "stream-1", new_data)
    assert out is old_state
    instance.input_batch.remove_request.assert_called_once_with("stream-1")
    assert out.prompt_token_ids == [1, 2, 3, 7, 8, 9]
    assert out.block_ids == ([4, 5],)
    assert out.num_computed_tokens == 5
    # Previous outputs folded into the prompt: output list cleared.
    assert out.output_token_ids == []
    # num_prompt_tokens derived from the new prompt length.
    assert out.num_prompt_tokens == 6
    instance.late_interaction_runner.register_request.assert_called_once_with(
        "stream-1", None
    )


def test_update_streaming_request_recomputes_mrope(monkeypatch):
    from vllm.v1.core.sched.output import NewRequestData

    instance = object.__new__(MR.ModelRunnerFL)
    old_state = MR.CachedRequestState(
        req_id="s2",
        prompt_token_ids=[1],
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=0,
        output_token_ids=[],
    )
    instance.requests = {"s2": old_state}
    instance.input_batch = Mock()
    instance.uses_mrope = True
    instance.late_interaction_runner = Mock()
    calls = []
    monkeypatch.setattr(
        MR.ModelRunnerFL,
        "_init_mrope_positions",
        lambda self, state: calls.append(state),
    )
    new_data = NewRequestData(
        req_id="s2",
        prompt_token_ids=[1, 2],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=1,
        lora_request=None,
    )
    MR.ModelRunnerFL._update_streaming_request(instance, "s2", new_data)
    assert calls == [old_state]


# ---------------------------------------------------------------------------
# _calc_spec_decode_metadata and index math
# ---------------------------------------------------------------------------


def test_calc_spec_decode_metadata_index_math(monkeypatch):
    instance = object.__new__(MR.ModelRunnerFL)
    instance.device = torch.device("cpu")
    instance.arange_np = np.arange(256, dtype=np.int64)
    # Prepare a scratch buffer wide enough for the test batch.
    instance._arange_scratch = np.empty(256, dtype=np.int64)
    # input_ids.gpu must cover the largest gathered index (103 in the oracle
    # below); a 256-slot tensor is more than enough.
    instance.input_ids = NS(gpu=torch.zeros(256, dtype=torch.int32))
    monkeypatch.setattr(
        MR, "async_tensor_h2d", lambda data, device: torch.as_tensor(
            np.asarray(data), dtype=torch.int32
        )
    )

    # cu_num_scheduled_tokens: [4, 104, 107, 207, 209]
    # num_draft_tokens:        [3,   0,   2,   0,   1]
    cu_scheduled = np.array([4, 104, 107, 207, 209], dtype=np.int32)
    num_draft = np.array([3, 0, 2, 0, 1], dtype=np.int32)

    meta = MR.ModelRunnerFL._calc_spec_decode_metadata(
        instance, num_draft, cu_scheduled
    )
    assert meta.num_draft_tokens == [3, 0, 2, 0, 1]
    # logits_indices gather every sampled position (drafts + bonus).
    assert meta.logits_indices.tolist() == [
        0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208,
    ]
    # bonus indices are the last sampled position of each request.
    assert meta.bonus_logits_indices.tolist() == [3, 4, 7, 8, 10]
    # target indices point at the draft positions of each request.
    assert meta.target_logits_indices.tolist() == [0, 1, 2, 5, 6, 9]


def test_get_cumsum_and_arange_oracle():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.arange_np = np.arange(64, dtype=np.int64)
    arange_out = np.zeros(16, dtype=np.int64)

    cu = MR.ModelRunnerFL._get_cumsum_and_arange(
        instance, np.array([2, 5, 3]), arange_out
    )
    np.testing.assert_array_equal(cu, [2, 7, 10])
    np.testing.assert_array_equal(
        arange_out[:10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    )


def test_compute_prev_positions_maps_new_and_returning():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.input_batch = NS(
        prev_req_id_to_index={"keep": 1, "moved": 2},
        req_ids=["fresh", "keep", "moved", "also-fresh"],
    )
    instance.prev_positions = NS(np=np.zeros(8, dtype=np.int64))
    MR.ModelRunnerFL._compute_prev_positions(instance, 4)
    np.testing.assert_array_equal(
        instance.prev_positions.np[:4], [-1, 1, 2, -1]
    )


def test_compute_prev_positions_fills_minus_one_without_prev_mapping():
    instance = object.__new__(MR.ModelRunnerFL)
    instance.input_batch = NS(prev_req_id_to_index=None, req_ids=["a"])
    instance.prev_positions = NS(np=np.zeros(4, dtype=np.int64))
    MR.ModelRunnerFL._compute_prev_positions(instance, 1)
    assert instance.prev_positions.np[0] == -1


# ---------------------------------------------------------------------------
# _get_slot_mappings
# ---------------------------------------------------------------------------


class TestGetSlotMappings:
    def _make(self, cpu_env, n_groups=1):
        instance = object.__new__(MR.ModelRunnerFL)
        instance.device = torch.device("cpu")
        req = cpu_env.req_state("r1", prompt_len=8, block_ids=[0, 1])
        batch = cpu_env.make_batch([req])
        instance.input_batch = batch
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheGroupSpec,
        )

        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
        )
        instance.kv_cache_config = NS(
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=[f"layer-{g}"], kv_cache_spec=spec)
                for g in range(n_groups)
            ]
        )
        return instance

    def test_no_kv_groups_returns_none_pair(self, cpu_env):
        instance = self._make(cpu_env)
        instance.kv_cache_config = NS(kv_cache_groups=[])
        by_gid, by_layer = MR.ModelRunnerFL._get_slot_mappings(
            instance, num_tokens_padded=4, num_reqs_padded=1,
            num_tokens_unpadded=2,
        )
        assert by_gid is None and by_layer is None

    def test_missing_kv_cache_config_returns_none(self, cpu_env):
        instance = self._make(cpu_env)
        del instance.kv_cache_config
        assert "kv_cache_config" not in instance.__dict__
        by_gid, by_layer = MR.ModelRunnerFL._get_slot_mappings(
            instance, num_tokens_padded=4, num_reqs_padded=1,
            num_tokens_unpadded=2,
        )
        assert by_gid is None and by_layer is None

    def test_slot_mapping_pads_tail_with_minus_one(self, cpu_env):
        instance = self._make(cpu_env)
        by_gid, by_layer = MR.ModelRunnerFL._get_slot_mappings(
            instance, num_tokens_padded=6, num_reqs_padded=1,
            num_tokens_unpadded=4,
        )
        assert set(by_gid) == {0}
        slot = by_gid[0]
        assert slot.shape == (6,)
        # Only the unpadded prefix may hold real slot values; the pad tail
        # must be filled with -1 for full-graph capture.
        assert bool((slot[4:] == -1).all())
        assert by_layer == {"layer-0": slot}

    def test_multiple_groups_share_mapping_per_layer(self, cpu_env):
        instance = self._make(cpu_env, n_groups=2)
        # The InputBatch only owns one block table (one block size); a second
        # KV cache group needs a second entry. Clone the existing table.
        second = instance.input_batch.block_table[0]
        instance.input_batch.block_table.block_tables.append(second)
        by_gid, by_layer = MR.ModelRunnerFL._get_slot_mappings(
            instance, num_tokens_padded=4, num_reqs_padded=1,
            num_tokens_unpadded=4,
        )
        assert set(by_gid) == {0, 1}
        assert set(by_layer) == {"layer-0", "layer-1"}

    def test_encoder_only_spec_uses_zero_slot_mapping(self, cpu_env):
        from vllm.v1.kv_cache_interface import (
            EncoderOnlyAttentionSpec,
            KVCacheGroupSpec,
        )

        instance = self._make(cpu_env)
        instance.kv_cache_config.kv_cache_groups[0] = KVCacheGroupSpec(
            layer_names=["enc-0"],
            kv_cache_spec=EncoderOnlyAttentionSpec(
                block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
            ),
        )
        by_gid, by_layer = MR.ModelRunnerFL._get_slot_mappings(
            instance, num_tokens_padded=4, num_reqs_padded=1,
            num_tokens_unpadded=2,
        )
        # Encoder-only groups get a zero tensor, then the -1 pad tail.
        assert bool((by_gid[0][:2] == 0).all())
        assert bool((by_gid[0][2:] == -1).all())


# ---------------------------------------------------------------------------
# _get_encoder_seq_lens
# ---------------------------------------------------------------------------


def test_get_encoder_seq_lens_ignores_non_cross_attention():
    instance = object.__new__(MR.ModelRunnerFL)
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    out = MR.ModelRunnerFL._get_encoder_seq_lens(
        instance, {"r1": 4}, spec, num_reqs=1
    )
    assert out == (None, None)


def test_get_encoder_seq_lens_builds_per_request_lengths(cpu_env):
    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    instance = object.__new__(MR.ModelRunnerFL)
    req = cpu_env.req_state("r1", prompt_len=8, block_ids=[0])
    instance.requests = {"r1": req}
    req.mm_features = [
        NS(mm_position=NS(length=5)),
        NS(mm_position=NS(length=7)),
    ]
    instance.input_batch = cpu_env.make_batch([req])
    instance.encoder_seq_lens = _buffer(8)

    spec = CrossAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    lens, lens_cpu = MR.ModelRunnerFL._get_encoder_seq_lens(
        instance, {"r1": 4}, spec, num_reqs=1
    )
    assert lens_cpu[0] == 12  # sum of the two features' lengths
    assert lens.shape == (1,)


def _buffer(n, dtype=torch.int32):
    """Small stand-in for the encoder_seq_lens CpuGpuBuffer (CPU-only)."""
    from vllm.v1.utils import CpuGpuBuffer

    saved = CpuGpuBuffer.__init__.__kwdefaults__["pin_memory"]
    CpuGpuBuffer.__init__.__kwdefaults__["pin_memory"] = False
    try:
        buf = CpuGpuBuffer(n, dtype=dtype, device=torch.device("cpu"))
    finally:
        CpuGpuBuffer.__init__.__kwdefaults__["pin_memory"] = saved
    return buf


def test_get_encoder_seq_lens_capture_mode_uses_max_encoder_len(
    cpu_env,
):
    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    instance = object.__new__(MR.ModelRunnerFL)
    req = cpu_env.req_state("r1", prompt_len=8, block_ids=[0])
    instance.requests = {"r1": req}
    req.mm_features = [NS(mm_position=NS(length=5))]
    instance.input_batch = cpu_env.make_batch([req])
    instance.encoder_seq_lens = _buffer(8)
    instance.model_config = NS(hf_config=NS(max_source_positions=40))
    instance.max_encoder_len = 17

    spec = CrossAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    _, lens_cpu = MR.ModelRunnerFL._get_encoder_seq_lens(
        instance, {"r1": 4}, spec, num_reqs=2,
        for_cudagraph_capture=True,
    )
    # hf_config.max_source_positions wins over max_encoder_len, and the
    # pad row (second request) is also set to that value.
    assert lens_cpu.tolist() == [40, 40]
