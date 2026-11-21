# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for the batch/state-machine methods of ModelRunnerFL.

``vllm_fl.worker.model_runner`` imports ``vllm_fl.compilation.graph`` whose
``Graph`` class raises ``NotImplementedError`` at class-body level unless the
current platform's ``device_type`` is cuda/npu/musa/gcu/ptpu/txda.  Every test
here therefore installs a delegating proxy on ``vllm.platforms.current_platform``
that fakes only the CUDA-ness attributes and forwards everything else to the
real platform, before the first import of the module under test.

The heavy stateful methods (``_update_states``, ``_prepare_inputs``,
``_build_attention_metadata``, ...) are driven on top of *real* vLLM v1
objects: a real ``InputBatch`` with real ``CachedRequestState`` entries, real
``SchedulerOutput`` dataclasses and real KV-cache specs.  Only true external
boundaries are replaced:

* the triton ``_COMPUTE_SLOT_MAPPING_KERNEL`` in ``vllm.v1.worker.block_table``
  (a triton JIT launcher that cannot run on a CPU-only build) is proxied with a
  pure-torch implementation of the documented slot-mapping formula;
* GPU-only torch APIs (streams/events) are faked where a method touches them.

Every numeric expectation is computed with an independent plain-Python or
plain-torch oracle, then compared to the method output.
"""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

try:  # pragma: no cover - dual-environment guard
    import vllm  # noqa: F401

    HAS_VLLM = True
except ImportError:  # pragma: no cover - Windows machines without vllm
    HAS_VLLM = False

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.skipif(not HAS_VLLM, reason="requires real vllm"),
]


class _FakeCudaPlatform:
    """Delegate to the real platform; fake only CUDA-ness for import checks."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    @property
    def device_type(self):
        return "cuda"

    @property
    def dispatch_key(self):
        return "CUDA"

    @property
    def vendor_name(self):
        return None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


class _CpuSlotMappingKernel:
    """Pure-torch stand-in for the triton slot-mapping kernel boundary.

    Implements the documented mapping for the single-rank, non-interleaved
    configuration used by these tests: ``slot = block_number * block_size +
    offset_in_block``, with the ``PAD_SLOT_ID`` sentinel (-1) in the tail
    region between ``num_tokens`` and ``max_num_tokens``.  Everything but
    ``__call__`` is delegated to the real triton kernel object so metadata
    registrations inside ``BlockTable.__init__`` keep working.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __call__(
        self,
        num_reqs,
        num_tokens,
        max_num_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table_stride,
        slot_mapping,
        kv_cache_block_size,
        total_cp_world_size,
        total_cp_rank,
        cp_kv_cache_interleave_size,
    ):
        pad_id = -1
        for i in range(num_tokens, max_num_tokens):
            slot_mapping[i] = pad_id
        flat_block_table = block_table.reshape(-1)
        for req_idx in range(num_reqs):
            start = int(query_start_loc[req_idx])
            end = int(query_start_loc[req_idx + 1])
            for i in range(start, end):
                pos = int(positions[i])
                vbi = pos // kv_cache_block_size
                vbo = pos - vbi * kv_cache_block_size
                bn = int(flat_block_table[req_idx * block_table_stride + vbi])
                slot_mapping[i] = bn * kv_cache_block_size + vbo


class _FakeEvent:
    """Minimal stand-in for a torch device Event."""

    def __init__(self, blocking=False):
        self.blocking = blocking
        self.record_count = 0
        self.synchronized = False

    def record(self):
        self.record_count += 1

    def synchronize(self):
        self.synchronized = True


@pytest.fixture
def model_runner_module(monkeypatch):
    """Import vllm_fl.worker.model_runner with CUDA-ness faked at import time.

    The compilation graph module validates ``device_type`` while its class
    body executes, so the proxy is installed before the first import and the
    real platform object is restored afterwards.
    """
    pytest.importorskip("vllm", reason="requires real vllm")
    platforms = import_module("vllm.platforms")
    real = platforms.current_platform
    monkeypatch.setattr(platforms, "current_platform", _FakeCudaPlatform(real))
    module = import_module("vllm_fl.worker.model_runner")
    yield module
    monkeypatch.setattr(platforms, "current_platform", real)


@pytest.fixture
def pp_last_rank(monkeypatch):
    """Make get_pp_group() report a single-rank last-stage pipeline group.

    ``get_pp_group`` reads the private ``_PP`` module global; patching it with
    a tiny stand-in avoids initializing the real distributed groups on a CPU
    test machine while leaving the scheduling logic under test untouched.
    """
    parallel_state = import_module("vllm.distributed.parallel_state")
    monkeypatch.setattr(
        parallel_state,
        "_PP",
        SimpleNamespace(is_last_rank=True, is_first_rank=True, ranks=[0]),
    )


@pytest.fixture
def cpu_slot_mapping_kernel(monkeypatch):
    """Replace the triton slot-mapping kernel with a pure-torch oracle.

    The kernel launcher inspects triton JIT parameters, which fails on a
    CPU-only torch build; the formula is re-implemented independently in
    torch so ``BlockTable.compute_slot_mapping`` stays fully exercised.
    """
    block_table_module = import_module("vllm.v1.worker.block_table")
    real = block_table_module._COMPUTE_SLOT_MAPPING_KERNEL
    proxy = _CpuSlotMappingKernel(real)
    monkeypatch.setattr(block_table_module, "_COMPUTE_SLOT_MAPPING_KERNEL", proxy)
    return proxy


# ---------------------------------------------------------------------------
# Shared builders for real InputBatch / SchedulerOutput graphs.
# ---------------------------------------------------------------------------

CPU_DEVICE = "cpu"
VOCAB = 32


def _make_input_batch(model_runner_module, *, is_pooling_model=False, num_spec=0):
    """Build a real InputBatch sized for the tests in this file."""
    torch = import_module("torch")
    InputBatch = import_module("vllm.v1.worker.gpu_input_batch").InputBatch
    batch = InputBatch(
        max_num_reqs=8,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device(CPU_DEVICE),
        vocab_size=VOCAB,
        block_sizes=[16],
        max_num_blocks_per_req=[4],
        is_pooling_model=is_pooling_model,
        num_spec_tokens=num_spec,
    )
    return batch


def _make_req_state(model_runner_module, req_id, prompt, block, **overrides):
    """Build a real CachedRequestState with greedy sampling params."""
    SamplingParams = import_module("vllm.sampling_params").SamplingParams
    SamplingType = import_module("vllm.sampling_params").SamplingType
    CachedRequestState = import_module(
        "vllm.v1.worker.gpu_input_batch"
    ).CachedRequestState

    params = SamplingParams()
    params.sampling_type = SamplingType.GREEDY
    state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=params,
        generator=None,
        block_ids=(list(block),),
        num_computed_tokens=0,
        output_token_ids=[],
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _scheduler_output(
    model_runner_module,
    *,
    new_reqs=(),
    cached_reqs=None,
    num_scheduled_tokens=None,
    total_num_scheduled_tokens=0,
    spec_tokens=None,
    finished_req_ids=None,
    scheduled_encoder_inputs=None,
    free_encoder_mm_hashes=None,
):
    """Build a real SchedulerOutput dataclass with the given contents."""
    CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData
    SchedulerOutput = import_module("vllm.v1.core.sched.output").SchedulerOutput

    if cached_reqs is None:
        cached_reqs = CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )

    return SchedulerOutput(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=dict(num_scheduled_tokens or {}),
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=dict(spec_tokens or {}),
        scheduled_encoder_inputs=dict(scheduled_encoder_inputs or {}),
        finished_req_ids=set(finished_req_ids or ()),
        free_encoder_mm_hashes=list(free_encoder_mm_hashes or ()),
    )


def _new_req_data(
    model_runner_module,
    req_id,
    prompt,
    block,
    sampling_params="greedy",
    pooling_params=None,
    **overrides,
):
    """Build a real NewRequestData."""
    SamplingParams = import_module("vllm.sampling_params").SamplingParams
    SamplingType = import_module("vllm.sampling_params").SamplingType
    NewRequestData = import_module("vllm.v1.core.sched.output").NewRequestData

    if sampling_params == "greedy":
        params = SamplingParams()
        params.sampling_type = SamplingType.GREEDY
    else:
        params = sampling_params

    data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=None if prompt is None else list(prompt),
        mm_features=[],
        sampling_params=params,
        pooling_params=pooling_params,
        block_ids=(list(block),),
        num_computed_tokens=0,
        lora_request=None,
    )
    for key, value in overrides.items():
        setattr(data, key, value)
    return data


class _NullLateInteractionRunner:
    """Recording stand-in for the LateInteractionRunner boundary."""

    def __init__(self):
        self.finished_calls = []
        self.registered = []

    def on_requests_finished(self, req_ids):
        self.finished_calls.append(list(req_ids))

    def register_request(self, req_id, pooling_params):
        self.registered.append((req_id, pooling_params))


def _base_runner(model_runner_module, input_batch, **overrides):
    """A SimpleNamespace runner with the common _update_states attributes.

    ``_update_streaming_request`` is the real method bound to this runner so
    the streaming path of ``_update_states`` runs the actual implementation
    rather than a stub.
    """
    MethodType = import_module("types").MethodType
    torch = import_module("torch")
    runner = SimpleNamespace(
        device=torch.device(CPU_DEVICE),
        input_batch=input_batch,
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=_NullLateInteractionRunner(),
        encoder_cache={},
        speculative_config=None,
        is_pooling_model=False,
        uses_mrope=False,
        use_async_scheduling=False,
        use_async_spec_decode=False,
        kv_caches=[],
        kv_cache_config=SimpleNamespace(num_blocks=8, kv_cache_groups=[0]),
        _on_request_state_removed=lambda *a: None,
        _zero_block_ids=lambda ids: None,
        _process_encoder_cache_scheduler_output=lambda so: None,
        _may_reorder_batch=lambda so: None,
        # ngram-GPU bookkeeping is exercised through the closure tests below.
        token_ids_gpu_tensor=None,
        num_tokens_no_spec_gpu=None,
        _ngram_pinned_idx_buf=None,
        _ngram_pinned_val_buf=None,
        _num_valid_draft_tokens_event=None,
        _num_valid_draft_tokens_cpu=None,
    )
    runner._update_streaming_request = MethodType(
        model_runner_module.ModelRunnerFL._update_streaming_request, runner
    )
    for name in ("_get_cumsum_and_arange", "_compute_prev_positions"):
        setattr(
            runner,
            name,
            MethodType(getattr(model_runner_module.ModelRunnerFL, name), runner),
        )
    runner.arange_np = np.arange(256, dtype=np.int32)
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


# ---------------------------------------------------------------------------
# _update_states: finished / unscheduled removal + new request registration.
# ---------------------------------------------------------------------------


class TestUpdateStatesBasics:
    """ModelRunnerFL._update_states with real InputBatch state."""

    def test_finished_request_is_removed_from_cache_and_batch(
        self, model_runner_module, pp_last_rank
    ):
        """A finished req_id leaves requests, num_prompt_logprobs and batch."""
        import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()

        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = state
        runner.num_prompt_logprobs["r1"] = 5

        sched = _scheduler_output(
            model_runner_module,
            finished_req_ids=["r1"],
        )
        result = model_runner_module.ModelRunnerFL._update_states(runner, sched)

        assert result is None
        assert "r1" not in runner.requests
        assert "r1" not in runner.num_prompt_logprobs
        assert runner.late_interaction_runner.finished_calls == [["r1"]]
        assert batch.num_reqs == 0
        assert batch.req_id_to_index == {}

    def test_new_request_registered_and_added_to_batch(
        self, model_runner_module, pp_last_rank
    ):
        """A new request gets a CachedRequestState and joins the batch."""
        batch = _make_input_batch(model_runner_module)
        runner = _base_runner(model_runner_module, batch)

        new_data = _new_req_data(model_runner_module, "r1", (5, 6, 7), (9,))
        sched = _scheduler_output(
            model_runner_module,
            new_reqs=[new_data],
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        assert "r1" in runner.requests
        req_state = runner.requests["r1"]
        assert req_state.prompt_token_ids == [5, 6, 7]
        assert req_state.block_ids == ([9],)
        assert batch.req_id_to_index == {"r1": 0}
        assert runner.late_interaction_runner.registered == [("r1", None)]

    def test_unscheduled_request_removed_but_state_kept(
        self, model_runner_module, pp_last_rank
    ):
        """A cached request absent from the schedule leaves the batch only."""
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()

        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = state

        # r1 is cached but not scheduled; r2 is scheduled so it stays.
        other = _make_req_state(model_runner_module, "r2", (1,), (10,))
        batch.add_request(other)
        batch.refresh_metadata()
        runner.requests["r2"] = other

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r2": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        assert batch.req_id_to_index == {"r2": 0}
        # The cached state survives for a later resume.
        assert "r1" in runner.requests
        assert "r2" in runner.requests

    def test_prompt_logprobs_none_and_minus_one_mapping(
        self, model_runner_module, pp_last_rank
    ):
        """prompt_logprobs==-1 maps to vocab_size, otherwise kept verbatim."""
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        SamplingType = import_module("vllm.sampling_params").SamplingType

        batch = _make_input_batch(model_runner_module)
        runner = _base_runner(model_runner_module, batch)

        def seeded_params(seed, prompt_logprobs):
            params = SamplingParams()
            params.sampling_type = SamplingType.GREEDY
            params.seed = seed
            params.prompt_logprobs = prompt_logprobs
            return params

        new_full = _new_req_data(
            model_runner_module,
            "rA",
            (5, 6),
            (9,),
            sampling_params=seeded_params(11, 7),
        )
        new_minus_one = _new_req_data(
            model_runner_module,
            "rB",
            (7, 8),
            (10,),
            sampling_params=seeded_params(12, -1),
        )
        sched = _scheduler_output(
            model_runner_module,
            new_reqs=[new_full, new_minus_one],
            num_scheduled_tokens={"rA": 2, "rB": 2},
            total_num_scheduled_tokens=4,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        assert runner.num_prompt_logprobs == {"rA": 7, "rB": VOCAB}

    def test_random_seed_sampling_creates_generator(
        self, model_runner_module, pp_last_rank
    ):
        """RANDOM_SEED sampling builds a generator seeded from the params."""
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        SamplingType = import_module("vllm.sampling_params").SamplingType

        batch = _make_input_batch(model_runner_module)
        runner = _base_runner(model_runner_module, batch)

        params = SamplingParams()
        params.sampling_type = SamplingType.RANDOM_SEED
        params.seed = 4242
        new_data = _new_req_data(
            model_runner_module, "r1", (5,), (9,), sampling_params=params
        )
        sched = _scheduler_output(
            model_runner_module,
            new_reqs=[new_data],
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        generator = runner.requests["r1"].generator
        assert generator is not None
        # The generator state matches a manually seeded twin (oracle).
        twin = torch_generator(model_runner_module, 4242)
        torch_mod = import_module("torch")
        assert torch_mod.equal(generator.get_state(), twin.get_state())

    def test_non_random_sampling_has_no_generator(
        self, model_runner_module, pp_last_rank
    ):
        """Greedy requests carry a None generator."""
        batch = _make_input_batch(model_runner_module)
        runner = _base_runner(model_runner_module, batch)
        new_data = _new_req_data(model_runner_module, "r1", (5,), (9,))
        sched = _scheduler_output(
            model_runner_module,
            new_reqs=[new_data],
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)
        assert runner.requests["r1"].generator is None


def torch_generator(model_runner_module, seed):
    """Independent oracle: a manually seeded torch.Generator on CPU."""
    torch = import_module("torch")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


# ---------------------------------------------------------------------------
# _update_states: running-request bookkeeping (blocks, tokens, discards).
# ---------------------------------------------------------------------------


class TestUpdateStatesRunningRequests:
    """_update_states over scheduled_cached_reqs entries."""

    def test_append_blocks_updates_batch_row(self, model_runner_module, pp_last_rank):
        """Appended blocks extend req_state.block_ids and the block table."""
        CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData

        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = state

        sched = _scheduler_output(
            model_runner_module,
            cached_reqs=CachedRequestData(
                req_ids=["r1"],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[([12, 13],)],
                num_computed_tokens=[3],
                num_output_tokens=[0],
            ),
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        assert state.block_ids == ([9, 12, 13],)
        # The persistent batch row mirrors the appended blocks.
        block_row = batch.block_table[0].block_table.np[0].tolist()
        assert block_row[:3] == [9, 12, 13]
        assert batch.num_computed_tokens_cpu[0] == 3

    def test_discard_output_tokens_trims_state_and_batch(
        self, model_runner_module, pp_last_rank
    ):
        """num_output_tokens < len(output_token_ids) trims both views."""
        CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData

        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        state.output_token_ids = [40, 41, 42, 43]
        batch.add_request(state)
        batch.refresh_metadata()
        # The batch sees the same 4 output tokens.
        batch.num_tokens_no_spec[0] = 3 + 4
        batch.num_prompt_tokens[0] = 3

        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = state

        sched = _scheduler_output(
            model_runner_module,
            cached_reqs=CachedRequestData(
                req_ids=["r1"],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[None],
                num_computed_tokens=[7],
                num_output_tokens=[2],
            ),
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        assert state.output_token_ids == [40, 41]
        assert batch.num_tokens_no_spec[0] == 3 + 2

    def test_resumed_request_replaces_blocks_and_rejoins_batch(
        self, model_runner_module, pp_last_rank
    ):
        """Resumed-from-preemption requests swap in the new block ids."""
        CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData

        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = state

        # r1 is unscheduled first (removed from the batch) and then resumed
        # with brand new blocks in the same scheduler output.
        sched = _scheduler_output(
            model_runner_module,
            cached_reqs=CachedRequestData(
                req_ids=["r1"],
                resumed_req_ids={"r1"},
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[([21, 22],)],
                num_computed_tokens=[2],
                num_output_tokens=[0],
            ),
            num_scheduled_tokens={"r1": 2},
            total_num_scheduled_tokens=2,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        assert state.block_ids == ([21, 22],)
        assert batch.req_id_to_index == {"r1": 0}
        assert batch.block_table[0].block_table.np[0].tolist()[:2] == [21, 22]

    def test_async_scheduling_recovers_output_tokens_on_resume(
        self, model_runner_module, pp_last_rank
    ):
        """Resumed async requests re-fetch output ids from all_token_ids."""
        CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData

        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = _base_runner(model_runner_module, batch, use_async_scheduling=True)
        runner.requests["r1"] = state

        all_tokens = {"r1": [5, 6, 7, 40, 41]}
        sched = _scheduler_output(
            model_runner_module,
            cached_reqs=CachedRequestData(
                req_ids=["r1"],
                resumed_req_ids={"r1"},
                new_token_ids=[],
                all_token_ids=all_tokens,
                new_block_ids=[([21],)],
                num_computed_tokens=[5],
                num_output_tokens=[2],
            ),
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        assert state.output_token_ids == [40, 41]


class TestUpdateStatesNotLastRank:
    """_update_states under pipeline parallelism (not last rank)."""

    def _run_with_pp(self, model_runner_module, monkeypatch, new_token_ids, expected):
        CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData

        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = state

        monkeypatch.setattr(
            model_runner_module,
            "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=False),
        )

        sched = _scheduler_output(
            model_runner_module,
            cached_reqs=CachedRequestData(
                req_ids=["r1"],
                resumed_req_ids=set(),
                new_token_ids=new_token_ids,
                all_token_ids={},
                new_block_ids=[None],
                num_computed_tokens=[3],
                num_output_tokens=[0],
            ),
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)
        assert state.output_token_ids == expected
        return batch

    def test_new_tokens_appended_single_fast_path(
        self, model_runner_module, pp_last_rank, monkeypatch
    ):
        """num_new_tokens==1 appends via the single-token fast path."""
        batch = self._run_with_pp(
            model_runner_module, monkeypatch, [[44]], expected=[44]
        )
        # token_ids_cpu row gained the sampled token and the is_token_ids mark.
        assert batch.token_ids_cpu[0, 3] == 44
        assert batch.is_token_ids[0, 3]
        assert batch.num_tokens_no_spec[0] == 4

    def test_new_tokens_appended_multi(
        self, model_runner_module, pp_last_rank, monkeypatch
    ):
        """num_new_tokens>1 extends with the trailing slice."""
        self._run_with_pp(
            model_runner_module,
            monkeypatch,
            [[44, 45]],
            expected=[44, 45],
        )

    def test_no_new_tokens_async_pp(
        self, model_runner_module, pp_last_rank, monkeypatch
    ):
        """Empty new_token_ids (async PP broadcast) appends nothing."""
        self._run_with_pp(model_runner_module, monkeypatch, [], expected=[])


class TestUpdateStatesStreaming:
    """_update_states + _update_streaming_request for re-submitted req ids."""

    def test_streaming_request_replaces_prompt_state(
        self, model_runner_module, pp_last_rank
    ):
        """A re-scheduled req_id in self.requests takes the streaming path."""
        batch = _make_input_batch(model_runner_module)
        old_state = _make_req_state(
            model_runner_module, "r1", (5, 6), (9,), output_token_ids=[40]
        )
        batch.add_request(old_state)
        batch.refresh_metadata()
        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = old_state
        # The late-interaction runner must be re-registered with the new
        # pooling params on the streaming path.
        runner.late_interaction_runner.registered.clear()

        new_data = _new_req_data(
            model_runner_module,
            "r1",
            (5, 6, 7, 40, 41),
            (30,),
            num_computed_tokens=4,
        )
        sched = _scheduler_output(
            model_runner_module,
            new_reqs=[new_data],
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        # The same object was updated in place and re-added to the batch.
        assert runner.requests["r1"] is old_state
        assert old_state.prompt_token_ids == [5, 6, 7, 40, 41]
        assert old_state.block_ids == ([30],)
        assert old_state.num_computed_tokens == 4
        assert old_state.output_token_ids == []
        assert batch.req_id_to_index == {"r1": 0}

    def test_streaming_request_with_embeds_counts_prompt_len(
        self, model_runner_module, pp_last_rank
    ):
        """prompt length falls back to prompt_embeds rows when ids are None."""
        torch = import_module("torch")

        batch = _make_input_batch(model_runner_module)
        old_state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        batch.add_request(old_state)
        batch.refresh_metadata()
        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = old_state

        embeds = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        new_data = _new_req_data(
            model_runner_module,
            "r1",
            None,
            (30,),
            prompt_embeds=embeds,
        )
        sched = _scheduler_output(
            model_runner_module,
            new_reqs=[new_data],
            num_scheduled_tokens={"r1": 2},
            total_num_scheduled_tokens=2,
        )
        model_runner_module.ModelRunnerFL._update_states(runner, sched)

        # num_prompt_tokens is derived from the embeds length (3 rows).
        assert runner.requests["r1"].num_prompt_tokens == 3


class TestUpdateStatesAsyncSpecDecode:
    """_update_states optimistic spec-decode corrections closure."""

    def _setup(self, model_runner_module):
        CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()

        prev_draft = model_runner_module.CpuGpuBuffer(
            8, dtype=torch.int32, device=torch.device(CPU_DEVICE)
        )
        runner = _base_runner(
            model_runner_module,
            batch,
            use_async_scheduling=True,
            prev_num_draft_tokens=prev_draft,
        )
        runner.requests["r1"] = state
        return runner, state, CachedRequestData

    def test_optimistic_extend_and_deferred_correction(
        self, model_runner_module, pp_last_rank
    ):
        """prev_num_draft_len>0 optimistically extends then corrects back."""
        import_module("torch")
        runner, state, CachedRequestData = self._setup(model_runner_module)

        state.prev_num_draft_len = 2
        sched = _scheduler_output(
            model_runner_module,
            cached_reqs=CachedRequestData(
                req_ids=["r1"],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[None],
                num_computed_tokens=[5],
                num_output_tokens=[3],
            ),
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        correct = model_runner_module.ModelRunnerFL._update_states(runner, sched)

        # Optimistic extend added 2 placeholder -1 tokens.
        assert state.output_token_ids == [-1, -1]
        assert correct is not None

        # Give the closure a valid sampled token count: 3 accepted tokens
        # for the single request at previous index 0 -> num_accepted = 2.
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner._get_valid_sampled_token_count = lambda: [3]

        correct()

        # num_accepted = 3 - 1 = 2; correction = 2 - 2 = 0, no change.
        assert state.num_computed_tokens == 5

    def test_correction_subtracts_rejected_draft_tokens(
        self, model_runner_module, pp_last_rank
    ):
        """Fewer accepted tokens than the optimistic count pulls computed back."""
        import_module("torch")
        runner, state, CachedRequestData = self._setup(model_runner_module)

        state.prev_num_draft_len = 2
        sched = _scheduler_output(
            model_runner_module,
            cached_reqs=CachedRequestData(
                req_ids=["r1"],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[None],
                num_computed_tokens=[5],
                num_output_tokens=[3],
            ),
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        correct = model_runner_module.ModelRunnerFL._update_states(runner, sched)

        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        # Only 2 sampled tokens -> num_accepted = 1; correction = 2 - 1 = 1.
        runner._get_valid_sampled_token_count = lambda: [2]

        computed_before = int(batch_num_computed(runner))
        correct()

        assert state.num_computed_tokens == 5 - 1
        assert int(batch_num_computed(runner)) == computed_before - 1

    def test_no_correction_without_valid_counts(
        self, model_runner_module, pp_last_rank
    ):
        """An empty valid count list leaves the closure a no-op."""
        runner, state, CachedRequestData = self._setup(model_runner_module)
        state.prev_num_draft_len = 1
        sched = _scheduler_output(
            model_runner_module,
            cached_reqs=CachedRequestData(
                req_ids=["r1"],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[None],
                num_computed_tokens=[5],
                num_output_tokens=[2],
            ),
            num_scheduled_tokens={"r1": 2},
            total_num_scheduled_tokens=2,
        )
        correct = model_runner_module.ModelRunnerFL._update_states(runner, sched)
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner._get_valid_sampled_token_count = lambda: []
        correct()
        assert state.num_computed_tokens == 5


def batch_num_computed(runner):
    """Read the persistent batch's num_computed_tokens_cpu for row 0."""
    return runner.input_batch.num_computed_tokens_cpu[0]


# ---------------------------------------------------------------------------
# _update_states_after_model_execute
# ---------------------------------------------------------------------------


class TestUpdateStatesAfterModelExecute:
    """ModelRunnerFL._update_states_after_model_execute acceptance counting."""

    def _runner(self, model_runner_module, *, hybrid=True, mode="none"):
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        spec = SimpleNamespace(method="ngram") if True else None
        runner = _base_runner(
            model_runner_module,
            batch,
            speculative_config=spec,
            model_config=SimpleNamespace(is_hybrid=hybrid),
            cache_config=SimpleNamespace(mamba_cache_mode=mode),
            num_accepted_tokens=model_runner_module.CpuGpuBuffer(
                8, dtype=torch.int32, device=torch.device(CPU_DEVICE)
            ),
            num_accepted_tokens_event=_FakeEvent(),
        )
        return runner

    def test_early_return_without_spec_config(self, model_runner_module, pp_last_rank):
        """No speculative config: nothing is written at all."""
        torch = import_module("torch")
        runner = self._runner(model_runner_module)
        runner.speculative_config = None

        output_token_ids = torch.tensor([[3, 4, -1, -1]])
        before = runner.num_accepted_tokens.gpu.clone()
        model_runner_module.ModelRunnerFL._update_states_after_model_execute(
            runner, output_token_ids, _scheduler_output(model_runner_module)
        )
        assert torch.equal(runner.num_accepted_tokens.gpu, before)

    def test_early_return_for_non_hybrid_model(self, model_runner_module, pp_last_rank):
        """A non-hybrid model skips the acceptance bookkeeping."""
        torch = import_module("torch")
        runner = self._runner(model_runner_module, hybrid=False)

        output_token_ids = torch.tensor([[3, 4, -1, -1]])
        before = runner.num_accepted_tokens.gpu.clone()
        model_runner_module.ModelRunnerFL._update_states_after_model_execute(
            runner, output_token_ids, _scheduler_output(model_runner_module)
        )
        assert torch.equal(runner.num_accepted_tokens.gpu, before)

    def test_counts_accepted_tokens_per_request(
        self, model_runner_module, pp_last_rank
    ):
        """Non-(-1) rows are counted into num_accepted_tokens.gpu."""
        torch = import_module("torch")
        runner = self._runner(model_runner_module)

        output_token_ids = torch.tensor(
            [[3, 4, -1, -1], [5, 6, 7, -1], [-1, -1, -1, -1]]
        )
        model_runner_module.ModelRunnerFL._update_states_after_model_execute(
            runner, output_token_ids, _scheduler_output(model_runner_module)
        )

        # Independent oracle: count non -1 entries per row.
        expected = (output_token_ids != -1).sum(dim=1)
        assert torch.equal(runner.num_accepted_tokens.gpu[:3], expected.to(torch.int32))
        # The non-align branch copies into the CPU tensor and records the event.
        assert torch.equal(
            runner.input_batch.num_accepted_tokens_cpu_tensor[:3],
            expected.to(torch.int32),
        )
        assert runner.num_accepted_tokens_event.record_count == 1


# ---------------------------------------------------------------------------
# _update_streaming_request (direct invocation)
# ---------------------------------------------------------------------------


class TestUpdateStreamingRequest:
    """ModelRunnerFL._update_streaming_request field refresh."""

    def test_updates_all_fields_and_clears_outputs(
        self, model_runner_module, pp_last_rank
    ):
        """Every refreshed field comes from the new request data."""
        import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(
            model_runner_module,
            "r1",
            (5, 6),
            (9,),
            output_token_ids=[40, 41],
        )
        batch.add_request(state)
        batch.refresh_metadata()
        runner = _base_runner(model_runner_module, batch)
        runner.requests["r1"] = state

        new_data = _new_req_data(
            model_runner_module,
            "r1",
            (5, 6, 7, 40, 41, 42),
            (31, 32),
            num_computed_tokens=5,
        )
        result = model_runner_module.ModelRunnerFL._update_streaming_request(
            runner, "r1", new_data
        )

        assert result is state
        assert state.prompt_token_ids == [5, 6, 7, 40, 41, 42]
        assert state.block_ids == ([31, 32],)
        assert state.num_computed_tokens == 5
        assert state.output_token_ids == []
        # num_prompt_tokens recomputed from the new prompt length.
        assert state.num_prompt_tokens == 6
        # Re-registered with the (None) pooling params; the batch removal is
        # this method's contract (the caller re-adds the request afterwards).
        assert runner.late_interaction_runner.registered == [("r1", None)]
        assert batch.req_id_to_index == {}

    def test_mrope_request_recomputes_positions(
        self, model_runner_module, pp_last_rank, monkeypatch
    ):
        """uses_mrope runners recompute mrope positions on refresh."""
        mrope_states = []
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = _base_runner(model_runner_module, batch, uses_mrope=True)
        runner.requests["r1"] = state
        runner._init_mrope_positions = lambda req_state: mrope_states.append(
            req_state.req_id
        )

        new_data = _new_req_data(model_runner_module, "r1", (5, 6, 7, 8), (31,))
        model_runner_module.ModelRunnerFL._update_streaming_request(
            runner, "r1", new_data
        )
        assert mrope_states == ["r1"]


# ---------------------------------------------------------------------------
# _init_model_kwargs (pooling token_type_ids)
# ---------------------------------------------------------------------------


class TestInitModelKwargs:
    """ModelRunnerFL._init_model_kwargs for pooling models."""

    def _pooling_runner(self, model_runner_module, extra_kwargs_list, seq_lens):
        torch = import_module("torch")
        PoolingParams = import_module("vllm.pooling_params").PoolingParams

        PoolingStates = import_module("vllm.v1.pool.metadata").PoolingStates
        batch = _make_input_batch(model_runner_module, is_pooling_model=True)
        params = []
        for i, extra in enumerate(extra_kwargs_list):
            param = PoolingParams(task="embed")
            if extra is not None:
                param.extra_kwargs = dict(extra)
            params.append(param)
            state = _make_req_state(
                model_runner_module, f"r{i}", tuple(range(3 + i)), (9 + i,)
            )
            # InputBatch picks the pooling branch only when sampling_params
            # is None, exactly like the scheduler does for pooling requests.
            state.sampling_params = None
            state.pooling_params = param
            state.pooling_states = PoolingStates()
            batch.add_request(state)

        runner = _base_runner(
            model_runner_module,
            batch,
            is_pooling_model=True,
            optimistic_seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        )
        # get_pooling_params must line up with req order.
        batch.refresh_metadata()
        return runner

    def test_non_pooling_model_returns_empty(self, model_runner_module):
        """Generation models get an empty kwargs dict."""
        runner = _base_runner(
            model_runner_module, _make_input_batch(model_runner_module)
        )
        result = model_runner_module.ModelRunnerFL._init_model_kwargs(runner)
        assert result == {}

    def test_no_token_type_requests_returns_empty(self, model_runner_module):
        """No compressed_token_type_ids extras: empty kwargs."""
        runner = self._pooling_runner(
            model_runner_module, [None, None], seq_lens=[5, 3]
        )
        result = model_runner_module.ModelRunnerFL._init_model_kwargs(runner)
        assert result == {}

    def test_token_type_ids_built_from_positions(self, model_runner_module):
        """ids are 1 where the arange position is at/after the split point."""
        torch = import_module("torch")
        # r0 has compressed ids starting at position 2 of a 5-token seq.
        runner = self._pooling_runner(
            model_runner_module,
            [{"compressed_token_type_ids": 2}, None],
            seq_lens=[5, 3],
        )
        result = model_runner_module.ModelRunnerFL._init_model_kwargs(runner)

        assert set(result) == {"token_type_ids"}
        token_type_ids = result["token_type_ids"]
        # Independent oracle: >= split for the request with extras; a
        # request without extras defaults pos to its full seq_len (all 0).
        expected_r0 = [0, 0, 1, 1, 1]
        expected_r1 = [0, 0, 0]
        assert token_type_ids.tolist() == expected_r0 + expected_r1
        assert token_type_ids.dtype == torch.int32

    def test_all_requests_with_extras(self, model_runner_module):
        """Every request carrying extras is split at its own position."""
        runner = self._pooling_runner(
            model_runner_module,
            [{"compressed_token_type_ids": 1}, {"compressed_token_type_ids": 0}],
            seq_lens=[4, 2],
        )
        result = model_runner_module.ModelRunnerFL._init_model_kwargs(runner)
        # r0: pos 1 -> [0,1,1,1]; r1: pos 0 -> [1,1]
        assert result["token_type_ids"].tolist() == [0, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# _prepare_input_ids
# ---------------------------------------------------------------------------


class TestPrepareInputIds:
    """ModelRunnerFL._prepare_input_ids GPU assembly of sampled tokens."""

    def _runner(
        self,
        model_runner_module,
        batch,
        *,
        prev_sampled=None,
        enable_prompt_embeds=False,
        draft_token_ids=None,
        prev_num_spec_tokens=0,
        pp_recv_work=None,
    ):
        torch = import_module("torch")
        runner = _base_runner(
            model_runner_module,
            batch,
            enable_prompt_embeds=enable_prompt_embeds,
            input_ids=model_runner_module.CpuGpuBuffer(
                64, dtype=torch.int32, device=torch.device(CPU_DEVICE)
            ),
            inputs_embeds=model_runner_module.CpuGpuBuffer(
                (64, 4), dtype=torch.float32, device=torch.device(CPU_DEVICE)
            ),
            is_token_ids=model_runner_module.CpuGpuBuffer(
                64, dtype=torch.bool, device=torch.device(CPU_DEVICE)
            ),
            prev_positions=model_runner_module.CpuGpuBuffer(
                8, dtype=torch.int32, device=torch.device(CPU_DEVICE)
            ),
            _draft_token_ids=draft_token_ids,
            prev_num_spec_tokens=prev_num_spec_tokens,
            _pp_recv_work=pp_recv_work,
        )
        batch.prev_sampled_token_ids = prev_sampled
        return runner

    def test_normal_scheduling_copies_cpu_ids(self, model_runner_module):
        """Without prev sampled ids the CPU buffer is uploaded directly."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()

        runner = self._runner(model_runner_module, batch)
        runner.input_ids.cpu[:3] = torch.tensor([5, 6, 7], dtype=torch.int32)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        model_runner_module.ModelRunnerFL._prepare_input_ids(
            runner, sched, 1, 3, np.array([3])
        )
        assert runner.input_ids.gpu[:3].tolist() == [5, 6, 7]

    def test_normal_scheduling_with_prompt_embeds_uploads_all(
        self, model_runner_module
    ):
        """enable_prompt_embeds also uploads embeds and is_token_ids."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()

        runner = self._runner(model_runner_module, batch, enable_prompt_embeds=True)
        runner.input_ids.cpu[:3] = torch.tensor([5, 6, 7], dtype=torch.int32)
        runner.is_token_ids.cpu[:3] = torch.tensor([True, True, False])
        runner.inputs_embeds.cpu[:3] = torch.ones(3, 4)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        model_runner_module.ModelRunnerFL._prepare_input_ids(
            runner, sched, 1, 3, np.array([3])
        )
        assert runner.is_token_ids.gpu[:3].tolist() == [True, True, False]
        assert torch.equal(runner.inputs_embeds.gpu[:3], torch.ones(3, 4))

    def test_no_common_requests_returns_early(self, model_runner_module):
        """All-new requests: input_ids.cpu has everything, nothing scattered."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()

        prev_sampled = torch.zeros((4, 1), dtype=torch.int32)
        runner = self._runner(model_runner_module, batch, prev_sampled=prev_sampled)
        # Production calls _compute_prev_positions inside _prepare_inputs
        # before _prepare_input_ids; the all-zeros default of the scratch
        # buffer would otherwise look like a common request.
        model_runner_module.ModelRunnerFL._compute_prev_positions(runner, 1)
        # _prepare_inputs fills the CPU side before calling in; mirror that.
        runner.input_ids.cpu[:3] = torch.tensor([5, 6, 7], dtype=torch.int32)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        model_runner_module.ModelRunnerFL._prepare_input_ids(
            runner, sched, 1, 3, np.array([3])
        )
        # prev_positions is all -1 (new request): no sampled ids scattered,
        # the plain CPU upload went through untouched.
        assert runner.input_ids.gpu[:3].tolist() == [5, 6, 7]

    def test_common_decode_fast_path_copies_slice(self, model_runner_module):
        """Unchanged batch copies prev sampled ids as a single slice."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(
            model_runner_module, "r1", (5, 6, 7), (9,), num_computed_tokens=3
        )
        state.output_token_ids = [60]
        batch.add_request(state)
        batch.refresh_metadata()

        prev_sampled = torch.tensor([[60], [61]], dtype=torch.int32)
        runner = self._runner(model_runner_module, batch, prev_sampled=prev_sampled)
        # Mark the request as present at index 0 in the previous iteration.
        batch.prev_req_id_to_index = {"r1": 0}
        model_runner_module.ModelRunnerFL._compute_prev_positions(runner, 1)
        runner.input_ids.cpu[0] = 99  # placeholder that must be overwritten

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._prepare_input_ids(
            runner, sched, 1, 1, np.array([1])
        )
        assert runner.input_ids.gpu[0].item() == 60

    def test_reordered_batch_scatters_sampled_tokens(self, model_runner_module):
        """A condense-reordered batch takes the scatter path with indices."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        # r2 sits at batch index 0 now but was at prev index 1.
        state2 = _make_req_state(
            model_runner_module, "r2", (8, 9), (10,), num_computed_tokens=2
        )
        state2.output_token_ids = [70]
        batch.add_request(state2)
        state1 = _make_req_state(
            model_runner_module, "r1", (5, 6, 7), (9,), num_computed_tokens=3
        )
        state1.output_token_ids = [71]
        batch.add_request(state1)
        batch.refresh_metadata()

        prev_sampled = torch.tensor([[71], [70]], dtype=torch.int32)
        runner = self._runner(model_runner_module, batch, prev_sampled=prev_sampled)
        batch.prev_req_id_to_index = {"r1": 0, "r2": 1}
        model_runner_module.ModelRunnerFL._compute_prev_positions(runner, 2)
        runner.input_ids.cpu[:2] = torch.tensor([98, 99], dtype=torch.int32)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r2": 1, "r1": 1},
            total_num_scheduled_tokens=2,
        )
        model_runner_module.ModelRunnerFL._prepare_input_ids(
            runner, sched, 2, 2, np.array([1, 2])
        )
        # r2 (index 0) got prev sampled row 1; r1 (index 1) got row 0.
        assert runner.input_ids.gpu[:2].tolist() == [70, 71]

    def test_draft_tokens_scattered_after_sampled(self, model_runner_module):
        """Spec-decode draft tokens scatter from the _draft_token_ids buffer."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(
            model_runner_module, "r1", (5, 6, 7), (9,), num_computed_tokens=3
        )
        state.output_token_ids = [60]
        batch.add_request(state)
        batch.refresh_metadata()

        prev_sampled = torch.tensor([[60]], dtype=torch.int32)
        draft_ids = torch.tensor([[80, 81]], dtype=torch.int64)
        runner = self._runner(
            model_runner_module,
            batch,
            prev_sampled=prev_sampled,
            draft_token_ids=draft_ids,
            prev_num_spec_tokens=2,
        )
        batch.prev_req_id_to_index = {"r1": 0}
        model_runner_module.ModelRunnerFL._compute_prev_positions(runner, 1)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
            spec_tokens={"r1": [80, 81]},
        )
        model_runner_module.ModelRunnerFL._prepare_input_ids(
            runner, sched, 1, 3, np.array([3])
        )
        # slot 0: the accepted sample; slots 1-2: the two draft tokens.
        assert runner.input_ids.gpu[:3].tolist() == [60, 80, 81]

    def test_pp_recv_work_waited_and_cleared(self, model_runner_module):
        """A pending PP broadcast work is waited on exactly once."""
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()

        waits = []

        class _Work:
            def wait(self):
                waits.append("wait")

        runner = self._runner(model_runner_module, batch, pp_recv_work=_Work())
        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        model_runner_module.ModelRunnerFL._prepare_input_ids(
            runner, sched, 1, 3, np.array([3])
        )
        assert waits == ["wait"]
        assert runner._pp_recv_work is None


# ---------------------------------------------------------------------------
# _calc_spec_decode_metadata
# ---------------------------------------------------------------------------


class TestCalcSpecDecodeMetadata:
    """ModelRunnerFL._calc_spec_decode_metadata against the docstring oracle."""

    def _runner(self, model_runner_module):
        torch = import_module("torch")
        runner = _base_runner(
            model_runner_module,
            _make_input_batch(model_runner_module),
            input_ids=model_runner_module.CpuGpuBuffer(
                256, dtype=torch.int32, device=torch.device(CPU_DEVICE)
            ),
            _arange_scratch=np.zeros(256, dtype=np.int32),
        )
        return runner

    def test_docstring_example_indices(self, model_runner_module):
        """The docstring worked example is reproduced exactly."""
        runner = self._runner(model_runner_module)
        num_draft = np.array([3, 0, 2, 0, 1], dtype=np.int32)
        cu_scheduled = np.array([4, 104, 107, 207, 209], dtype=np.int64)

        metadata = model_runner_module.ModelRunnerFL._calc_spec_decode_metadata(
            runner, num_draft, cu_scheduled
        )

        torch = import_module("torch")
        assert metadata.num_draft_tokens == [3, 0, 2, 0, 1]
        torch.testing.assert_close(
            metadata.cu_num_draft_tokens,
            torch.tensor([3, 3, 5, 5, 6], dtype=metadata.cu_num_draft_tokens.dtype),
        )
        torch.testing.assert_close(
            metadata.cu_num_sampled_tokens,
            torch.tensor([4, 5, 8, 9, 11], dtype=metadata.cu_num_sampled_tokens.dtype),
        )
        torch.testing.assert_close(
            metadata.logits_indices,
            torch.tensor(
                [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208],
                dtype=metadata.logits_indices.dtype,
            ),
        )
        torch.testing.assert_close(
            metadata.target_logits_indices,
            torch.tensor(
                [0, 1, 2, 5, 6, 9], dtype=metadata.target_logits_indices.dtype
            ),
        )
        torch.testing.assert_close(
            metadata.bonus_logits_indices,
            torch.tensor([3, 4, 7, 8, 10], dtype=metadata.bonus_logits_indices.dtype),
        )

    def test_draft_token_ids_gathered_from_input_ids(self, model_runner_module):
        """draft_token_ids = input_ids[logits][target + 1] as documented."""
        torch = import_module("torch")
        runner = self._runner(model_runner_module)
        # Fill input_ids with position-flavoured values to trace gathers;
        # _prepare_inputs uploads the buffer before this runs, so mirror it.
        runner.input_ids.cpu[:] = torch.arange(256, dtype=torch.int32)
        runner.input_ids.copy_to_gpu()
        num_draft = np.array([2], dtype=np.int32)
        cu_scheduled = np.array([3], dtype=np.int64)

        metadata = model_runner_module.ModelRunnerFL._calc_spec_decode_metadata(
            runner, num_draft, cu_scheduled
        )
        torch = import_module("torch")
        # logits_indices for a 3-token sched with 2 drafts: [0, 1, 2];
        # target = [0, 1]; draft_token_ids = ids[[0,1,2]][[1, 2]] = [1, 2].
        torch.testing.assert_close(
            metadata.draft_token_ids, torch.tensor([1, 2], dtype=torch.int32)
        )


def _prepare_inputs_runner(model_runner_module, batch, requests, **flags):
    """A runner carrying every buffer _prepare_inputs touches.

    ``requests`` maps req_id -> CachedRequestState for the scheduled batch
    rows (the production runner keeps the same dict).
    """
    torch = import_module("torch")
    device = torch.device(CPU_DEVICE)
    CpuGpuBuffer = model_runner_module.CpuGpuBuffer

    def buf(*size, dtype=torch.int32):
        return CpuGpuBuffer(*size, dtype=dtype, device=device)

    runner = _base_runner(
        model_runner_module,
        batch,
        enable_prompt_embeds=flags.get("enable_prompt_embeds", False),
        use_ascend_attention=False,
        use_async_spec_decode=False,
        max_model_len=64,
        input_ids=buf(64),
        inputs_embeds=buf((64, 4), dtype=torch.float32),
        is_token_ids=buf(64, dtype=torch.bool),
        query_start_loc=buf(9),
        positions=torch.zeros(64, dtype=torch.int64, device=device),
        query_pos=buf(64),
        seq_lens=torch.zeros(8, dtype=torch.int32, device=device),
        optimistic_seq_lens_cpu=torch.zeros(8, dtype=torch.int32),
        num_computed_tokens=torch.zeros(8, dtype=torch.int32, device=device),
        prev_num_draft_tokens=buf(8),
        req_indices=buf(64, dtype=torch.int64),
        prev_positions=buf(8, dtype=torch.int64),
        num_scheduled_tokens=buf(8),
        num_decode_draft_tokens=buf(8),
        num_accepted_tokens=buf(8),
        encoder_seq_lens=buf(8),
        discard_request_mask=buf(8, dtype=torch.bool),
        _arange_scratch=np.zeros(64, dtype=np.int32),
        valid_sampled_token_count_gpu=None,
        num_accepted_tokens_event=None,
        lora_config=None,
        _kl_guard_on=lambda: False,
        _pp_recv_work=None,
        _draft_token_ids=None,
        prev_num_spec_tokens=0,
    )
    runner._calc_spec_decode_metadata = _bound(
        model_runner_module, runner, "_calc_spec_decode_metadata"
    )
    runner._prepare_input_ids = _bound(
        model_runner_module, runner, "_prepare_input_ids"
    )
    runner.requests = dict(requests)
    return runner


def _bound(model_runner_module, obj, name):
    """Bind a real ModelRunnerFL method onto a foreign object."""
    MethodType = import_module("types").MethodType
    return MethodType(getattr(model_runner_module.ModelRunnerFL, name), obj)


class TestPrepareInputs:
    """ModelRunnerFL._prepare_inputs end-to-end on a real InputBatch."""

    def _batch_with(self, model_runner_module, specs):
        """Build a batch from (req_id, prompt, blocks, computed, outputs)."""
        batch = _make_input_batch(model_runner_module)
        requests = {}
        for req_id, prompt, blocks, computed, outputs in specs:
            state = _make_req_state(
                model_runner_module,
                req_id,
                prompt,
                blocks,
                num_computed_tokens=computed,
                output_token_ids=list(outputs),
            )
            batch.add_request(state)
            requests[req_id] = state
        batch.refresh_metadata()
        return batch, requests

    def test_prefill_positions_tokens_and_logits_indices(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """A fresh prefill gathers prompt tokens at contiguous positions."""
        import_module("torch")
        batch, requests = self._batch_with(
            model_runner_module,
            [("r1", (5, 6, 7), (9,), 0, ())],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
        )
        logits_indices, spec_meta, max_sampled = (
            model_runner_module.ModelRunnerFL._prepare_inputs(
                runner, sched, np.array([3], dtype=np.int32)
            )
        )

        assert spec_meta is None
        assert max_sampled == 1
        assert logits_indices.tolist() == [2]
        assert runner.input_ids.gpu[:3].tolist() == [5, 6, 7]
        # positions run 0..2 for a fresh prefill.
        assert runner.positions[:3].tolist() == [0, 1, 2]
        # seq_lens = num_computed + scheduled = 3.
        assert runner.seq_lens[:1].tolist() == [3]
        # Slot mapping oracle: block 9, block size 16, offsets 0..2.
        assert runner.input_batch.block_table[0].slot_mapping.gpu[:3].tolist() == [
            9 * 16 + 0,
            9 * 16 + 1,
            9 * 16 + 2,
        ]
        # num_computed mirrored onto the GPU tensor.
        assert runner.num_computed_tokens[:1].tolist() == [0]
        # Optimistic len 3 == num_tokens 3: the last chunk is sampled
        # (not discarded).
        assert runner.discard_request_mask.gpu[:1].tolist() == [False]

    def test_decode_uses_computed_tokens_offset(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """A decode step positions at num_computed and reuses the block."""
        batch, requests = self._batch_with(
            model_runner_module,
            [("r1", (5, 6, 7), (9,), 3, (60,))],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        logits_indices, _, _ = model_runner_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([1], dtype=np.int32)
        )

        assert logits_indices.tolist() == [0]
        # The decode token is the last output token id in the row.
        assert runner.input_ids.gpu[0].item() == 60
        assert runner.positions[0].item() == 3
        assert runner.seq_lens[:1].tolist() == [4]
        # Slot for position 3 in block 9.
        assert (
            runner.input_batch.block_table[0].slot_mapping.gpu[0].item() == 9 * 16 + 3
        )

    def test_chunked_prefill_second_segment(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """The second chunk of a chunked prefill resumes at num_computed."""
        batch, requests = self._batch_with(
            model_runner_module,
            [("r1", (5, 6, 7, 8, 9), (9,), 3, ())],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 2},
            total_num_scheduled_tokens=2,
        )
        logits_indices, _, _ = model_runner_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([2], dtype=np.int32)
        )

        assert logits_indices.tolist() == [1]
        assert runner.input_ids.gpu[:2].tolist() == [8, 9]
        assert runner.positions[:2].tolist() == [3, 4]
        # Optimistic len 3+2 == num_tokens 5: the final chunk IS sampled.
        assert runner.discard_request_mask.gpu[:1].tolist() == [False]

    def test_multi_request_layouts_are_interleaved(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """req_indices/query_start_loc follow the flattened token layout."""
        batch, requests = self._batch_with(
            model_runner_module,
            [
                ("r1", (5, 6, 7), (9,), 3, (60,)),
                ("r2", (8, 9), (10,), 2, (61, 62)),
            ],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 1, "r2": 2},
            total_num_scheduled_tokens=3,
        )
        logits_indices, _, max_sampled = (
            model_runner_module.ModelRunnerFL._prepare_inputs(
                runner, sched, np.array([1, 2], dtype=np.int32)
            )
        )

        assert max_sampled == 1
        # r1 decode token, then r2's two decode tokens.
        assert runner.input_ids.gpu[:3].tolist() == [60, 61, 62]
        assert runner.req_indices.gpu[:3].tolist() == [0, 1, 1]
        assert runner.query_start_loc.gpu[:3].tolist() == [0, 1, 3]
        assert logits_indices.tolist() == [0, 2]
        assert runner.positions[:3].tolist() == [3, 2, 3]
        # seq lens: r1 = 3+1, r2 = 2+2.
        assert runner.seq_lens[:2].tolist() == [4, 4]

    def test_partial_request_is_discarded_from_sampling(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """A chunked prefill whose optimistic len reached num_tokens is kept,
        while one still short of its full prompt is marked discard."""
        batch, requests = self._batch_with(
            model_runner_module,
            [
                ("r1", (5, 6, 7, 8), (9,), 3, ()),  # optimistic 4 == num_tokens
                ("r2", (1, 2, 3, 4, 5), (10,), 2, ()),  # optimistic 3 < 5
            ],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 1, "r2": 1},
            total_num_scheduled_tokens=2,
        )
        model_runner_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([1, 1], dtype=np.int32)
        )

        # discard = optimistic < num_tokens: r1 False, r2 True.
        assert runner.discard_request_mask.gpu[:2].tolist() == [False, True]

    def test_spec_decode_path_builds_metadata(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """Scheduled draft tokens route through _calc_spec_decode_metadata."""
        batch, requests = self._batch_with(
            model_runner_module,
            [("r1", (5, 6, 7), (9,), 3, (60,))],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 3},
            total_num_scheduled_tokens=3,
            spec_tokens={"r1": [80, 81]},
        )
        logits_indices, spec_meta, max_sampled = (
            model_runner_module.ModelRunnerFL._prepare_inputs(
                runner, sched, np.array([3], dtype=np.int32)
            )
        )

        assert spec_meta is not None
        assert spec_meta.num_draft_tokens == [2]
        # num_sampled_tokens = drafts + 1 = 3.
        assert max_sampled == 3
        assert logits_indices.tolist() == spec_meta.logits_indices.tolist()
        # num_decode_draft_tokens marks decode-phase drafts (sched == len+1).
        assert runner.num_decode_draft_tokens.gpu[:1].tolist() == [2]

    def test_spec_decode_prefill_keeps_minus_one_mask(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """A prefill request with drafts keeps the -1 decode-draft mask."""
        batch, requests = self._batch_with(
            model_runner_module,
            [("r1", (5, 6, 7), (9,), 1, ())],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)

        # scheduled 3 != draft_len(2) + 1? 3 == 2+1 -> decode path; make the
        # scheduled count 4 so the prefill mask branch is taken.
        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 4},
            total_num_scheduled_tokens=4,
            spec_tokens={"r1": [80, 81]},
        )
        model_runner_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([4], dtype=np.int32)
        )
        # Prefill-with-drafts: the mask stays -1 (guided rollback sentinel).
        assert runner.num_decode_draft_tokens.gpu[:1].tolist() == [-1]

    def test_prompt_embeds_spliced_into_inputs_embeds(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """req_prompt_embeds rows land in inputs_embeds at the right slots."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(
            model_runner_module, "r1", (5, 6, 7), (9,), num_computed_tokens=1
        )
        # Three embed rows: row 0 was consumed, rows 1-2 are scheduled.
        state.prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        # Embeds-only request: __post_init__ already fixed num_prompt_tokens=3
        # from the original 3-token prompt, matching the embed row count, and
        # add_request marks is_token_ids False only when ids are absent.
        state.prompt_token_ids = None
        batch.add_request(state)
        batch.refresh_metadata()
        runner = _prepare_inputs_runner(
            model_runner_module,
            batch,
            {"r1": state},
            enable_prompt_embeds=True,
        )

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 2},
            total_num_scheduled_tokens=2,
        )
        model_runner_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([2], dtype=np.int32)
        )

        # Rows 1..2 of the prompt embeds were spliced into inputs_embeds
        # (row 0 was already consumed by the first chunk).
        assert torch.equal(runner.inputs_embeds.gpu[:2], state.prompt_embeds[1:3])
        # is_token_ids follows the batch row: positions 1..2 of an
        # embeds-backed prompt are marked False (not raw token ids).
        assert runner.is_token_ids.gpu[:2].tolist() == [False, False]

    def test_kl_guard_fills_positions_from_cpu(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """The kunlunxin guard path fills positions from the CPU array."""
        batch, requests = self._batch_with(
            model_runner_module,
            [("r1", (5, 6, 7), (9,), 3, (60,))],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)
        runner._kl_guard_on = lambda: True

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([1], dtype=np.int32)
        )
        assert runner.positions[:1].tolist() == [3]

    def test_query_start_loc_padded_region_filled(
        self, model_runner_module, cpu_slot_mapping_kernel, pp_last_rank
    ):
        """Rows past num_reqs repeat the final cumsum (non-decreasing)."""
        batch, requests = self._batch_with(
            model_runner_module,
            [("r1", (5, 6), (9,), 2, (60,))],
        )
        runner = _prepare_inputs_runner(model_runner_module, batch, requests)

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        model_runner_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([1], dtype=np.int32)
        )
        # [0, 1, 1, 1, ...]: the tail repeats the total.
        assert runner.query_start_loc.gpu[:4].tolist() == [0, 1, 1, 1]


# ---------------------------------------------------------------------------
# _build_attention_metadata
# ---------------------------------------------------------------------------


class _RecordingBuilder:
    """A minimal AttentionMetadataBuilder stand-in recording build calls.

    ``AttentionMetadataBuilder`` is an ABC whose abstract ``__init__`` and
    ``build`` must be overridden; this subclass stores the common metadata it
    was built with so tests can assert what the production loop passed down
    without touching any real GPU backend.  The recorded metadata objects are
    the *real* ``CommonAttentionMetadata`` instances produced by the method
    under test.
    """

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.builds = []
        self.capture_builds = []
        self.updates = []

    # Mirrors AttentionMetadataBuilder.supports_update_block_table (False).
    supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.builds.append((common_prefix_len, common_attn_metadata, kwargs))
        return {"cm": common_attn_metadata, "prefix_len": common_prefix_len}

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.capture_builds.append(common_attn_metadata)
        return {"cm": common_attn_metadata, "capture": True}

    def update_block_table(self, metadata, blk_table, slot_mapping):
        self.updates.append((metadata, blk_table, slot_mapping))
        return {"cm": metadata["cm"], "updated": True}


class _UpdateableBuilder(_RecordingBuilder):
    """Same recorder, but advertises block-table update support."""

    supports_update_block_table = True


def _full_spec():
    """A real FullAttentionSpec sized for the CPU test batch."""
    torch = import_module("torch")
    FullAttentionSpec = import_module("vllm.v1.kv_cache_interface").FullAttentionSpec
    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
    )


def _bam_runner(model_runner_module, batch, groups, kv_cache_specs, **overrides):
    """A runner wired for _build_attention_metadata.

    ``groups`` is a list (per kv cache group) of lists of
    ``(layer_names, builder)`` pairs; each becomes a real AttentionGroup with
    the real spec, and the runner's kv_cache_config holds real
    KVCacheGroupSpec entries.  ``batch`` must already contain its requests.
    """
    torch = import_module("torch")
    AttentionGroup = import_module("vllm.v1.worker.utils").AttentionGroup
    kv_interface = import_module("vllm.v1.kv_cache_interface")

    device = torch.device(CPU_DEVICE)
    attn_groups = []
    kv_cache_groups = []
    for gid, (group_list, spec) in enumerate(zip(groups, kv_cache_specs)):
        group_objects = []
        layer_names_all = []
        for layer_names, builder in group_list:
            group = AttentionGroup(
                backend=None,
                layer_names=list(layer_names),
                kv_cache_spec=spec,
                kv_cache_group_id=gid,
            )
            group.metadata_builders = [builder]
            group_objects.append(group)
            layer_names_all.extend(layer_names)
        attn_groups.append(group_objects)
        kv_cache_groups.append(
            kv_interface.KVCacheGroupSpec(
                layer_names=layer_names_all, kv_cache_spec=spec
            )
        )

    runner = _base_runner(
        model_runner_module,
        batch,
        kv_cache_config=kv_interface.KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        ),
        attn_groups=attn_groups,
        max_model_len=64,
        optimistic_seq_lens_cpu=torch.zeros(8, dtype=torch.int32),
        query_start_loc=model_runner_module.CpuGpuBuffer(
            9, dtype=torch.int32, device=device
        ),
        seq_lens=torch.zeros(8, dtype=torch.int32, device=device),
        positions=torch.zeros(64, dtype=torch.int64, device=device),
        use_ascend_attention=False,
        use_async_spec_decode=False,
        dcp_world_size=1,
        dcp_rank=0,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        is_mm_prefix_lm=False,
        cache_config=SimpleNamespace(use_replayssm=False),
        model_config=SimpleNamespace(rswa_window=None),
        routed_experts_initialized=False,
        speculative_config=None,
        drafter=None,
        num_accepted_tokens=model_runner_module.CpuGpuBuffer(
            8, dtype=torch.int32, device=device
        ),
        num_decode_draft_tokens=model_runner_module.CpuGpuBuffer(
            8, dtype=torch.int32, device=device
        ),
        encoder_seq_lens=model_runner_module.CpuGpuBuffer(
            8, dtype=torch.int32, device=device
        ),
    )
    runner._get_encoder_seq_lens = _bound(
        model_runner_module, runner, "_get_encoder_seq_lens"
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    # Mirror the production sequence: the block table is committed to the
    # (CPU) device buffer before _build_attention_metadata reads it.
    batch.block_table.commit_block_table(batch.num_reqs)
    return runner


def _fill_query_start_loc(runner, cu_tokens):
    """Write cu_num_tokens into the runner's query_start_loc buffer.

    ``cu_tokens`` holds the inclusive cumsum without the leading zero, so
    [3] becomes the buffer [0, 3, 3, ...] exactly like _prepare_inputs does.
    """
    torch = import_module("torch")
    vals = [0, *cu_tokens]
    padded = vals + [cu_tokens[-1]] * (9 - len(vals))
    runner.query_start_loc.cpu[:] = torch.tensor(padded, dtype=torch.int32)
    runner.query_start_loc.copy_to_gpu()


class TestBuildAttentionMetadata:
    """ModelRunnerFL._build_attention_metadata against real KV cache specs."""

    def _batch_with(self, model_runner_module, num_groups=1):
        """A batch of one request; ``num_groups`` block tables per request."""
        torch = import_module("torch")
        InputBatch = import_module("vllm.v1.worker.gpu_input_batch").InputBatch
        batch = InputBatch(
            max_num_reqs=8,
            max_model_len=64,
            max_num_batched_tokens=64,
            device=torch.device(CPU_DEVICE),
            vocab_size=VOCAB,
            block_sizes=[16] * num_groups,
            max_num_blocks_per_req=[4] * num_groups,
        )
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        if num_groups > 1:
            state.block_ids = ([9],) + ([9],) * (num_groups - 1)
        batch.add_request(state)
        batch.refresh_metadata()
        return batch, state

    def test_attention_free_model_returns_empty(self, model_runner_module):
        """Zero kv-cache groups short-circuit to ({}, None)."""
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        runner = _base_runner(
            model_runner_module,
            _make_input_batch(model_runner_module),
            kv_cache_config=kv_interface.KVCacheConfig(
                num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
            ),
        )
        result = model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3
        )
        assert result == ({}, None)

    def test_single_group_builds_layer_metadata(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """One full-attention group: every layer maps to the built metadata."""
        torch = import_module("torch")
        batch, _ = self._batch_with(model_runner_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            model_runner_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}

        attn_metadata, spec_cm = (
            model_runner_module.ModelRunnerFL._build_attention_metadata(
                runner, 3, 1, 3, slot_mappings=slot_mappings
            )
        )

        assert spec_cm is None
        assert builder.builds, "the builder must have been invoked"
        prefix_len, cm, kwargs = builder.builds[0]
        assert prefix_len == 0
        assert cm.num_reqs == 1
        assert cm.num_actual_tokens == 3
        assert cm.max_query_len == 3
        # max_seq_len comes from optimistic_seq_lens (3 here, not max_model_len).
        assert cm.max_seq_len == 3
        assert attn_metadata["layers.0"]["cm"] is cm

    def test_cudagraph_capture_uses_max_model_len(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """Capture mode builds via build_for_cudagraph_capture with
        max_seq_len == max_model_len (64) rather than the batch max (3)."""
        torch = import_module("torch")
        batch, _ = self._batch_with(model_runner_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            model_runner_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}

        attn_metadata, _ = model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner,
            3,
            1,
            3,
            for_cudagraph_capture=True,
            slot_mappings=slot_mappings,
        )

        assert builder.capture_builds, "capture build path must run"
        assert builder.capture_builds[0].max_seq_len == 64
        assert attn_metadata["layers.0"]["capture"] is True

    def test_padded_rows_get_null_block_id(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """Padded request rows of the block table are set to NULL_BLOCK_ID."""
        torch = import_module("torch")
        NULL_BLOCK_ID = 0
        batch, _ = self._batch_with(model_runner_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            model_runner_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        # Poison rows 2..3: they sit in the padded region (3 > num_reqs=1).
        batch.block_table[0].block_table.cpu[2, 0] = 77
        batch.block_table[0].block_table.cpu[3, 0] = 88
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}

        model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, num_reqs_padded=3, slot_mappings=slot_mappings
        )

        block_table_tensor = builder.builds[0][1].block_table_tensor
        # Row 0 keeps the real block id 9; padding rows are NULL_BLOCK_ID.
        assert block_table_tensor[0, 0].item() == 9
        assert block_table_tensor[1, 0].item() == NULL_BLOCK_ID
        assert block_table_tensor[2, 0].item() == NULL_BLOCK_ID

    def test_hybrid_groups_reuse_metadata_via_update_block_table(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """Two groups with identical spec+builder type: the second build goes
        through builder.update_block_table instead of a fresh build."""
        torch = import_module("torch")
        batch, _ = self._batch_with(model_runner_module, num_groups=2)
        spec = _full_spec()
        builder = _UpdateableBuilder(spec, ["layers.0"], None, None)
        builder2 = _UpdateableBuilder(spec, ["layers.1"], None, None)
        runner = _bam_runner(
            model_runner_module,
            batch,
            [[(["layers.0"], builder)], [(["layers.1"], builder2)]],
            [spec, spec],
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {
            0: torch.arange(64, dtype=torch.int64),
            1: torch.arange(64, dtype=torch.int64),
        }

        attn_metadata, _ = model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )

        # First group: fresh build; second group: cached update path.
        assert len(builder.builds) == 1
        assert len(builder2.updates) == 1, "identical spec must reuse via update"
        assert len(builder2.builds) == 0
        assert attn_metadata["layers.1"]["updated"] is True

    def test_encoder_only_spec_gets_zero_block_table(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """EncoderOnlyAttentionSpec groups read a zeros tensor, not the batch."""
        torch = import_module("torch")
        EncoderOnlyAttentionSpec = import_module(
            "vllm.v1.kv_cache_interface"
        ).EncoderOnlyAttentionSpec
        batch, _ = self._batch_with(model_runner_module)
        spec = EncoderOnlyAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )
        builder = _RecordingBuilder(spec, ["enc.0"], None, None)
        runner = _bam_runner(
            model_runner_module, batch, [[(["enc.0"], builder)]], [spec]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}

        model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )

        block_table_tensor = builder.builds[0][1].block_table_tensor
        assert block_table_tensor.shape == (1, 1)
        assert block_table_tensor.sum().item() == 0

    def test_ubatch_slices_build_per_ubatch(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """ubatch_slices produce a list of per-ubatch metadata dicts."""
        torch = import_module("torch")
        UBatchSlice = import_module("vllm.v1.worker.ubatch_utils").UBatchSlice
        batch = _make_input_batch(model_runner_module)
        state1 = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        state2 = _make_req_state(model_runner_module, "r2", (8, 9), (10,))
        batch.add_request(state1)
        batch.add_request(state2)
        batch.refresh_metadata()

        spec = _full_spec()
        # One metadata builder per ubatch, mirroring create_metadata_builders.
        builders = [
            _RecordingBuilder(spec, ["layers.0"], None, None),
            _RecordingBuilder(spec, ["layers.0"], None, None),
        ]
        runner = _bam_runner(
            model_runner_module, batch, [[(["layers.0"], builders[0])]], [spec]
        )
        runner.attn_groups[0][0].metadata_builders = builders
        # Two requests, two tokens each: [0,2) and [2,4).
        _fill_query_start_loc(runner, [2, 4])
        runner.optimistic_seq_lens_cpu[:2] = 2
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}

        ubatch_slices = [
            UBatchSlice(slice(0, 1), slice(0, 2)),
            UBatchSlice(slice(1, 2), slice(2, 4)),
        ]
        attn_metadata, _ = model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner,
            4,
            2,
            2,
            ubatch_slices=ubatch_slices,
            slot_mappings=slot_mappings,
        )

        assert isinstance(attn_metadata, list) and len(attn_metadata) == 2
        # Each ubatch built its own metadata through its own builder.
        assert len(builders[0].builds) == 1 and len(builders[1].builds) == 1
        cm0 = builders[0].builds[0][1]
        cm1 = builders[1].builds[0][1]
        assert cm0.num_reqs == 1 and cm1.num_reqs == 1
        assert cm0.num_actual_tokens == 2 and cm1.num_actual_tokens == 2
        assert attn_metadata[0]["layers.0"]["cm"] is cm0
        assert attn_metadata[1]["layers.0"]["cm"] is cm1

    def test_spec_decode_metadata_unpadded_when_padded(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """spec decode + padding: the drafter's common metadata is unpadded."""
        torch = import_module("torch")
        batch, _ = self._batch_with(model_runner_module)
        spec = _full_spec()
        builder = _RecordingBuilder(spec, ["layers.0"], None, None)
        # A drafter of a class not in the special list: every group's cm is
        # eligible, then unpadded because padding was requested.
        runner = _bam_runner(
            model_runner_module,
            batch,
            [[(["layers.0"], builder)]],
            [spec],
            speculative_config=SimpleNamespace(method="eagle"),
            drafter=SimpleNamespace(),
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}

        _, spec_cm = model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner,
            3,
            1,
            3,
            num_tokens_padded=8,
            num_reqs_padded=2,
            slot_mappings=slot_mappings,
        )

        assert spec_cm is not None
        assert spec_cm.num_reqs == 1
        assert spec_cm.num_actual_tokens == 3

    def test_eagle_drafter_selects_own_kv_group(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """An EagleProposer drafter only gets the cm of its own kv group.

        The EagleProposer branch of the spec-decode selection compares
        ``drafter.kv_cache_gid`` against each kv cache group id and keeps only
        the matching group's common metadata, instead of the first group's.
        """
        torch = import_module("torch")
        EagleProposer = import_module("vllm.v1.spec_decode.eagle").EagleProposer
        batch, _ = self._batch_with(model_runner_module, num_groups=2)
        spec0 = _full_spec()
        spec1 = _full_spec()
        builder0 = _RecordingBuilder(spec0, ["layers.0"], None, None)
        builder1 = _RecordingBuilder(spec1, ["layers.1"], None, None)
        # A bare EagleProposer: __new__ skips the model-loading __init__; the
        # production branch only reads ``kv_cache_gid`` off the instance.
        drafter = EagleProposer.__new__(EagleProposer)
        drafter.kv_cache_gid = 1
        runner = _bam_runner(
            model_runner_module,
            batch,
            [[(["layers.0"], builder0)], [(["layers.1"], builder1)]],
            [spec0, spec1],
            speculative_config=SimpleNamespace(method="eagle"),
            drafter=drafter,
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        # Distinguishable mappings so identity checks are meaningful.
        slot_mappings = {
            0: torch.arange(64, dtype=torch.int64),
            1: torch.arange(100, 164, dtype=torch.int64),
        }

        _, spec_cm = model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )

        # The drafter's cm is the group-1 copy: its slot_mapping is group 1's.
        assert spec_cm is not None
        assert spec_cm.slot_mapping is slot_mappings[1]
        assert spec_cm.block_table_tensor is builder1.builds[0][1].block_table_tensor

    def test_cross_attention_spec_gets_encoder_seq_lens(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """CrossAttentionSpec groups receive encoder_seq_lens from the batch."""
        torch = import_module("torch")
        CrossAttentionSpec = import_module(
            "vllm.v1.kv_cache_interface"
        ).CrossAttentionSpec
        mm_inputs = import_module("vllm.multimodal.inputs")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        # One mm feature of length 4 placed at offset 0.
        feature = mm_inputs.MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier="hash-1",
            mm_position=mm_inputs.PlaceholderRange(offset=0, length=4),
        )
        state.mm_features = [feature]
        batch.add_request(state)
        batch.refresh_metadata()

        spec = CrossAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )
        builder = _RecordingBuilder(spec, ["cross.0"], None, None)
        runner = _bam_runner(
            model_runner_module,
            batch,
            [[(["cross.0"], builder)]],
            [spec],
            requests={"r1": state},
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}

        model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner,
            3,
            1,
            3,
            num_scheduled_tokens={"r1": 3},
            slot_mappings=slot_mappings,
        )

        cm = builder.builds[0][1]
        assert cm.encoder_seq_lens is not None
        assert cm.encoder_seq_lens.tolist() == [4]
        assert cm.encoder_seq_lens_cpu.tolist() == [4]

    def test_uniform_type_specs_dereference_first_layer(
        self, model_runner_module, pp_last_rank, cpu_slot_mapping_kernel
    ):
        """UniformTypeKVCacheSpecs cache keys dereference to the inner spec.

        Two attention groups inside one uniform-spec kv cache group: the
        production loop dereferences ``kv_cache_specs[layer_names[0]]`` to
        build the builder cache key.  With an update-capable builder the
        second group must reuse the first group's metadata through
        ``update_block_table`` — proving both dereferenced the same inner
        FullAttentionSpec and hit one cache entry.
        """
        torch = import_module("torch")
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        inner = _full_spec()
        uniform = kv_interface.UniformTypeKVCacheSpecs(
            block_size=16, kv_cache_specs={"layers.0": inner, "layers.1": inner}
        )
        batch, _ = self._batch_with(model_runner_module)
        builder = _UpdateableBuilder(inner, ["layers.0"], None, None)
        builder2 = _UpdateableBuilder(inner, ["layers.1"], None, None)
        runner = _bam_runner(
            model_runner_module,
            batch,
            [[(["layers.0"], builder), (["layers.1"], builder2)]],
            [uniform],
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}

        attn_metadata, _ = model_runner_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        # The dereferenced inner spec matched, so the second group reused.
        assert len(builder.builds) == 1
        assert len(builder2.updates) == 1
        assert attn_metadata["layers.1"]["updated"] is True


# ---------------------------------------------------------------------------
# _batch_mm_inputs_from_scheduler / _execute_mm_encoder / _gather_mm_embeddings
# ---------------------------------------------------------------------------


def _mm_feature(modality, identifier, offset, length, data=None, is_embed=None):
    """Build a real MultiModalFeatureSpec with a real PlaceholderRange."""
    mm_inputs = import_module("vllm.multimodal.inputs")
    return mm_inputs.MultiModalFeatureSpec(
        data=data,
        modality=modality,
        identifier=identifier,
        mm_position=mm_inputs.PlaceholderRange(
            offset=offset, length=length, is_embed=is_embed
        ),
    )


def _mm_item(torch, values):
    """A real MultiModalKwargsItem with one batched-style ``pixels`` field.

    ``MultiModalBatchedField`` is the field type vLLM processors emit for
    per-item pixel tensors, so ``group_and_batch_mm_kwargs`` stacks
    consecutive items along dim 0 exactly as it would in production.
    """
    mm_inputs = import_module("vllm.multimodal.inputs")
    elem = mm_inputs.MultiModalFieldElem(
        data=torch.tensor(values),
        field=mm_inputs.MultiModalBatchedField(),
    )
    return mm_inputs.MultiModalKwargsItem({"pixels": elem})


def _pe_item(torch, embeds):
    """A real ``prompt_embeds`` item: a single ``embedding`` field."""
    mm_inputs = import_module("vllm.multimodal.inputs")
    elem = mm_inputs.MultiModalFieldElem(
        data=embeds,
        field=mm_inputs.MultiModalSharedField(batch_size=1),
    )
    return mm_inputs.MultiModalKwargsItem({"embedding": elem})


class _FakeMmModel:
    """Recording model boundary for the multimodal encoder path.

    ``embed_multimodal`` derives the item count from the leading dimension
    of the batched pixels (the contract of a batched field) and returns one
    2D tensor per item, satisfying ``sanity_check_mm_encoder_outputs``.
    The LoRA hooks return configurable tower/connector token counts.
    """

    def __init__(self, tower_tokens=3, connector_tokens=2):
        self.calls = []
        self._tower_tokens = tower_tokens
        self._connector_tokens = connector_tokens

    def embed_multimodal(self, **kwargs):
        torch = import_module("torch")
        num_items = kwargs["pixels"].shape[0]
        self.calls.append(kwargs)
        return [torch.full((2, 3), float(num_items + i)) for i in range(num_items)]

    def get_mm_lora_token_counts(self, modality, mm_kwargs, num_mm_embeds):
        return self._tower_tokens, self._connector_tokens

    def get_mm_mapping(self):
        return SimpleNamespace(connector=True)


class TestBatchMmInputsFromScheduler:
    """ModelRunnerFL._batch_mm_inputs_from_scheduler on real feature specs."""

    def _runner(self, model_runner_module, requests):
        runner = _base_runner(
            model_runner_module, _make_input_batch(model_runner_module)
        )
        runner.requests = dict(requests)
        return runner

    def test_empty_scheduler_inputs_return_triply_empty(self, model_runner_module):
        """No scheduled encoder inputs short-circuits to ([], [], [])."""
        runner = self._runner(model_runner_module, {})
        sched = _scheduler_output(model_runner_module)
        result = model_runner_module.ModelRunnerFL._batch_mm_inputs_from_scheduler(
            runner, sched
        )
        assert result == ([], [], [])

    def test_features_are_batched_with_hash_and_position(self, model_runner_module):
        """Each scheduled feature contributes hash, kwargs and lora ref."""
        torch = import_module("torch")
        state = _make_req_state(model_runner_module, "r1", (5,), (9,))
        state.mm_features = [
            _mm_feature("image", "h1", 1, 2, data=_mm_item(torch, [1.0, 2.0])),
            _mm_feature("audio", "h2", 4, 1, data=_mm_item(torch, [3.0])),
        ]
        runner = self._runner(model_runner_module, {"r1": state})

        sched = _scheduler_output(
            model_runner_module, scheduled_encoder_inputs={"r1": [0, 1]}
        )
        hashes, kwargs, lora_refs = (
            model_runner_module.ModelRunnerFL._batch_mm_inputs_from_scheduler(
                runner, sched
            )
        )

        assert hashes == ["h1", "h2"]
        assert [m for m, _ in kwargs] == ["image", "audio"]
        # The kwargs payload is the feature's data item verbatim.
        assert kwargs[0][1]["pixels"].data.tolist() == [1.0, 2.0]
        assert kwargs[1][1]["pixels"].data.tolist() == [3.0]
        assert [(req, pos.offset) for req, pos in lora_refs] == [("r1", 1), ("r1", 4)]

    def test_cached_features_with_none_data_are_skipped(self, model_runner_module):
        """data=None items (already processed) contribute nothing."""
        torch = import_module("torch")
        state = _make_req_state(model_runner_module, "r1", (5,), (9,))
        state.mm_features = [
            _mm_feature("image", "h1", 0, 2, data=None),
            _mm_feature("image", "h2", 2, 2, data=_mm_item(torch, [9.0])),
        ]
        runner = self._runner(model_runner_module, {"r1": state})

        sched = _scheduler_output(
            model_runner_module, scheduled_encoder_inputs={"r1": [0, 1]}
        )
        hashes, kwargs, lora_refs = (
            model_runner_module.ModelRunnerFL._batch_mm_inputs_from_scheduler(
                runner, sched
            )
        )
        assert hashes == ["h2"]
        assert [m for m, _ in kwargs] == ["image"]
        assert [(req, pos.offset) for req, pos in lora_refs] == [("r1", 2)]


class TestExecuteMmEncoder:
    """ModelRunnerFL._execute_mm_encoder with a recording model boundary."""

    def _runner(self, model_runner_module, batch, requests, model, **overrides):
        runner = _base_runner(
            model_runner_module,
            batch,
            requests=dict(requests),
            model=model,
            lora_config=None,
            is_multimodal_pruning_enabled=False,
            requires_sequential_video_encoding=False,
            encoder_cudagraph_manager=None,
            observability_config=None,
        )
        runner._batch_mm_inputs_from_scheduler = _bound(
            model_runner_module, runner, "_batch_mm_inputs_from_scheduler"
        )
        runner._cache_encoder_output = _bound(
            model_runner_module, runner, "_cache_encoder_output"
        )
        runner.timed_encoder_operation = _bound(
            model_runner_module, runner, "timed_encoder_operation"
        )
        # The real mixin staticmethod; it early-returns without EC transfer.
        runner.maybe_save_ec_to_connector = (
            model_runner_module.ModelRunnerFL.maybe_save_ec_to_connector
        )
        for key, value in overrides.items():
            setattr(runner, key, value)
        return runner

    def _sched(self, model_runner_module, mm_ids, num_tokens):
        return _scheduler_output(
            model_runner_module,
            scheduled_encoder_inputs={"r1": mm_ids},
            num_scheduled_tokens={"r1": num_tokens},
            total_num_scheduled_tokens=num_tokens,
        )

    def test_no_kwargs_returns_empty_list(self, model_runner_module):
        """No scheduled encoder inputs: nothing to encode."""
        batch = _make_input_batch(model_runner_module)
        runner = self._runner(model_runner_module, batch, {}, _FakeMmModel())
        sched = _scheduler_output(model_runner_module)
        out = model_runner_module.ModelRunnerFL._execute_mm_encoder(runner, sched)
        assert out == []

    def test_encoder_outputs_cached_by_hash(self, model_runner_module):
        """Model outputs land in the encoder cache keyed by mm hash."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        state.mm_features = [
            _mm_feature("image", "h1", 0, 2, data=_mm_item(torch, [1.0, 2.0]))
        ]
        batch.add_request(state)
        model = _FakeMmModel()
        runner = self._runner(model_runner_module, batch, {"r1": state}, model)

        out = model_runner_module.ModelRunnerFL._execute_mm_encoder(
            runner, self._sched(model_runner_module, [0], 2)
        )

        assert len(out) == 1
        assert out[0].shape == (2, 3)
        assert "h1" in runner.encoder_cache
        assert runner.encoder_cache["h1"] is out[0]
        # embed_multimodal received the batched pixels of the single item.
        assert len(model.calls) == 1
        assert model.calls[0]["pixels"].shape[0] == 1

    def test_two_images_batch_into_one_call(self, model_runner_module):
        """Consecutive same-modality items share one encoder invocation."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        state.mm_features = [
            _mm_feature("image", "h1", 0, 1, data=_mm_item(torch, [1.0])),
            _mm_feature("image", "h2", 1, 1, data=_mm_item(torch, [2.0])),
        ]
        batch.add_request(state)
        model = _FakeMmModel()
        runner = self._runner(model_runner_module, batch, {"r1": state}, model)

        out = model_runner_module.ModelRunnerFL._execute_mm_encoder(
            runner, self._sched(model_runner_module, [0, 1], 2)
        )

        # One stacked call with both rows, two outputs, two cached hashes.
        assert len(model.calls) == 1
        assert model.calls[0]["pixels"].shape == (2, 1)
        assert len(out) == 2
        assert "h1" in runner.encoder_cache
        assert "h2" in runner.encoder_cache

    def test_prompt_embeds_passthrough_skips_encoder(self, model_runner_module):
        """prompt_embeds items are cached directly without model calls."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5,), (9,))
        embeds = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        state.mm_features = [
            _mm_feature("prompt_embeds", "pe1", 0, 2, data=_pe_item(torch, embeds))
        ]
        batch.add_request(state)
        model = _FakeMmModel()
        runner = self._runner(model_runner_module, batch, {"r1": state}, model)

        out = model_runner_module.ModelRunnerFL._execute_mm_encoder(
            runner, self._sched(model_runner_module, [0], 1)
        )

        # No encoder ran: the passthrough returned nothing.
        assert out == []
        assert model.calls == []
        # But the embeds were placed in the encoder cache.
        assert torch.equal(runner.encoder_cache["pe1"], embeds)

    def test_prompt_embeds_and_image_mixed(self, model_runner_module):
        """A prompt_embeds item next to an image only encodes the image."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        embeds = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        state.mm_features = [
            _mm_feature("prompt_embeds", "pe1", 0, 2, data=_pe_item(torch, embeds)),
            _mm_feature("image", "img1", 2, 2, data=_mm_item(torch, [1.0, 2.0])),
        ]
        batch.add_request(state)
        model = _FakeMmModel()
        runner = self._runner(model_runner_module, batch, {"r1": state}, model)

        out = model_runner_module.ModelRunnerFL._execute_mm_encoder(
            runner, self._sched(model_runner_module, [0, 1], 2)
        )

        # Only the image went through the model.
        assert len(model.calls) == 1
        assert len(out) == 1
        assert "pe1" in runner.encoder_cache
        assert "img1" in runner.encoder_cache

    def test_different_modalities_encode_separately(self, model_runner_module):
        """An image and an audio item form two separate encoder groups."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        state.mm_features = [
            _mm_feature("image", "i1", 0, 1, data=_mm_item(torch, [1.0])),
            _mm_feature("audio", "a1", 1, 1, data=_mm_item(torch, [2.0])),
        ]
        batch.add_request(state)
        model = _FakeMmModel()
        runner = self._runner(model_runner_module, batch, {"r1": state}, model)

        out = model_runner_module.ModelRunnerFL._execute_mm_encoder(
            runner, self._sched(model_runner_module, [0, 1], 2)
        )

        # Two modality groups -> two model invocations, two cached hashes.
        assert len(model.calls) == 2
        assert len(out) == 2
        assert "i1" in runner.encoder_cache
        assert "a1" in runner.encoder_cache

    def test_sequential_video_encoding_runs_one_by_one(self, model_runner_module):
        """requires_sequential_video_encoding encodes each video alone."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        state.mm_features = [
            _mm_feature("video", "v1", 0, 1, data=_mm_item(torch, [1.0])),
            _mm_feature("video", "v2", 1, 1, data=_mm_item(torch, [2.0])),
        ]
        batch.add_request(state)
        model = _FakeMmModel()
        runner = self._runner(
            model_runner_module,
            batch,
            {"r1": state},
            model,
            requires_sequential_video_encoding=True,
        )

        out = model_runner_module.ModelRunnerFL._execute_mm_encoder(
            runner, self._sched(model_runner_module, [0, 1], 2)
        )

        # One micro-batch call per video, each seeing a single item.
        assert len(model.calls) == 2
        assert all(call["pixels"].shape[0] == 1 for call in model.calls)
        assert len(out) == 2
        assert "v1" in runner.encoder_cache
        assert "v2" in runner.encoder_cache

    def test_encoder_cudagraph_manager_short_circuits_model(self, model_runner_module):
        """A cudagraph manager handling the modality replaces the model call."""

        class _CgManager:
            def __init__(self):
                self.executed = []

            def supports_modality(self, modality):
                return modality == "image"

            def execute(self, mm_kwargs_batch):
                self.executed.append(mm_kwargs_batch)
                torch = import_module("torch")
                return [torch.full((2, 3), 7.0)]

        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        state.mm_features = [
            _mm_feature("image", "i1", 0, 2, data=_mm_item(torch, [1.0, 2.0]))
        ]
        batch.add_request(state)
        model = _FakeMmModel()
        manager = _CgManager()
        runner = self._runner(
            model_runner_module,
            batch,
            {"r1": state},
            model,
            encoder_cudagraph_manager=manager,
        )

        out = model_runner_module.ModelRunnerFL._execute_mm_encoder(
            runner, self._sched(model_runner_module, [0], 2)
        )

        assert len(manager.executed) == 1
        assert model.calls == []
        assert torch.equal(runner.encoder_cache["i1"], torch.full((2, 3), 7.0))
        assert len(out) == 1
        assert torch.equal(out[0], torch.full((2, 3), 7.0))

    def test_lora_tower_and_connector_mappings(self, model_runner_module):
        """Tower + connector LoRA mappings are built from the encoder batch."""
        LoRAMappingType = import_module("vllm.lora.layers").LoRAMappingType
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        state.mm_features = [
            _mm_feature("image", "i1", 0, 2, data=_mm_item(torch, [1.0, 2.0]))
        ]
        batch.add_request(state)
        # Row 0 of the batch carries lora id 7, mapped to one lora request.
        batch.request_lora_mapping[0] = 7
        batch.lora_id_to_lora_request[7] = "lora-req"

        model = _FakeMmModel(tower_tokens=3, connector_tokens=2)
        seen = []

        class _LoraManager:
            def supports_tower_connector_lora(self):
                return True

            def set_active_adapters(self, requests, mapping):
                seen.append((list(requests), mapping))

        runner = self._runner(
            model_runner_module,
            batch,
            {"r1": state},
            model,
            lora_config=SimpleNamespace(enabled=True),
            lora_manager=_LoraManager(),
        )

        model_runner_module.ModelRunnerFL._execute_mm_encoder(
            runner, self._sched(model_runner_module, [0], 2)
        )

        assert len(seen) == 2
        tower_reqs, tower = seen[0]
        conn_reqs, conn = seen[1]
        # The lora request of id 7 is active for both mappings.
        assert tower_reqs == ["lora-req"]
        assert conn_reqs == ["lora-req"]
        # Tower: 3 tokens of lora id 7, one prompt entry.
        assert list(tower.index_mapping) == [7, 7, 7]
        assert list(tower.prompt_mapping) == [7]
        assert tower.is_prefill is True
        assert tower.type is LoRAMappingType.TOWER
        # Connector: 2 tokens of lora id 7, same prompt entry.
        assert list(conn.index_mapping) == [7, 7]
        assert list(conn.prompt_mapping) == [7]
        assert conn.type is LoRAMappingType.CONNECTOR
        # Encoding still ran and cached the output.
        assert "i1" in runner.encoder_cache


class TestGatherMmEmbeddings:
    """ModelRunnerFL._gather_mm_embeddings against hand-computed oracles."""

    def _runner(self, model_runner_module, batch, requests, encoder_cache):
        runner = _base_runner(
            model_runner_module,
            batch,
            requests=dict(requests),
            uses_mrope=False,
            is_multimodal_pruning_enabled=False,
            encoder_cache=encoder_cache,
        )
        runner._get_encoder_output_from_cache = _bound(
            model_runner_module, runner, "_get_encoder_output_from_cache"
        )
        return runner

    def test_text_only_batch_returns_empty(self, model_runner_module):
        """A batch without mm features gathers nothing."""
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        batch.add_request(state)
        runner = self._runner(model_runner_module, batch, {"r1": state}, {})

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 2},
                total_num_scheduled_tokens=2,
            ),
        )
        assert embeds == []
        assert mask.tolist() == [False, False]

    def test_full_feature_window_gathered(self, model_runner_module):
        """A feature fully inside the scheduled window is gathered whole."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        state.mm_features = [_mm_feature("image", "i1", 0, 2)]
        batch.add_request(state)
        cached = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        runner = self._runner(model_runner_module, batch, {"r1": state}, {"i1": cached})

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 2},
                total_num_scheduled_tokens=2,
            ),
        )
        assert len(embeds) == 1
        assert torch.equal(embeds[0], cached)
        assert mask.tolist() == [True, True]

    def test_partial_window_slices_rows(self, model_runner_module):
        """A chunked prefill gathers only the rows in the computed window."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(
            model_runner_module, "r1", (5, 6, 7, 8), (9,), num_computed_tokens=1
        )
        state.mm_features = [_mm_feature("image", "i1", 0, 3)]
        batch.add_request(state)
        cached = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        runner = self._runner(model_runner_module, batch, {"r1": state}, {"i1": cached})

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 2},
                total_num_scheduled_tokens=2,
            ),
        )
        # Window [1, 3): rows 1..2 of the cached encoder output.
        assert len(embeds) == 1
        assert torch.equal(embeds[0], cached[1:3])
        assert mask.tolist() == [True, True]

    def test_fully_consumed_feature_not_refetched(self, model_runner_module):
        """A feature entirely before the window is not gathered again."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(
            model_runner_module, "r1", (5, 6, 7), (9,), num_computed_tokens=2
        )
        state.mm_features = [_mm_feature("image", "i1", 0, 2)]
        batch.add_request(state)
        runner = self._runner(
            model_runner_module, batch, {"r1": state}, {"i1": torch.ones(2, 3)}
        )

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 1},
                total_num_scheduled_tokens=1,
            ),
        )
        assert embeds == []
        assert mask.tolist() == [False]

    def test_feature_beyond_window_is_skipped(self, model_runner_module):
        """A feature starting past the scheduled window contributes nothing."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(
            model_runner_module,
            "r1",
            (5, 6, 7, 8, 9, 10, 11),
            (9,),
            num_computed_tokens=3,
        )
        state.mm_features = [_mm_feature("image", "i1", 5, 2)]
        batch.add_request(state)
        runner = self._runner(
            model_runner_module, batch, {"r1": state}, {"i1": torch.ones(2, 3)}
        )

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 1},
                total_num_scheduled_tokens=1,
            ),
        )
        assert embeds == []
        assert mask.tolist() == [False]

    def test_drafter_lookahead_missing_output_is_skipped(self, model_runner_module):
        """A reachable-only-via-look-ahead feature may miss the cache."""
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7, 8), (9,))
        state.mm_features = [_mm_feature("image", "later", 2, 1)]
        batch.add_request(state)
        runner = self._runner(model_runner_module, batch, {"r1": state}, {})

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 2},
                total_num_scheduled_tokens=2,
            ),
            # The drafter's +1 look-ahead: the window [1, 3) reaches the
            # feature even though its start sits at the un-shifted boundary.
            shift_computed_tokens=1,
        )
        # No error: the feature lies at/after the un-shifted boundary.
        assert embeds == []
        assert mask.tolist() == [False, False]

    def test_missing_cache_raises_for_reachable_feature(self, model_runner_module):
        """A reachable feature without a cached output raises RuntimeError."""
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        state.mm_features = [_mm_feature("image", "missing", 0, 2)]
        batch.add_request(state)
        runner = self._runner(model_runner_module, batch, {"r1": state}, {})

        with pytest.raises(RuntimeError, match="Encoder cache miss"):
            model_runner_module.ModelRunnerFL._gather_mm_embeddings(
                runner,
                _scheduler_output(
                    model_runner_module,
                    num_scheduled_tokens={"r1": 2},
                    total_num_scheduled_tokens=2,
                ),
            )

    def test_is_embed_mask_selects_embed_rows(self, model_runner_module):
        """is_embed masks gather only embedded rows and OR into the mask."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        # Positions 0 and 2 are embedded; position 1 keeps its token id.
        is_embed = torch.tensor([True, False, True])
        state.mm_features = [_mm_feature("image", "i1", 0, 3, is_embed=is_embed)]
        batch.add_request(state)
        # One encoder-output row per embedded position.
        cached = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        runner = self._runner(model_runner_module, batch, {"r1": state}, {"i1": cached})

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 3},
                total_num_scheduled_tokens=3,
            ),
        )
        # Both embedded rows are gathered, the middle token is not.
        assert len(embeds) == 1
        assert torch.equal(embeds[0], cached[0:2])
        assert mask.tolist() == [True, False, True]

    def test_two_requests_offsets_are_batch_flattened(self, model_runner_module):
        """Per-request offsets accumulate across the flattened token axis."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        s1 = _make_req_state(model_runner_module, "r1", (5, 6), (9,))
        s1.mm_features = [_mm_feature("image", "i1", 0, 2)]
        s2 = _make_req_state(model_runner_module, "r2", (8, 9), (10,))
        s2.mm_features = [_mm_feature("image", "i2", 0, 2)]
        batch.add_request(s1)
        batch.add_request(s2)
        runner = self._runner(
            model_runner_module,
            batch,
            {"r1": s1, "r2": s2},
            {"i1": torch.zeros(2, 3), "i2": torch.ones(2, 3)},
        )

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 2, "r2": 2},
                total_num_scheduled_tokens=4,
            ),
        )
        # Row assignment: r1's embeds then r2's, in flattened token order.
        assert len(embeds) == 2
        assert torch.equal(embeds[0], torch.zeros(2, 3))
        assert torch.equal(embeds[1], torch.ones(2, 3))
        assert mask.tolist() == [True, True, True, True]

    def test_shift_computed_tokens_moves_window(self, model_runner_module):
        """shift_computed_tokens slides the gather window forward."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        state.mm_features = [_mm_feature("image", "i1", 0, 3)]
        batch.add_request(state)
        cached = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        runner = self._runner(model_runner_module, batch, {"r1": state}, {"i1": cached})

        sched = _scheduler_output(
            model_runner_module,
            num_scheduled_tokens={"r1": 2},
            total_num_scheduled_tokens=2,
        )
        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner, sched, shift_computed_tokens=1
        )
        # Window [1, 3) after the shift.
        assert len(embeds) == 1
        assert torch.equal(embeds[0], cached[1:3])
        assert mask.tolist() == [True, True]
        """A window holding only non-embedded positions gathers nothing."""
        torch = import_module("torch")
        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(
            model_runner_module, "r1", (5, 6, 7), (9,), num_computed_tokens=1
        )
        # Only position 2 is embedded; the scheduled window [1, 2) covers
        # position 1, which keeps its token id.
        is_embed = torch.tensor([False, False, True])
        state.mm_features = [_mm_feature("image", "i1", 0, 3, is_embed=is_embed)]
        batch.add_request(state)
        cached = torch.arange(2, dtype=torch.float32).reshape(1, 2)
        runner = self._runner(model_runner_module, batch, {"r1": state}, {"i1": cached})

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 1},
                total_num_scheduled_tokens=1,
            ),
        )
        # No embeddings fall in [1, 2): nothing gathered, nothing marked.
        assert embeds == []
        assert mask.tolist() == [False]

    def test_mrope_pruning_recomputes_positions(self, model_runner_module):
        """Pruning + mrope routes gathered embeds through recompute."""
        torch = import_module("torch")

        class _MropeModel:
            def __init__(self):
                self.recomputed = []

            def recompute_mrope_positions(
                self,
                input_ids,
                multimodal_embeddings,
                mrope_positions,
                num_computed_tokens,
            ):
                self.recomputed.append(
                    (input_ids, list(multimodal_embeddings), num_computed_tokens)
                )
                # Shift the stored positions by one: an observable delta.
                new_positions = mrope_positions + 1
                return multimodal_embeddings, new_positions, 2

        batch = _make_input_batch(model_runner_module)
        state = _make_req_state(model_runner_module, "r1", (5, 6, 7), (9,))
        state.mm_features = [_mm_feature("image", "i1", 0, 3)]
        batch.add_request(state)
        # Prompt positions matrix: 3 rows x (prompt len + 1 dummy column).
        mrope = torch.arange(9, dtype=torch.int64).reshape(3, 3)
        state.mrope_positions = mrope.clone()
        state.mrope_position_delta = 0

        cached = torch.zeros(3, 4)
        model = _MropeModel()
        runner = _base_runner(
            model_runner_module,
            batch,
            requests={"r1": state},
            uses_mrope=True,
            is_multimodal_pruning_enabled=True,
            encoder_cache={"i1": cached},
        )
        runner._get_encoder_output_from_cache = _bound(
            model_runner_module, runner, "_get_encoder_output_from_cache"
        )
        runner._calc_mrope_positions = _bound(
            model_runner_module, runner, "_calc_mrope_positions"
        )
        runner.mrope_positions = model_runner_module.CpuGpuBuffer(
            (3, 65), dtype=torch.int64, device=torch.device(CPU_DEVICE)
        )
        runner.model = model

        embeds, mask = model_runner_module.ModelRunnerFL._gather_mm_embeddings(
            runner,
            _scheduler_output(
                model_runner_module,
                num_scheduled_tokens={"r1": 3},
                total_num_scheduled_tokens=3,
            ),
        )

        # The model saw the gathered embeds and the pre-shift positions.
        assert len(model.recomputed) == 1
        _, embeds_seen, _ = model.recomputed[0]
        assert torch.equal(embeds_seen[0], cached)
        # recompute returned the embeds unmodified; positions shifted by +1.
        assert torch.equal(state.mrope_positions, mrope + 1)
        assert state.mrope_position_delta == 2
        # The embeds carry the modality tag and mask marks the span.
        assert getattr(embeds[0], "modality", None) == "image"
        assert mask.tolist() == [True, True, True]
        # _calc_mrope_positions staged the recomputed (+1) prompt rows and the
        # full-position copy_to_gpu(total) carried all three mrope dims over.
        assert runner.mrope_positions.cpu[:, :3].tolist() == [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        assert runner.mrope_positions.gpu[:, :3].tolist() == [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
