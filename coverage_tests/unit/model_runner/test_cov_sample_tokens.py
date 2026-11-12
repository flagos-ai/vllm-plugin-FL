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

"""Coverage tests for ModelRunnerFL.sample_tokens and draft-id plumbing.

``sample_tokens`` consumes the ExecuteModelState written by
``execute_model``: it samples, bookkeeps, proposes drafts (before or after
bookkeeping depending on the speculative method), finalizes the KV
connector, and returns either the synchronous output, an async snapshot, or
a connector-only output. Collaborators are recorders on a namespace runner;
the GPUs' stream/event layer is not reached.
"""

from contextlib import ExitStack
from types import SimpleNamespace

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


class _Recorder:
    def __init__(self, result=None, result_factory=None):
        self.calls = []
        self.result = result
        self.result_factory = result_factory

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.result_factory is not None:
            return self.result_factory(*args, **kwargs)
        return self.result


class _FakeOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def with_kv_conn_output_only(cls, kv):
        obj = cls()
        obj.kv_connector_output = kv
        return obj


class _FakeAsyncOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sampled_token_ids_cpu = torch.tensor([[9]])
        self.async_copy_ready_event = "ready-event"


class _FakeProposer:
    """Passes the isinstance checks for the EAGLE/draft proposer union."""

    def __init__(self):
        self.calls = []

    def dummy_run(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def prepare_next_token_ids_padded(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return torch.tensor([3, 4], dtype=torch.int32), torch.tensor(
            [1, 1], dtype=torch.int32
        )


class _FakeNgramDrafter:
    def __init__(self):
        self.calls = []
        self.updates = []

    def update_token_ids_ngram(self, *args, **kwargs):
        self.updates.append((args, kwargs))
        return (
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([2, 2], dtype=torch.int32),
            torch.tensor([5, 6], dtype=torch.int32),
        )

    def propose(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return torch.tensor([[7]], dtype=torch.int32), torch.tensor(
            [1], dtype=torch.int32
        )


def _sampler_output():
    return SimpleNamespace(
        sampled_token_ids=torch.tensor([[5], [6]], dtype=torch.int32),
        logprobs_tensors=None,
    )


def _state(logits=None, common=None):
    return (
        SimpleNamespace(total_num_scheduled_tokens=2, num_spec_tokens_to_schedule=1),
        logits if logits is not None else torch.zeros(2, 4),
        SimpleNamespace(kind="spec"),
        common if common is not None else SimpleNamespace(max_seq_len=7),
        torch.zeros(2, 3),  # hidden_states
        torch.zeros(2, 3),  # sample_hidden_states
        None,  # aux
        SimpleNamespace(kind="ec"),  # ec connector output
        SimpleNamespace(kind="stats"),
        {"layer": torch.zeros(1)},  # slot mappings
    )


def _runner(state=None, **overrides):
    runner = SimpleNamespace(
        execute_model_state=state,
        kv_connector_output=SimpleNamespace(kind="kv"),
        use_async_scheduling=False,
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(kind="sampling"),
            prev_sampled_token_ids=torch.zeros(2, 1),
            req_ids=["a", "b"],
            vocab_size=11,
            set_async_sampled_token_ids=_Recorder(),
        ),
        _sample=_Recorder(result=_sampler_output()),
        _update_states_after_model_execute=_Recorder(),
        broadcast_pp_output=False,
        _pp_broadcast_prev_sampled_token_ids=_Recorder(),
        _pp_receive_prev_sampled_token_ids_to_input_batch=_Recorder(),
        _draft_token_ids=None,
        _draft_probs=None,
        _draft_prob_req_ids=None,
        _draft_token_req_ids=None,
        valid_sampled_token_count_gpu=None,
        speculative_config=None,
        num_spec_tokens=2,
        requests={},
        discard_request_mask=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.bool)),
        token_ids_gpu_tensor=torch.zeros(2, 4, dtype=torch.int32),
        num_tokens_no_spec_gpu=torch.zeros(2, dtype=torch.int32),
        _copy_valid_sampled_token_count=_Recorder(),
        _copy_draft_token_ids_to_cpu=_Recorder(),
        propose_draft_token_ids=_Recorder(result=[[1], [2]]),
        _bookkeeping_sync=_Recorder(result=_default_bookkeeping()),
        finalize_kv_connector=_Recorder(),
        eplb_step=_Recorder(),
        supports_mm_inputs=False,
        routed_experts_initialized=False,
        routed_experts_cpu=torch.zeros(2, 3, dtype=torch.int32),
        routed_experts_slot_mapping_cpu=torch.zeros(2, dtype=torch.int32),
        routed_experts_slot_mapping_device=torch.tensor([4, 5], dtype=torch.int32),
        routed_experts_capturer=SimpleNamespace(
            get_device_buffer=lambda: torch.tensor(
                [[1, 2, 3], [4, 5, 6]], dtype=torch.int32
            )
        ),
        _get_or_create_async_output_copy_stream=_Recorder(result="copy-stream"),
        device=torch.device("cpu"),
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def _default_bookkeeping():
    return (
        {"layer": 0},
        None,
        [[5], [6]],
        {"a": None},
        ["a", "b"],
        {"a": 0, "b": 1},
        [],
    )


def _spec(**flags):
    defaults = dict(
        use_eagle=lambda: False,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: False,
        use_ngram_gpu=lambda: False,
        disable_padded_drafter_batch=False,
        enforce_eager=False,
        method="ngram",
    )
    defaults.update(flags)
    return SimpleNamespace(**defaults)


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


def _proposer_patches(drafter):
    """Patch every proposer class to the fake for the isinstance checks."""
    return _PatchGroup(
        [
            *(
                _Patch(mr_module, name, _FakeProposer)
                for name in (
                    "EagleProposer",
                    "DFlashProposer",
                    "DraftModelProposer",
                    "ExtractHiddenStatesProposer",
                    "Gemma4Proposer",
                )
            ),
            _Patch(mr_module, "NgramProposerGPU", _FakeNgramDrafter),
        ]
    )


def test_sample_tokens_without_state_returns_connector_only_output():
    """No pending state returns the KV-connector-only wrapper and resets it."""
    runner = _runner(state=None)
    kv = runner.kv_connector_output
    with _Patch(mr_module, "ModelRunnerOutput", _FakeOutput):
        out = ModelRunnerFL.sample_tokens(runner, None)

    assert isinstance(out, _FakeOutput)
    assert out.kv_connector_output is kv
    assert runner.kv_connector_output is None


def test_sample_tokens_without_state_async_non_last_rank_receives_tokens():
    """Async scheduling receives the previous sampled ids on non-last ranks."""
    runner = _runner(state=None, use_async_scheduling=True)
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(
            mr_module,
            "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=False),
        ),
    ):
        out = ModelRunnerFL.sample_tokens(runner, None)

    assert isinstance(out, _FakeOutput)
    assert len(runner._pp_receive_prev_sampled_token_ids_to_input_batch.calls) == 1


def test_sample_tokens_sync_path_returns_output_and_advances_io_step():
    """The synchronous path samples, bookkeeps, and builds the output object."""
    state = _state()
    runner = _runner(state=state)
    kv = runner.kv_connector_output
    sampler_output = runner._sample.result
    io_calls = []
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "advance_io_step", lambda: io_calls.append(True)),
        _Patch(
            mr_module,
            "record_function_or_nullcontext",
            lambda name: _null_context(),
        ),
    ):
        out = ModelRunnerFL.sample_tokens(runner, None)

    assert runner.execute_model_state is None
    assert out.kwargs["req_ids"] == ["a", "b"]
    assert out.kwargs["sampled_token_ids"] == [[5], [6]]
    assert out.kwargs["kv_connector_output"] is kv
    assert out.kwargs["ec_connector_output"] is None
    assert out.kwargs["num_nans_in_logits"] == {"layer": 0}
    assert out.kwargs["routed_experts"] is None
    assert runner.input_batch.prev_sampled_token_ids is None
    assert runner._draft_token_ids is None and runner._draft_probs is None
    assert io_calls == [True]
    # Sample used the unpacked logits and spec metadata.
    sample_args = runner._sample.calls[0][0]
    assert sample_args[0] is state[1]
    assert sample_args[1] is state[2]
    assert runner._update_states_after_model_execute.calls[0][0][0] is (
        sampler_output.sampled_token_ids
    )
    book_args = runner._bookkeeping_sync.calls[0][0]
    assert book_args[1] is sampler_output
    assert book_args[4] == 2


def test_sample_tokens_applies_grammar_bitmask_before_sampling():
    """A grammar output patches the logits through apply_grammar_bitmask."""
    state = _state()
    runner = _runner(state=state)
    grammar = SimpleNamespace(kind="grammar")
    bitmask = _Recorder()
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "advance_io_step", lambda: None),
        _Patch(mr_module, "apply_grammar_bitmask", bitmask),
        _Patch(mr_module, "record_function_or_nullcontext", lambda n: _null_context()),
    ):
        ModelRunnerFL.sample_tokens(runner, grammar)

    call_args = bitmask.calls[0][0]
    assert call_args[0] is state[0]
    assert call_args[1] is grammar
    assert call_args[2] is runner.input_batch
    assert call_args[3] is state[1]


def test_sample_tokens_sync_routed_experts_are_wrapped_as_lists():
    """Sync scheduling exposes routed experts as numpy slices of the CPU buffers."""
    state = _state()
    runner = _runner(state=state, routed_experts_initialized=True)
    wrapped = _Recorder(result="routed-lists")
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "RoutedExpertsLists", wrapped),
        _Patch(mr_module, "advance_io_step", lambda: None),
        _Patch(mr_module, "record_function_or_nullcontext", lambda n: _null_context()),
    ):
        out = ModelRunnerFL.sample_tokens(runner, None)

    assert out.routed_experts == "routed-lists"
    routing = wrapped.calls[0][1]["routing_data"]
    slot = wrapped.calls[0][1]["slot_mapping"]
    assert routing.shape == (2, 3)
    assert slot.shape == (2,)


def test_sample_tokens_async_path_snapshots_routed_experts():
    """Async scheduling clones the device snapshot and registers the CPU copy."""
    state = _state()
    runner = _runner(state=state, use_async_scheduling=True)
    runner.routed_experts_initialized = True
    source = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    runner.routed_experts_capturer = SimpleNamespace(get_device_buffer=lambda: source)
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "AsyncGPUModelRunnerOutput", _FakeAsyncOutput),
        _Patch(mr_module, "advance_io_step", lambda: None),
        _Patch(
            mr_module,
            "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=True, world_size=1),
        ),
        _Patch(mr_module, "record_function_or_nullcontext", lambda n: _null_context()),
    ):
        out = ModelRunnerFL.sample_tokens(runner, None)

    assert isinstance(out, _FakeAsyncOutput)
    snapshot = out.kwargs["routed_experts"]
    assert snapshot.routing_data.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert snapshot.slot_mapping.tolist() == [4, 5]
    # The snapshot must not alias the shared capturer buffer.
    assert snapshot.routing_data.data_ptr() != source.data_ptr()
    set_args = runner.input_batch.set_async_sampled_token_ids.calls[0][0]
    assert set_args[0] is out.sampled_token_ids_cpu
    assert set_args[1] == "ready-event"
    assert out.kwargs["async_output_copy_stream"] == "copy-stream"
    assert out.kwargs["invalid_req_indices"] == []


def test_sample_tokens_async_last_rank_broadcasts_sampled_ids():
    """Async PP on the last rank broadcasts sampled ids when not broadcast-mode."""
    state = _state()
    runner = _runner(state=state, use_async_scheduling=True)
    broadcast = _Recorder()
    runner._pp_broadcast_prev_sampled_token_ids = broadcast
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "AsyncGPUModelRunnerOutput", _FakeAsyncOutput),
        _Patch(mr_module, "advance_io_step", lambda: None),
        _Patch(
            mr_module,
            "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=True, world_size=2),
        ),
        _Patch(mr_module, "record_function_or_nullcontext", lambda n: _null_context()),
    ):
        ModelRunnerFL.sample_tokens(runner, None)

    assert broadcast.calls[0][0][0] is runner._sample.result.sampled_token_ids


def test_sample_tokens_eagle_waits_for_counts_when_batch_too_long():
    """EAGLE with a non-fitting batch stages next-token counts and zeros drafts."""
    state = _state()
    drafter = _FakeProposer()
    runner = _runner(
        state=state,
        speculative_config=_spec(use_eagle=lambda: True),
        drafter=drafter,
        valid_sampled_token_count_event=object(),
    )
    runner._input_fits_in_drafter = _Recorder(result=False)
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "advance_io_step", lambda: None),
        _Patch(mr_module, "record_function_or_nullcontext", lambda n: _null_context()),
        _proposer_patches(drafter),
    ):
        ModelRunnerFL.sample_tokens(runner, None)

    prep = drafter.calls[0][0]
    assert prep[0] is runner._sample.result.sampled_token_ids
    assert prep[1] is runner.requests
    assert prep[2] is runner.input_batch
    assert prep[3] is runner.discard_request_mask.gpu
    copy_args = runner._copy_valid_sampled_token_count.calls[0][0]
    assert copy_args[0].tolist() == [3, 4]
    assert copy_args[1].tolist() == [1, 1]
    assert runner._draft_token_ids.shape == (2, 2)
    assert runner._draft_token_ids.eq(0).all()
    assert runner._draft_prob_req_ids is None
    assert runner._copy_draft_token_ids_to_cpu.calls[0][1] == {"zeros_only": True}
    assert len(runner.finalize_kv_connector.calls) == 1


def test_sample_tokens_ngram_gpu_updates_counts_when_batch_too_long():
    """The GPU ngram proposer updates token ids and stages valid counts."""
    state = _state()
    drafter = _FakeNgramDrafter()
    runner = _runner(
        state=state,
        speculative_config=_spec(
            method="ngram_gpu", use_ngram_gpu=lambda: True
        ),
        drafter=drafter,
        valid_sampled_token_count_event=object(),
    )
    runner._input_fits_in_drafter = _Recorder(result=False)
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "advance_io_step", lambda: None),
        _Patch(mr_module, "record_function_or_nullcontext", lambda n: _null_context()),
        _proposer_patches(drafter),
    ):
        ModelRunnerFL.sample_tokens(runner, None)

    upd = drafter.updates[0][0]
    assert upd[0] is runner._sample.result.sampled_token_ids
    assert upd[1] is runner.input_batch
    assert upd[2] is runner.token_ids_gpu_tensor
    assert upd[3] is runner.num_tokens_no_spec_gpu
    assert upd[4] is runner.discard_request_mask.gpu
    copy_args = runner._copy_valid_sampled_token_count.calls[0][0]
    assert copy_args[0].tolist() == [1, 2]
    # Drafts are zeroed for the next scheduler step.
    assert runner._copy_draft_token_ids_to_cpu.calls[0][1] == {"zeros_only": True}


def test_sample_tokens_defers_draft_proposal_after_bookkeeping():
    """CPU-side proposers run with the booked valid tokens, after bookkeeping."""
    order = []
    state = _state()
    runner = _runner(state=state, speculative_config=_spec())
    runner._input_fits_in_drafter = _Recorder(result=True)
    runner._bookkeeping_sync = _Recorder(
        result_factory=lambda *a, **k: (order.append("bookkeep"), _default_bookkeeping())[1]
    )
    runner.propose_draft_token_ids = _Recorder(
        result_factory=lambda *a, **k: order.append("draft") or [[8], [9]]
    )
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "advance_io_step", lambda: None),
        _Patch(mr_module, "record_function_or_nullcontext", lambda n: _null_context()),
        _proposer_patches(None),
    ):
        ModelRunnerFL.sample_tokens(runner, None)

    assert order == ["bookkeep", "draft"]
    propose_args = runner.propose_draft_token_ids.calls[0][0]
    assert propose_args[1] == [[5], [6]]
    assert runner._draft_token_ids == [[8], [9]]
    assert len(runner._copy_draft_token_ids_to_cpu.calls) == 1
    assert runner._copy_draft_token_ids_to_cpu.calls[0][1] == {}


def test_sample_tokens_gpu_proposer_runs_before_bookkeeping():
    """GPU-token proposers run right after sampling, before bookkeeping."""
    order = []
    state = _state()
    drafter = _FakeProposer()
    runner = _runner(
        state=state,
        speculative_config=_spec(use_eagle=lambda: True),
        drafter=drafter,
    )
    runner._input_fits_in_drafter = _Recorder(result=True)
    runner._bookkeeping_sync = _Recorder(
        result_factory=lambda *a, **k: (order.append("bookkeep"), _default_bookkeeping())[1]
    )
    runner.propose_draft_token_ids = _Recorder(
        result_factory=lambda *a, **k: order.append("draft") or [[8], [9]]
    )
    with (
        _Patch(mr_module, "ModelRunnerOutput", _FakeOutput),
        _Patch(mr_module, "advance_io_step", lambda: None),
        _Patch(mr_module, "record_function_or_nullcontext", lambda n: _null_context()),
        _proposer_patches(drafter),
    ):
        ModelRunnerFL.sample_tokens(runner, None)

    assert order == ["draft", "bookkeep"]
    propose_args = runner.propose_draft_token_ids.calls[0][0]
    assert propose_args[1] is runner._sample.result.sampled_token_ids


def test_get_spec_decode_draft_probs_returns_none_without_cache():
    """No cached probabilities yields None regardless of the metadata."""
    runner = SimpleNamespace(_draft_probs=None, _draft_prob_req_ids=None)
    assert ModelRunnerFL._get_spec_decode_draft_probs(runner, object()) is None


def test_get_spec_decode_draft_probs_gathers_per_request_rows():
    """Draft probs are concatenated in batch order, trimmed to draft counts."""
    probs = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    runner = SimpleNamespace(
        _draft_probs=probs,
        _draft_prob_req_ids=["a", "b", "c"],
        input_batch=SimpleNamespace(req_ids=["b", "a", "c"]),
    )
    metadata = SimpleNamespace(num_draft_tokens=[2, 1, 0])
    result = ModelRunnerFL._get_spec_decode_draft_probs(runner, metadata)

    # "b" is row 1 (2 columns), "a" is row 0 (1 column), "c" has no drafts.
    expected = torch.cat([probs[1, :2], probs[0, :1]], dim=0)
    torch.testing.assert_close(result, expected)
    assert result.is_contiguous()


def test_get_spec_decode_draft_probs_missing_request_falls_back():
    """A request absent from the cache disables the cached-prob path."""
    runner = SimpleNamespace(
        _draft_probs=torch.zeros(2, 3),
        _draft_prob_req_ids=["a"],
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
    )
    metadata = SimpleNamespace(num_draft_tokens=[1, 1])
    assert ModelRunnerFL._get_spec_decode_draft_probs(runner, metadata) is None


def test_get_spec_decode_draft_probs_no_drafts_returns_none():
    """Metadata with no draft tokens produces no concatenated tensor."""
    runner = SimpleNamespace(
        _draft_probs=torch.zeros(1, 3),
        _draft_prob_req_ids=["a"],
        input_batch=SimpleNamespace(req_ids=["a"]),
    )
    metadata = SimpleNamespace(num_draft_tokens=[0])
    assert ModelRunnerFL._get_spec_decode_draft_probs(runner, metadata) is None


def test_input_fits_in_drafter_requires_metadata():
    """Without common attention metadata the drafter cannot run."""
    runner = SimpleNamespace(speculative_config=object())
    assert ModelRunnerFL._input_fits_in_drafter(runner, None) is False


def test_input_fits_in_drafter_counts_dflash_bonus_token():
    """DFlash reserves one extra query token beyond the spec tokens."""
    runner = SimpleNamespace(
        num_spec_tokens=3,
        effective_drafter_max_model_len=10,
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
    )
    metadata = SimpleNamespace(max_seq_len=6)
    # 6 + 3 + 1 = 10 fits exactly.
    assert ModelRunnerFL._input_fits_in_drafter(runner, metadata) is True
    metadata.max_seq_len = 7
    assert ModelRunnerFL._input_fits_in_drafter(runner, metadata) is False

    runner.speculative_config = SimpleNamespace(use_dflash=lambda: False)
    assert ModelRunnerFL._input_fits_in_drafter(runner, metadata) is True
    metadata.max_seq_len = 8
    assert ModelRunnerFL._input_fits_in_drafter(runner, metadata) is False


def test_is_all_reqs_chunked_prefill_checks_the_live_mask():
    """Only a fully set discard mask reports an all-chunked batch."""
    mask = torch.zeros(4, dtype=torch.bool)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=2),
        discard_request_mask=SimpleNamespace(np=mask),
    )
    assert ModelRunnerFL._is_all_reqs_chunked_prefill(runner) is False
    mask[:2] = True
    assert ModelRunnerFL._is_all_reqs_chunked_prefill(runner) is True


def test_propose_draft_token_ids_ngram_gpu_copies_valid_counts():
    """The GPU ngram proposal caches draft counts and copies them asynchronously."""
    drafter = _FakeNgramDrafter()
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="ngram_gpu",
            use_ngram_gpu=lambda: True,
        ),
        drafter=drafter,
        _draft_probs=object(),
        _draft_prob_req_ids=["stale"],
        token_ids_gpu_tensor=torch.zeros(2, 3, dtype=torch.int32),
        num_tokens_no_spec_gpu=torch.tensor([1, 2], dtype=torch.int32),
        discard_request_mask=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.bool)),
        input_batch=SimpleNamespace(num_reqs=2),
        _copy_valid_sampled_token_count=_Recorder(),
        _num_valid_draft_tokens_cpu=torch.zeros(2, dtype=torch.int32),
        _num_valid_draft_tokens_copy_stream="stream",
        _num_valid_draft_tokens_event="event",
    )
    copy_valid = _Recorder()
    sched = SimpleNamespace(
        total_num_scheduled_tokens=2, num_spec_tokens_to_schedule=1
    )
    sampled = torch.tensor([[3], [4]], dtype=torch.int32)
    with (
        _Patch(mr_module, "NgramProposerGPU", _FakeNgramDrafter),
        _Patch(mr_module, "copy_num_valid_draft_tokens", copy_valid),
    ):
        draft = ModelRunnerFL.propose_draft_token_ids(
            runner,
            sched,
            sampled,
            SimpleNamespace(kind="sampling"),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            None,
            None,
            SimpleNamespace(kind="common"),
            {"layer": torch.zeros(1)},
        )

    assert draft.tolist() == [[7]]
    assert runner._draft_probs is None and runner._draft_prob_req_ids is None
    update_args = drafter.updates[0][0]
    assert update_args[0] is sampled
    propose_args = drafter.calls[0][0]
    assert propose_args[0] == 1
    torch.testing.assert_close(
        propose_args[1], runner.num_tokens_no_spec_gpu[:2]
    )
    torch.testing.assert_close(
        propose_args[2], runner.token_ids_gpu_tensor[:2]
    )
    assert runner._num_valid_draft_tokens.tolist() == [1]
    copied = copy_valid.calls[0][0]
    assert copied[0] is runner._num_valid_draft_tokens_cpu
    assert copied[1] == "stream"
    assert copied[2] == "event"
    assert copied[3] is runner._num_valid_draft_tokens
    assert copied[4] == 2


def _null_context():
    from contextlib import nullcontext

    return nullcontext()
