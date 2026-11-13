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

"""Pure-CPU contracts for the PLE n-gram context and token history.

The runners keep one left-context window per request:
``ngram_context[r]`` is the last ``ngram_size - 1`` committed tokens before the
scheduled chunk, with EOS filling unavailable positions. ``PLETokenHistory``
retains consumed tokens independently of the movable CPU ``InputBatch`` rows.
Every expected window below is derived from those rules.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import (
    legacy_model_state as lms,
    model_state as ms,
)
from vllm_fl.models.qwen3_8_flash_next.gpu.ple_token_history import PLETokenHistory

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# PLETokenHistory
# ---------------------------------------------------------------------------


def _history(max_num_reqs=3, max_model_len=8, max_num_tokens=6, context_len=2):
    return PLETokenHistory(
        max_num_reqs,
        max_model_len,
        max_num_tokens,
        context_len,
        0,
        CPU,
        False,
    )


def _seed_cpu(num_reqs, max_len, rows):
    cpu = torch.full((num_reqs, max_len), -1, dtype=torch.int32)
    for row, values in rows.items():
        cpu[row, : len(values)] = torch.tensor(values, dtype=torch.int32)
    return cpu


class TestTokenHistoryAllocation:
    def test_sentinel_eos_and_free_slots_are_initialized(self):
        """Verifies sentinel eos and free slots are initialized."""
        history = _history()
        assert history.sentinel == 3 * 8
        assert history.tokens[history.sentinel].item() == 0
        assert history._free_slots == [2, 1, 0]

    def test_forget_unknown_request_is_a_no_op(self):
        """Verifies forget unknown request is a no op."""
        history = _history()
        history.forget(["missing"])
        assert history._slots == {}
        assert history._free_slots == [2, 1, 0]


class TestTokenHistoryPrepare:
    def test_reads_left_context_and_stores_current_token(self):
        """Verifies reads left context and stores current token."""
        history = _history()
        cpu = _seed_cpu(3, 8, {0: [5, 6]})
        context = torch.empty((1, 2), dtype=torch.int32)
        history.prepare(
            req_ids=["a"],
            num_computed_tokens=np.array([2], dtype=np.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            token_ids_cpu=cpu,
            is_token_ids=None,
            req_indices=np.array([0]),
            query_positions=np.array([0]),
            input_ids=torch.tensor([7], dtype=torch.int32),
            context=context,
        )
        assert context.tolist() == [[5, 6]]
        assert history._slots == {"a": 0}
        assert history._valid_lengths == {"a": 3}
        assert history.tokens[0:3].tolist() == [5, 6, 7]

    def test_second_step_reads_the_previously_written_tokens(self):
        """Verifies second step reads the previously written tokens."""
        history = _history()
        history.prepare(
            req_ids=["a"],
            num_computed_tokens=np.array([2], dtype=np.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            token_ids_cpu=_seed_cpu(3, 8, {0: [5, 6]}),
            is_token_ids=None,
            req_indices=np.array([0]),
            query_positions=np.array([0]),
            input_ids=torch.tensor([7], dtype=torch.int32),
            context=torch.empty((1, 2), dtype=torch.int32),
        )
        context = torch.empty((1, 2), dtype=torch.int32)
        history.prepare(
            req_ids=["a"],
            num_computed_tokens=np.array([3], dtype=np.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            token_ids_cpu=torch.full((3, 8), -1, dtype=torch.int32),
            is_token_ids=None,
            req_indices=np.array([0]),
            query_positions=np.array([0]),
            input_ids=torch.tensor([8], dtype=torch.int32),
            context=context,
        )
        assert context.tolist() == [[6, 7]]
        assert history.tokens[3].item() == 8

    def test_token_mask_replaces_non_token_placeholders_with_eos(self):
        """Verifies token mask replaces non token placeholders with eos."""
        history = _history(max_num_reqs=2, max_num_tokens=4)
        cpu = _seed_cpu(2, 8, {0: [5, 6]})
        context = torch.empty((1, 2), dtype=torch.int32)
        history.prepare(
            req_ids=["b"],
            num_computed_tokens=np.array([2], dtype=np.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            token_ids_cpu=cpu,
            is_token_ids=np.array([[True, False]]),
            req_indices=np.array([0]),
            query_positions=np.array([0]),
            input_ids=torch.tensor([9], dtype=torch.int32),
            context=context,
        )
        assert context.tolist() == [[5, 0]]

    def test_negative_async_placeholder_is_rejected(self):
        """Verifies negative async placeholder is rejected."""
        history = _history()
        with pytest.raises(RuntimeError, match="async placeholders"):
            history.prepare(
                req_ids=["a"],
                num_computed_tokens=np.array([2], dtype=np.int32),
                num_scheduled_tokens=np.array([1], dtype=np.int32),
                token_ids_cpu=_seed_cpu(3, 8, {0: [-1, 6]}),
                is_token_ids=None,
                req_indices=np.array([0]),
                query_positions=np.array([0]),
                input_ids=torch.tensor([7], dtype=torch.int32),
                context=torch.empty((1, 2), dtype=torch.int32),
            )

    @pytest.mark.parametrize("start", [-1, 8])
    def test_positions_outside_model_capacity_are_rejected(self, start):
        """Verifies positions outside model capacity are rejected."""
        history = _history()
        with pytest.raises(ValueError, match="exceed model capacity"):
            history.prepare(
                req_ids=["a"],
                num_computed_tokens=np.array([start], dtype=np.int32),
                num_scheduled_tokens=np.array([1], dtype=np.int32),
                token_ids_cpu=_seed_cpu(3, 8, {}),
                is_token_ids=None,
                req_indices=np.array([0]),
                query_positions=np.array([0]),
                input_ids=torch.tensor([7], dtype=torch.int32),
                context=torch.empty((1, 2), dtype=torch.int32),
            )

    def test_empty_batch_fills_all_context_rows_with_eos(self):
        """Verifies empty batch fills all context rows with eos."""
        history = _history()
        context = torch.empty((2, 2), dtype=torch.int32)
        history.prepare(
            req_ids=[],
            num_computed_tokens=np.array([], dtype=np.int32),
            num_scheduled_tokens=np.array([], dtype=np.int32),
            token_ids_cpu=_seed_cpu(3, 8, {}),
            is_token_ids=None,
            req_indices=np.array([], dtype=np.int64),
            query_positions=np.array([], dtype=np.int64),
            input_ids=torch.empty(0, dtype=torch.int32),
            context=context,
        )
        assert context.tolist() == [[0, 0], [0, 0]]

    def test_requests_missing_from_the_batch_release_their_slot(self):
        """Verifies requests missing from the batch release their slot."""
        history = _history()
        history.prepare(
            req_ids=["a", "b"],
            num_computed_tokens=np.array([0, 0], dtype=np.int32),
            num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
            token_ids_cpu=_seed_cpu(3, 8, {}),
            is_token_ids=None,
            req_indices=np.array([0, 1]),
            query_positions=np.array([0, 0]),
            input_ids=torch.tensor([1, 2], dtype=torch.int32),
            context=torch.empty((2, 2), dtype=torch.int32),
        )
        b_slot = history._slots["b"]
        history.prepare(
            req_ids=["a"],
            num_computed_tokens=np.array([1], dtype=np.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            token_ids_cpu=_seed_cpu(3, 8, {}),
            is_token_ids=None,
            req_indices=np.array([0]),
            query_positions=np.array([0]),
            input_ids=torch.tensor([3], dtype=torch.int32),
            context=torch.empty((1, 2), dtype=torch.int32),
        )
        assert "b" not in history._slots
        assert "b" not in history._valid_lengths
        assert b_slot in history._free_slots


# ---------------------------------------------------------------------------
# Qwen3_8FlashNextModelState
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_state(monkeypatch):
    def fake_init(self, vllm_config, model, encoder_cache, device):
        self.model_config = vllm_config.model_config
        self.max_num_reqs = vllm_config.scheduler_config.max_num_reqs
        self.device = device

    monkeypatch.setattr(ms.MambaHybridModelState, "__init__", fake_init)
    return ms


def _vllm_config(
    *, ple_layer_ids=(1,), ngram_size=3, eos_token_id=2, pp=1, max_num_reqs=4
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                ple_layer_ids=list(ple_layer_ids),
                ngram_size=ngram_size,
                eos_token_id=eos_token_id,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_reqs=max_num_reqs),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
    )


class TestModelStateConstruction:
    def test_non_ngram_model_skips_context_buffers(self, base_state):
        """Verifies non ngram model skips context buffers."""
        state = base_state.Qwen3_8FlashNextModelState(
            _vllm_config(ple_layer_ids=()), None, None, CPU
        )
        assert state.uses_ngram_embedding is False
        assert state.ngram_context_len == 0
        assert state.ngram_eos_token_id == 0
        assert not hasattr(state, "ngram_context")

    def test_ngram_model_allocates_context_and_offsets(self, base_state):
        """Verifies ngram model allocates context and offsets."""
        state = base_state.Qwen3_8FlashNextModelState(_vllm_config(), None, None, CPU)
        assert state.uses_ngram_embedding is True
        assert state.ngram_context_len == 2
        assert state.ngram_context.shape == (4, 2)
        assert (state.ngram_context == 2).all()
        assert state.ngram_context_offsets.tolist() == [-2, -1]
        assert state.ple_query_start_loc.shape == (5,)

    def test_pipeline_parallel_is_rejected(self, base_state):
        """Verifies pipeline parallel is rejected."""
        with pytest.raises(RuntimeError, match="pipeline_parallel_size=1"):
            base_state.Qwen3_8FlashNextModelState(_vllm_config(pp=2), None, None, CPU)

    def test_ngram_size_one_has_no_left_context(self, base_state):
        """Verifies ngram size one has no left context."""
        with pytest.raises(ValueError, match="context length >= 1"):
            base_state.Qwen3_8FlashNextModelState(
                _vllm_config(ngram_size=1), None, None, CPU
            )


def _req_states(token_store, computed):
    return SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor(computed)),
        all_token_ids=SimpleNamespace(gpu=token_store),
    )


class TestPrepareNgramContext:
    def test_gathers_left_context_and_eos_fills_invalid_positions(self, base_state):
        """Verifies gathers left context and eos fills invalid positions."""
        state = base_state.Qwen3_8FlashNextModelState(_vllm_config(), None, None, CPU)
        token_store = torch.tensor(
            [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int32
        )
        input_batch = SimpleNamespace(
            num_reqs=2,
            num_reqs_after_padding=3,
            idx_mapping=torch.tensor([1, 0, 9]),
        )
        req_states = _req_states(token_store, [2, 1])
        context = state._prepare_ngram_context(input_batch, req_states)
        # Request 1 has computed 1 token: [-1, 0] -> EOS, token 20.
        # Request 0 has computed 2 tokens: [0, 1] -> tokens 10, 11.
        # Row 2 is graph padding and stays EOS.
        assert context.tolist() == [[2, 20], [10, 11], [2, 2]]

    def test_empty_batch_returns_eos_filled_padding(self, base_state):
        """Verifies empty batch returns eos filled padding."""
        state = base_state.Qwen3_8FlashNextModelState(_vllm_config(), None, None, CPU)
        input_batch = SimpleNamespace(
            num_reqs=0,
            num_reqs_after_padding=2,
            idx_mapping=torch.tensor([0, 0]),
        )
        req_states = _req_states(torch.zeros(1, 4, dtype=torch.int32), [0])
        context = state._prepare_ngram_context(input_batch, req_states)
        assert context.tolist() == [[2, 2], [2, 2]]


class TestModelStateInputs:
    def _state(self, base_state, **kwargs):
        return base_state.Qwen3_8FlashNextModelState(
            _vllm_config(**kwargs), None, None, CPU
        )

    def test_prepare_inputs_adds_padded_query_offsets_and_context(
        self, base_state, monkeypatch
    ):
        """Verifies prepare inputs adds padded query offsets and context."""
        monkeypatch.setattr(
            ms.MambaHybridModelState,
            "prepare_inputs",
            lambda self, input_batch, req_states: {"base": 1},
        )
        state = self._state(base_state)
        input_batch = SimpleNamespace(
            num_reqs=2,
            num_reqs_after_padding=3,
            idx_mapping=torch.tensor([0, 1, 9]),
            query_start_loc=torch.tensor([0, 2, 5, 5, 5], dtype=torch.int32),
        )
        req_states = _req_states(torch.zeros(2, 5, dtype=torch.int32), [0, 0])
        inputs = state.prepare_inputs(input_batch, req_states)
        assert inputs["base"] == 1
        assert inputs["query_start_loc"].tolist() == [0, 2, 5, 5]
        assert inputs["ngram_context"].tolist() == [[2, 2], [2, 2], [2, 2]]

    def test_non_ngram_prepare_inputs_returns_base_mapping(
        self, base_state, monkeypatch
    ):
        """Verifies non ngram prepare inputs returns base mapping."""
        monkeypatch.setattr(
            ms.MambaHybridModelState,
            "prepare_inputs",
            lambda self, input_batch, req_states: {"base": 2},
        )
        state = self._state(base_state, ple_layer_ids=())
        assert state.prepare_inputs(SimpleNamespace(), SimpleNamespace()) == {"base": 2}

    @pytest.mark.parametrize(
        ("num_reqs", "num_tokens", "expected"),
        [(2, 5, [0, 2, 5]), (2, 4, [0, 2, 4]), (1, 4, [0, 4])],
    )
    def test_dummy_inputs_split_tokens_and_fill_eos(
        self, base_state, monkeypatch, num_reqs, num_tokens, expected
    ):
        """Verifies dummy inputs split tokens and fill eos."""
        monkeypatch.setattr(
            ms.MambaHybridModelState,
            "prepare_dummy_inputs",
            lambda self, num_reqs, num_tokens: {"base": 3},
        )
        state = self._state(base_state)
        inputs = state.prepare_dummy_inputs(num_reqs, num_tokens)
        assert inputs["base"] == 3
        assert inputs["query_start_loc"].tolist() == expected
        assert inputs["ngram_context"].shape[0] == num_reqs
        assert (inputs["ngram_context"] == 2).all()


# ---------------------------------------------------------------------------
# Qwen3_8FlashNextLegacyModelState
# ---------------------------------------------------------------------------


def _make_buffer(device):
    def buffer(*shape, dtype):
        cpu = torch.zeros(shape, dtype=dtype)
        gpu = torch.zeros(shape, dtype=dtype, device=device)

        def copy_to_gpu(n=None):
            if n is None:
                gpu.copy_(cpu)
            else:
                gpu[:n].copy_(cpu[:n])

        return SimpleNamespace(gpu=gpu, np=cpu.numpy(), copy_to_gpu=copy_to_gpu)

    return buffer


def _runner(device, *, ple_layer_ids=(1,), ngram_size=3, is_encoder_decoder=False):
    batch = SimpleNamespace(
        req_ids=["a"],
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        token_ids_cpu_tensor=torch.full((4, 32), -1, dtype=torch.int32),
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                ple_layer_ids=list(ple_layer_ids),
                ngram_size=ngram_size,
                eos_token_id=0,
            ),
            is_encoder_decoder=is_encoder_decoder,
        ),
        max_num_reqs=4,
        max_model_len=32,
        max_num_tokens=32,
        device=device,
        pin_memory=False,
        _make_buffer=_make_buffer(device),
        input_batch=batch,
        enable_prompt_embeds=False,
        query_start_loc=_make_buffer(device)(5, dtype=torch.int32),
        num_scheduled_tokens=SimpleNamespace(np=np.array([2], dtype=np.int32)),
        req_indices=SimpleNamespace(np=np.array([0, 0])),
        query_pos=SimpleNamespace(np=np.array([0, 1])),
        input_ids=SimpleNamespace(
            gpu=torch.tensor([7, 8], dtype=torch.int32, device=device)
        ),
    )


def _first_rank(monkeypatch, ranks=(0,)):
    monkeypatch.setattr(
        lms,
        "get_pp_group",
        lambda: SimpleNamespace(ranks=list(ranks), is_first_rank=ranks[0] == 0),
    )


class TestLegacyModelState:
    def test_non_ngram_model_returns_empty_kwargs(self, monkeypatch):
        """Verifies non ngram model returns empty kwargs."""
        _first_rank(monkeypatch)
        runner = _runner(CPU, ple_layer_ids=())
        state = lms.Qwen3_8FlashNextLegacyModelState(runner)
        assert state.uses_ngram_embedding is False
        assert state.prepare_inputs(1, 4, None) == {}
        assert state.prepare_dummy_inputs(1, 4, np.array([2])) == {}
        state.remove_requests(["a"])  # no history to forget

    def test_pipeline_parallel_is_rejected_for_ngram(self, monkeypatch):
        """Verifies pipeline parallel is rejected for ngram."""
        _first_rank(monkeypatch, ranks=(0, 1))
        with pytest.raises(RuntimeError, match="pipeline_parallel_size=1"):
            lms.Qwen3_8FlashNextLegacyModelState(_runner(CPU))

    def test_ngram_size_one_is_rejected(self, monkeypatch):
        """Verifies ngram size one is rejected."""
        _first_rank(monkeypatch)
        with pytest.raises(ValueError, match="context length >= 1"):
            lms.Qwen3_8FlashNextLegacyModelState(_runner(CPU, ngram_size=1))

    def test_missing_input_ids_on_first_rank_is_rejected(self, monkeypatch):
        """Verifies missing input ids on first rank is rejected."""
        _first_rank(monkeypatch)
        state = lms.Qwen3_8FlashNextLegacyModelState(_runner(CPU))
        with pytest.raises(RuntimeError, match="token ids"):
            state.prepare_inputs(1, 4, None)

    def test_encoder_decoder_skips_ngram_kwargs(self, monkeypatch):
        """Verifies encoder decoder skips ngram kwargs."""
        _first_rank(monkeypatch)
        state = lms.Qwen3_8FlashNextLegacyModelState(
            _runner(CPU, is_encoder_decoder=True)
        )
        assert state.prepare_inputs(1, 4, None) == {}

    def test_dummy_inputs_fill_query_offsets_and_eos_context(self, monkeypatch):
        """Verifies dummy inputs fill query offsets and eos context."""
        _first_rank(monkeypatch)
        runner = _runner(CPU)
        state = lms.Qwen3_8FlashNextLegacyModelState(runner)
        inputs = state.prepare_dummy_inputs(1, 4, np.array([2], dtype=np.int32))
        assert inputs["query_start_loc"].cpu().tolist() == [0, 2, 2, 2, 2]
        assert torch.count_nonzero(inputs["ngram_context"]) == 0
        assert state.ple_token_history._slots == {}

    def test_ngram_kwargs_require_an_offset_source(self, monkeypatch):
        """Verifies ngram kwargs require an offset source."""
        _first_rank(monkeypatch)
        state = lms.Qwen3_8FlashNextLegacyModelState(_runner(CPU))
        with pytest.raises(RuntimeError, match="query_start_loc is required"):
            state._maybe_add_ngram_kwargs(
                {},
                num_reqs=1,
                num_reqs_padded=2,
                is_first_rank=True,
                is_encoder_decoder=False,
                use_dummy_context=True,
                query_start_loc=None,
                num_scheduled_tokens=None,
            )
