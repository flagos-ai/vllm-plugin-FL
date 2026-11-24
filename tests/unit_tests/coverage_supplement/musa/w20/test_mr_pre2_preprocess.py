# SPDX-License-Identifier: Apache-2.0
"""Extra branch coverage for ModelRunnerFL._preprocess (model_runner.py 3726-3869).

Complements ``test_mr_exec_core.py`` (w3c) which already covers the text-only,
prompt-embeds-with-tokens, non-first-rank and encoder-decoder happy paths.
Here we target the branches it leaves untouched:

* speculative scheduler placeholder clamping (``speculative_config is not None``)
* the multimodal + prompt-embeds merge path
* the "all scheduled positions are already embeddings" short circuit
* ``uses_mrope`` position gathering
* the pipeline-parallel intermediate-tensor precondition
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr

ModelRunnerFL = mr.ModelRunnerFL


def _runner():
    return object.__new__(ModelRunnerFL)


def _pre_runner():
    runner = _runner()
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.speculative_config = None
    runner.supports_mm_inputs = False
    runner.enable_prompt_embeds = False
    runner.uses_mrope = False
    runner.maybe_get_ec_connector_output = lambda *a, **k: contextlib.nullcontext()
    runner._init_model_kwargs = lambda: {"init": 1}
    runner._extract_mm_kwargs = lambda _so: {"mm": 2}
    runner._execute_mm_encoder = lambda _so: None
    return runner


def _sched(total=4, encoder_inputs=None):
    return SimpleNamespace(
        total_num_scheduled_tokens=total,
        scheduled_encoder_inputs=encoder_inputs or {},
    )


# ---------------------------------------------------------------------------
# speculative placeholder clamping
# ---------------------------------------------------------------------------


def test_preprocess_clamps_negative_spec_placeholders(monkeypatch):
    runner = _pre_runner()
    runner.speculative_config = SimpleNamespace()
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([-1, 2, -3, 4]))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(4, 2))
    runner.positions = torch.zeros(4, dtype=torch.long)
    input_ids, embeds, _, _, _, _ = runner._preprocess(_sched(4), 4)
    # -1 placeholders are clamped to 0 in place before the embedding lookup.
    assert input_ids.tolist() == [0, 2, 0, 4]
    assert runner.input_ids.gpu.tolist() == [0, 2, 0, 4]
    assert embeds is None


# ---------------------------------------------------------------------------
# multimodal + prompt embeds merge path
# ---------------------------------------------------------------------------


def test_preprocess_mm_with_prompt_embeds_merges_only_token_positions(monkeypatch):
    runner = _pre_runner()
    runner.supports_mm_inputs = True
    runner.enable_prompt_embeds = True
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    runner.encoder_cache = {}
    runner.input_batch = SimpleNamespace(req_prompt_embeds={"r0": True})
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([1, 2, 0, 0]))
    target = torch.zeros(4, 2)
    target[2] = 9
    target[3] = 9
    runner.inputs_embeds = SimpleNamespace(gpu=target)
    # Positions 0-1 are real token ids, positions 2-3 are precomputed embeds.
    runner.is_token_ids = SimpleNamespace(gpu=torch.tensor([True, True, False, False]))
    runner.positions = torch.zeros(4, dtype=torch.long)

    seen = {}

    def embed(ids, multimodal_embeddings=None, is_multimodal=None):
        seen["mm"] = (multimodal_embeddings, is_multimodal)
        return ids.to(torch.float32).unsqueeze(-1).expand(-1, 2) + 100

    runner.model = SimpleNamespace(
        requires_raw_input_tokens=False, embed_input_ids=embed
    )
    runner._gather_mm_embeddings = lambda _so: ("M", "I")

    input_ids, embeds, _, _, kwargs, _ = runner._preprocess(_sched(4), 4)
    assert input_ids is None
    assert seen["mm"] == ("M", "I")
    # Token-id positions are re-embedded, prompt-embeds positions are preserved.
    assert embeds[0].tolist() == [101.0, 101.0]
    assert embeds[1].tolist() == [102.0, 102.0]
    assert embeds[2].tolist() == [9.0, 9.0]
    assert embeds[3].tolist() == [9.0, 9.0]
    assert kwargs == {"init": 1, "mm": 2}


# ---------------------------------------------------------------------------
# prompt embeds with no token-id positions
# ---------------------------------------------------------------------------


def test_preprocess_prompt_embeds_all_positions_already_embedded(monkeypatch):
    runner = _pre_runner()
    runner.enable_prompt_embeds = True
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    # No token-id positions at all, so embed_input_ids must not be reached.
    runner.is_token_ids = SimpleNamespace(np=np.array([False, False]))
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([5, 6]))
    embeds = torch.full((2, 2), 3.0)
    runner.inputs_embeds = SimpleNamespace(gpu=embeds)
    runner.positions = torch.zeros(2, dtype=torch.long)
    runner.device = "cpu"

    input_ids, out_embeds, _, _, kwargs, _ = runner._preprocess(_sched(2), 2)
    assert input_ids is None
    assert out_embeds.tolist() == [[3.0, 3.0], [3.0, 3.0]]
    assert kwargs == {"init": 1}


# ---------------------------------------------------------------------------
# uses_mrope position gathering
# ---------------------------------------------------------------------------


def test_preprocess_uses_mrope_positions_slice(monkeypatch):
    runner = _pre_runner()
    runner.uses_mrope = True
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    runner.input_ids = SimpleNamespace(gpu=torch.arange(4))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(4, 2))
    runner.mrope_positions = SimpleNamespace(
        gpu=torch.arange(12, dtype=torch.long).reshape(3, 4)
    )
    runner.positions = torch.zeros(4, dtype=torch.long)

    input_ids, embeds, positions, _, _, _ = runner._preprocess(_sched(4), 2)
    assert input_ids.tolist() == [0, 1]
    assert embeds is None
    # mrope_positions.gpu[:, :num_input_tokens]
    assert positions.tolist() == [[0, 1], [4, 5], [8, 9]]


# ---------------------------------------------------------------------------
# pipeline-parallel intermediate-tensor precondition
# ---------------------------------------------------------------------------


def test_preprocess_non_first_rank_requires_intermediate(monkeypatch):
    runner = _pre_runner()
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False))
    runner.input_ids = SimpleNamespace(gpu=torch.arange(3))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(3, 2))
    runner.positions = torch.zeros(3, dtype=torch.long)
    with pytest.raises(AssertionError):
        runner._preprocess(_sched(3), 3, intermediate_tensors=None)
