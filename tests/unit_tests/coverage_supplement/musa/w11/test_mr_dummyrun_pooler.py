"""Branch coverage for _dummy_pooler_run_task (6691-6748) and
_dummy_pooler_run (6748-6778)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_mr_dummyrun_common import (  # noqa: E402
    FakePoolingMetadata,
    FakePoolingParams,
    FakePoolingStates,
    LightModel,
    LightPooler,
    mr,
    new_runner,
)

MR = mr.ModelRunnerFL


def _install_pooling_stubs(monkeypatch):
    monkeypatch.setattr(mr, "PoolingParams", FakePoolingParams)
    monkeypatch.setattr(mr, "PoolingMetadata", FakePoolingMetadata)
    monkeypatch.setattr(mr, "PoolingStates", FakePoolingStates)


def _pooler_runner(pooler=None, max_num_seqs=4):
    pooler = pooler or LightPooler()
    model = LightModel(num_tokens=6, vocab=8)
    model.pooler = pooler
    runner = new_runner(
        model=model,
        device=torch.device("cpu"),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=SimpleNamespace(model="dummy"),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)
        ),
    )
    return runner, pooler


def test_dummy_pooler_run_task_builds_metadata_and_calls_pooler(monkeypatch):
    _install_pooling_stubs(monkeypatch)
    runner, pooler = _pooler_runner()
    hidden = torch.zeros(6, 4)

    out = MR._dummy_pooler_run_task(runner, hidden, "embed")

    assert out[0].shape == (1,)
    assert len(pooler.calls) == 1
    metadata = pooler.calls[0].pooling_metadata
    assert metadata.prompt_lens.tolist() == [1, 1, 1, 3]
    assert metadata.prompt_token_ids.shape == (4, 1)
    assert metadata.prompt_token_ids.dtype == torch.int32
    assert len(metadata.pooling_params) == 4
    assert all(p.task == "embed" for p in metadata.pooling_params)
    assert len(metadata.pooling_states) == 4

    cursor = metadata.cursor
    assert cursor.device == torch.device("cpu")
    assert np.array_equal(
        cursor.num_scheduled_tokens, np.array([1, 1, 1, 3])
    )
    assert cursor.seq_lens_cpu is metadata.prompt_lens
    assert pooler.applied[0][0] == "embed"
    assert all(p.verified_with is runner.model_config for p in metadata.pooling_params)


def test_dummy_pooler_run_task_oom_rewrapped(monkeypatch):
    _install_pooling_stubs(monkeypatch)
    runner, _ = _pooler_runner(
        pooler=LightPooler(error=RuntimeError("CUDA out of memory"))
    )

    with pytest.raises(RuntimeError, match="warming up pooler"):
        MR._dummy_pooler_run_task(runner, torch.zeros(6, 4), "embed")


def test_dummy_pooler_run_task_other_error_passes_through(monkeypatch):
    _install_pooling_stubs(monkeypatch)
    runner, _ = _pooler_runner(pooler=LightPooler(error=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        MR._dummy_pooler_run_task(runner, torch.zeros(6, 4), "embed")


def test_dummy_pooler_run_mm_encoder_only(monkeypatch):
    _install_pooling_stubs(monkeypatch)
    runner, _ = _pooler_runner()
    runner.vllm_config.model_config.multimodal_config = SimpleNamespace(
        mm_encoder_only=True
    )

    out = MR._dummy_pooler_run(runner, torch.zeros(6, 4))

    assert out.numel() == 0


def test_dummy_pooler_run_requires_supported_tasks(monkeypatch):
    _install_pooling_stubs(monkeypatch)
    runner, _ = _pooler_runner()
    runner.get_supported_pooling_tasks = lambda: []

    with pytest.raises(RuntimeError, match="does not support"):
        MR._dummy_pooler_run(runner, torch.zeros(6, 4))


def test_dummy_pooler_run_picks_largest_task_and_reruns(monkeypatch):
    _install_pooling_stubs(monkeypatch)
    pooler = LightPooler(
        task_sizes={"embed": 2, "classify": 8}, include_none=True
    )
    runner, _ = _pooler_runner(pooler=pooler)
    runner.get_supported_pooling_tasks = lambda: ["embed", "classify"]

    out = MR._dummy_pooler_run(runner, torch.zeros(6, 4))

    tasks = [c.pooling_metadata.pooling_params[0].task for c in pooler.calls]
    assert tasks == ["embed", "classify", "classify"]
    assert out[0].shape == (8,)
