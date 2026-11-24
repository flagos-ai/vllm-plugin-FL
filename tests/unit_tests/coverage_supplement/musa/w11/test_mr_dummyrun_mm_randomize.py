"""Branch coverage for _get_mm_dummy_batch (6072-6108) and
maybe_randomize_inputs (6023-6070)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_mr_dummyrun_common import mr, new_runner  # noqa: E402

MR = mr.ModelRunnerFL


def _install_group(monkeypatch):
    captured = []

    def group_and_batch_mm_kwargs(mm_inputs, device=None, pin_memory=None):
        captured.append(
            SimpleNamespace(
                mm_inputs=mm_inputs, device=device, pin_memory=pin_memory
            )
        )
        yield ("image", 1, {"pixel_values": torch.zeros(1, 2)})

    monkeypatch.setattr(
        mr, "group_and_batch_mm_kwargs", group_and_batch_mm_kwargs
    )
    return captured


def test_get_mm_dummy_batch_uses_budget_profile_inputs(monkeypatch):
    captured = _install_group(monkeypatch)
    profile = MagicMock(return_value="PROFILE_INPUTS")
    budget = SimpleNamespace(
        get_dummy_encoder_profile_inputs=profile, cache="CACHE"
    )
    runner = new_runner(mm_budget=budget, device="musa:0")

    out = MR._get_mm_dummy_batch(runner, "image", 3)

    assert "pixel_values" in out
    profile.assert_called_once_with("image", 3)
    assert captured[0].mm_inputs == "PROFILE_INPUTS"
    assert captured[0].device == "musa:0"
    assert captured[0].pin_memory == mr.PIN_MEMORY


def test_get_mm_dummy_batch_falls_back_to_registry(monkeypatch):
    captured = _install_group(monkeypatch)
    item = {"pixel_values": torch.zeros(1, 2)}
    registry = SimpleNamespace(
        get_dummy_mm_inputs=MagicMock(
            return_value={"mm_kwargs": {"image": [item]}}
        )
    )
    budget = SimpleNamespace(cache="CACHE")
    runner = new_runner(
        mm_budget=budget,
        mm_registry=registry,
        model_config="MODEL_CONFIG",
        device="cpu",
    )

    out = MR._get_mm_dummy_batch(runner, "image", 2)

    assert "pixel_values" in out
    registry.get_dummy_mm_inputs.assert_called_once_with(
        "MODEL_CONFIG", mm_counts={"image": 1}, cache="CACHE"
    )
    assert captured[0].mm_inputs == [("image", item), ("image", item)]


def test_get_mm_dummy_batch_rejects_already_cached_item(monkeypatch):
    _install_group(monkeypatch)
    registry = SimpleNamespace(
        get_dummy_mm_inputs=MagicMock(
            return_value={"mm_kwargs": {"image": [None]}}
        )
    )
    runner = new_runner(
        mm_budget=SimpleNamespace(cache="CACHE"),
        mm_registry=registry,
        model_config="MODEL_CONFIG",
        device="cpu",
    )

    with pytest.raises(AssertionError):
        MR._get_mm_dummy_batch(runner, "image", 1)


# ---------------------------------------------------------------------------
# maybe_randomize_inputs
# ---------------------------------------------------------------------------
def _randomize_runner(data_parallel_size=1, vocab=1000):
    return new_runner(
        input_ids=SimpleNamespace(gpu=torch.full((8,), 5, dtype=torch.long)),
        inputs_embeds=SimpleNamespace(gpu=torch.full((8, 4), 5.0)),
        model_config=SimpleNamespace(get_vocab_size=lambda: vocab),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=data_parallel_size
            )
        ),
    )


def test_maybe_randomize_inputs_disabled_leaves_ids(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False)
    runner = _randomize_runner(data_parallel_size=1)
    ids = torch.full((4,), 5, dtype=torch.long)

    with MR.maybe_randomize_inputs(runner, ids, None):
        pass

    assert torch.equal(ids, torch.full((4,), 5, dtype=torch.long))


def test_maybe_randomize_inputs_env_and_dp_randomizes_ids(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", True)
    monkeypatch.setattr(
        mr.torch,
        "randint_like",
        lambda t, low=0, high=None: torch.full_like(t, 7),
    )
    runner = _randomize_runner(data_parallel_size=2)
    ids = torch.full((4,), 5, dtype=torch.long)

    with MR.maybe_randomize_inputs(runner, ids, None):
        assert torch.equal(ids, torch.full((4,), 7, dtype=torch.long))

    assert torch.equal(ids, torch.zeros(4, dtype=torch.long))


def test_maybe_randomize_inputs_explicit_embeds(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False)
    monkeypatch.setattr(
        mr.torch, "randn_like", lambda t: torch.full_like(t, 3.0)
    )
    runner = _randomize_runner()
    embeds = torch.full((4, 4), 5.0)

    with MR.maybe_randomize_inputs(
        runner, None, embeds, randomize_inputs=True
    ):
        assert torch.equal(embeds, torch.full((4, 4), 3.0))

    assert torch.equal(embeds, torch.zeros(4, 4))


def test_maybe_randomize_inputs_dp_size_one_stays_disabled(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", True)
    runner = _randomize_runner(data_parallel_size=1)
    embeds = torch.full((4, 4), 5.0)

    with MR.maybe_randomize_inputs(runner, None, embeds):
        pass

    assert torch.equal(embeds, torch.full((4, 4), 5.0))
