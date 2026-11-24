"""Branch coverage for ModelRunnerFL._dummy_sampler_run (6558-6691)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_mr_dummyrun_common import (  # noqa: E402
    LightModel,
    LightSampler,
    mr,
    new_runner,
)

MR = mr.ModelRunnerFL


def _sampler_runner(sampler=None, model=None, spec=None, mm_encoder_only=None):
    model = model or LightModel(num_tokens=5, vocab=8)
    multimodal_config = (
        SimpleNamespace(mm_encoder_only=True)
        if mm_encoder_only is not None
        else None
    )
    return new_runner(
        device=torch.device("cpu"),
        model=model,
        sampler=sampler or LightSampler(),
        rejection_sampler=MagicMock(),
        speculative_config=spec,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=multimodal_config)
        ),
    )


def _set_vendor(monkeypatch, name):
    import vllm_fl.platform as platform_mod

    monkeypatch.setattr(platform_mod.PlatformFL, "vendor_name", name)


def test_dummy_sampler_run_mm_encoder_only(monkeypatch):
    runner = _sampler_runner(mm_encoder_only=True)
    out = MR._dummy_sampler_run(runner, torch.zeros(5, 4))
    assert out.numel() == 0
    assert runner.sampler.calls == []


def test_dummy_sampler_run_runs_real_and_forward_native_paths(monkeypatch):
    _set_vendor(monkeypatch, "musa")
    sampler = LightSampler(logprobs_mode="raw_logits")
    runner = _sampler_runner(sampler=sampler)

    out = MR._dummy_sampler_run(runner, torch.zeros(5, 4))

    assert out.shape == (5,)
    assert len(sampler.calls) == 2

    first = sampler.calls[0]
    assert first.logits.shape == (5, 8)
    assert first.metadata.all_greedy is False
    assert first.metadata.top_k.tolist() == [7, 7, 7, 7, 7]
    assert torch.allclose(
        first.metadata.temperature, torch.full((5,), 0.5)
    )

    second = sampler.calls[1].metadata
    assert set(second.generators) == {0}
    assert second.generators[0].device == torch.device("cpu")


def test_dummy_sampler_run_processed_mode_skips_extra_call(monkeypatch):
    _set_vendor(monkeypatch, "musa")
    monkeypatch.setattr(mr, "PROCESSED_LOGPROBS_MODES", {"processed_logits"})
    sampler = LightSampler(logprobs_mode="processed_logits")
    runner = _sampler_runner(sampler=sampler)

    MR._dummy_sampler_run(runner, torch.zeros(5, 4))

    assert len(sampler.calls) == 1


def test_dummy_sampler_run_kunlunxin_forces_greedy(monkeypatch):
    _set_vendor(monkeypatch, "kunlunxin")
    sampler = LightSampler()
    runner = _sampler_runner(sampler=sampler)

    MR._dummy_sampler_run(runner, torch.zeros(5, 4))

    assert sampler.calls[0].metadata.all_greedy is True


def test_dummy_sampler_run_oom_is_rewrapped(monkeypatch):
    _set_vendor(monkeypatch, "musa")
    sampler = LightSampler(error=RuntimeError("CUDA out of memory"))
    runner = _sampler_runner(sampler=sampler)

    with pytest.raises(RuntimeError, match="warming up sampler"):
        MR._dummy_sampler_run(runner, torch.zeros(5, 4))


def test_dummy_sampler_run_other_runtime_error_passes_through(monkeypatch):
    _set_vendor(monkeypatch, "musa")
    sampler = LightSampler(error=RuntimeError("boom"))
    runner = _sampler_runner(sampler=sampler)

    with pytest.raises(RuntimeError, match="boom"):
        MR._dummy_sampler_run(runner, torch.zeros(5, 4))


def _spec(rejection="standard", draft="probabilistic"):
    return SimpleNamespace(
        rejection_sample_method=rejection,
        draft_sample_method=draft,
    )


def _install_spec_metadata(monkeypatch):
    factory = MagicMock(return_value="DUMMY_SPEC_METADATA")
    monkeypatch.setattr(
        mr, "SpecDecodeMetadata", SimpleNamespace(make_dummy=factory)
    )
    monkeypatch.setattr(
        mr.torch,
        "accelerator",
        SimpleNamespace(synchronize=MagicMock()),
        raising=False,
    )
    return factory


def test_dummy_sampler_run_spec_decode_probabilistic(monkeypatch):
    _set_vendor(monkeypatch, "musa")
    factory = _install_spec_metadata(monkeypatch)
    sampler = LightSampler()
    runner = _sampler_runner(sampler=sampler, spec=_spec())

    MR._dummy_sampler_run(runner, torch.zeros(3, 4))

    factory.assert_called_once_with([[0], [0], [0]], torch.device("cpu"))
    assert runner.rejection_sampler.call_count == 2

    first = runner.rejection_sampler.call_args_list[0].args
    draft_probs, logits = first[1], first[2]
    assert draft_probs is not None
    assert draft_probs.shape == (3, 8)
    assert torch.allclose(
        draft_probs.sum(dim=-1), torch.ones(3), atol=1e-5
    )
    assert logits.shape == (6, 8)

    second_metadata = runner.rejection_sampler.call_args_list[1].args[3]
    assert second_metadata.all_greedy is True
    assert second_metadata.temperature is None
    mr.torch.accelerator.synchronize.assert_called()


def test_dummy_sampler_run_spec_decode_greedy_draft_skips_draft_probs(monkeypatch):
    _set_vendor(monkeypatch, "musa")
    _install_spec_metadata(monkeypatch)
    sampler = LightSampler()
    runner = _sampler_runner(sampler=sampler, spec=_spec(draft="greedy"))

    MR._dummy_sampler_run(runner, torch.zeros(3, 4))

    first = runner.rejection_sampler.call_args_list[0].args
    assert first[1] is None
