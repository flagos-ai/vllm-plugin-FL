"""Scoped sampler binding, native fallbacks and explicit request precedence."""

from unittest.mock import Mock

import torch

from vllm.entrypoints.openai.completion.protocol import CompletionRequest

from vllm_fl.ops.kimi_k3 import sampling
from vllm_fl.patches.kimi_k3 import sampling as hook


def test_request_values_override_model_sampling_defaults():
    defaults = {"temperature": 1.0, "top_p": 0.95}
    request = CompletionRequest(model="kimi-k3", prompt=[1], max_tokens=16)
    selected = request.to_sampling_params(16, defaults)
    assert (selected.temperature, selected.top_p) == (1.0, 0.95)
    greedy = CompletionRequest(
        model="kimi-k3", prompt=[1], max_tokens=16, temperature=0.0
    ).to_sampling_params(16, defaults)
    assert (greedy.temperature, greedy.top_p) == (0.0, 1.0)


def test_unsupported_filter_preserves_original_call(monkeypatch):
    original = Mock(return_value=object())
    monkeypatch.setattr(sampling, "baseline_filter", original)
    logits = torch.empty(1, 10)
    assert sampling.apply_top_k_top_p(logits, None, None) is original.return_value
    original.assert_called_once_with(logits, None, None)


def test_hook_is_scoped_idempotent_and_rejects_partial_binding(monkeypatch):
    import vllm.v1.sample.ops.topk_topp_sampler as native
    import vllm.v1.worker.gpu.sample.sampler as runner

    monkeypatch.setattr(native, "apply_top_k_top_p", sampling.baseline_filter)
    monkeypatch.setattr(runner, "apply_top_k_top_p", sampling.baseline_filter)
    monkeypatch.setattr(hook, "is_kimi_k3_profile", lambda: False)
    hook.install()
    assert native.apply_top_k_top_p is sampling.baseline_filter
    monkeypatch.setattr(hook, "is_kimi_k3_profile", lambda: True)
    hook.install()
    hook.install()
    assert (
        native.apply_top_k_top_p
        is runner.apply_top_k_top_p
        is sampling.apply_top_k_top_p
    )
    monkeypatch.setattr(runner, "apply_top_k_top_p", sampling.baseline_filter)
    import pytest

    with pytest.raises(RuntimeError, match="pinned native/V2"):
        hook.install()
    assert native.apply_top_k_top_p is sampling.apply_top_k_top_p
    assert runner.apply_top_k_top_p is sampling.baseline_filter


def test_prepare_validates_without_mutating_bindings(monkeypatch):
    import vllm.v1.sample.ops.topk_topp_sampler as native
    import vllm.v1.worker.gpu.sample.sampler as runner

    monkeypatch.setattr(hook, "is_kimi_k3_profile", lambda: True)
    for module in (native, runner):
        monkeypatch.setattr(module, "apply_top_k_top_p", sampling.baseline_filter)
    assert hook.prepare() == (native, runner)
    assert (
        native.apply_top_k_top_p is runner.apply_top_k_top_p is sampling.baseline_filter
    )
