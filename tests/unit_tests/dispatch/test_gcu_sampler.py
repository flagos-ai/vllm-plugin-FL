# Copyright (c) 2026 BAAI. All rights reserved.

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_test_platform = os.environ.get("FL_TEST_PLATFORM")
if _test_platform and _test_platform != "enflame":
    pytest.skip("GCU-specific unit tests only run on Enflame", allow_module_level=True)

from vllm.v1.sample.ops import topk_topp_sampler

from vllm_fl.dispatch.backends.vendor.gcu.impl import sampler


def test_apply_top_k_top_p_on_cpu_preserves_native_semantics():
    logits = torch.tensor(
        [[0.1, 2.0, 0.4, 1.2], [3.0, -1.0, 0.5, 2.0]],
        dtype=torch.float16,
    )
    k = torch.tensor([2, 3], dtype=torch.int32)
    p = torch.tensor([0.95, 0.8], dtype=torch.float32)

    expected = topk_topp_sampler.apply_top_k_top_p_pytorch(
        logits.float().clone(), k, p, allow_cpu_sync=True
    )
    actual = sampler._apply_top_k_top_p_on_cpu(
        logits,
        k,
        p,
        topk_topp_sampler.apply_top_k_top_p_pytorch,
    )

    assert actual.dtype == logits.dtype
    assert actual.device == logits.device
    assert torch.equal(actual.float(), expected)


def test_apply_sampler_cpu_detour_wraps_once(monkeypatch):
    calls = []

    def native_sampler(logits, k, p, allow_cpu_sync=False):
        calls.append((logits.device.type, allow_cpu_sync))
        return logits

    def original_sampler(logits, k, p):
        calls.append(("original", logits.device.type))
        return logits

    module = SimpleNamespace(
        apply_top_k_top_p=original_sampler,
        apply_top_k_top_p_pytorch=native_sampler,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.sample.ops",
        SimpleNamespace(topk_topp_sampler=module),
    )

    assert sampler.apply_sampler_cpu_detour() is True
    wrapped = module.apply_top_k_top_p
    assert sampler.apply_sampler_cpu_detour() is False
    assert module.apply_top_k_top_p is wrapped
    assert wrapped._vllm_fl_gcu_cpu_detour is True


def test_sampler_wrapper_only_detours_gcu_top_p(monkeypatch):
    current_calls = []
    detour_calls = []

    def original(logits, k, p):
        current_calls.append((logits.device.type, p))
        return "original"

    module = SimpleNamespace(
        apply_top_k_top_p=original,
        apply_top_k_top_p_pytorch=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.sample.ops",
        SimpleNamespace(topk_topp_sampler=module),
    )
    monkeypatch.setattr(
        sampler,
        "_apply_top_k_top_p_on_cpu",
        lambda *args: detour_calls.append(args) or "detoured",
    )

    sampler.apply_sampler_cpu_detour()
    wrapped = module.apply_top_k_top_p
    gcu_logits = SimpleNamespace(device=SimpleNamespace(type="gcu"))
    cpu_logits = SimpleNamespace(device=SimpleNamespace(type="cpu"))

    assert wrapped(gcu_logits, None, torch.tensor([0.9])) == "detoured"
    assert wrapped(gcu_logits, torch.tensor([2]), None) == "original"
    assert wrapped(cpu_logits, torch.tensor([2]), torch.tensor([0.9])) == "original"
    assert len(detour_calls) == 1
    assert current_calls[0] == ("gcu", None)
    assert current_calls[1][0] == "cpu"
    assert torch.equal(current_calls[1][1], torch.tensor([0.9]))
