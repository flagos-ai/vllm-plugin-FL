# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.v2_greedy_sampler import (
    apply_v2_greedy_sampler,
)


@pytest.fixture
def sampler_classes(monkeypatch):
    class Sampler:
        def __init__(self, vllm_config, temperatures):
            self.sampling_states = SimpleNamespace(
                temperature=SimpleNamespace(np=np.array(temperatures))
            )
            self.fallback_args = None

        def _sample_random(self, *args):
            self.fallback_args = args
            return None, args[0]

    modules = (
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.sample",
        "vllm.v1.worker.gpu.sample.sampler",
    )
    for name in modules:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    model_runner = ModuleType("vllm.v1.worker.gpu.model_runner")
    model_runner.Sampler = Sampler
    sys.modules[modules[0]].model_runner = model_runner
    sys.modules[modules[-1]].Sampler = Sampler

    apply_v2_greedy_sampler()
    replacement = model_runner.Sampler
    apply_v2_greedy_sampler()
    assert model_runner.Sampler is replacement
    assert sys.modules[modules[-1]].Sampler is Sampler
    return replacement, Sampler


def test_v2_greedy_uses_processed_logits_for_each_request(sampler_classes):
    sampler_cls, _ = sampler_classes
    sampler = sampler_cls(
        SimpleNamespace(speculative_config=None), [0.8, 0.0, 0.0, 0.0]
    )
    logits = torch.tensor(
        [[2, 2, 0], [3, 2, 4], [2, 8, 3]],
        dtype=torch.bfloat16,
    )
    slots = np.array([3, 1, 2])
    sampled, processed = sampler._sample_random(
        logits, torch.tensor(slots), slots, torch.zeros(3), None, None, False
    )
    assert sampled.tolist() == [0, 2, 1]
    assert sampled.dtype == torch.int64
    assert processed is logits
    assert sampler.fallback_args is None


@pytest.mark.parametrize(
    "condition", ["mixed", "speculative", "top_k", "top_p", "fused", "expanded"]
)
def test_v2_greedy_delegates_other_batches(sampler_classes, condition):
    sampler_cls, _ = sampler_classes
    sampler = sampler_cls(
        SimpleNamespace(
            speculative_config=object() if condition == "speculative" else None
        ),
        [0.0, 0.7 if condition == "mixed" else 0.0],
    )
    logits = torch.ones(3 if condition == "expanded" else 2, 4)
    slots = np.array([1, 0])
    top_k = torch.tensor([2, 2]) if condition == "top_k" else None
    top_p = torch.tensor([0.5, 0.5]) if condition == "top_p" else None
    args = (
        logits,
        torch.tensor(slots),
        slots,
        torch.zeros(2),
        top_k,
        top_p,
        condition == "fused",
    )
    sampled, processed = sampler._sample_random(*args)
    assert sampled is None
    assert processed is logits
    assert all(
        actual is expected for actual, expected in zip(sampler.fallback_args, args)
    )
