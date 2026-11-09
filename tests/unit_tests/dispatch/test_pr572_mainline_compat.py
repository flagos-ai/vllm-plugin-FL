# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for PR572 integration with the newer mainline APIs."""

import importlib

import pytest
import torch


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_raw_topk_preserves_provider_arguments(monkeypatch, phase):
    adapters = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.top_k_per_row"
    )
    providers = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.mla_ops"
    )
    calls = []
    logits = torch.empty((2, 8), dtype=torch.float32)
    indices = torch.empty((2, 3), dtype=torch.int32)
    lengths = torch.tensor([5, 7], dtype=torch.int32)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    args = (
        (logits, starts, lengths, indices, 2, 8, 1, 3)
        if phase == "prefill"
        else (logits, 1, lengths, indices, 2, 8, 1, 3)
    )

    def provider(*received):
        calls.append(received)

    def resolve(name):
        assert name == f"top_k_per_row_{phase}"
        return provider

    monkeypatch.setattr(providers, "get_operator", resolve)
    getattr(adapters, f"top_k_per_row_{phase}")(*args)
    assert len(calls) == 1
    assert len(calls[0]) == len(args)
    for actual, expected in zip(calls[0], args):
        if isinstance(expected, torch.Tensor):
            assert actual is expected
        else:
            assert actual == expected


def test_raw_topk_does_not_retry_execution_failure(monkeypatch):
    adapters = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.top_k_per_row"
    )
    providers = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.impl.mla_ops"
    )
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("provider execution failed")

    monkeypatch.setattr(providers, "get_operator", lambda _: fail)
    with pytest.raises(RuntimeError, match="provider execution failed"):
        adapters.top_k_per_row_decode(
            torch.empty((1, 8)), 1, torch.tensor([5], dtype=torch.int32),
            torch.empty((1, 2), dtype=torch.int32), 1, 8, 1, 2,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_reference_raw_topk_preserves_inactive_output(phase):
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.reference.impl.top_k_per_row"
    )
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0], [9.0, 8.0, 7.0, 6.0]])
    indices = torch.full((2, 3), -99, dtype=torch.int32)
    if phase == "prefill":
        module.top_k_per_row_prefill(
            logits, torch.tensor([1, 0], dtype=torch.int32),
            torch.tensor([4, 2], dtype=torch.int32), indices, 1, 4, 1, 2,
        )
        assert indices[0].tolist() == [0, 2, -99]
    else:
        module.top_k_per_row_decode(
            logits, 1, torch.tensor([3], dtype=torch.int32),
            indices, 1, 4, 1, 2,
        )
        assert indices[0].tolist() == [1, 2, -99]
    assert indices[1].tolist() == [-99, -99, -99]


def test_mainline_and_pr_clamp_contracts_register_separately(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops as flaggems
    from vllm_fl.dispatch.backends.reference import register_ops as reference
    from vllm_fl.dispatch.registry import OpRegistry

    monkeypatch.setattr(flaggems, "use_flaggems_op", lambda _: True)
    registry = OpRegistry()
    flaggems.register_builtins(registry)
    reference.register_builtins(registry)
    entries = registry.snapshot().impls_by_op
    for name in (
        "silu_and_mul_with_clamp", "bounded_swiglu",
        "top_k_per_row_prefill", "top_k_per_row_decode",
    ):
        assert {impl.impl_id for impl in entries[name]} == {
            "default.flagos", "reference.torch",
        }


def test_cached_ops_do_not_retain_unused_call_sites():
    import gc
    import weakref
    from vllm_fl.dispatch import CachedOp

    cached = CachedOp("temporary_call_site")
    reference = weakref.ref(cached)
    del cached
    gc.collect()
    assert reference() is None
