"""Routing and fallback tests for the opt-in top-p selector."""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.ops import topp_select


def test_startup_failure_keeps_original_binding(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler

    original = topk_topp_sampler.apply_top_k_top_p
    monkeypatch.setattr(topp_select, "_INSTALLED", False)
    monkeypatch.setattr(topp_select, "_WRAPPER", None)
    monkeypatch.setattr(
        topp_select.device_profiles, "lookup", lambda _: {"topp_select": True}
    )
    monkeypatch.setattr(topp_select.envs, "switch", lambda *args, **kwargs: True)
    monkeypatch.setattr(topp_select, "_startup_gate", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.ops._topp_select_triton",
        SimpleNamespace(apply=lambda logits, p: logits),
    )

    assert not topp_select.install(torch.device("cuda:0"), 130560)
    assert topk_topp_sampler.apply_top_k_top_p is original


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires CUDA",
)
@pytest.mark.gpu
def test_cuda_small_fp64_l1_cases():
    from vllm_fl.ops._topp_select_triton import apply

    assert topp_select._validate_small_cases(apply, torch.device("cuda"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires CUDA",
)
@pytest.mark.gpu
def test_cuda_duplicate_boundary_row21_regression():
    from vllm_fl.ops._topp_select_triton import apply

    vocab = 130560
    logits = torch.full((1, vocab), -float("inf"), dtype=torch.float32)
    start = 78177
    logits[0, start : start + 157] = -0.64453125
    logits = logits.cuda()
    p = torch.tensor([1.0 - 90.000001 / 157], dtype=torch.float32, device="cuda")

    expected, expected_keep, _, _, _ = topp_select._cpu_reference(logits.cpu(), p.cpu())
    actual = apply(logits, p)
    torch.cuda.synchronize()
    assert int((~expected_keep[0, start : start + 157]).sum()) == 90
    assert int(torch.isneginf(actual[0, start : start + 157]).sum()) == 90
    assert torch.equal(actual.cpu().view(torch.int32), expected.view(torch.int32))
    assert torch.isneginf(actual[0, 78255])

    eager = actual
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed = apply(logits, p)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(eager.view(torch.int32), graphed.view(torch.int32))
