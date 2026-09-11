"""Tests for the default-on HY4 skinny-N FP32 projection."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("HY4 HC projection tests require CUDA", allow_module_level=True)

from vllm_fl.ops.hy4_hc_projection import (  # noqa: E402
    HC_N8_ENABLE_ENV,
    _is_candidate,
    _is_large_m_rms_candidate,
    hc_n8_projection,
)

K = 24576
N = 8
EPS = 1e-6


def _literal(flat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = flat.float()
    return F.linear(flat, weight) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + EPS
    )


@pytest.mark.parametrize("m", [1, 2, 8, 64, 65, 257])
@pytest.mark.parametrize("n", [4, 8])
def test_hy4_hc_literal_correctness(monkeypatch: pytest.MonkeyPatch, m: int, n: int):
    monkeypatch.setenv(HC_N8_ENABLE_ENV, "1")
    torch.manual_seed(101 + m + n)
    flat = torch.randn((m, K), device="cuda", dtype=torch.float32)
    weight = torch.randn((n, K), device="cuda", dtype=torch.float32)
    expected = _literal(flat, weight)
    actual = hc_n8_projection(flat, weight, EPS)
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected, atol=3e-4, rtol=3e-4)
    assert _is_candidate(flat, weight) is (n == N and 2 <= m <= 64)
    assert _is_large_m_rms_candidate(flat, weight) is (m > 64)


def test_hy4_hc_large_m_avoids_full_square_allocation(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(HC_N8_ENABLE_ENV, "1")
    m = 1024
    flat = torch.randn((m, K), device="cuda", dtype=torch.float32)
    weight = torch.randn((N, K), device="cuda", dtype=torch.float32)
    expected = _literal(flat, weight)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = hc_n8_projection(flat, weight, EPS)
    torch.cuda.synchronize()
    extra_peak = torch.cuda.max_memory_allocated() - baseline
    assert torch.allclose(actual, expected, atol=3e-4, rtol=3e-4)
    # The former square() allocation is 96 MiB at this shape.  Allow room for
    # the GEMM workspace while proving that no input-sized temporary survives.
    assert extra_peak < 32 * 1024 * 1024


def test_hy4_hc_n8_graph_replay_same_address_different_data(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(HC_N8_ENABLE_ENV, "1")
    torch.manual_seed(202)
    flat = torch.randn((8, K), device="cuda", dtype=torch.float32)
    weight = torch.randn((N, K), device="cuda", dtype=torch.float32)
    for _ in range(30):
        hc_n8_projection(flat, weight, EPS)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = hc_n8_projection(flat, weight, EPS)
    address = output.data_ptr()

    changed = torch.randn_like(flat)
    flat.copy_(changed)
    graph.replay()
    torch.cuda.synchronize()
    expected = _literal(changed, weight)
    assert output.data_ptr() == address
    assert torch.allclose(output, expected, atol=3e-4, rtol=3e-4)


def test_hy4_hc_large_m_graph_replay(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(HC_N8_ENABLE_ENV, "1")
    torch.manual_seed(303)
    flat = torch.randn((65, K), device="cuda", dtype=torch.float32)
    weight = torch.randn((N, K), device="cuda", dtype=torch.float32)
    for _ in range(10):
        hc_n8_projection(flat, weight, EPS)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = hc_n8_projection(flat, weight, EPS)
    address = output.data_ptr()

    changed = torch.randn_like(flat)
    flat.copy_(changed)
    graph.replay()
    torch.cuda.synchronize()
    expected = _literal(changed, weight)
    assert output.data_ptr() == address
    assert torch.allclose(output, expected, atol=3e-4, rtol=3e-4)


def test_hy4_hc_n8_explicit_disable_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(HC_N8_ENABLE_ENV, "0")
    flat = torch.randn((8, K), device="cuda", dtype=torch.float32)
    weight = torch.randn((N, K), device="cuda", dtype=torch.float32)
    assert not _is_candidate(flat, weight)
    actual = hc_n8_projection(flat, weight, EPS)
    expected = _literal(flat, weight)
    assert torch.equal(actual, expected)


def test_hy4_hc_n8_default_is_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(HC_N8_ENABLE_ENV, raising=False)
    flat = torch.randn((8, K), device="cuda", dtype=torch.float32)
    weight = torch.randn((N, K), device="cuda", dtype=torch.float32)
    assert _is_candidate(flat, weight)
