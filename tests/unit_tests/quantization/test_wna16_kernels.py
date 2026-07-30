# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch

from vllm_fl.quantization.wna16.kernels import gemm, moe


def test_gemm_is_unavailable_without_plugin_operator(monkeypatch):
    monkeypatch.setattr(gemm, "_resolve_wna16_gemm", lambda: None)
    assert gemm.is_wna16_gemm_available() is False
    with pytest.raises(RuntimeError, match="not built"):
        gemm.wna16_gemm(
            torch.empty(1, 8),
            torch.empty(1, 1, dtype=torch.int32),
            torch.empty(1, 1),
            8,
        )


def test_gemm_calls_fixed_plugin_operator(monkeypatch):
    calls = []

    def kernel(*args):
        calls.append(args)
        return torch.ones(2, 3)

    monkeypatch.setattr(gemm, "_resolve_wna16_gemm", lambda: kernel)
    result = gemm.wna16_gemm(
        torch.empty(2, 8),
        torch.empty(3, 1, dtype=torch.int32),
        torch.empty(3, 1),
        8,
    )
    assert gemm.is_wna16_gemm_available() is True
    assert result.shape == (2, 3)
    assert len(calls) == 1


def test_moe_calls_fixed_plugin_operator(monkeypatch):
    calls = []

    def kernel(*args):
        calls.append(args)
        return torch.ones(1)

    monkeypatch.setattr(moe, "_resolve_wna16_moe", lambda: kernel)
    assert moe.is_wna16_moe_available() is True
    result = moe.wna16_moe(
        torch.empty(1, 8),
        torch.empty(2, 4, 1, dtype=torch.int32),
        torch.empty(2, 4, 1, dtype=torch.int32),
        torch.empty(2, 4, 1),
        torch.empty(2, 4, 1),
        torch.empty(1, 2),
        torch.empty(1, 2, dtype=torch.int32),
        4,
        8,
        "silu",
        False,
        2,
        None,
        True,
    )
    assert result.shape == (1,)
    assert len(calls) == 1
