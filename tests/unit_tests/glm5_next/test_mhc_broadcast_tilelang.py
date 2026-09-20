# SPDX-License-Identifier: Apache-2.0
"""Correctness and CUDA Graph tests for first-layer mHC broadcast."""

from __future__ import annotations

import pytest
import torch


def test_mhc_broadcast_weight_collapse_is_algebraically_exact() -> None:
    torch.manual_seed(20260827)
    tokens, hidden_size, streams = 7, 32, 4
    mix_size = streams * (2 + streams)
    residual = torch.randn(tokens, hidden_size, dtype=torch.float64)
    fn = torch.randn(mix_size, streams * hidden_size, dtype=torch.float64)

    expanded = residual.unsqueeze(1).expand(-1, streams, -1).reshape(tokens, -1)
    collapsed = fn.view(mix_size, streams, hidden_size).sum(dim=1)

    torch.testing.assert_close(
        expanded @ fn.t(),
        residual @ collapsed.t(),
        rtol=1e-12,
        atol=1e-12,
    )


def _require_cuda_broadcast_kernel():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not is_deep_gemm_supported():
        pytest.skip("requires the DeepGEMM TF32 mHC projection")

    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang

    from vllm_fl.kernels.glm5_next.mhc_broadcast_tilelang import (
        mhc_pre_broadcast_tilelang,
    )

    return mhc_pre_tilelang, mhc_pre_broadcast_tilelang


def _make_inputs(tokens: int):
    torch.manual_seed(20260827 + tokens)
    hidden_size, streams = 4096, 4
    mix_size = streams * (2 + streams)
    fn = (
        torch.randn(
            mix_size,
            streams * hidden_size,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.01
    ).contiguous()
    return {
        "residual": torch.randn(
            tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        ),
        "fn": fn,
        "fn_broadcast": fn.view(mix_size, streams, hidden_size).sum(dim=1).contiguous(),
        "scale": torch.tensor([0.8, 0.7, 0.6], device="cuda", dtype=torch.float32),
        "base": (
            torch.randn(mix_size, device="cuda", dtype=torch.float32) * 0.1
        ).contiguous(),
        "norm_weight": (
            1 + torch.randn(hidden_size, device="cuda", dtype=torch.float32) * 0.05
        )
        .to(torch.bfloat16)
        .contiguous(),
        "streams": streams,
    }


def _generic_outputs(mhc_pre_tilelang, values):
    expanded = (
        values["residual"].unsqueeze(1).expand(-1, values["streams"], -1).contiguous()
    )
    post, comb, layer_input = mhc_pre_tilelang(
        expanded,
        values["fn"],
        values["scale"],
        values["base"],
        1e-5,
        1e-6,
        1e-6,
        2.5,
        20,
        norm_weight=values["norm_weight"],
        norm_eps=1e-5,
    )
    return expanded, post, comb, layer_input


def _broadcast_outputs(mhc_pre_broadcast_tilelang, values):
    return mhc_pre_broadcast_tilelang(
        values["residual"],
        values["fn"],
        values["scale"],
        values["base"],
        1e-5,
        1e-6,
        1e-6,
        2.5,
        20,
        norm_weight=values["norm_weight"],
        norm_eps=1e-5,
        fn_broadcast=values["fn_broadcast"],
    )


def _assert_outputs_close(reference, actual) -> None:
    assert torch.equal(reference[0], actual[0])
    torch.testing.assert_close(reference[1], actual[1], rtol=2e-3, atol=1e-3)
    torch.testing.assert_close(reference[2], actual[2], rtol=2e-3, atol=1e-3)
    torch.testing.assert_close(reference[3], actual[3], rtol=2e-2, atol=4e-2)


@pytest.mark.parametrize("tokens", [1, 24, 257, 1024])
def test_mhc_broadcast_matches_materialized_reference(tokens: int) -> None:
    generic, broadcast = _require_cuda_broadcast_kernel()
    values = _make_inputs(tokens)

    reference = _generic_outputs(generic, values)
    actual = _broadcast_outputs(broadcast, values)
    repeated = _broadcast_outputs(broadcast, values)
    torch.cuda.synchronize()

    _assert_outputs_close(reference, actual)
    for lhs, rhs in zip(actual, repeated):
        assert torch.equal(lhs, rhs)


def test_mhc_broadcast_cuda_graph_replay() -> None:
    generic, broadcast = _require_cuda_broadcast_kernel()
    values = _make_inputs(24)

    # Compile both DeepGEMM and TileLang before stream capture.
    _broadcast_outputs(broadcast, values)
    torch.cuda.synchronize()

    static_residual = values["residual"].clone()
    graph_values = dict(values, residual=static_residual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = _broadcast_outputs(broadcast, graph_values)

    next_residual = torch.randn_like(static_residual)
    static_residual.copy_(next_residual)
    graph.replay()
    torch.cuda.synchronize()
    replayed = tuple(value.clone() for value in graph_outputs)

    reference_values = dict(values, residual=next_residual)
    reference = _generic_outputs(generic, reference_values)
    torch.cuda.synchronize()
    _assert_outputs_close(reference, replayed)
