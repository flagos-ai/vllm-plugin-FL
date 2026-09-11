"""Correctness and graph-safety tests for the HYV4 HC P1 fusion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_fl.models.hy_v4 import HYV4HyperConnection
from vllm_fl.ops import hy_v4_hc as hc

pytestmark = pytest.mark.gpu


def _literal_read_reference(
    streams: torch.Tensor,
    connection: HYV4HyperConnection,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal Torch reference, including the FP32 norm and F.linear."""
    flat = streams.flatten(1).float()
    norm = torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + connection.normalize_eps
    )
    gates = F.linear(flat, connection.hc_fn) * norm
    read, write = gates.split(connection.num_streams, dim=-1)
    read = torch.sigmoid(
        read * connection.hc_scale[0] + connection.hc_base[: connection.num_streams]
    )
    write = connection.magnitude * torch.sigmoid(
        write * connection.hc_scale[1] + connection.hc_base[connection.num_streams :]
    )
    read = read + connection.hc_eps
    write = write + connection.hc_eps
    collapsed = (read.unsqueeze(-1) * streams.float()).sum(dim=1)
    return collapsed.to(streams.dtype), write


def _connection(hidden_size: int, num_streams: int = 4) -> HYV4HyperConnection:
    config = SimpleNamespace(
        hc_mult=num_streams,
        hidden_size=hidden_size,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )
    connection = HYV4HyperConnection(config).cuda()
    torch.manual_seed(19)
    with torch.no_grad():
        connection.hc_fn.normal_()
        connection.hc_base.normal_()
        connection.hc_scale.normal_()
    return connection


def test_hy4_hc_explicit_disable_keeps_literal_torch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hc.HYV4_HC_FUSION_ENV, "0")
    monkeypatch.delenv(hc.HYV4_HC_FUSION_LEGACY_ENV, raising=False)
    monkeypatch.delenv(hc.HYV4_HC_FUSION_ALIAS_ENV, raising=False)
    connection = _connection(hidden_size=33)
    streams = torch.randn(
        1, connection.num_streams, 33, device="cuda", dtype=torch.bfloat16
    )
    reference = _literal_read_reference(streams, connection)
    result = connection.read(streams)
    torch.testing.assert_close(result[0], reference[0], atol=0, rtol=0)
    torch.testing.assert_close(result[1], reference[1], atol=0, rtol=0)


def test_hy4_hc_default_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hc.HYV4_HC_FUSION_ENV, raising=False)
    monkeypatch.delenv(hc.HYV4_HC_FUSION_LEGACY_ENV, raising=False)
    monkeypatch.delenv(hc.HYV4_HC_FUSION_ALIAS_ENV, raising=False)
    assert hc._env_enabled()


@pytest.mark.parametrize("rows", [1, 8, 64])
def test_hy4_hc_read_matches_literal_torch_reference(
    monkeypatch: pytest.MonkeyPatch, rows: int
) -> None:
    monkeypatch.setenv(hc.HYV4_HC_FUSION_ENV, "1")
    hidden_size, num_streams = 129, 4
    # Padding makes the actual row stride larger than ``num_streams * H``.
    backing = torch.randn(
        rows, num_streams, hidden_size + 7, device="cuda", dtype=torch.bfloat16
    )
    streams = backing[..., :hidden_size]
    connection = _connection(hidden_size, num_streams)

    reference, reference_write = _literal_read_reference(streams, connection)
    result, result_write = connection.read(streams)

    max_error = (result.float() - reference.float()).abs().max().item()
    avg_error = (result.float() - reference.float()).abs().mean().item()
    gate_max_error = (result_write - reference_write).abs().max().item()
    # The final collapse is stored as BF16; Triton's FP32 sigmoid/reduction
    # order can differ from Torch CUDA by one BF16 ulp while retaining a tiny
    # mean error.
    assert max_error <= 1e-1, (rows, max_error, avg_error)
    assert avg_error <= 1e-2, (rows, max_error, avg_error)
    assert gate_max_error <= 2e-6, (rows, gate_max_error)


@pytest.mark.parametrize("rows", [1, 8, 64])
def test_hy4_hc_writeback_matches_literal_torch_reference(rows: int) -> None:
    hidden_size, num_streams = 129, 4
    backing = torch.randn(
        rows, num_streams, hidden_size + 5, device="cuda", dtype=torch.bfloat16
    )
    streams = backing[..., :hidden_size]
    delta = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    write = torch.randn(rows, num_streams, device="cuda", dtype=torch.float32)

    result = hc.writeback(streams, delta, write, num_streams, use_fusion=True)
    reference = (
        streams.float() + write.float().unsqueeze(-1) * delta.float().unsqueeze(1)
    ).to(delta.dtype)
    max_error = (result.float() - reference.float()).abs().max().item()
    avg_error = (result.float() - reference.float()).abs().mean().item()
    assert max_error <= 1e-1, (rows, max_error, avg_error)
    assert avg_error <= 1e-2, (rows, max_error, avg_error)


def test_hy4_hc_cuda_graph_replay_same_shape_different_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hc._CUSTOM_OP_REGISTERED:
        pytest.skip("vLLM direct custom-op registration is unavailable")
    monkeypatch.setenv(hc.HYV4_HC_FUSION_ENV, "1")
    rows, hidden_size, num_streams = 8, 129, 4
    scale = torch.randn(2, device="cuda", dtype=torch.float32)
    base = torch.randn(2 * num_streams, device="cuda", dtype=torch.float32)
    streams = torch.empty(
        rows, num_streams, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    gates = torch.empty(rows, 2 * num_streams, device="cuda", dtype=torch.float32)
    delta = torch.empty(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    write = torch.empty(rows, num_streams, device="cuda", dtype=torch.float32)

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        collapsed, write_out = hc.read_post_linear(
            streams,
            gates,
            scale,
            base,
            2.0,
            1e-6,
            num_streams,
        )
        written = hc.writeback(streams, delta, write, num_streams)
        return collapsed, write_out, written

    streams.normal_()
    gates.normal_()
    delta.normal_()
    write.normal_()
    run()  # eager warmup compiles the Triton kernels before capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        captured = run()
    captured_ptrs = tuple(t.data_ptr() for t in captured)

    for seed in (101, 202):
        torch.manual_seed(seed)
        streams_value = torch.randn_like(streams)
        gates_value = torch.randn_like(gates)
        delta_value = torch.randn_like(delta)
        write_value = torch.randn_like(write)
        streams.copy_(streams_value)
        gates.copy_(gates_value)
        delta.copy_(delta_value)
        write.copy_(write_value)
        graph.replay()
        torch.cuda.synchronize()

        reference, reference_write = hc._torch_read_post_linear(
            streams_value,
            gates_value,
            scale,
            base,
            2.0,
            1e-6,
            num_streams,
        )
        reference_written = hc._torch_writeback(streams_value, delta_value, write_value)
        expected = (reference, reference_write, reference_written)
        max_errors = [
            (actual.float() - target.float()).abs().max().item()
            for actual, target in zip(captured, expected)
        ]
        avg_errors = [
            (actual.float() - target.float()).abs().mean().item()
            for actual, target in zip(captured, expected)
        ]
        assert max(max_errors) <= 1e-1, (seed, max_errors, avg_errors)
        assert max(avg_errors) <= 1e-2, (seed, max_errors, avg_errors)
        assert tuple(t.data_ptr() for t in captured) == captured_ptrs


def test_hy4_hc_custom_ops_survive_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hc._CUSTOM_OP_REGISTERED:
        pytest.skip("vLLM direct custom-op registration is unavailable")
    monkeypatch.setenv(hc.HYV4_HC_FUSION_ENV, "1")
    rows, hidden_size, num_streams = 8, 129, 4
    scale = torch.randn(2, device="cuda", dtype=torch.float32)
    base = torch.randn(2 * num_streams, device="cuda", dtype=torch.float32)

    def run(
        streams: torch.Tensor,
        gates: torch.Tensor,
        delta: torch.Tensor,
        write: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        collapsed, write_out = hc.read_post_linear(
            streams,
            gates,
            scale,
            base,
            2.0,
            1e-6,
            num_streams,
        )
        return collapsed, write_out, hc.writeback(streams, delta, write, num_streams)

    compiled = torch.compile(run, backend="inductor", dynamic=False, fullgraph=True)
    streams = torch.randn(
        rows, num_streams, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    gates = torch.randn(rows, 2 * num_streams, device="cuda")
    delta = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    write = torch.randn(rows, num_streams, device="cuda")
    result = compiled(streams, gates, delta, write)
    reference, reference_write = hc._torch_read_post_linear(
        streams, gates, scale, base, 2.0, 1e-6, num_streams
    )
    expected = (reference, reference_write, hc._torch_writeback(streams, delta, write))
    max_errors = [
        (actual.float() - target.float()).abs().max().item()
        for actual, target in zip(result, expected)
    ]
    avg_errors = [
        (actual.float() - target.float()).abs().mean().item()
        for actual, target in zip(result, expected)
    ]
    assert max(max_errors) <= 1e-1, (max_errors, avg_errors)
    assert max(avg_errors) <= 1e-2, (max_errors, avg_errors)
