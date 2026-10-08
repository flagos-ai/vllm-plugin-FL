# SPDX-License-Identifier: Apache-2.0
"""Default HC model math and graph replay preserve the checkpoint contract."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.models.hy_v4 import HYV4HyperConnection, HYV4HyperHead


def _config(hidden=33):
    return SimpleNamespace(
        hc_mult=4, hidden_size=hidden, hc_magnitude=2.0, hc_eps=1e-6, rms_norm_eps=1e-5
    )


def _initialize(module):
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(std=0.1)
    return module


def _read_reference(streams, connection):
    flat = streams.float().flatten(1)
    inv_norm = torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + connection.normalize_eps
    )
    gates = (flat @ connection.hc_fn.T) * inv_norm
    read = torch.sigmoid(gates[:, :4] * connection.hc_scale[0] + connection.hc_base[:4])
    write = connection.magnitude * torch.sigmoid(
        gates[:, 4:] * connection.hc_scale[1] + connection.hc_base[4:]
    )
    collapsed = ((read + connection.hc_eps)[..., None] * streams.float()).sum(1)
    return collapsed.to(streams.dtype), write + connection.hc_eps


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 7])
def test_default_hc_read_write_and_head(dtype, rows):
    torch.manual_seed(19)
    connection = _initialize(HYV4HyperConnection(_config()))
    head = _initialize(HYV4HyperHead(_config()))
    streams = torch.randn(rows, 4, 40, dtype=dtype)[..., :33]
    before = streams.clone()
    actual, write = connection.read(streams)
    expected, expected_write = _read_reference(streams, connection)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(write, expected_write, rtol=0, atol=0)
    delta = torch.randn(rows, 33, dtype=torch.float16)
    written = connection.write(streams, delta, write)
    expected_written = (streams.float() + write[..., None] * delta.float()[:, None]).to(
        delta.dtype
    )
    assert written.dtype == delta.dtype
    torch.testing.assert_close(written, expected_written, rtol=0, atol=0)
    flat = streams.float().flatten(1)
    norm = torch.rsqrt(flat.square().mean(-1, keepdim=True) + head.normalize_eps)
    gates = torch.sigmoid(
        (flat @ head.hc_head_fn.T) * norm * head.hc_head_scale + head.hc_head_base
    )
    expected_head = (
        ((gates + head.hc_eps)[..., None] * streams.float()).sum(1).to(dtype)
    )
    torch.testing.assert_close(head(streams), expected_head, rtol=0, atol=0)
    torch.testing.assert_close(streams, before, rtol=0, atol=0)


@pytest.mark.gpu
def test_hc_graph_replay_uses_refreshed_inputs():
    connection = _initialize(HYV4HyperConnection(_config())).cuda()
    streams = torch.randn(7, 4, 33, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(7, 33, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        collapsed, write = connection.read(streams)
        connection.write(streams, delta, write)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        collapsed, write = connection.read(streams)
        written = connection.write(streams, delta, write)
    pointers = tuple(t.data_ptr() for t in (collapsed, write, written))
    for seed in (11, 23):
        torch.manual_seed(seed)
        streams.normal_()
        delta.normal_()
        expected, expected_write = _read_reference(streams, connection)
        expected_written = (
            streams.float() + expected_write[..., None] * delta.float()[:, None]
        ).to(delta.dtype)
        for out in (collapsed, write, written):
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for actual, target in zip(
            (collapsed, write, written), (expected, expected_write, expected_written)
        ):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        assert tuple(t.data_ptr() for t in (collapsed, write, written)) == pointers
