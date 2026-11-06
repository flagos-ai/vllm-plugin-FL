"""Exercise real stream routing and replay for both DV4 parallel helpers."""

import contextlib

import pytest
import torch

import vllm_fl
from vllm.utils import multi_stream_utils


@pytest.mark.parametrize("helper", ["many", "two"])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("override", ["0", "1"])
def test_aux_stream_routing_and_replay(monkeypatch, helper, capture, override):
    monkeypatch.setenv("VLLM_MULTI_STREAM_DURING_CAPTURE", override)
    vllm_fl._patch_multi_stream_capture_guard()
    aux = torch.cuda.Stream()
    start, done = torch.cuda.Event(), torch.cuda.Event()
    output = torch.zeros(2, device="cuda")
    torch.cuda.synchronize()
    called_streams = []

    def main_fn():
        called_streams.append(("main", torch.cuda.current_stream().cuda_stream))
        output[0].add_(1)

    def aux_fn():
        called_streams.append(("aux", torch.cuda.current_stream().cuda_stream))
        output[1].add_(2)

    graph = torch.cuda.CUDAGraph() if capture else None
    with torch.cuda.graph(graph) if capture else contextlib.nullcontext():
        caller = torch.cuda.current_stream().cuda_stream
        if helper == "many":
            multi_stream_utils.execute_in_parallel(
                main_fn, [aux_fn], start, [done], [aux], enable=True
            )
        else:
            multi_stream_utils.maybe_execute_in_parallel(
                main_fn, aux_fn, start, done, aux
            )
    if graph is not None:
        graph.replay()
    torch.cuda.synchronize()
    assert output.tolist() == [1.0, 2.0]
    assert dict(called_streams)["main"] == caller
    expected_aux = caller if capture and override == "0" else aux.cuda_stream
    assert dict(called_streams)["aux"] == expected_aux
