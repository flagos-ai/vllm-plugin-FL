from datetime import timedelta
from contextlib import nullcontext
import argparse
from vendor_pg_observations import observe_runtime
import json
import os

import torch
import torch.distributed as dist
import sys


def emit(*values, **ignored):
    os.write(1, (" ".join(map(str, values)) + "\n").encode())


def test_graphs(group, source, segments, ops, replay_stream):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = []
    with torch.cuda.stream(stream):
        for segment in range(segments):
            graph = torch.cuda.CUDAGraph()
            emit("CAPTURE", dist.get_rank(), segments, ops, segment, flush=True)
            with torch.cuda.graph(graph, stream=stream):
                outputs = []
                for _ in range(ops):
                    out = source.clone()
                    dist.all_reduce(out, group=group)
                    outputs.append(out)
            captured.append((graph, outputs))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    expected = torch.full(source.shape, 10, dtype=source.dtype)
    print("STREAMS", dist.get_rank(), "capture", stream.cuda_stream, "caller", torch.cuda.current_stream().cuda_stream, "mode", replay_stream, flush=True)
    for replay in range(1, 13):
        emit("START_REPLAY", dist.get_rank(), segments, ops, replay, flush=True)
        context = torch.cuda.stream(stream) if replay_stream == "capture" else nullcontext()
        with context:
            for graph_index, (graph, _) in enumerate(captured):
                graph.replay()
                print("AFTER_REPLAY_ENQUEUE", dist.get_rank(), segments, ops, replay, graph_index, flush=True)
            torch.cuda.synchronize()
            print("AFTER_SYNC", dist.get_rank(), segments, ops, replay, flush=True)
        for _, outputs in captured:
            for out in outputs:
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        emit("DONE_REPLAY", dist.get_rank(), segments, ops, replay, flush=True)
    del captured
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-stream", choices=("original", "capture"), default="original")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_properties(device).pci_bus_id == (132, 182)[local]
    assert "kccl" in dist.Backend.backend_list or "bccl" in dist.Backend.backend_list
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    group = dist.new_group(ranks=[0,1,2,3], backend="nccl", timeout=timedelta(seconds=90))
    assert dist.get_backend(group) == "nccl"
    backend_type = str(type(group._get_backend(device)))
    assert "ProcessGroupXCCL" in backend_type, backend_type
    print("NATIVE_VENDOR_BACKEND", rank, backend_type, flush=True)
    assert not any(name.startswith(("flagcx", "vllm_fl")) for name in sys.modules)
    source = torch.full((1,2048), rank + 1, dtype=torch.bfloat16, device=device)
    warmup = source.clone()
    dist.all_reduce(warmup, group=group)
    torch.cuda.synchronize()
    torch.testing.assert_close(warmup.cpu(), torch.full((1,2048),10,dtype=torch.bfloat16), rtol=0, atol=0)
    print(json.dumps({"rank":rank,"backend":dist.get_backend(group),"eager_warmup":10}), flush=True)
    observe_runtime(rank, group)
    for segments, ops in ((1,1),(8,16)):
        test_graphs(group, source, segments, ops, args.replay_stream)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
