from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist
import flagcx


def test_graphs(group, source, segments, ops):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = []
    with torch.cuda.stream(stream):
        for segment in range(segments):
            graph = torch.cuda.CUDAGraph()
            print("CAPTURE", dist.get_rank(), segments, ops, segment, flush=True)
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
    for replay in range(1, 13):
        print("START_REPLAY", dist.get_rank(), segments, ops, replay, flush=True)
        for graph, _ in captured:
            graph.replay()
        torch.cuda.synchronize()
        for _, outputs in captured:
            for out in outputs:
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        print("DONE_REPLAY", dist.get_rank(), segments, ops, replay, flush=True)
    del captured
    torch.cuda.synchronize()


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_properties(device).pci_bus_id == (132, 182)[local]
    assert "flagcx" in dist.Backend.backend_list
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    group = dist.new_group(ranks=[0,1,2,3], backend="flagcx", timeout=timedelta(seconds=90))
    assert dist.get_backend(group) == "flagcx"
    source = torch.full((1,2048), rank + 1, dtype=torch.bfloat16, device=device)
    warmup = source.clone()
    dist.all_reduce(warmup, group=group)
    torch.cuda.synchronize()
    torch.testing.assert_close(warmup.cpu(), torch.full((1,2048),10,dtype=torch.bfloat16), rtol=0, atol=0)
    print(json.dumps({"rank":rank,"backend":dist.get_backend(group),"eager_warmup":10}), flush=True)
    for segments, ops in ((1,1),(8,16)):
        test_graphs(group, source, segments, ops)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
