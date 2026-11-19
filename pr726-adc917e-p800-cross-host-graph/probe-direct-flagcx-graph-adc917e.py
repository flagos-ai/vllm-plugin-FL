from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist
from vllm_fl.distributed.device_communicators import flagcx


def emit(*values):
    os.write(1, (" ".join(map(str, values)) + "\n").encode())


def test_graphs(comm, source, segments, ops):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = []
    with torch.cuda.stream(stream):
        for segment in range(segments):
            graph = torch.cuda.CUDAGraph()
            emit("CAPTURE", dist.get_rank(), segments, ops, segment)
            with torch.cuda.graph(graph, stream=stream):
                outputs = []
                for _ in range(ops):
                    out = comm.all_reduce(source, stream=stream)
                    assert out is not None
                    outputs.append(out)
            captured.append((graph, outputs))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    expected = torch.full(source.shape, 10, dtype=source.dtype)
    for replay in range(1, 13):
        emit("START_REPLAY", dist.get_rank(), segments, ops, replay)
        for graph, _ in captured:
            graph.replay()
        torch.cuda.synchronize()
        for _, outputs in captured:
            for out in outputs:
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        emit("DONE_REPLAY", dist.get_rank(), segments, ops, replay)
    del captured
    torch.cuda.synchronize()


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_properties(device).pci_bus_id == (132, 182)[local]
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    assert flagcx._flagcx_kunlunxin and flagcx._flagcx_packaged_library
    comm = flagcx.PyFlagcxCommunicator(dist.group.WORLD, device)
    assert not comm.disabled, comm.init_error
    assert flagcx._api.__name__ == "flagcx.api"
    source = torch.full((1,2048), rank + 1, dtype=torch.bfloat16, device=device)
    warmup = comm.all_reduce(source)
    assert warmup is not None
    torch.cuda.synchronize()
    torch.testing.assert_close(warmup.cpu(), torch.full((1,2048),10,dtype=torch.bfloat16), rtol=0, atol=0)
    emit(json.dumps({"rank":rank,"backend":"FlagCX direct","eager_warmup":10,"library":comm.flagcx.lib._name}))
    for segments, ops in ((1,1),(8,16)):
        test_graphs(comm, source, segments, ops)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
