# P800 XCCL stalls on repeated cross-host FlagCX graph replay

## Reproduction

- Two P800 hosts with two cards each, four ranks, XCCL RDMA through FlagCX, and a Gloo control process group.
- P800 driver 5.0.21.47, XRE 5.37.1, PyTorch 2.9.0+cu129 and FlagCX wheel 0.14.1+xre5.37.1. The wheel's loaded library reports version code 10000.
- Image digest `sha256:93caedbfc36d665c2e7d9f70a38bbb2501a860c9389be660888b0b6d66bc4402`. Plugin commit `adc917edea0933a8823b7dbe067debaeb8dc582a` supplies the direct Python wrapper control.
- The captured workload is one BF16 `[1, 2048]` all-reduce. Each rank contributes its index plus one. The expected output is 10 in every element on every replay.

Launch two processes per host with `torchrun --nnodes=2 --nproc-per-node=2 --node-rank "$NODE_RANK" --master-addr "$HEAD_IP" --master-port "$RENDEZVOUS_PORT" repro.py`. Configure the same working XCCL RDMA NICs and Gloo socket interface on both hosts. Run once with `ROUTE=process_group` and once with `ROUTE=direct`.

```python
from datetime import timedelta
import os

import flagcx
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group("gloo", timeout=timedelta(seconds=90))

if os.environ["ROUTE"] == "process_group":
    group = dist.new_group(ranks=[0, 1, 2, 3], backend="flagcx")

    def reduce(x, stream=None):
        y = x.clone()
        dist.all_reduce(y, group=group)
        return y

else:
    from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

    comm = PyFlagcxCommunicator(dist.group.WORLD, device)
    assert not comm.disabled

    def reduce(x, stream=None):
        return comm.all_reduce(x, stream=stream)

x = torch.full((1, 2048), rank + 1, device=device, dtype=torch.bfloat16)
expected = torch.full(x.shape, 10, dtype=x.dtype)
warm = reduce(x)
torch.cuda.synchronize()
torch.testing.assert_close(warm.cpu(), expected, rtol=0, atol=0)

graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    with torch.cuda.graph(graph, stream=stream):
        y = reduce(x, stream)
torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()

for replay in range(1, 13):
    print(f"rank={rank} start replay={replay}", flush=True)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    print(f"rank={rank} done replay={replay}", flush=True)
```

## Observations

Eager warmup and replay 1 return 10 on all four ranks for both FlagCX bindings. All four ranks enter replay 2 and stop progressing at the graph replay or completion boundary. Logs contain zero XRE 299 and NoC timeout markers. Both routes were stopped after the 355-second test deadline. Each launcher exited 137 after forced teardown, and their containers were removed.

A real 35B BF16 Runner V2 text-only TP4 run on the same two hosts completed FULL decode capture. The first original request recorded four 1-to-1 FULL replays before generation stopped. It returned zero successful responses for nine original requests. The same deployment's Eager text gate passed 9/9, and the independently checked two-host eager FP32/BF16 all-reduce and all-gather passed.

Can the adaptor owners confirm whether repeated CUDA graph replay with cross-host XCCL RDMA is supported on this wheel and XRE combination, and diagnose the stall at replay 2?
