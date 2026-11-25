"""Model-free two-node probe. Original H4 sequence with nonblocking event evidence."""
import argparse
import concurrent.futures
import faulthandler
import os
import time
import torch

p = argparse.ArgumentParser()
p.add_argument('--rank', type=int, required=True)
p.add_argument('--master', required=True)
p.add_argument('--port', type=int, default=29643)
p.add_argument('--steps', type=int, default=20)
p.add_argument('--pressure', type=float, default=.9)
p.add_argument('--completion', choices=('output', 'compute'), default='output')
p.add_argument('--execution', choices=('mixed', 'graph', 'eager'), default='mixed')
p.add_argument('--split-comm', action='store_true')
p.add_argument('--via-plugin', action='store_true')
p.add_argument('--vary-inputs', action='store_true')
args = p.parse_args()
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.v1.outputs import ModelRunnerOutput
from vllm_fl.worker.model_runner import AsyncGPUModelRunnerOutput

torch.cuda.set_device(0)
print('START', args.rank, os.getpid(), flush=True)
group = StatelessProcessGroup.create(args.master, args.port, args.rank, 2, store_timeout=60)
owner = None
if args.via_plugin:
    assert not args.split_comm
    import importlib
    import torch.distributed as dist
    from vllm_fl.platform import PlatformFL
    name = PlatformFL.get_device_communicator_cls()
    assert name == 'vllm_fl.distributed.hygon_split_graph_comm.HygonSplitGraphCommunicator', name
    module, cls = name.rsplit('.',1)
    dist.init_process_group('gloo', init_method=f'tcp://{args.master}:{args.port+1}', rank=args.rank, world_size=2)
    owner = getattr(importlib.import_module(module), cls)(dist.group.WORLD,
        device=torch.device('cuda',0), unique_name='tp:probe', tcp_store_group=group)
    comm = owner.pynccl_comm
    z = torch.full((4,),args.rank+1,dtype=torch.bfloat16,device='cuda')
    assert torch.all(owner.all_reduce(z) == 3).item()
else:
    comm = PyNcclCommunicator(group, 0, library_path='/opt/dtk/lib/librccl.so.1')
assert comm.available and not comm.disabled
graph_comm = (PyNcclCommunicator(group, 0, library_path='/opt/dtk/lib/librccl.so.1')
              if args.split_comm else comm)
assert graph_comm.available and not graph_comm.disabled
capture = torch.cuda.Stream()
copy_stream = torch.cuda.Stream()
graphs = []
graph_expected = [3]*7
for n in (1, 2, 4, 8, 16, 32, 64):
    x = torch.full((n, 6144), args.rank+1, dtype=torch.bfloat16, device='cuda')
    y = torch.empty_like(x)
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        graph_comm.all_reduce(x, y)
    capture.synchronize()
    assert torch.all(y == 3).item()
    graph = torch.cuda.CUDAGraph()
    group.barrier()
    with torch.cuda.graph(graph, stream=capture):
        for _ in range(156):
            graph_comm.all_reduce(x, y)
    graphs.append((graph, x, y))
    print('CAPTURED', args.rank, n, flush=True)
free, total = torch.cuda.mem_get_info()
ballast = torch.empty(max(0,int(total*args.pressure)-(total-free)), dtype=torch.uint8, device='cuda')
print('READY', args.rank, 'ballast', ballast.numel(), flush=True)
events = []
start = time.monotonic()

def collect(output, n, expected):
    torch.cuda.set_device(0)
    assert output.get_output().sampled_token_ids == [[expected]]*n

def collect_compute(event):
    torch.cuda.set_device(0)
    event.synchronize()

pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
pending = []
try:
    for step in range(args.steps):
        if len(pending) == 2:
            print('WAIT', args.rank, step-2, flush=True)
            pending.pop(0).result(timeout=30)
        print('SUBMIT', args.rank, step, 'barrier', flush=True)
        group.barrier()
        print('SUBMIT', args.rank, step, 'compute', flush=True)
        delta = step % 5 if args.vary_inputs else 0
        expected = 3 + 2*delta
        if args.execution == 'eager' or (args.execution == 'mixed' and step % 8 == 0):
            n = (4096,8192,2048,64)[(step//8)%4]
            a = torch.full((n,6144), args.rank+1+delta, dtype=torch.bfloat16, device='cuda')
            b = torch.empty_like(a)
            for _ in range(156):
                comm.all_reduce(a,b)
            sampled = b[:64,0].to(torch.int64).reshape(-1,1)
            del a,b
        else:
            graph,x,y = graphs[step%len(graphs)]
            if args.vary_inputs:
                x.fill_(args.rank+1+delta)
            graph_expected[step%len(graphs)] = expected
            graph.replay()
            sampled = y[:,0].to(torch.int64).reshape(-1,1)
        ce = torch.cuda.Event()
        ce.record()
        print('SUBMIT', args.rank, step, 'copy', flush=True)
        if args.completion == 'output':
            output = AsyncGPUModelRunnerOutput(ModelRunnerOutput(req_ids=[],req_id_to_index={}),
                         sampled,None,[],copy_stream,10)
            pending.append(pool.submit(collect,output,sampled.shape[0],expected))
        else:
            pending.append(pool.submit(collect_compute,ce))
        events.append((step,ce,sampled,expected))
        print('SUBMITTED', args.rank, step, flush=True)
    for future in pending:
        future.result(timeout=30)
    pool.shutdown()
    torch.cuda.synchronize()
    for (_,_,y),expected in zip(graphs,graph_expected):
        assert torch.all(y == expected).item()
    for step,ce,sampled,expected in events:
        assert torch.all(sampled == expected).item(), step
    group.barrier()
    print('PASS',args.rank,args.steps,time.monotonic()-start,flush=True)
    for graph,x,y in graphs:
        graph.reset()
    group.barrier()
    if graph_comm is not comm:
        graph_comm.destroy()
    if owner is not None:
        owner.destroy()
        dist.destroy_process_group()
    else:
        comm.destroy()
    print('EXIT',args.rank,flush=True)
except BaseException as exc:
    print('FAIL', args.rank, repr(exc), flush=True)
    faulthandler.dump_traceback()
    os._exit(2)

