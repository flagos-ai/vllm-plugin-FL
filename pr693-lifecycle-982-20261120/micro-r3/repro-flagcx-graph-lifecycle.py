"""Tiny real FlagCX graph replay retained by Dynamo's native code cache.

Use identical computation for implicit, destroy and clear cleanup variants.
This is a lifecycle diagnostic, not a model/adaptation Gate.
"""
import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import weakref

import torch
import torch.distributed as dist
from vllm.platforms import current_platform

assert getattr(current_platform, 'vendor_name', None) == 'metax'
assert current_platform.device_type == 'cuda'

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.compilation.cuda_graph import CUDAGraphEntry, CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

parser = argparse.ArgumentParser()
parser.add_argument('--cleanup', choices=['implicit', 'destroy', 'clear'], required=True)
args = parser.parse_args()
assert md.version('vllm') == '0.32.0+empty'
assert hashlib.sha256(Path('/opt/FlagCX/build/lib/libflagcx.so').read_bytes()).hexdigest() == '33b3c77248366928f89b6b75ad6b5eb579126512d39381ebd59a4c245896f0ad'
assert hashlib.sha256(Path('/opt/FlagCX/plugin/interservice/flagcx_wrapper.py').read_bytes()).hexdigest() == 'a791806a0ab36da872cd200b5466c36001ef798c8fb444fc62b1d1f6805ca5cf'
rank = int(os.environ['RANK'])
local = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local)
device = torch.device('cuda', local)
dist.init_process_group('gloo')
world = dist.get_world_size()
assert world == 2
comm = PyFlagcxCommunicator(dist.group.WORLD, device)
assert comm.available and not comm.disabled
reduced_outputs = []
graph_refs = []


def registry_state():
    return {
        'native_wrappers': len(CUDAGraphWrapper._all_instances),
        'native_graphs': sum(len(w.concrete_cudagraph_entries) for w in CUDAGraphWrapper._all_instances),
        'breakable_wrappers': len(BreakableCUDAGraphWrapper._all_instances),
        'live_graph_objects': sum(ref() is not None for ref in graph_refs),
    }


def report(stage, **fields):
    record = {'stage': stage, 'rank': rank, 'pid': os.getpid(), 'time_unix': time.time(),
              'cleanup': args.cleanup, **fields}
    print(json.dumps(record), flush=True)
    Path(f'/work/rank-{rank}-{stage}.json').write_text(json.dumps(record, indent=2))


report('runtime', torch=torch.__version__, vllm=md.version('vllm'),
       library_sha256='33b3c77248366928f89b6b75ad6b5eb579126512d39381ebd59a4c245896f0ad',
       device=str(device), world=world)


def backend(gm, inputs):
    assert len(inputs) == 1 and isinstance(inputs[0], torch.Tensor)
    static_inputs = [inputs[0].detach().clone()]
    # Warm the exact real submission and native Torch add before capture.
    for _ in range(2):
        comm.all_reduce(static_inputs[0])
        gm(*static_inputs)
    torch.cuda.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reduced = comm.all_reduce(static_inputs[0])
        output = gm(*static_inputs)
    reference = weakref.ref(graph)
    graph_refs.append(reference)
    reduced_outputs.append(weakref.ref(reduced))
    output_references = tuple(weakref.ref(item) for item in output)

    def replay(*new_inputs):
        static_inputs[0].copy_(new_inputs[0])
        captured = reference()
        assert captured is not None
        captured.replay()
        values = tuple(reference() for reference in output_references)
        assert all(value is not None for value in values)
        return values

    config = SimpleNamespace(compilation_config=SimpleNamespace())
    holder = CUDAGraphWrapper(replay, config, CUDAGraphMode.PIECEWISE)
    descriptor = BatchDescriptor(num_tokens=16)
    holder.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
        batch_descriptor=descriptor, cudagraph=graph, output=(output, reduced))
    return holder


def function(value):
    return value + 1


compiled = torch.compile(function, backend=backend, fullgraph=True)
base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
for multiplier in (1, 2, 3):
    value = ((base + rank + 1) * multiplier).to(device)
    result = compiled(value)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(result.cpu(), (base + rank + 1) * multiplier + 1, rtol=0, atol=0)
    expected = (world * base + world * (world + 1) // 2) * multiplier
    torch.testing.assert_close(reduced_outputs[-1]().cpu(), expected, rtol=0, atol=0)
report('replay_pass', replays=3, exact_output_and_collective_oracles=True)
del compiled, value, result
gc.collect()
state = registry_state()
assert state['native_wrappers'] == 1 and state['native_graphs'] == 1
assert state['live_graph_objects'] == 1
report('retained_after_gc', registry=state)
dist.barrier()
if args.cleanup == 'clear':
    torch.cuda.synchronize(device)
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
    cleared = registry_state()
    assert cleared['native_graphs'] == 0 and cleared['live_graph_objects'] == 0
    report('registries_cleared', registry=cleared)

report('destroy_begin', registry=registry_state())
if args.cleanup != 'implicit':
    handle = comm.comm
    comm.comm = None
    comm.available = False
    comm.disabled = True
    comm.flagcx.flagcxCommDestroy(handle)
    report('native_destroy_returned')
else:
    # This is the existing MetaX PyFlagcxCommunicator behavior (a no-op).
    comm.destroy()
dist.destroy_process_group()
report('exit_python', registry=registry_state())
