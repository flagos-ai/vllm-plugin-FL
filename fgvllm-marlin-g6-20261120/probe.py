"""Frozen native-reference Marlin G6 baseline and profiling diagnostics."""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path

import torch
import triton
import vllm

from tests.test_fused_marlin_moe import compute_max_diff

SHAPES = [(1, 4, 512, 512, 2)] + [
    (t, 8, 4096, 14336, 2) for t in (1, 4, 16, 64, 256, 1024)
]
parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true')
parser.add_argument('--accuracy-only', action='store_true')
parser.add_argument('--variant', default='baseline')
parser.add_argument('--only', default='all')
args = parser.parse_args()
kernel = importlib.import_module('flaggems_vllm.ops.fused_marlin_moe')
print(json.dumps(dict(event='runtime', torch=torch.__version__, vllm=vllm.__version__,
    triton=triton.__version__, kernel=kernel.__file__,
    source_sha256=hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
    shapes=SHAPES, dtype='bfloat16', tolerance=0.04, warmup_ms=10, rep_ms=50,
    gate=0.9, variant=args.variant)), flush=True)
for precision in ('int8', 'fp8'):
    module = importlib.import_module(f'benchmark.test_fused_marlin_moe_w8a16_{precision}')
    cls = getattr(module, f'FusedMarlinMoEW8A16{precision.upper()}Benchmark')
    native = getattr(module, f'_vllm_baseline_{precision}')
    actual = getattr(module, f'_gems_call_{precision}')
    bench = cls('fused_marlin_moe_w8a16', native, [torch.bfloat16])
    for shape in SHAPES:
        if args.only != 'all' and args.only != f'{precision}-{shape[0]}' and args.only != f't{shape[0]}':
            continue
        if args.profile and (precision, shape) not in (
            ('int8', SHAPES[2]), ('fp8', SHAPES[1])
        ):
            continue
        torch.manual_seed(20261119 + sum(shape))
        inputs = next(bench._gen(shape, torch.bfloat16))
        expected, result = native(*inputs), actual(*inputs)
        error = float(compute_max_diff(result.float(), expected.float()))
        print(json.dumps(dict(event='accuracy', precision=precision, shape=shape,
                              max_diff=error)), flush=True)
        assert error < 0.04
        for _ in range(5):
            native(*inputs)
            actual(*inputs)
        torch.cuda.synchronize()
        if args.accuracy_only:
            print(json.dumps(dict(event='compiled_accuracy', precision=precision,
                                  shape=shape)), flush=True)
        elif args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                for _ in range(3):
                    with torch.profiler.record_function('gems_marlin'):
                        actual(*inputs)
                torch.cuda.synchronize()
            dest = Path('/probe/outputs') / f'profile-{precision}-{shape[0]}.json'
            prof.export_chrome_trace(str(dest))
            print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=25), flush=True)
        else:
            rounds = []
            for i in range(3):
                values = {}
                for name, op in ((('native', native), ('gems', actual)) if i % 2 == 0
                                 else (('gems', actual), ('native', native))):
                    values[name] = triton.testing.do_bench(lambda: op(*inputs),
                        warmup=10, rep=50, return_mode='median')
                values['speedup'] = values['native'] / values['gems']
                rounds.append(values)
            print(json.dumps(dict(event='timing', precision=precision, shape=shape,
                                  rounds=rounds)), flush=True)
        del inputs, expected, result
        gc.collect()
        torch.cuda.empty_cache()
