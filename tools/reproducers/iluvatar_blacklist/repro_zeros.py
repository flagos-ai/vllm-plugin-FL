"""Standalone native vs FlagGems reproducer; nonzero exit means failure.
Run in an Iluvatar environment with PYTHONPATH unset (see README.md).
"""
import argparse
import importlib.metadata as metadata
import json
import torch
import flag_gems

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--small', action='store_true', help='small allocation control')
args = parser.parse_args()
torch.manual_seed(args.seed)
print(json.dumps({'torch': torch.__version__, 'flag_gems': metadata.version('flag_gems'),
                  'source': flag_gems.__file__, 'device': torch.cuda.get_device_name(0),
                  'seed': args.seed}), flush=True)
size = (1 << 20) if args.small else (16 << 30)
x = torch.zeros(size, device='cuda', dtype=torch.int8)
torch.cuda.synchronize()
for chunk in x.split(16 << 20):
    assert torch.count_nonzero(chunk).item() == 0
del chunk, x
print('NATIVE_PASS', flush=True)
x = flag_gems.ops.zeros((size,), device='cuda', dtype=torch.int8)
torch.cuda.synchronize()
for chunk in x.split(16 << 20):
    assert torch.count_nonzero(chunk).item() == 0
print('NO_FAILURE_REPRODUCED', flush=True)
