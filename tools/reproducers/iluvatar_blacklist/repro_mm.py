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
a = torch.randn(17, 5120, device='cuda', dtype=torch.bfloat16)
b = torch.randn(4096, 5120, device='cuda', dtype=a.dtype).t()
expected = torch.mm(a, b)
def replacement():
    return flag_gems.ops.mm(a, b)

torch.cuda.synchronize()
print('NATIVE_PASS', flush=True)
actual = replacement()
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.02)
print('NO_FAILURE_REPRODUCED', 'max_abs_error',
      (actual.float() - expected.float()).abs().max().item(), flush=True)
