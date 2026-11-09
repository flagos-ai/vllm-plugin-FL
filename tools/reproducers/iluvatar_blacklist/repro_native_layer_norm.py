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
x = torch.randn(256, 1152, device='cuda', dtype=torch.bfloat16)
w = torch.randn(1152, device='cuda', dtype=x.dtype)
b = torch.randn(1152, device='cuda', dtype=x.dtype)
expected = torch.native_layer_norm(x, (1152,), w, b, 1e-6)
torch.cuda.synchronize()
print('NATIVE_PASS', [(tuple(value.shape), str(value.dtype)) for value in expected], flush=True)
actual = flag_gems.ops.native_layer_norm(x, (1152,), w, b, 1e-6)
torch.cuda.synchronize()
print('GEMS_OUTPUTS', [(tuple(value.shape), str(value.dtype)) for value in actual], flush=True)
torch.testing.assert_close(actual, expected)
print('NO_FAILURE_REPRODUCED', flush=True)
