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
q, k, v = [torch.randn(1, 4, 256, 72, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
# Force the mathematical reference: no nested flash redispatch in the baseline.
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.MATH):
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
def replacement():
    return flag_gems.ops._scaled_dot_product_flash_attention(q, k, v)[0]

torch.cuda.synchronize()
print('NATIVE_PASS', flush=True)
actual = replacement()
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.02)
print('NO_FAILURE_REPRODUCED', 'max_abs_error',
      (actual.float() - expected.float()).abs().max().item(), flush=True)
