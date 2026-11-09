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
a = torch.randn(256, 1152, device='cuda', dtype=torch.bfloat16)
b = torch.randn(4608, 1152, device='cuda', dtype=a.dtype).t()
bias = torch.randn(4608, device='cuda', dtype=a.dtype)
expected = torch.addmm(bias, a, b)
def replacement():
    return flag_gems.ops.addmm(bias, a, b)

torch.cuda.synchronize()
print('NATIVE_PASS', flush=True)
actual = replacement()
torch.cuda.synchronize()
# BI-V150 does not provide a usable FP64 GPU reference; compute on CPU.
reference = torch.addmm(bias.cpu().double(), a.cpu().double(), b.cpu().double()).to(device=a.device, dtype=a.dtype)
print('HIGH_PRECISION_CHECK', 'native_max_abs',
      (expected.float() - reference.float()).abs().max().item(),
      'gems_max_abs', (actual.float() - reference.float()).abs().max().item(), flush=True)
# PyTorch 2.10 default BF16 tolerance: rtol=0.016, atol=1e-5.
# Reference errors are diagnostic: a native mismatch alone does not establish
# that Gems is less accurate. Both BF16 paths round differently near zero.
torch.testing.assert_close(actual, expected)
print('NO_FAILURE_REPRODUCED', 'max_abs_error',
      (actual.float() - expected.float()).abs().max().item(), flush=True)
