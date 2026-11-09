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
x = torch.zeros(8, device='cuda', dtype=torch.bool)
x[1:5].fill_(True)
assert x.cpu().tolist() == [False, True, True, True, True, False, False, False]
x.zero_()
print('NATIVE_PASS', flush=True)
with flag_gems.use_gems(include=['slice']):
    view = x[1:5]
    view.fill_(True)
assert x.cpu().tolist() == [False, True, True, True, True, False, False, False]
print('NO_FAILURE_REPRODUCED', flush=True)
