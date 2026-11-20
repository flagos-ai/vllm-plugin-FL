"""Unmodified vLLM 0.32 native kernels; no FlagGems/plugin import or provider patch."""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

os.environ['VLLM_PLUGINS'] = ''
os.environ['USE_FLAGGEMS'] = '0'
import torch
import triton

op = sys.argv[1]
assert op in ('grammar', 'penalties', 'bincount')
assert not any(name.startswith(('flaggems_vllm', 'vllm_fl')) for name in sys.modules)
module = importlib.import_module('vllm.v1.worker.gpu.structured_outputs' if op == 'grammar' else 'vllm.v1.worker.gpu.sample.penalties')
assert not getattr(module, '_vllm_fl_kunlunxin_penalties_provider', False)
fn = getattr(module, '_apply_grammar_bitmask_kernel' if op == 'grammar' else 'apply_penalties' if op == 'penalties' else 'bincount')
py_fn = fn.fn if op == 'grammar' else fn
assert py_fn.__module__.startswith('vllm.')
source = Path(inspect.getsourcefile(py_fn))
print(json.dumps({'op': op, 'native_module': py_fn.__module__, 'native_source': str(source), 'native_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'torch': torch.__version__, 'triton': triton.__version__, 'provider_imported': False, 'timing_run': False}), flush=True)
device = 'cuda:0'
torch.cuda.set_device(0)

if op == 'grammar':
    # Existing benchmark shape; vLLM's original BLOCK_SIZE=8192.
    rows, vocab = 1, 167
    logits = torch.zeros((rows, vocab), dtype=torch.float32, device=device)
    mask = torch.randint(-(1 << 31), 1 << 31, (rows, triton.cdiv(vocab, 32)), dtype=torch.int32, generator=torch.Generator().manual_seed(873)).to(device)
    mapping = torch.arange(rows, dtype=torch.int32, device=device)
    offsets = torch.arange(rows + 1, dtype=torch.int32, device=device)
    fn[(rows, triton.cdiv(vocab, 8192))](logits, logits.stride(0), mapping, offsets, mask, mask.stride(0), vocab, MASK_STRIDE=1, BLOCK_SIZE=8192)
elif op == 'penalties':
    # Existing penalties benchmark shape; native wrapper selects BLOCK_SIZE=8192.
    rows, vocab = 1, 1031
    logits = torch.randn((rows, vocab), generator=torch.Generator().manual_seed(873)).to(device)
    mapping = torch.arange(rows, dtype=torch.int32, device=device)
    zeros = torch.zeros_like(mapping)
    repetition = torch.full((rows,), 1.001, device=device)
    frequency = torch.full((rows,), 0.01, device=device)
    presence = frequency.clone()
    packed = torch.zeros((rows, triton.cdiv(vocab, 32)), dtype=torch.int32, device=device)
    counts = torch.ones((rows, vocab), dtype=torch.int32, device=device)
    fn(logits, mapping, zeros, zeros, repetition, frequency, presence, packed, counts)
else:
    # Existing UT shape and noncontiguous output strides; native BLOCK_SIZE=1024.
    vocab, limit = 167, 185
    ids = torch.randint(vocab, (4, 190), dtype=torch.int32, generator=torch.Generator().manual_seed(147))
    ids[:, :4] = torch.tensor([31, 31, 12, vocab - 1])
    mapping = torch.tensor([3, 1], dtype=torch.int64, device=device)
    prompt_lens = torch.tensor([20, 135, 0, 72], dtype=torch.int32, device=device)
    prefill_lens = torch.tensor([44, 185, 0, 175], dtype=torch.int32, device=device)
    words = triton.cdiv(vocab, 32)
    packed = torch.full((4, words + 2), 7, dtype=torch.int32, device=device)
    counts = torch.full((4, vocab + 5), 7, dtype=torch.int32, device=device)
    fn(mapping, ids.to(device), prompt_lens, prefill_lens, packed[:, :words], counts[:, :vocab], limit)

torch.cuda.synchronize()
print('NATIVE_EXECUTION_COMPLETED', flush=True)
