import hashlib
import importlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import traceback

overrides = ['TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH', 'TRITON_CUOBJDUMP_PATH',
             'TRITON_NVDISASM_PATH', 'TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH',
             'TRITON_CUPTI_INCLUDE_PATH', 'TRITON_CUPTI_LIB_PATH', 'CPATH',
             'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH']
initial_overrides = {k: os.environ[k] for k in overrides if k in os.environ}
assert not initial_overrides, initial_overrides

import torch
import triton
import triton.experimental.tle as tle
from triton import knobs
from triton.experimental.tle.language import communication

import flaggems_vllm as gems
from flaggems_vllm.utils.triton_version_utils import HAS_TLE

prefix = Path('/compiler1540')
assert Path(triton.__file__).resolve().is_relative_to(prefix / 'triton')
assert version('flagtree') == '0.7.0+git6ed3ec13'
assert triton.__version__ == '3.6.0' and HAS_TLE
assert torch.__version__.startswith('2.13') and version('vllm') == '0.32.0'
import_generated_overrides = {k: os.environ[k] for k in overrides if k in os.environ}
assert import_generated_overrides in ({}, {'TRITON_PTXAS_BLACKWELL_PATH': '/usr/local/cuda/bin/ptxas'}), import_generated_overrides
tools = {k: getattr(knobs.nvidia, k).path
         for k in ['ptxas', 'ptxas_blackwell', 'cuobjdump', 'nvdisasm']}
assert all(Path(p).resolve().is_relative_to(prefix / 'triton/backends/nvidia/bin')
           for k, p in tools.items() if k != 'ptxas_blackwell')
assert (prefix / 'triton/backends/nvidia/include/cuda.h').is_file()
libtriton = prefix / 'triton/_C/libtriton.so'
libtriton_sha = hashlib.sha256(libtriton.read_bytes()).hexdigest()
assert libtriton_sha == '9026d97c42075b0c0125bfbedf6e375a1fa16bc65fa37de26eaae03add51db78'
assert Path(communication.FLAGCX_LIB_PATH).resolve() == Path('/compiler/flagcx/lib')
flagcx_sha = hashlib.sha256(Path('/compiler/flagcx/lib/libflagcx.so').read_bytes()).hexdigest()
assert flagcx_sha == 'c768f5fa077c6d932eb23542131bc3a5ae436b0beebb96bac9581aed58279b48'

m = importlib.import_module('flaggems_vllm.ops.mhc.mhc_pre')
source = Path(m.__file__)
kernel = m.mhc_pre_packed_reader_kernel
while not isinstance(kernel, triton.runtime.JITFunction):
    kernel = kernel.fn
temperature = os.environ['NUMERIC_TEMPERATURE']
receipt = {'temperature': temperature, 'flagtree': version('flagtree'),
           'triton': triton.__version__, 'triton_path': triton.__file__,
           'tle': HAS_TLE, 'tle_path': tle.__file__, 'torch': torch.__version__,
           'vllm': version('vllm'), 'source': str(source),
           'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'libtriton_sha256': libtriton_sha, 'flagcx_sha256': flagcx_sha,
           'source_merge': '6ed3ec1307eb37c6f4f00f92c6018b15e1b4f1ee',
           'source_head': 'c0e8d152cde3194ce4a4c09fe3596abf7e36535b',
           'complete_wheel_sha256': 'fe97e2d3931a583e27e9d907866c5a055d26989568ab73d6467d11d21db8049c',
           'default_tools': tools, 'external_cuda_asset_overrides': False,
           'initial_asset_overrides': initial_overrides,
           'import_generated_asset_env_preserved': import_generated_overrides,
           'active_sm90_tool': tools['ptxas'],
           'coefficient_rtol': 2e-5, 'coefficient_atol': 2e-5,
           'value_rtol': .01, 'value_atol': .01, 'rows': []}
print('RUNTIME', json.dumps({k: v for k, v in receipt.items() if k != 'rows'}), flush=True)


def save(row):
    receipt['rows'].append(row)
    print('RESULT', json.dumps(row), flush=True)
    Path(f'/probe/numeric-{temperature}.json').write_text(json.dumps(receipt, indent=2))


def error(observed, reference, tolerance):
    diff = (observed.float() - reference.float()).abs()
    invalid = ~torch.isfinite(observed) | ~torch.isfinite(reference)
    return {'bad': int((invalid | (diff > tolerance + tolerance * reference.float().abs())).sum()),
            'nonfinite': int(invalid.sum()),
            'max_abs': diff.max().item()}


configs = [(64, 64, 4, 2), (64, 32, 8, 3)]
for K, splits in [(5120, 10), (10240, 16), (16384, 16), (32768, 16)]:
    M = 512
    torch.manual_seed(731)
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = (torch.randn(K, 32, device='cuda') * .003).bfloat16()
    b[:, 24:] = 0
    expected = (a.double() @ b.double()).float()
    expected_sq = a.float().square().sum(-1)
    for BM, BK, W, ST in configs:
        out = torch.empty(splits, M, 32, device='cuda', dtype=torch.float32)
        sq = torch.empty(splits, M, device='cuda', dtype=torch.float32)
        row = {'kind': 'raw_projection', 'M': M, 'K': K, 'splits': splits,
               'block_m': BM, 'block_k': BK, 'warps': W, 'stages': ST, 'iterations': []}
        try:
            for repeat in range(5):
                compiled = kernel[(triton.cdiv(M, BM), 1, splits)](
                    a, b, out, sq, M, K, K_SPLIT_LEN=K // splits,
                    BLOCK_N=32, BLOCK_M=BM, BLOCK_K=BK, num_warps=W, num_stages=ST)
                row['iterations'].append({'projection': error(out.sum(0), expected, 2e-5),
                                           'square_max_abs': (sq.sum(0) - expected_sq).abs().max().item()})
            row.update(n_regs=compiled.n_regs, n_spills=compiled.n_spills,
                       shared=compiled.metadata.shared)
            for ext in ['ttgir', 'ptx']:
                Path(f'/probe/{temperature}-M{M}-K{K}-W{W}-S{ST}.{ext}').write_text(compiled.asm[ext])
            row['passed'] = all(x['projection']['bad'] == 0 for x in row['iterations'])
        except Exception:
            row.update(passed=False, error=traceback.format_exc())
        save(row)

spec = importlib.util.spec_from_file_location('oracle', '/candidate/tests/test_mhc_with_norm.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
for hidden in [1280, 2560, 4096, 8192]:
    r, fn, s, b, n = ref._inputs(tokens=512, hidden=hidden)
    x = torch.randn(512, hidden, dtype=torch.bfloat16, device='cuda')
    p = torch.randn(512, 4, 1, device='cuda') * .3
    c = torch.randn(512, 4, 4, device='cuda') * .2
    r = gems.mhc_post(x, r, p, c)
    expected = ref._pre_reference(r, fn, s, b, repeats=10)
    for BM, BK, W, ST in configs:
        original = m.mhc_pre_packed_reader_kernel
        m.mhc_pre_packed_reader_kernel = triton.autotune(
            configs=[triton.Config({'BLOCK_M': BM, 'BLOCK_K': BK}, num_warps=W, num_stages=ST)],
            key=['M', 'stride_am'])(kernel)
        row = {'kind': 'strong_post_coefficients_and_bf16_value', 'shape': [512, hidden],
               'config': f'M{BM}K{BK}W{W}S{ST}', 'sinkhorn_repeats': 10, 'iterations': []}
        try:
            for repeat in range(3):
                actual = gems.mhc_pre(r, fn, s, b, 1e-6, 1e-6, 1e-6, 1., 10)
                row['iterations'].append({'post': error(actual[0], expected[0], 2e-5),
                                           'comb': error(actual[1], expected[1], 2e-5),
                                           'bf16_value': error(actual[2], expected[2], .01)})
            row['passed'] = all(v['bad'] == 0 for one in row['iterations'] for v in one.values())
        except Exception:
            row.update(passed=False, error=traceback.format_exc())
        finally:
            m.mhc_pre_packed_reader_kernel = original
        save(row)

receipt['all_passed'] = all(r['passed'] for r in receipt['rows'])
Path(f'/probe/numeric-{temperature}.json').write_text(json.dumps(receipt, indent=2))
print('SUMMARY', json.dumps({'all_passed': receipt['all_passed'], 'rows': len(receipt['rows'])}), flush=True)
raise SystemExit(0 if receipt['all_passed'] else 1)
