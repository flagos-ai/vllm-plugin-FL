"""C500 MiniCPM5 W8A8 Triton launch table, inside the existing linear op."""
from functools import lru_cache
from importlib import import_module
import json
import os
from pathlib import Path


OPTIONS = dict(num_stages=1, enable_fp_fusion=False)
SCHEDULED = (' -mllvm -metaxgpu-sched-regpressure=false -mllvm -metaxgpu-vectorize-slp=true'
             ' -mllvm -metaxgpu-sched-mma-maxnum=3 -mllvm -metaxgpu-mma-unroll-count=1'
             ' -mllvm -metaxgpu-mma-sched=true ')
LIVENESS = (' -mllvm -metaxgpu-ldg-bsm-mtreg-reduce=true -mllvm -metaxgpu-igroup=true ')
PROJECTIONS = {(2048,2048):'o',(2560,2048):'qkv',(12288,2048):'gate',(2048,6144):'down'}
_observed = set()


@lru_cache(maxsize=None)
def _entry(module, symbol):
    return getattr(import_module(module),symbol)


def _ceil(value, tile):
    return (value+tile-1)//tile


def route(m,n,k):
    p = PROJECTIONS.get((n,k))
    if p is None or m <= 0:
        raise ValueError(f'Unsupported C500 W8A8 shape {(m,n,k)}')
    if m==2048:
        if p=='gate':
            return ('triton_w8a8_gate_m2048_generated_r1236','w8a8_gate_m2048',(8,96,2),4,None)
        return (f'triton_w8a8_{p}_m2048_generated_r1238',f'w8a8_{p}_m2048',
                (376 if p=='qkv' else 304,1,1),4,None)
    if m==512:
        if p=='gate':
            return ('triton_w8a8_gate_m512_generated_r1234','w8a8_gate_m512',(2,48,1),8,None)
        if p=='o':
            return ('triton_w8a8_o_m512_generated_r1194','w8a8_o_m512',(200,1,1),4,SCHEDULED)
        if p=='down':
            return ('triton_w8a8_down_m512_ordered_r1212','w8a8_down_m512',(4,16,1),4,SCHEDULED+LIVENESS)
        return ('triton_w8a8_n128_ordered_r1191','w8a8_n128',(8,20,1),4,SCHEDULED)
    if m<=16:
        if p=='gate':
            return ('triton_w8a8_gate_generated_r1145','w8a8_gate',(1,n//128,1),4,None)
        return ('triton_w8a8_decode_16x32_r1137','w8a8_decode',(1,n//32,1),4,None)
    if p=='gate':
        mod = 'triton_w8a8_n128_ordered_r1191' if m==128 else 'triton_w8a8_n128_generated_r1157'
        return (mod,'w8a8_n128',(_ceil(m,64),n//128,1),4,SCHEDULED)
    if p=='qkv':
        return ('triton_w8a8_qkv_m128_generated_r1156','w8a8_qkv_m128',(_ceil(m,64),n//64,1),4,None)
    return ('triton_w8a8_prefill_n96_generated_r1147','w8a8_prefill_n96',(_ceil(m,32),_ceil(n,96),1),4,None)


def launch(a,b,sa,sb,out,m,n,k):
    module,symbol,grid,warps,flags = route(m,n,k)
    kwargs = dict(OPTIONS,num_warps=warps)
    if flags is not None:
        kwargs['extra_options'] = flags
    compiled = _entry(module,symbol)[grid](a,b,sa,sb,out,m,n,k,**kwargs)
    path = os.environ.get('C500_W8A8_ROUTE_EVIDENCE')
    key = m,n,k
    if path and key not in _observed:
        from hashlib import sha256
        record=dict(mnk=key,module=module,symbol=symbol,grid=grid,num_warps=warps,
                    fatbin_sha256=sha256(compiled.asm['mcfatbin']).hexdigest())
        with Path(path).open('a') as log:
            log.write(json.dumps(record)+'\n')
        _observed.add(key)
    return compiled
