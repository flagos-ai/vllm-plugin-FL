"""One physical BF16 weight copy for prefill fusion and native decode GEMM.

Original logical rows are recovered by a bijective block64 permutation. Native
MM keeps its shape, configuration selection and K recurrence. Its packed output
is paired by the activation reader without an extra output permutation kernel.
"""
import hashlib
import json
import os
import time
import types
from pathlib import Path

import torch
import triton
import triton.language as tl

try:
    from .packed_gate_up_kernel import blockpaired_gate_up
except ImportError:
    from .packed_gate_up_kernel import blockpaired_gate_up

_seen=set()
_weight_layers={}


@triton.jit
def packed_activation(X,Y,M:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=offset//N;col=offset%N
    gate_index=(col//64)*128+col%64
    gate=tl.load(X+row*(2*N)+gate_index,row<M,0).to(tl.float32)
    up=tl.load(X+row*(2*N)+gate_index+64,row<M,0).to(tl.float32)
    sigmoid=tl.fdiv(1.0,1.0+tl.exp(-gate))
    result=((gate*sigmoid)*up).to(tl.bfloat16)
    tl.store(Y+offset,result,row<M)


@torch.library.custom_op('ts_r857::gate_up_silu',mutates_args=())
def gate_up_silu(x:torch.Tensor,packed:torch.Tensor)->torch.Tensor:
    import flag_gems
    m=x.shape[0]
    assert x.ndim==2 and x.shape[1]==2048 and x.dtype==torch.bfloat16
    assert x.is_contiguous() and packed.shape==(12288,2048) and packed.is_contiguous()
    output=x.new_empty((m,6144))
    prefill_fused=513<=m<=2048
    small_decode_fused=17<=m<=32
    decode_fused=33<=m<=64
    low_decode_fused=m==8
    eligible=prefill_fused or decode_fused or small_decode_fused or low_decode_fused
    if low_decode_fused:
        kernel=blockpaired_gate_up[(96,)](
            x,packed,output,output,output,m,6144,2048,2048,16,128,64,1,False,
            num_warps=4,num_stages=1)
    elif small_decode_fused:
        kernel=blockpaired_gate_up[(96,)](
            x,packed,output,output,output,m,6144,2048,2048,32,128,64,1,False,
            num_warps=8,num_stages=1)
    elif decode_fused:
        kernel=blockpaired_gate_up[(48,)](
            x,packed,output,output,output,m,6144,2048,2048,64,256,64,1,False,
            num_warps=16,num_stages=1)
    elif prefill_fused:
        kernel=blockpaired_gate_up[(triton.cdiv(m,256)*48,)](
            x,packed,output,output,output,m,6144,2048,2048,256,256,32,4,False,
            num_warps=16,num_stages=2)
    else:
        raw=flag_gems.mm(x,packed.t())
        packed_activation[(triton.cdiv(m*6144,1024),)](raw,output,m,6144,1024,num_warps=4)
    directory=os.environ.get('TIANSHU_FUSED_PREFILL_RECEIPTS')
    key=(packed.data_ptr(),m)
    if directory and key not in _seen:
        _seen.add(key)
        root=Path(directory);root.mkdir(exist_ok=True)
        row={'layer':_weight_layers.get(packed.data_ptr()),'M':m,'fused':eligible,
             'graph_capture':torch.cuda.is_current_stream_capturing(),
             'low_decode_fused':low_decode_fused,'prefill_fused':prefill_fused,'decode_fused':decode_fused,'small_decode_fused':small_decode_fused,
             'weight_storage':'single_packed_parameter','output_shape':list(output.shape)}
        if eligible:row.update(shared_bytes=kernel.metadata.shared,kernel_hash=kernel.hash)
        (root/f'{packed.data_ptr()}_{m}.json').write_text(json.dumps(row,indent=2))
    return output


@gate_up_silu.register_fake
def _(x,packed):return x.new_empty((x.shape[0],6144))

