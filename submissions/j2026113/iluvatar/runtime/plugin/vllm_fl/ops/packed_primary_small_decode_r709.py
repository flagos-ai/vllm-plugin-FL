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
    from .fused_blockpaired_gate_up_r533 import blockpaired_gate_up
except ImportError:
    from fused_blockpaired_gate_up_r533 import blockpaired_gate_up

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


@torch.library.custom_op('ts_r709::gate_up_silu',mutates_args=())
def gate_up_silu(x:torch.Tensor,packed:torch.Tensor)->torch.Tensor:
    import flag_gems
    m=x.shape[0]
    assert x.ndim==2 and x.shape[1]==2048 and x.dtype==torch.bfloat16
    assert x.is_contiguous() and packed.shape==(12288,2048) and packed.is_contiguous()
    output=x.new_empty((m,6144))
    prefill_fused=1793<=m<=2048
    small_decode_fused=17<=m<=32
    decode_fused=33<=m<=64
    eligible=prefill_fused or decode_fused or small_decode_fused
    if small_decode_fused:
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
             'prefill_fused':prefill_fused,'decode_fused':decode_fused,'small_decode_fused':small_decode_fused,
             'weight_storage':'single_packed_parameter','output_shape':list(output.shape)}
        if eligible:row.update(shared_bytes=kernel.metadata.shared,kernel_hash=kernel.hash)
        (root/f'{packed.data_ptr()}_{m}.json').write_text(json.dumps(row,indent=2))
    return output


@gate_up_silu.register_fake
def _(x,packed):return x.new_empty((x.shape[0],6144))


def packed_forward(self,x):
    result=gate_up_silu(x,self.gate_up_proj.weight)
    result,_=self.down_proj(result)
    return result


def install(model):
    from vllm.model_executor.models.llama import LlamaMLP
    from vllm.model_executor.layers.activation import SiluAndMul
    torch.cuda.synchronize();before=torch.cuda.memory_allocated();start=time.perf_counter()
    rows=[]
    for name,module in model.named_modules():
        if not isinstance(module,LlamaMLP):continue
        assert not getattr(module,'_ts_primary_packed',False)
        projection=module.gate_up_proj;weight=projection.weight
        assert weight.shape==(12288,2048) and weight.dtype==torch.bfloat16 and weight.is_contiguous()
        assert projection.tp_size==1 and projection.bias is None
        assert type(projection.quant_method).__name__=='UnquantizedLinearMethod'
        assert isinstance(module.act_fn,SiluAndMul)
        packed=torch.stack([weight[:6144].reshape(96,64,2048),weight[6144:].reshape(96,64,2048)],1).reshape(12288,2048)
        view=packed.reshape(96,2,64,2048)
        assert torch.equal(view[:,0].reshape(6144,2048),weight[:6144])
        assert torch.equal(view[:,1].reshape(6144,2048),weight[6144:])
        # Retain the Parameter and all its vLLM metadata; replace only storage.
        weight.data=packed
        module.forward=types.MethodType(packed_forward,module)
        module._ts_primary_packed=True
        _weight_layers[weight.data_ptr()]=name
        rows.append({'name':name,'packing_exactly_reversible':True,'weight_bytes':weight.numel()*weight.element_size()})
    assert len(rows)==42
    del packed,view
    torch.cuda.synchronize()
    return {'layer_count':42,'layers':rows,'packing_seconds':time.perf_counter()-start,
            'allocated_before':before,'allocated_after':torch.cuda.memory_allocated(),
            'allocated_delta':torch.cuda.memory_allocated()-before,'duplicate_weight_bytes':0,
            'logical_weight_bytes':sum(v['weight_bytes'] for v in rows),
            'packed_bytes':0,'scope':'One packed Parameter replaces original storage in independent runtime only; all M use paired activation; 17..64and1793..2048useprojectionfusion; downprojection unchanged.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
