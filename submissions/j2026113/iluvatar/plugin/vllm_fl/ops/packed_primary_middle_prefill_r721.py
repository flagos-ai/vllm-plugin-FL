"""Compatibility import for the canonical FlagGems fused implementation."""
import hashlib
import json
import os
import time
import types
from pathlib import Path
import torch
import triton
import triton.language as tl
from flag_gems.fused.packed_gate_up_silu import (
    gate_up_silu, packed_activation, blockpaired_gate_up, _seen, _weight_layers,
)

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
            'packed_bytes':0,'scope':'One packed Parameter replaces original storage in independent runtime only; all M use paired activation; 17..64and513..2048useprojectionfusion; downprojection unchanged.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
