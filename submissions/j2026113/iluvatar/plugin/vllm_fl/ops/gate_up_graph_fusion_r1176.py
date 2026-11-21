"""Fuse the ORIGINAL traced MLP graph while retaining r974 kernels/storage.

No LlamaMLP.forward replacement, public operator/provider reassignment, new
mathematical kernel, weight clone at replay, or eager execution of packed graph.
All registered packed weights must match before any graph is changed.
"""
import hashlib
import inspect
import json
import operator
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flag_gems.fused.packed_gate_up_silu import gate_up_silu, _weight_layers

_MODEL_STATE = None


def install(model):
    """Keep the prior exact packing, leave every original forward untouched."""
    global _MODEL_STATE
    from vllm.model_executor.models.llama import LlamaMLP
    from vllm.model_executor.layers.activation import SiluAndMul
    assert _MODEL_STATE is None
    modules=[(name,m) for name,m in model.named_modules() if isinstance(m,LlamaMLP)]
    assert len(modules)==42
    for name,m in modules:
        p=m.gate_up_proj;w=p.weight
        assert m.forward.__func__ is LlamaMLP.forward
        assert w.shape==(12288,2048) and w.dtype==torch.bfloat16 and w.is_contiguous()
        assert p.tp_size==1 and p.bias is None
        assert type(p.quant_method).__name__=='UnquantizedLinearMethod'
        assert isinstance(m.act_fn,SiluAndMul)
        assert getattr(m.act_fn._forward_method,'__name__',None)=='forward_native'
        assert not getattr(m,'_ts_primary_packed',False)
    torch.cuda.synchronize();before=torch.cuda.memory_allocated();start=time.perf_counter()
    rows=[]
    for name,m in modules:
        w=m.gate_up_proj.weight
        packed=torch.stack([w[:6144].reshape(96,64,2048),w[6144:].reshape(96,64,2048)],1).reshape(12288,2048)
        view=packed.reshape(96,2,64,2048)
        assert torch.equal(view[:,0].reshape(6144,2048),w[:6144])
        assert torch.equal(view[:,1].reshape(6144,2048),w[6144:])
        w.data=packed
        m._ts_primary_packed=True
        _weight_layers[w.data_ptr()]=name
        rows.append({'name':name,'packing_exactly_reversible':True,'weight_bytes':w.numel()*w.element_size()})
        assert m.forward.__func__ is LlamaMLP.forward
    del packed,view
    torch.cuda.synchronize()
    _MODEL_STATE={'expected_layers':42,'layer_names':[n for n,_ in modules]}
    return {'layer_count':42,'layers':rows,'packing_seconds':time.perf_counter()-start,
        'allocated_before':before,'allocated_after':torch.cuda.memory_allocated(),
        'allocated_delta':torch.cuda.memory_allocated()-before,'duplicate_weight_bytes':0,
        'logical_weight_bytes':sum(x['weight_bytes'] for x in rows),'packed_bytes':0,
        'model_forward_replaced':False,'original_activation_forward_retained':True,
        'compiler_fusion_required_before_execution':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _tensor(node):
    if not isinstance(node,torch.fx.Node):return None
    return node.meta.get('example_value',node.meta.get('val'))


def _is_slice(node,source,start,stop):
    if not (node.op=='call_function' and node.target is operator.getitem and node.args[0] is source):return False
    index=node.args[1]
    if not isinstance(index,tuple) or len(index)!=2 or index[0] is not Ellipsis:return False
    part=index[1]
    return isinstance(part,slice) and part.start==start and part.stop==stop and part.step is None


def fuse_graph(gm,expected_layers,require_model_names=True):
    """Fail closed on missed patterns, side users, wrong metadata or weights."""
    graph=gm.graph
    candidates=[]
    for n in graph.nodes:
        if n.op!='call_function' or n.target is not torch._C._nn.linear:continue
        if len(n.args)<2 or (len(n.args)>2 and n.args[2] is not None):continue
        x,w=n.args[:2];weight=_tensor(w);inp=_tensor(x)
        if not (isinstance(weight,torch.Tensor) and tuple(weight.shape)==(12288,2048)):continue
        if require_model_names and 'mlp_modules_gate_up_proj_parameters_weight' not in w.name:continue
        assert weight.dtype==torch.bfloat16 and weight.is_contiguous()
        assert isinstance(inp,torch.Tensor) and inp.ndim==2 and inp.shape[1]==2048
        assert inp.dtype==torch.bfloat16 and inp.is_contiguous()
        candidates.append((n,x,w))
    assert len(candidates)==expected_layers,('Expected every packed MLP projection in original graph',len(candidates),expected_layers)
    assert len({w for _,_,w in candidates})==expected_layers
    matched=[]
    for projection,x,w in candidates:
        users=list(projection.users)
        assert len(users)==2,('Projection has unaccounted users',projection.name)
        gate=[u for u in users if _is_slice(u,projection,None,6144)]
        up=[u for u in users if _is_slice(u,projection,6144,None)]
        assert len(gate)==len(up)==1
        gate,up=gate[0],up[0]
        assert len(gate.users)==1 and len(up.users)==1
        silu=next(iter(gate.users));mul=next(iter(up.users))
        assert silu.op=='call_function' and silu.target is F.silu
        assert silu.args==(gate,) and not silu.kwargs.get('inplace',False)
        assert len(silu.users)==1 and next(iter(silu.users)) is mul
        assert mul.op=='call_function' and mul.target in (operator.mul,torch.mul)
        assert set(mul.args)=={silu,up} and len(mul.args)==2 and not mul.kwargs
        # The consumed output must remain the input to the original down
        # projection; the pass doesn't change that projection or other users.
        assert len(mul.users)==1
        down=next(iter(mul.users))
        assert down.op=='call_function' and down.target is torch._C._nn.linear and down.args[0] is mul
        dw=_tensor(down.args[1]);assert isinstance(dw,torch.Tensor) and tuple(dw.shape)==(2048,6144)
        matched.append((projection,x,w,gate,up,silu,mul,down))
    # Do all validation before mutating even one node.
    names=[]
    for projection,x,w,gate,up,silu,mul,down in matched:
        with graph.inserting_before(projection):
            fused=graph.call_function(torch.ops.ts_r857.gate_up_silu.default,(x,w))
        fused.meta=dict(mul.meta)
        names.append({'weight':w.name,'projection':projection.name,'down_projection':down.name,'new_node':fused.name})
        mul.replace_all_uses_with(fused)
        for dead in [mul,silu,up,gate,projection]:
            assert not dead.users
            graph.erase_node(dead)
    graph.lint();gm.recompile()
    assert len([n for n in graph.nodes if n.op=='call_function' and n.target==torch.ops.ts_r857.gate_up_silu.default])==expected_layers
    return {'matched_layers':expected_layers,'matches':names,'model_forward_replaced':False,
        'existing_down_projections_retained':True,'same_r974_custom_operator':True,
        'note':'Compile-time graph rewrite only; registration/matches are not graph replay execution counts.'}


def apply_model_fusion(gm):
    if _MODEL_STATE is None:return None
    receipt=fuse_graph(gm,_MODEL_STATE['expected_layers'],True)
    root=os.environ.get('TIANSHU_CANDIDATE_RECEIPT')
    if root:
        p=Path(root)/('MLP_graph_fusion_'+str(os.getpid())+'.json')
        p.write_text(json.dumps(receipt,indent=2))
    return receipt
