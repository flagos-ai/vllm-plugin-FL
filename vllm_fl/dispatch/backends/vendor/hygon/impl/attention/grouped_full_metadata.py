"""Exact runtime grouping eligibility; no cached shape-specific decisions."""
import torch,triton
import triton.language as tl

@triton.jit
def _group_refs(IX,REF,FULL,T:tl.constexpr,SLOTS:tl.constexpr):
    group=tl.program_id(0)
    r=tl.arange(0,4)
    j=tl.arange(0,2048)
    rows=group*4+r
    ix=tl.load(IX+rows[:,None]*2048+j[None,:],rows[:,None]<T,other=-1)
    valid=(ix>=0)&(ix<SLOTS)&(rows[:,None]<T)
    count=tl.sum(valid.to(tl.int32),1)
    tl.store(FULL+rows,count==2048,rows<T)
    candidate=tl.min(tl.where(count==tl.max(count,0),r,4),0)
    ref_row=group*4+candidate
    common=tl.load(IX+ref_row*2048+j,ref_row<T,other=-1)
    match=(~valid)|(ix==common[None,:])
    ok=tl.sum(tl.sum((~match).to(tl.int32),1),0)==0
    # A partial final group always uses the original per-query path.
    tl.store(REF+group,tl.where(ok&(group*4+3<T),ref_row,-1))

def group_refs(indices,slots):
    if indices.ndim!=2 or indices.shape[1]!=2048 or indices.dtype!=torch.int32 or not indices.is_cuda or not indices.is_contiguous():
        raise ValueError('Expected contiguous GPU int32 [T,2048]')
    out=torch.empty((triton.cdiv(len(indices),4),),device=indices.device,dtype=torch.int32)
    full=torch.empty((len(indices),),device=indices.device,dtype=torch.int32)
    if len(indices):_group_refs[(len(out),)](indices,out,full,len(indices),int(slots),num_warps=4)
    return out,full
