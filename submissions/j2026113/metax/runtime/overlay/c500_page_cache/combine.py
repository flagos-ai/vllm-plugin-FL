"""Two-query pair metadata and native partial; caller supplies validated buffers.

Pair slots are packed across requests without pairing across a boundary.
Inactive pair slots have count0 and the native CTA returns uniformly.
Each valid query writes original per-query scratch. Merge skips padded query
capacity entirely, preserving inactive outputs and avoiding stale scratch.
"""
import ctypes
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl


@triton.jit
def _query_pair_metadata(Cu,Ends,Owners,Bases,Counts,FirstVisible,
                         CAP:tl.constexpr,BATCH:tl.constexpr,BLOCK:tl.constexpr):
    pair=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    owner=tl.full((BLOCK,),0,tl.int32);base=tl.full((BLOCK,),0,tl.int32)
    count=tl.full((BLOCK,),0,tl.int32);visible=tl.full((BLOCK,),0,tl.int32)
    pair_prefix=0
    for request in range(BATCH):
        begin=tl.load(Cu+request);finish=tl.load(Cu+request+1);qlen=finish-begin
        pairs=tl.cdiv(qlen,2)
        active=(pair>=pair_prefix)&(pair<pair_prefix+pairs)&(pair<CAP)
        query_base=begin+(pair-pair_prefix)*2
        owner=tl.where(active,request,owner);base=tl.where(active,query_base,base)
        count=tl.where(active,tl.minimum(2,finish-query_base),count)
        end=tl.load(Ends+request)
        visible=tl.where(active,end-qlen+query_base-begin+1,visible)
        pair_prefix+=pairs
    tl.store(Owners+pair,owner,mask=pair<CAP);tl.store(Bases+pair,base,mask=pair<CAP)
    tl.store(Counts+pair,count,mask=pair<CAP);tl.store(FirstVisible+pair,visible,mask=pair<CAP)


@triton.jit
def _query_pair_combine(Part,Maxima,Sums,Cu,Out,BATCH:tl.constexpr,SPLITS:tl.constexpr,SP:tl.constexpr):
    g=tl.program_id(0);head=tl.program_id(1);total=tl.load(Cu+BATCH)
    if g//2<total:
        s=tl.arange(0,SP);d=tl.arange(0,128);offset=(g*SPLITS+s)*8+head
        mx=tl.load(Maxima+offset,mask=s<SPLITS,other=float('-inf'))
        ls=tl.load(Sums+offset,mask=s<SPLITS,other=0.)
        maximum=tl.max(mx,0);maximum=tl.where(maximum==float('-inf'),0.,maximum)
        weight=tl.where(ls>0,tl.exp2(mx-maximum),0.)
        partial=tl.load(Part+offset[:,None]*128+d[None,:],mask=s[:,None]<SPLITS,other=0.)
        numerator=tl.sum(partial*weight[:,None],0);denominator=tl.sum(ls*weight,0)
        tl.store(Out+(g*8+head)*128+d,numerator/tl.where(denominator>0,denominator,1.))


class NativeQueryPairs:
    def __init__(self,library,digest):
        p=Path(library);assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
        self.lib=ctypes.CDLL(str(p),mode=ctypes.RTLD_LOCAL)
        self.fn=self.lib.c500_query_pairs_forward;self.fn.argtypes=[ctypes.c_void_p]*13+[ctypes.c_int]*4+[ctypes.c_void_p];self.fn.restype=ctypes.c_int
        self.lib.c500_query_pairs_abi.restype=ctypes.c_int;assert self.lib.c500_query_pairs_abi()==3
        self.error=self.lib.c500_query_pairs_error;self.error.argtypes=[ctypes.c_int];self.error.restype=ctypes.c_char_p

    def prepare(self,cu,ends,owners,bases,counts,visible):
        _query_pair_metadata[(triton.cdiv(owners.numel(),128),)](
            cu,ends,owners,bases,counts,visible,owners.numel(),ends.numel(),128,num_warps=1,num_stages=1)

    def into(self,q,k,v,qs,ks,cu,owners,bases,counts,visible,table,part,maxima,sums,out,*,max_len,splits):
        if splits not in (1,4,8,16,32,64):raise ValueError('unsupported splits')
        status=self.fn(*[t.data_ptr() for t in (q,k,v,qs,ks,visible,table,owners,bases,counts,part,maxima,sums)],
                       owners.numel(),table.shape[1],max_len,splits,torch.cuda.current_stream(q.device).cuda_stream)
        if status:raise RuntimeError('native query pairs launch: '+self.error(status).decode())
        _query_pair_combine[(q.shape[0]*2,8)](part,maxima,sums,cu,out,table.shape[0],splits,
                                            triton.next_power_of_2(splits),num_warps=1,num_stages=1)
