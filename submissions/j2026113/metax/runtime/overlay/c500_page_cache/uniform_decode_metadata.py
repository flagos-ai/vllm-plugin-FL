"""Per-call stable task compaction for declared qlen in {0,1}.

Same four INT32 arrays as ABI4 _tile_metadata(TILE=4). No new persistent
state, cross-layer reuse or synchronization. Caller guarantees Cu differences
are zero/one, bounded Ends, and disjoint input/output arrays. Probe-only.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _uniform_decode_metadata(Cu,Ends,Owners,Bases,Counts,Visible,
                             BATCH:tl.constexpr,CAP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.arange(0,BLOCK)
    begin=tl.load(Cu+row,mask=row<BATCH,other=0)
    finish=tl.load(Cu+row+1,mask=row<BATCH,other=0)
    active=(row<BATCH)&(finish-begin==1)
    work=active.to(tl.int32)
    prefix=tl.cumsum(work,0)
    total=tl.sum(work,0)
    # Active destinations [0,min(total,CAP)) and zero tail [total,CAP)
    # are disjoint; inactive lanes never write a compact destination.
    target=tl.maximum(prefix-1,0)
    store=active&(target<CAP)
    end=tl.load(Ends+row,mask=store,other=0)
    tl.store(Owners+target,row,mask=store)
    tl.store(Bases+target,begin,mask=store)
    tl.store(Counts+target,1,mask=store)
    tl.store(Visible+target,end,mask=store)
    tail=(row>=total)&(row<CAP)
    tl.store(Owners+row,0,mask=tail)
    tl.store(Bases+row,0,mask=tail)
    tl.store(Counts+row,0,mask=tail)
    tl.store(Visible+row,0,mask=tail)


def prepare(cu,ends,owners,bases,counts,visible):
    batch=ends.numel();cap=owners.numel()
    if not (1<=batch<=1024 and 1<=cap<=2048):
        raise ValueError('Probe contract requires batch1..1024, capacity1..2048')
    if cu.shape!=(batch+1,) or ends.ndim!=1:
        raise ValueError('Cu/Ends extent mismatch')
    tensors=(cu,ends,owners,bases,counts,visible)
    if any(t.dtype!=torch.int32 or not t.is_contiguous() or t.device!=cu.device for t in tensors) or not cu.is_cuda:
        raise ValueError('Requires contiguous INT32 arrays on same GPU')
    if any(t.shape!=(cap,) for t in (owners,bases,counts,visible)):
        raise ValueError('Metadata output extents must match')
    _uniform_decode_metadata[(1,)](cu,ends,owners,bases,counts,visible,batch,cap,
                                   triton.next_power_of_2(max(batch,cap)),
                                   num_warps=1 if max(batch,cap)<=128 else 4,num_stages=1)
