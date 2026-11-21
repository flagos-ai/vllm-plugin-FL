"""Two exact runner gathers without per-input-shape generated tuning objects.

The runner already constructs distinct destination positions for each common
request. Repeated source rows are supported; duplicate destination positions
are outside that original scatter contract. No buffer lifetime is extended.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _sampled_token_scatter(src,source_rows,dest_rows,out,N,
                          SRC_STRIDE:tl.constexpr,SOURCE_INDEX_STRIDE:tl.constexpr,
                          DEST_INDEX_STRIDE:tl.constexpr,OUT_STRIDE:tl.constexpr,
                          BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=i<N
    source=tl.load(source_rows+i*SOURCE_INDEX_STRIDE,mask=mask,other=0)
    dest=tl.load(dest_rows+i*DEST_INDEX_STRIDE,mask=mask,other=0)
    value=tl.load(src+source*SRC_STRIDE,mask=mask,other=0)
    tl.store(out+dest*OUT_STRIDE,value,mask=mask)

def copy_sampled_tokens(src,source_rows,dest_rows,out):
    assert src.ndim==2 and src.shape[1]>=1 and out.ndim==1
    assert source_rows.ndim==dest_rows.ndim==1
    assert src.dtype==out.dtype==torch.int32
    assert source_rows.dtype==dest_rows.dtype==torch.int64
    assert src.is_cuda and src.device==source_rows.device==dest_rows.device==out.device
    assert source_rows.numel()==dest_rows.numel()
    n=source_rows.numel()
    if n:
        _sampled_token_scatter[(triton.cdiv(n,256),)](
            src,source_rows,dest_rows,out,n,src.stride(0),source_rows.stride(0),
            dest_rows.stride(0),out.stride(0),BLOCK=256,num_warps=4,num_stages=1)

@triton.jit
def _hidden_rows(src,indices,out,N,H,
                 STRIDE0:tl.constexpr,STRIDE1:tl.constexpr,
                 INDEX_STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    source=tl.load(indices+row*INDEX_STRIDE)
    values=tl.load(src+source.to(tl.int64)*STRIDE0+col*STRIDE1,mask=col<H,other=0)
    tl.store(out+row.to(tl.int64)*H+col,values,mask=col<H)

def select_hidden_rows(src,indices):
    assert src.ndim==2 and indices.ndim==1
    assert src.dtype==torch.bfloat16 and indices.dtype in (torch.int32,torch.int64)
    assert src.is_cuda and indices.device==src.device
    n,h=indices.numel(),src.shape[1]
    out=torch.empty((n,h),device=src.device,dtype=src.dtype)
    if n and h:
        _hidden_rows[(n,triton.cdiv(h,256))](src,indices,out,n,h,src.stride(0),src.stride(1),
                                           indices.stride(0),BLOCK=256,num_warps=4,num_stages=1)
    return out
