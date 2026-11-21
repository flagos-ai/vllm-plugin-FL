"""Fused gather of source column zero followed by distinct-destination scatter.

This explicit fused API has the runner's original qualified contract: int32
source/output; int64 valid nonnegative indices; distinct destination positions;
source and output storage do not overlap. Repeated source rows are supported.
It does not materialize a gather temporary or replace registered index/scatter
operators. Invalid/aliased inputs are outside this specialized fused contract.
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


def gather_scatter(src,source_rows,dest_rows,out):
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

