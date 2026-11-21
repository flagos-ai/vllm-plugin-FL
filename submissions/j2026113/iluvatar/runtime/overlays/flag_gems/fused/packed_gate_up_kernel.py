# Copyright 2026 FlagOS Contributors
# Derived from the Apache-2.0 FlagGems native Iluvatar MM tile recurrence.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""Interleaved BF16 gate/up weights, single GEMM, pairwise output activation."""
import triton
import triton.language as tl


@triton.jit
def blockpaired_gate_up(X,W,Y,GATE_DEBUG,UP_DEBUG,M,N:tl.constexpr,K:tl.constexpr,
                      STRIDE_XM:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                      GROUP:tl.constexpr,DEBUG:tl.constexpr):
    pid=tl.program_id(0)
    grid_m=tl.cdiv(M,BM);grid_n=tl.cdiv(2*N,BN)
    if GROUP==1:
        tile_m=pid//grid_n;tile_n=pid%grid_n
    else:
        width=GROUP*grid_n;group=pid//width;first=group*GROUP
        size=tl.minimum(grid_m-first,GROUP)
        local=pid%width;tile_m=first+local%size;tile_n=local//size
    rows=tile_m*BM+tl.arange(0,BM)
    cols=tile_n*BN+tl.arange(0,BN)
    read_rows=tl.max_contiguous(tl.multiple_of(rows%M,BM),BM)
    read_cols=tl.max_contiguous(tl.multiple_of(cols%(2*N),BN),BN)
    kk=tl.arange(0,BK)
    a_pointer=X+read_rows[:,None]*STRIDE_XM+kk[None,:]
    b_pointer=W+read_cols[None,:]*K+kk[:,None]
    acc=tl.zeros((BM,BN),tl.float32)
    tl.static_assert(K%BK==0 and BN%2==0)
    for step in range(0,tl.cdiv(K,BK)):
        a=tl.load(a_pointer);b=tl.load(b_pointer)
        acc=tl.dot(a,b,acc,out_dtype=tl.float32,input_precision=None)
        a_pointer+=BK;b_pointer+=BK
    # Explicit projection rounding remains before the pointwise activation.
    bf16=acc.to(tl.bfloat16)
    tl.static_assert(BN%128==0)
    # Physical column bit6 pairs gate/up. For the qualified prefill4x4MMA
    # mapping it is register-owned, unlike the adjacent-column lane bit.
    groups=tl.reshape(bf16,(BM,BN//128,2,64))
    pairs=tl.reshape(tl.permute(groups,(0,1,3,2)),(BM,BN//2,2))
    gate_bf16,up_bf16=tl.split(pairs)
    gate=gate_bf16.to(tl.float32);up=up_bf16.to(tl.float32)
    sigmoid=tl.fdiv(1.0,1.0+tl.exp(-gate))
    activated=((gate*sigmoid)*up).to(tl.bfloat16)
    output_cols=tile_n*(BN//2)+tl.arange(0,BN//2)
    offsets=rows[:,None]*N+output_cols[None,:]
    valid=(rows[:,None]<M)&(output_cols[None,:]<N)
    tl.store(Y+offsets,activated,mask=valid)
    if DEBUG:
        tl.store(GATE_DEBUG+offsets,gate_bf16,mask=valid)
        tl.store(UP_DEBUG+offsets,up_bf16,mask=valid)
