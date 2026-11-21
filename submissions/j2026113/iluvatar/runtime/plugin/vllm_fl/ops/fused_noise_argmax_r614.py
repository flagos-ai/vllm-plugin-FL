# SPDX-License-Identifier: Apache-2.0
"""Fuse existing FP32 probability/noise division and stable row argmax.

Softmax, filtering and exponential-noise generation are outside this operator.
WRITE_SCORES preserves original inplace probability mutation when requested.
"""
import triton
import triton.language as tl


@triton.jit
def noise_argmax(P,Q,SCORE,OUT,PART_VALUE,PART_INDEX,N:tl.constexpr,
                 STRIDE_P:tl.constexpr,STRIDE_Q:tl.constexpr,
                 BLOCK:tl.constexpr,SPLITS:tl.constexpr,WRITE_SCORES:tl.constexpr):
    row=tl.program_id(0);part=tl.program_id(1)
    tile_count=tl.cdiv(N,BLOCK)
    per_part=tl.cdiv(tile_count,SPLITS)
    start=part*per_part
    stop=tl.minimum((part+1)*per_part,tile_count)
    offsets=tl.arange(0,BLOCK)
    maximum=-float('inf')
    index=start*BLOCK
    for tile in range(start,stop):
        column=tile*BLOCK+offsets;valid=column<N
        probability=tl.load(P+row*STRIDE_P+column,valid,0.)
        noise=tl.load(Q+row*STRIDE_Q+column,valid,1.)
        score=probability/noise
        if WRITE_SCORES:
            tl.store(SCORE+row*STRIDE_P+column,score,valid)
        score=tl.where(valid,score,-float('inf'))
        tile_maximum,tile_index=tl.max(score,0,return_indices=True,return_indices_tie_break_left=True)
        update=tile_maximum>maximum
        maximum=tl.where(update,tile_maximum,maximum)
        index=tl.where(update,tile*BLOCK+tile_index,index)
    if SPLITS==1:
        tl.store(OUT+row,index.to(tl.int64))
    else:
        tl.store(PART_VALUE+row*SPLITS+part,maximum)
        tl.store(PART_INDEX+row*SPLITS+part,index.to(tl.int64))


@triton.jit
def reduce_noise_argmax(PART_VALUE,PART_INDEX,OUT,SPLITS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);offset=tl.arange(0,BLOCK)
    values=tl.load(PART_VALUE+row*SPLITS+offset,offset<SPLITS,-float('inf'))
    indexes=tl.load(PART_INDEX+row*SPLITS+offset,offset<SPLITS,0x7fffffffffffffff)
    maximum=tl.max(values,0)
    answer=tl.min(tl.where(values==maximum,indexes,0x7fffffffffffffff),0)
    tl.store(OUT+row,answer)
