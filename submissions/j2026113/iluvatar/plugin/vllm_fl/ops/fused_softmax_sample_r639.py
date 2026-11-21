# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Online normalization recurrence derived from FlagGems ops/softmax.py.
"""Native softmax recurrence with a sampling epilogue, fixed supplied noise."""
import triton
import triton.language as tl
from flag_gems.ops.softmax import prev_multiple_of


@triton.jit
def native_row_stats(input_ptr,N,TILE_N:tl.constexpr):
    m=tl.full([TILE_N],value=float('-inf'),dtype=tl.float32)
    z=tl.full([TILE_N],value=0.,dtype=tl.float32)
    previous_multiple=prev_multiple_of(N,TILE_N)
    for start_n in range(0,previous_multiple,TILE_N):
        n_offsets=start_n+tl.arange(0,TILE_N)
        inp=tl.load(input_ptr+n_offsets)
        m_new=tl.maximum(m,inp)
        all_neg_inf=m_new==float('-inf')
        z=tl.where(all_neg_inf,z,z*tl.exp(m-m_new)+tl.exp(inp-m_new))
        m=m_new
    for start_n in range(previous_multiple,N,TILE_N):
        n_offsets=start_n+tl.arange(0,TILE_N)
        mask=n_offsets<N
        inp=tl.load(input_ptr+n_offsets,mask=mask,other=-float('inf'))
        m_new=tl.maximum(m,inp)
        all_neg_inf=m_new==float('-inf')
        z=tl.where(all_neg_inf,z,z*tl.exp(m-m_new)+tl.exp(inp-m_new))
        m=m_new
    m_reduced=tl.max(m,0)
    z=tl.sum(z*tl.exp(m-m_reduced),0)
    return m_reduced,z


@triton.jit
def fused_row(X,Q,OUT,PROB_DEBUG,SCORE_DEBUG,M,N,TILE_N:tl.constexpr,DEBUG:tl.constexpr):
    row=tl.program_id(0)
    input_ptr=X+row*N;noise_ptr=Q+row*N
    m,z=native_row_stats(input_ptr,N,TILE_N)
    previous_multiple=prev_multiple_of(N,TILE_N)
    maximum=-float('inf');index=0
    # Preserve native softmax's reverse writeback traversal. Tie handling
    # explicitly chooses the earliest logical index across reverse tiles.
    for start_n in range(0,N,TILE_N):
        n_offsets=(previous_multiple-start_n)+tl.arange(0,TILE_N)
        valid=n_offsets<N
        inp=tl.load(input_ptr+n_offsets,valid,-float('inf'),eviction_policy='evict_first')
        probability=(tl.exp(inp-m)/z).to(tl.float32)
        noise=tl.load(noise_ptr+n_offsets,valid,1.)
        score=probability/noise
        if DEBUG:
            tl.store(PROB_DEBUG+row*N+n_offsets,probability,valid)
            tl.store(SCORE_DEBUG+row*N+n_offsets,score,valid)
        masked=tl.where(valid,score,-float('inf'))
        local_max,local_index=tl.max(masked,0,return_indices=True,return_indices_tie_break_left=True)
        candidate_index=(previous_multiple-start_n)+local_index
        update=(local_max>maximum)|((local_max==maximum)&(candidate_index<index))
        maximum=tl.where(update,local_max,maximum)
        index=tl.where(update,candidate_index,index)
    tl.store(OUT+row,index.to(tl.int64))


@triton.jit
def row_stats_kernel(X,STATS,N,TILE_N:tl.constexpr):
    row=tl.program_id(0)
    m,z=native_row_stats(X+row*N,N,TILE_N)
    tl.store(STATS+row*2,m);tl.store(STATS+row*2+1,z)


@triton.jit
def output_sample(X,Q,STATS,PART_VALUE,PART_INDEX,PROB_DEBUG,SCORE_DEBUG,
                  N:tl.constexpr,BLOCK:tl.constexpr,SPLITS:tl.constexpr,DEBUG:tl.constexpr):
    row=tl.program_id(0);part=tl.program_id(1)
    maximum=tl.load(STATS+row*2);normalizer=tl.load(STATS+row*2+1)
    tiles=tl.cdiv(N,BLOCK);per_part=tl.cdiv(tiles,SPLITS)
    start=part*per_part;stop=tl.minimum((part+1)*per_part,tiles)
    current=-float('inf');index=start*BLOCK
    for tile in range(start,stop):
        column=tile*BLOCK+tl.arange(0,BLOCK);valid=column<N
        logits=tl.load(X+row*N+column,valid,-float('inf'),eviction_policy='evict_first')
        probability=(tl.exp(logits-maximum)/normalizer).to(tl.float32)
        noise=tl.load(Q+row*N+column,valid,1.)
        score=probability/noise
        if DEBUG:
            tl.store(PROB_DEBUG+row*N+column,probability,valid)
            tl.store(SCORE_DEBUG+row*N+column,score,valid)
        masked=tl.where(valid,score,-float('inf'))
        local_max,local_index=tl.max(masked,0,return_indices=True,return_indices_tie_break_left=True)
        update=local_max>current
        current=tl.where(update,local_max,current)
        index=tl.where(update,tile*BLOCK+local_index,index)
    tl.store(PART_VALUE+row*SPLITS+part,current)
    tl.store(PART_INDEX+row*SPLITS+part,index.to(tl.int64))
