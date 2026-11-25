# Copyright (c) 2026 BAAI. All rights reserved.
"""Batch tokens in the existing DV4 FP8/E8M0 + BF16 paged-cache layout."""
import os
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["OS0", "TS", "CS", "OFFSET"])
def gather(O,C,S,G,T,OS0,OS1:tl.constexpr,TS,CS,
           BLOCK:tl.constexpr,OFFSET,FNUZ:tl.constexpr,BT:tl.constexpr,WORKERS:tl.constexpr):
    req=tl.program_id(0);worker=tl.program_id(1)
    seq=tl.load(S+req);length=seq if G is None else tl.load(G+req)
    start=seq-length
    d=tl.arange(0,512);bd=tl.arange(0,64)
    for base in range(worker*BT,length,WORKERS*BT):
        i=base+tl.arange(0,BT);valid=i<length;pos=start+i
        block=tl.load(T+req*TS+pos//BLOCK,valid,other=0).to(tl.int64)
        at=block*CS+(pos%BLOCK)*576
        scale_at=block*CS+BLOCK*576+(pos%BLOCK)*8
        data=tl.load(C+at[:,None]+d[None,:],valid[:,None]&(d[None,:]<448),other=0)
        if FNUZ:x=data.to(tl.float8e4b8,bitcast=True).to(tl.float32)
        else:x=data.to(tl.float8e4nv,bitcast=True).to(tl.float32)
        scale=tl.load(C+scale_at[:,None]+d[None,:]//64,valid[:,None]&(d[None,:]<448),other=127)
        y=x*tl.exp2(scale.to(tl.float32)-127.)
        tl.store(O+req*OS0+(OFFSET+i[:,None])*OS1+d[None,:],y.to(tl.bfloat16),valid[:,None]&(d[None,:]<448))
        tail=(C+at[:,None]+448).to(tl.pointer_type(tl.bfloat16))+bd[None,:]
        value=tl.load(tail,valid[:,None],other=0)
        tl.store(O+req*OS0+(OFFSET+i[:,None])*OS1+448+bd[None,:],value,valid[:,None])


def _paged_cache_layout_supported(cache, block_size):
    """Pages are packed internally; padding/other layers may separate pages.

    vLLM exposes [pages, block_size, 584] views with a physical page stride.
    Requiring whole-tensor contiguity rejects those views. Also accept the
    equivalent flat-page ABI, without copying or flattening the cache.
    """
    if cache.dtype != torch.uint8 or cache.ndim not in (2, 3):
        return False
    if cache.ndim == 3:
        if (cache.shape[1:] != (block_size, 584)
                or cache.stride(1) != 584 or cache.stride(2) != 1):
            return False
    elif cache.shape[1] < block_size * 584 or cache.stride(1) != 1:
        return False
    return (cache.stride(0) >= block_size * 584
            and cache.stride(0) % 2 == 0 and cache.storage_offset() % 2 == 0)


def dequantize_and_gather_k_cache_hygon(out, k_cache, seq_lens, gather_lens,
                                      block_table, block_size, offset,
                                      use_fnuz=False):
    eligible = (
        os.getenv('VLLM_FL_HYGON_PREFILL_KV_GATHER', '1').lower() not in ('0', 'off', 'false')
        and not use_fnuz
        and out.is_cuda and out.dtype == torch.bfloat16 and out.ndim == 3
        and out.shape[-1] == 512 and out.stride(-1) == 1
        and _paged_cache_layout_supported(k_cache, block_size)
        and block_size in (2, 4, 16, 32, 64, 128, 256)
        and offset >= 0 and offset <= out.shape[1]
        and seq_lens.ndim == 1 and seq_lens.is_contiguous()
        and seq_lens.dtype in (torch.int32, torch.int64)
        and block_table.ndim == 2 and block_table.stride(1) == 1
        and block_table.dtype in (torch.int32, torch.int64)
        and seq_lens.shape[0] == out.shape[0] == block_table.shape[0]
        and (gather_lens is None or (
            gather_lens.shape == seq_lens.shape and gather_lens.is_contiguous()
            and gather_lens.dtype in (torch.int32, torch.int64)))
        and all(t.device == out.device for t in (k_cache, seq_lens, block_table))
        and (gather_lens is None or gather_lens.device == out.device)
    )
    if not eligible:
        from vllm.models.deepseek_v4.common.ops.cache_utils import dequantize_and_gather_k_cache
        return dequantize_and_gather_k_cache(out, k_cache, seq_lens, gather_lens,
                                           block_table, block_size, offset, use_fnuz)
    if seq_lens.numel() == 0:
        return
    tokens = 4 if block_size < 16 else 32
    gather[(seq_lens.numel(), 128)](
        out, k_cache, seq_lens, gather_lens, block_table,
        out.stride(0), out.stride(1), block_table.stride(0), k_cache.stride(0),
        block_size, offset, use_fnuz, tokens, 128, num_warps=4)


def patch_prefill_kv_gather():
    from vllm.models.deepseek_v4.amd import rocm
    rocm.dequantize_and_gather_k_cache = dequantize_and_gather_k_cache_hygon
