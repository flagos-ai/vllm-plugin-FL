"""656-byte packed KV writer with HCU's unrounded FP32 group scales."""
import torch
import triton
import triton.language as tl

@triton.jit
def _write(KV,ROPE,CACHE,SLOTS,KS:tl.constexpr,RS:tl.constexpr,BS:tl.constexpr,TS:tl.constexpr,PAGE:tl.constexpr):
    t=tl.program_id(0);g=tl.program_id(1)
    slot=tl.load(SLOTS+t)
    if slot>=0:
        base=(slot//PAGE)*BS+(slot%PAGE)*TS
        d=tl.arange(0,128)
        v=tl.load(KV+t*KS+g*128+d).to(tl.float32)
        scale=tl.maximum(tl.max(tl.abs(v),0)/448.,1.1754943508222875e-38)
        q=(v/scale).to(tl.float8e4nv).to(tl.uint8,bitcast=True)
        tl.store(CACHE+base+g*128+d,q)
        sp=tl.cast(CACHE+base+512,tl.pointer_type(tl.float32))
        tl.store(sp+g,scale)
        if g==0:
            rd=tl.arange(0,64)
            r=tl.load(ROPE+t*RS+rd)
            rp=tl.cast(CACHE+base+528,tl.pointer_type(tl.bfloat16))
            tl.store(rp+rd,r)

@torch.library.custom_op('vllm_fl::hcu_ref_fp8_cache_write',mutates_args=('cache',))
def write_hcu_cache(kv:torch.Tensor,rope:torch.Tensor,cache:torch.Tensor,slots:torch.Tensor)->None:
    if kv.dtype!=torch.bfloat16 or rope.dtype!=torch.bfloat16 or cache.dtype!=torch.uint8:
        raise ValueError('Expected BF16 inputs and uint8 packed cache')
    if kv.shape[-1]!=512 or rope.shape[-1]!=64 or cache.ndim!=3 or cache.shape[-1]!=656:
        raise ValueError('Expected NoPE512/RoPE64 and paged 656-byte cache')
    if slots.numel()>kv.shape[0] or slots.numel()>rope.shape[0]:
        raise ValueError('Slot count exceeds available KV or RoPE rows')
    if slots.numel()==0:return
    _write[(slots.numel(),4)](kv,rope,cache,slots,kv.stride(0),rope.stride(0),cache.stride(0),cache.stride(1),cache.shape[1],num_warps=4)

@write_hcu_cache.register_fake
def _fake(kv,rope,cache,slots): return None
