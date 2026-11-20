"""GLM/DeepSeek-V3.2 656-byte FP8 cache for the qualified gfx936 FlashMLA."""
import functools
import hashlib
import os
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _write_cache(KV, ROPE, CACHE, SLOTS, KV_STRIDE:tl.constexpr,
                 ROPE_STRIDE:tl.constexpr, BLOCK_STRIDE:tl.constexpr,
                 TOKEN_STRIDE:tl.constexpr, PAGE:tl.constexpr):
    token=tl.program_id(0)
    group=tl.program_id(1)
    slot=tl.load(SLOTS+token)
    if slot >= 0:
        base=(slot//PAGE)*BLOCK_STRIDE+(slot%PAGE)*TOKEN_STRIDE
        dims=tl.arange(0,128)
        values=tl.load(KV+token*KV_STRIDE+group*128+dims).to(tl.float32)
        scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(values),0)/448.,1.e-4))))
        quant=(values/scale).to(tl.float8e4nv).to(tl.uint8,bitcast=True)
        tl.store(CACHE+base+group*128+dims,quant)
        sp=tl.cast(CACHE+base+512,tl.pointer_type(tl.float32))
        tl.store(sp+group,scale)
        if group==0:
            dims_r=tl.arange(0,64)
            rope=tl.load(ROPE+token*ROPE_STRIDE+dims_r)
            rp=tl.cast(CACHE+base+528,tl.pointer_type(tl.bfloat16))
            tl.store(rp+dims_r,rope)


@torch.library.custom_op('vllm_fl::hygon_glm_fp8_cache_write',mutates_args=('cache',))
def write_fp8_cache(kv:torch.Tensor,rope:torch.Tensor,cache:torch.Tensor,
                    slots:torch.Tensor)->None:
    if kv.dtype!=torch.bfloat16 or rope.dtype!=torch.bfloat16 or cache.dtype!=torch.uint8:
        raise ValueError('FP8 packed writer requires BF16 inputs and uint8 cache')
    if kv.shape[-1]!=512 or rope.shape[-1]!=64 or cache.ndim!=3 or cache.shape[-1]!=656:
        raise ValueError('Expected GLM NoPE512/RoPE64 and paged 656-byte cache')
    if kv.stride(-1)!=1 or rope.stride(-1)!=1 or cache.stride(-1)!=1 or slots.ndim!=1:
        raise ValueError('Invalid KV strides or slot shape')
    if kv.shape[0]!=rope.shape[0] or kv.shape[0]!=slots.numel():
        raise ValueError(f'Token count mismatch: kv={tuple(kv.shape)}, rope={tuple(rope.shape)}, slots={tuple(slots.shape)}')
    if slots.numel()==0:return
    _write_cache[(slots.numel(),4)](kv,rope,cache,slots,kv.stride(0),rope.stride(0),
        cache.stride(0),cache.stride(1),cache.shape[1],num_warps=4)


@write_fp8_cache.register_fake
def _fake_write(kv,rope,cache,slots):
    return None


@functools.lru_cache(maxsize=1)
def qualified_interface():
    import flash_mla.cuda as binary
    import flash_mla.flash_mla_interface as interface
    for filename,expected in [
        (binary.__file__,'4af1d9c8eef36210a8ffedfcb836d9dd9412520edfa9948b05c35817dd1f3ebd'),
        (interface.__file__,'0c7e927224f28c87439b4493a8a6764e69221aef4277a6350c047a04d4648945')]:
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Unqualified FlashMLA artifact: '+filename)
    if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith('gfx936'):
        raise RuntimeError('GLM FP8 route qualified only on gfx936')
    return interface


# Only eager-owned metadata is retained. A graph capture miss uses the original
# per-call planner, so graph-pool allocations never escape into another graph.
# This cache is specific to the qualified sparse FP8 path: all length arguments
# are None and top-k remains 2048. Changing those semantics requires a new key.
_SCHEDULERS = {}
_MAX_SCHEDULERS = 128


def _scheduler_key(q, cache, indices):
    return (threading.get_ident(), q.device.index,
            tuple(q.shape), tuple(q.stride()), tuple(cache.shape),
            tuple(cache.stride()), tuple(indices.shape), tuple(indices.stride()))


@torch.library.custom_op('vllm_fl::hygon_glm_fp8_sparse_mla',mutates_args=())
def sparse_fp8_mla(q:torch.Tensor,cache:torch.Tensor,indices:torch.Tensor,
                   scale:float)->torch.Tensor:
    if q.ndim!=3 or q.shape[1:]!=(4,576) or q.dtype!=torch.bfloat16:
        raise ValueError('GLM FP8 path requires Q[T,4,576] BF16')
    if cache.ndim!=3 or cache.shape[-1]!=656 or cache.dtype!=torch.uint8 or not cache.is_contiguous():
        raise ValueError('GLM FP8 path requires contiguous paged uint8 cache[...,656]')
    if indices.shape!=(q.shape[0],2048) or indices.dtype!=torch.int32:
        raise ValueError('GLM FP8 path requires int32 indices[T,2048]')
    route=os.environ.get('VLLM_FL_GLM_FP8_MLA_IMPL','vendor')
    if route not in ('vendor','triton_grouped'):
        raise ValueError('Unknown FP8 MLA implementation: '+route)
    if route=='triton_grouped':
        # Scoped to this module's power-of-two-scale cache writer and gfx936.
        if not torch.cuda.get_device_properties(q.device).gcnArchName.startswith('gfx936'):
            raise RuntimeError('Grouped MLA qualified only on gfx936')
        from .grouped_fullvalid import sparse_mla
        return sparse_mla(q,cache,indices,scale)
    interface=qualified_interface()
    enabled=os.environ.get('VLLM_FL_GLM_FP8_REUSE_METADATA','0')=='1'
    key=_scheduler_key(q,cache,indices) if enabled else None
    entry=_SCHEDULERS.get(key) if enabled else None
    if entry is None:
        meta,splits=interface.get_mla_metadata()
    else:
        meta,ready=entry
        # Initialization may have happened on a different warmup stream.
        torch.cuda.current_stream(q.device).wait_event(ready)
        splits=None
    result=interface.flash_mla_with_kvcache(q.unsqueeze(1),cache.unsqueeze(2),
        None,None,512,meta,splits,softmax_scale=scale,is_fp8_kvcache=True,
        indices=indices.unsqueeze(1))[0]
    if (enabled and entry is None and len(_SCHEDULERS)<_MAX_SCHEDULERS
            and not torch.cuda.is_current_stream_capturing()):
        ready=torch.cuda.Event()
        ready.record(torch.cuda.current_stream(q.device))
        _SCHEDULERS[key]=(meta,ready)
    return result.squeeze(1)


@sparse_fp8_mla.register_fake
def _fake_sparse(q,cache,indices,scale):
    return q.new_empty((q.shape[0],4,512))
