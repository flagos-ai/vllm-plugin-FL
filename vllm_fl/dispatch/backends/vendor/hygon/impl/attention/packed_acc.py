"""EXPERIMENT ONLY: requires power-of-two scales in GLM writer range; arbitrary scales unsupported."""
"""Experimental gfx936 fused sparse FP8 MLA with explicit Triton Gluon LDS reuse.

Packed KV is decoded only for the current tile. No full-cache BF16 temporary.
QK consumes decoded fragments, PV reloads the same fragments from shared memory.
"""
import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout, DotOperandLayout, SwizzledSharedLayout
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import mfma, buffer_load
from .baseline import _merge


@g.jit
def decode_fp8(raw):
    u=raw.to(gl.uint32)
    mag=u&127
    normal=((mag<<20)+0x3c000000).to(gl.float32,bitcast=True)
    # For exponent zero, 2*(2^-7 + mantissa*2^-10)-2^-6
    # is the exact E4M3 subnormal mantissa*2^-9.
    absolute=gl.where(mag<8,normal*2.0-0.015625,normal)
    bits=gl.where(mag==127,0x7fc00000,absolute.to(gl.uint32,bitcast=True))
    return (bits|((u&128)<<24)).to(gl.float32,bitcast=True)


@g.jit
def packed_mul(a,b):
    a,b=gl.broadcast(a,b)
    R:gl.constexpr=a.shape[0]
    C:gl.constexpr=a.shape[1]
    a0,a1=gl.split(gl.reshape(a,(R,C//2,2)))
    b0,b1=gl.split(gl.reshape(b,(R,C//2,2)))
    aa=a0.to(gl.uint32,bitcast=True).to(gl.uint64)|(a1.to(gl.uint32,bitcast=True).to(gl.uint64)<<32)
    bb=b0.to(gl.uint32,bitcast=True).to(gl.uint64)|(b1.to(gl.uint32,bitcast=True).to(gl.uint64)<<32)
    z=gl.inline_asm_elementwise("v_pk_mul_f32 $0, $1, $2",constraints="=v,v,v",args=[aa,bb],dtype=gl.uint64,is_pure=True,pack=1)
    z0=z.to(gl.uint32).to(gl.float32,bitcast=True)
    z1=(z>>32).to(gl.uint32).to(gl.float32,bitcast=True)
    return gl.reshape(gl.join(z0,z1),(R,C))


@g.jit
def packed_acc_scale(a, b):
    # MMA lanes own columns separated by four. Pair these register-local
    # values; adjacent logical columns would require inter-lane exchange.
    R:gl.constexpr=a.shape[0]
    C:gl.constexpr=a.shape[1]
    x=gl.reshape(gl.permute(gl.reshape(a,(R,C//8,2,4)),(0,1,3,2)),(R,C))
    _,b=gl.broadcast(a,b)
    b=gl.reshape(gl.permute(gl.reshape(b,(R,C//8,2,4)),(0,1,3,2)),(R,C))
    x=packed_mul(x,b)
    y=gl.reshape(gl.permute(gl.reshape(x,(R,C//8,4,2)),(0,1,3,2)),(R,C))
    return gl.convert_layout(y,a.type.layout)


@g.jit
def pv_fragment(desc,LOAD:gl.constexpr,DB:gl.constexpr,STAGED:gl.constexpr):
    if STAGED:
        return gl.convert_layout(desc.load(LOAD),DB)
    else:
        return desc.load(DB)


@g.jit
def packed_acc_attention(Q,CACHE,IX,OUT,PART,LSE,TOKENS:gl.constexpr,SCALE:gl.constexpr,
                    SPLITS:gl.constexpr,BN:gl.constexpr,NW:gl.constexpr,ORDER:gl.constexpr,TRANS:gl.constexpr,STAGED:gl.constexpr):
    t=gl.program_id(0)
    split=gl.program_id(1)
    MMA:gl.constexpr=AMDMFMALayout(3,[16,16,16],TRANS,[1,NW])
    LOAD:gl.constexpr=BlockedLayout([1,4],[4,16],[1,NW],[1,0])
    SHARED:gl.constexpr=SwizzledSharedLayout(4,1,16,[ORDER,1-ORDER])
    DA:gl.constexpr=DotOperandLayout(0,MMA,4)
    DB:gl.constexpr=DotOperandLayout(1,MMA,4)
    h=gl.arange(0,16,layout=SliceLayout(1,LOAD))
    hc=gl.arange(0,4,layout=SliceLayout(1,LOAD))
    d=gl.arange(0,128,layout=SliceLayout(0,LOAD))
    r=gl.arange(0,64,layout=SliceLayout(0,LOAD))
    n=gl.arange(0,BN,layout=SliceLayout(1,LOAD))
    hs=gl.arange(0,16,layout=SliceLayout(1,MMA))
    ds=gl.arange(0,128,layout=SliceLayout(0,MMA))
    qr=buffer_load(Q,t*2304+h[:,None]*576+512+r[None,:],h[:,None]<4,0)
    qr=gl.convert_layout(qr,DA)
    q0=buffer_load(Q,t*2304+hc[:,None]*576+0+d[None,:])
    q1=buffer_load(Q,t*2304+hc[:,None]*576+128+d[None,:])
    q2=buffer_load(Q,t*2304+hc[:,None]*576+256+d[None,:])
    q3=buffer_load(Q,t*2304+hc[:,None]*576+384+d[None,:])
    kv=gl.allocate_shared_memory(gl.bfloat16,[BN,256],SHARED)
    m=gl.full((16,),float('-inf'),gl.float32,SliceLayout(1,MMA))
    l=gl.full((16,),0,gl.float32,SliceLayout(1,MMA))
    a0=gl.full((16,128),0,gl.float32,MMA)
    a1=gl.full((16,128),0,gl.float32,MMA)
    a2=gl.full((16,128),0,gl.float32,MMA)
    a3=gl.full((16,128),0,gl.float32,MMA)
    SPAN:gl.constexpr=triton.cdiv(2048,SPLITS*BN)*BN
    for start in range(split*SPAN,(split+1)*SPAN,BN):
        slot=start+n
        ix=buffer_load(IX,t*2048+slot,slot<2048,-1)
        valid=(ix>=0)&(ix<TOKENS)&(slot<2048)
        if gl.sum(valid.to(gl.int32),0)>0:
            base=gl.maximum(ix,0)*656
            scores=gl.full((16,BN),0,gl.float32,MMA)
            for c in gl.static_range(4):
                raw=buffer_load(CACHE,base[:,None]+c*128+d[None,:],valid[:,None],0)
                scale=buffer_load(CACHE.to(gl.pointer_type(gl.float32)),base//4+128+c,valid,0)
                k=packed_mul(decode_fp8(raw),scale[:,None]).to(gl.bfloat16,fp_downcast_rounding="rtz")
                kv.slice((c%2)*128,128,1).store(k)
                if c==0:q=q0
                elif c==1:q=q1
                elif c==2:q=q2
                else:q=q3
                q=gl.gather(q,gl.broadcast((h%4)[:,None],gl.full((16,128),0,gl.int32,LOAD))[0],0)
                q=gl.where(h[:,None]<4,q,0)
                scores=mfma(gl.convert_layout(q,DA),gl.convert_layout(gl.permute(k,(1,0)),DB),scores)
                if c==1:
                    pv0=pv_fragment(kv.slice(0,128,1),LOAD,DB,STAGED)
                    pv1=pv_fragment(kv.slice(128,128,1),LOAD,DB,STAGED)
            kr=buffer_load(CACHE.to(gl.pointer_type(gl.bfloat16)),base[:,None]//2+264+r[None,:],valid[:,None],0)
            scores=mfma(qr,gl.convert_layout(gl.permute(kr,(1,0)),DB),scores)
            vs=gl.convert_layout(valid,SliceLayout(0,MMA))
            scores=gl.where(vs[None,:],scores*(SCALE*1.4426950408889634),float('-inf'))
            nm=gl.maximum(m,gl.max(scores,1))
            alpha=gl.exp2(m-nm)
            p=gl.exp2(scores-nm[:,None])
            pb=gl.convert_layout(p.to(gl.bfloat16),DA)
            a0=mfma(pb,pv0,packed_acc_scale(a0,alpha[:,None]))
            a1=mfma(pb,pv1,packed_acc_scale(a1,alpha[:,None]))
            a2=mfma(pb,pv_fragment(kv.slice(0,128,1),LOAD,DB,STAGED),packed_acc_scale(a2,alpha[:,None]))
            a3=mfma(pb,pv_fragment(kv.slice(128,128,1),LOAD,DB,STAGED),packed_acc_scale(a3,alpha[:,None]))
            l=l*alpha+gl.sum(p,1)
            m=nm
    inv=1/gl.where(l>0,l,1)
    if SPLITS==1:
        dest=OUT+t*2048+hs[:,None]*512+ds[None,:]
    else:
        dest=PART+((t*SPLITS+split)*4+hs[:,None])*512+ds[None,:]
        gl.store(LSE+(t*SPLITS+split)*4+hs,gl.where(l>0,m+gl.log2(l),float('-inf')),hs<4)
    gl.store(dest,a0*inv[:,None],hs[:,None]<4)
    gl.store(dest+128,a1*inv[:,None],hs[:,None]<4)
    gl.store(dest+256,a2*inv[:,None],hs[:,None]<4)
    gl.store(dest+384,a3*inv[:,None],hs[:,None]<4)


def sparse_mla(q,cache,indices,scale,*,splits=None,block_n=64,num_warps=4,shared_order=1,transposed=False,staged_pv=False):
    if q.ndim!=3 or q.shape[1:]!=(4,576) or q.dtype!=torch.bfloat16:
        raise ValueError('Expected Q[T,4,576] BF16')
    if cache.ndim!=3 or cache.shape[-1]!=656 or cache.dtype!=torch.uint8:
        raise ValueError('Expected packed uint8 KV[pages,page_size,656]')
    if indices.shape!=(q.shape[0],2048) or indices.dtype!=torch.int32:
        raise ValueError('Expected indices[T,2048] int32')
    if not all(x.is_cuda and x.is_contiguous() and x.device==q.device for x in (q,cache,indices)):
        raise ValueError('Inputs must be contiguous on one GPU')
    if any(x.numel()*x.element_size()>=2**31 for x in (q,cache,indices)):
        raise ValueError('32-bit buffer offsets require each input byte extent <2**31')
    if splits is None:splits=16 if len(q)<128 else 1
    if splits not in (1,2,4,8,16) or block_n not in (16,32,64) or num_warps not in (1,2,4):
        raise ValueError('Unsupported split/block/wave configuration')
    out=torch.empty((len(q),4,512),dtype=q.dtype,device=q.device)
    if len(q)==0:return out
    if splits>1:
        part=torch.empty((len(q),splits,4,512),device=q.device,dtype=torch.float32)
        lse=torch.empty((len(q),splits,4),device=q.device,dtype=torch.float32)
    else:part=out;lse=out
    with torch.cuda.device(q.device):
        packed_acc_attention[(len(q),splits)](q,cache,indices,out,part,lse,cache.shape[0]*cache.shape[1],float(scale),splits,block_n,num_warps,shared_order,transposed,staged_pv,num_warps=num_warps,num_stages=1)
        if splits>1:_merge[(len(q),4)](part,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
    return out
