"""Experimental fused sparse MLA for GLM's packed 656-byte FP8 KV cache.

Independent implementation, not a binary-compatible vendor extension.
Q [T,4,576] BF16; cache [pages,page_size,656] uint8 contiguous;
indices [T,2048] int32 global token slots; negative/out-of-range slots ignored.
The 512 FP8 E4M3FN values use four FP32 scales; 64 BF16 RoPE values follow.
No dropout, backward, or separate causal mask (indices specify visible tokens).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode_e4m3fn(raw):
    # Normal E4M3FN values map exactly to BF16 by rebiasing the exponent.
    # Handle subnormals, signed zero and the two NaN encodings separately.
    u=raw.to(tl.uint16)
    mag=u&127
    bits=((u&128)<<8)|((mag<<4)+15360)
    normal=bits.to(tl.bfloat16,bitcast=True).to(tl.float32)
    sub=mag.to(tl.float32)*0.001953125
    sub=(sub.to(tl.uint32,bitcast=True)|((u.to(tl.uint32)&128)<<24)).to(tl.float32,bitcast=True)
    value=tl.where(mag<8,sub,normal)
    return tl.where(mag==127,float('nan'),value)


@triton.jit
def _load_k(CACHE,base,valid,offset):
    d=offset+tl.arange(0,128)
    raw=tl.load(CACHE+base[None,:]+d[:,None],valid[None,:],0)
    sf=tl.load((CACHE+base+512+4*(offset//128)).to(tl.pointer_type(tl.float32)),valid,0)
    return (_decode_e4m3fn(raw)*sf[None,:]).to(tl.bfloat16)


@triton.jit
def _attention(Q,CACHE,IX,OUT,PART,LSE, TOKENS:tl.constexpr,
               SCALE:tl.constexpr,SPLITS:tl.constexpr,BN:tl.constexpr):
    t=tl.program_id(0)
    split=tl.program_id(1)
    h=tl.arange(0,16)
    d=tl.arange(0,128)
    r=tl.arange(0,64)
    qr=tl.load(Q+t*2304+h[:,None]*576+512+r[None,:],h[:,None]<4,0)
    m=tl.full((16,),float('-inf'),tl.float32)
    l=tl.full((16,),0,tl.float32)
    a0=tl.full((16,128),0,tl.float32)
    a1=tl.full((16,128),0,tl.float32)
    a2=tl.full((16,128),0,tl.float32)
    a3=tl.full((16,128),0,tl.float32)
    SPAN:tl.constexpr=triton.cdiv(2048,SPLITS*BN)*BN
    for start in range(split*SPAN,(split+1)*SPAN,BN):
        slot=start+tl.arange(0,BN)
        ix=tl.load(IX+t*2048+slot,slot<2048,-1)
        valid=(ix>=0)&(ix<TOKENS)&(slot<2048)
        if tl.sum(valid.to(tl.int32),0)>0:
            base=tl.maximum(ix,0)*656
            scores=tl.full((16,BN),0,tl.float32)
            for chunk in range(4):
                q=tl.load(Q+t*2304+h[:,None]*576+chunk*128+d[None,:],h[:,None]<4,0)
                k=_load_k(CACHE,base,valid,chunk*128)
                scores=tl.dot(q,k,scores)
            kr=tl.load((CACHE+base[None,:]+528+2*r[:,None]).to(tl.pointer_type(tl.bfloat16)),valid[None,:],0)
            scores=tl.dot(qr,kr,scores)
            scores=tl.where(valid[None,:],scores*(SCALE*1.4426950408889634),float('-inf'))
            new_m=tl.maximum(m,tl.max(scores,1))
            alpha=tl.exp2(m-new_m)
            p=tl.exp2(scores-new_m[:,None])
            pb=p.to(tl.bfloat16)
            a0=a0*alpha[:,None]+tl.dot(pb,tl.trans(_load_k(CACHE,base,valid,0)))
            a1=a1*alpha[:,None]+tl.dot(pb,tl.trans(_load_k(CACHE,base,valid,128)))
            a2=a2*alpha[:,None]+tl.dot(pb,tl.trans(_load_k(CACHE,base,valid,256)))
            a3=a3*alpha[:,None]+tl.dot(pb,tl.trans(_load_k(CACHE,base,valid,384)))
            l=l*alpha+tl.sum(p,1)
            m=new_m
    inv=1/tl.where(l>0,l,1)
    if SPLITS==1:
        dest=OUT+t*2048+h[:,None]*512+d[None,:]
    else:
        dest=PART+((t*SPLITS+split)*4+h[:,None])*512+d[None,:]
        tl.store(LSE+(t*SPLITS+split)*4+h,tl.where(l>0,m+tl.log2(l),float('-inf')),h<4)
    tl.store(dest,a0*inv[:,None],h[:,None]<4)
    tl.store(dest+128,a1*inv[:,None],h[:,None]<4)
    tl.store(dest+256,a2*inv[:,None],h[:,None]<4)
    tl.store(dest+384,a3*inv[:,None],h[:,None]<4)


@triton.jit
def _merge(PART,LSE,OUT,SPLITS:tl.constexpr,BS:tl.constexpr):
    t=tl.program_id(0)
    h=tl.program_id(1)
    s=tl.arange(0,BS)
    d=tl.arange(0,512)
    logs=tl.load(LSE+(t*SPLITS+s)*4+h,s<SPLITS,float('-inf'))
    m=tl.max(logs,0)
    safe_m=tl.where(m==float('-inf'),0,m)
    w=tl.exp2(logs-safe_m)
    denom=tl.sum(w,0)
    values=tl.load(PART+((t*SPLITS+s[:,None])*4+h)*512+d[None,:],s[:,None]<SPLITS,0)
    out=tl.sum(values*w[:,None],0)/tl.where(denom>0,denom,1)
    tl.store(OUT+t*2048+h*512+d,out)


def sparse_mla(q,cache,indices,scale,*,splits=None,block_n=16,mmac_layout=-1,num_warps=None):
    if q.ndim!=3 or q.shape[1:]!=(4,576) or q.dtype!=torch.bfloat16:
        raise ValueError('Expected Q[T,4,576] BF16')
    if cache.ndim!=3 or cache.shape[-1]!=656 or cache.dtype!=torch.uint8:
        raise ValueError('Expected packed uint8 KV[pages,page_size,656]')
    if indices.shape!=(q.shape[0],2048) or indices.dtype!=torch.int32:
        raise ValueError('Expected indices[T,2048] int32')
    if not all(x.is_cuda and x.is_contiguous() and x.device==q.device for x in (q,cache,indices)):
        raise ValueError('Inputs must be contiguous on one GPU')
    if num_warps is None:num_warps=4 if q.shape[0]<128 else 1
    if splits is None:splits=4 if q.shape[0]<128 else 1
    if splits not in (1,2,4,8,16) or block_n not in (16,32,64):
        raise ValueError('Unsupported split or block size')
    out=torch.empty((q.shape[0],4,512),dtype=q.dtype,device=q.device)
    if q.shape[0]==0:return out
    if splits>1:
        part=torch.empty((q.shape[0],splits,4,512),device=q.device,dtype=torch.float32)
        lse=torch.empty((q.shape[0],splits,4),device=q.device,dtype=torch.float32)
    else:part=out;lse=out
    with torch.cuda.device(q.device):
        _attention[(q.shape[0],splits)](q,cache,indices,out,part,lse,cache.shape[0]*cache.shape[1],float(scale),splits,block_n,num_warps=num_warps,num_stages=1,mmac_layout_force=mmac_layout)
        if splits>1:_merge[(q.shape[0],4)](part,lse,out,splits,triton.next_power_of_2(splits),num_warps=4)
    return out
