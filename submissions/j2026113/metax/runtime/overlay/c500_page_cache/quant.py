"""Independent page16 GQA decode: direct INT8 QK, BF16 PV; no model integration."""
import torch
import triton
import triton.language as tl
from vllm_fl.dispatch.backends.flaggems.impl.quantization import _round_half_to_even
@triton.jit(do_not_specialize=[3])
def _quant_rows(X,Y,S,ROWS,BLOCK:tl.constexpr):
 r=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);d=tl.arange(0,128);x=tl.load(X+r[:,None]*128+d[None,:],mask=r[:,None]<ROWS,other=0).to(tl.float32);peak=tl.max(tl.abs(x),1);scale=tl.maximum(peak/127.,tl.full((BLOCK,),1.1754943508222875e-38,tl.float32)).to(tl.bfloat16);scale=tl.where(peak==0,1.,scale).to(tl.bfloat16);y=_round_half_to_even(tl.div_rn(x,scale[:,None].to(tl.float32)));y=tl.minimum(tl.maximum(y,-127.),127.).to(tl.int8);tl.store(Y+r[:,None]*128+d[None,:],y,mask=r[:,None]<ROWS);tl.store(S+r,scale,mask=r<ROWS)
def quantize(x):
 assert x.dtype==torch.bfloat16 and x.shape[-1]==128 and x.is_contiguous();y=torch.empty(x.shape,device=x.device,dtype=torch.int8);s=torch.empty(x.shape[:-1],device=x.device,dtype=torch.bfloat16);_quant_rows[(triton.cdiv(x.numel()//128,16),)](x,y,s,x.numel()//128,16,num_warps=4);return y,s
def quantize_into(x,y,s):_quant_rows[(triton.cdiv(x.numel()//128,16),)](x,y,s,x.numel()//128,16,num_warps=4)
@triton.jit
def _partial(Q,K,V,QS,KS,Lens,Table,P,Mx,Ls,COLS:tl.constexpr,SPLITS:tl.constexpr,MAXL:tl.constexpr,N:tl.constexpr,INTEGER:tl.constexpr):
 split=tl.program_id(0);g=tl.program_id(1);b=g//2;h=g%2;m=tl.arange(0,16);n=tl.arange(0,N);d=tl.arange(0,128);length=tl.load(Lens+b)
 q=tl.load(Q+(g*8+m[:,None])*128+d[None,:],mask=m[:,None]<8,other=0)
 if INTEGER:qscale=tl.load(QS+g*8+m,mask=m<8,other=1).to(tl.float32)
 mi=tl.full((16,),float('-inf'),tl.float32);li=tl.zeros((16,),tl.float32);acc=tl.zeros((16,128),tl.float32)
 tiles=tl.cdiv(MAXL,N);per=tl.cdiv(tiles,SPLITS);start=split*per;stop=tl.minimum((split+1)*per,tl.cdiv(length,N))
 for ti in range(start,stop):
  positions=ti*N+n;valid=positions<length;page=tl.load(Table+b*COLS+positions//16,mask=valid,other=0);row=(page*16+positions%16)*2+h
  kval=tl.load(K+row[None,:]*128+d[:,None],mask=valid[None,:],other=0)
  scores=tl.dot(q,kval).to(tl.float32)
  if INTEGER:
   kscale=tl.load(KS+row,mask=valid,other=1).to(tl.float32);scores*=qscale[:,None]*kscale[None,:]
  scores*=0.12751743084247367
  scores=tl.where(valid[None,:],scores,float('-inf'));newmax=tl.maximum(mi,tl.max(scores,1));alpha=tl.exp2(mi-newmax);prob=tl.exp2(scores-newmax[:,None]);li=li*alpha+tl.sum(prob,1);acc=acc*alpha[:,None];vv=tl.load(V+row[:,None]*128+d[None,:],mask=valid[:,None],other=0);acc=tl.dot(prob.to(tl.bfloat16),vv,acc);mi=newmax
 offset=(g*SPLITS+split)*8;tl.store(P+(offset+m[:,None])*128+d[None,:],acc,mask=m[:,None]<8);tl.store(Mx+offset+m,mi,mask=m<8);tl.store(Ls+offset+m,li,mask=m<8)
@triton.jit
def _combine(P,Mx,Ls,O,SPLITS:tl.constexpr,SP:tl.constexpr):
 g=tl.program_id(0);m=tl.program_id(1);s=tl.arange(0,SP);d=tl.arange(0,128);off=(g*SPLITS+s)*8+m;mx=tl.load(Mx+off,mask=s<SPLITS,other=float('-inf'));ls=tl.load(Ls+off,mask=s<SPLITS,other=0.);maxall=tl.max(mx,0);maxall=tl.where(maxall==float('-inf'),0.,maxall);weight=tl.where(ls>0,tl.exp2(mx-maxall),0.);psum=tl.load(P+off[:,None]*128+d[None,:],mask=s[:,None]<SPLITS,other=0.);num=tl.sum(psum*weight[:,None],0);den=tl.sum(ls*weight,0);out=num/tl.where(den>0,den,1.);tl.store(O+(g*8+m)*128+d,out)
def decode(q,k,v,qs,ks,lengths,table,*,max_len,splits=8,block_n=32,warps=2,integer=True,guards=False):
 B=q.shape[0];assert q.shape==(B,1,16,128) and q.is_contiguous();assert k.shape==v.shape and k.shape[1:]==(16,2,128) and k.is_contiguous() and v.is_contiguous();assert v.dtype==torch.bfloat16 and q.dtype==(torch.int8 if integer else torch.bfloat16) and k.dtype==q.dtype;assert lengths.shape==(B,) and lengths.dtype==torch.int32 and lengths.is_contiguous();assert table.shape[0]==B and table.dtype==torch.int32 and table.is_contiguous();assert 1<=max_len<=16*table.shape[1] and splits in [1,2,4,8,16,32,64];assert block_n in [16,32,64];checks=[]
 if integer:assert qs.shape==(B,1,16) and ks.shape==k.shape[:-1] and qs.is_contiguous() and ks.is_contiguous()
 def alloc(shape,dtype):
  if not guards:return torch.empty(shape,device=q.device,dtype=dtype)
  import math
  size=math.prod(shape);base=torch.full((size+128,),-123.,device=q.device,dtype=dtype);checks.append((base,size));return base[:size].view(shape)
 part=alloc((B*2,splits,8,128),torch.float32);mx=alloc((B*2,splits,8),torch.float32);ls=alloc((B*2,splits,8),torch.float32);out=alloc((B,1,16,128),torch.bfloat16)
 _partial[(splits,B*2)](q,k,v,qs,ks,lengths,table,part,mx,ls,table.shape[1],splits,max_len,block_n,integer,num_warps=warps,num_stages=1)
 _combine[(B*2,8)](part,mx,ls,out,splits,triton.next_power_of_2(splits),num_warps=4,num_stages=1)
 return (out,checks) if guards else out

