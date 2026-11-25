# Copyright (c) 2026 BAAI. All rights reserved.

"""BF16 prefill MQA; sequential head accumulation preserves the reference order."""
import triton
import triton.language as tl

# Exact request lengths and the compact logits stride change every mixed step.
# Suppress both constexpr and implicit integer specialization for these values.
@triton.jit(do_not_specialize=["M", "N", "OS"])
def mqa_headloop(Q,K,W,KS,KE,O,M,N,OS,BQ:tl.constexpr,BN:tl.constexpr,BH:tl.constexpr,GROUP:tl.constexpr,QS:tl.constexpr,HS:tl.constexpr,KN:tl.constexpr,KD:tl.constexpr):
 pid=tl.program_id(0);nq=tl.cdiv(M,BQ);nn=tl.cdiv(N,BN)
 group=pid//(GROUP*nn);first=group*GROUP;gs=tl.minimum(nq-first,GROUP)
 qi=first+(pid%(GROUP*nn))%gs;ni=(pid%(GROUP*nn))//gs
 qs=qi*BQ+tl.arange(0,BQ);ns=ni*BN+tl.arange(0,BN)
 starts=tl.load(KS+qs,qs<M,other=N);ends=tl.load(KE+qs,qs<M,other=0)
 valid=(qs[:,None]<M)&(ns[None,:]<N)&(ns[None,:]>=starts[:,None])&(ns[None,:]<ends[:,None])
 active=ni*BN<tl.max(ends,0) and (ni+1)*BN>tl.min(starts,0)
 acc=tl.zeros((BQ,BN),tl.float32)
 if active:
  rh=tl.arange(0,BQ*BH);query=qi*BQ+rh//BH;head=rh%BH;dd=tl.arange(0,128)
  b=tl.load(K+ns[None,:]*KN+dd[:,None]*KD,ns[None,:]<N,other=0)
  for h in range(64//BH):
   rr=query*64+head+h*BH
   a=tl.load(Q+query[:,None]*QS+(head[:,None]+h*BH)*HS+dd[None,:],query[:,None]<M,other=0)
   dot=tl.dot(a,b)
   w=tl.load(W+rr,query<M,other=0)
   vals=tl.maximum(dot,0)*w[:,None]
   acc+=tl.sum(tl.reshape(vals,(BQ,BH,BN)),1)
 acc=tl.where(valid,acc,float('-inf'))
 tl.store(O+qs[:,None]*OS+ns[None,:],acc,(qs[:,None]<M)&(ns[None,:]<N))

def prefill_mqa(q, k, weights, starts, ends, out):
    m, n = q.shape[0], k.shape[0]
    mqa_headloop[(triton.cdiv(m, 64) * triton.cdiv(n, 128),)](
        q, k, weights, starts, ends, out, m, n, out.stride(0),
        64, 128, 1, 8, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        num_warps=4, num_stages=3, matrix_instr_nonkdim=16)
