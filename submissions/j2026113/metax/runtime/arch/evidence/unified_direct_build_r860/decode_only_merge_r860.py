import triton
import triton.language as tl
@triton.jit
def decode_only_merge(Part,Maxima,Sums,Cu,Out,SPLITS:tl.constexpr,SP:tl.constexpr):
    request=tl.program_id(0);head=tl.program_id(1)
    begin=tl.load(Cu+request);finish=tl.load(Cu+request+1)
    if finish-begin==1:
        s=tl.arange(0,SP);d=tl.arange(0,128)
        offset=((begin*2+head//8)*SPLITS+s)*8+head%8
        mx=tl.load(Maxima+offset,mask=s<SPLITS,other=float('-inf'))
        ls=tl.load(Sums+offset,mask=s<SPLITS,other=0.)
        maximum=tl.max(mx,0);maximum=tl.where(maximum==float('-inf'),0.,maximum)
        weight=tl.where(ls>0,tl.exp2(mx-maximum),0.)
        partial=tl.load(Part+offset[:,None]*128+d[None,:],mask=s[:,None]<SPLITS,other=0.)
        numerator=tl.sum(partial*weight[:,None],0);denominator=tl.sum(ls*weight,0)
        tl.store(Out+(begin*16+head)*128+d,numerator/tl.where(denominator>0,denominator,1.))
