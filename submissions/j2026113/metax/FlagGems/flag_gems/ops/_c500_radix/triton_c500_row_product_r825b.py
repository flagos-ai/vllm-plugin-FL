"""Express native64-bit row product as the original32-bit component operations.

This is exact two's-complement multiplication of sign-extended32-bit inputs:
low=a_lo*b_lo; high=umulhi(a_lo,b_lo)+a_lo*b_hi+a_hi*b_lo (mod2**32).
Native actual MIR uses SMUL_HI_U32, two sign corrections and low SMUL_I32.
No approximation, extra memory, alternative rank or recovery path.
"""
import triton
import triton.language as tl


@triton.jit
def original_row_product(Row,N):
    a=Row.to(tl.int32)
    b=N.to(tl.int32)
    au=a.to(tl.uint32,bitcast=True)
    bu=b.to(tl.uint32,bitcast=True)
    low=au*bu
    ah=(a>>31).to(tl.uint32,bitcast=True)
    bh=(b>>31).to(tl.uint32,bitcast=True)
    unsigned_high=((au.to(tl.uint64)*bu.to(tl.uint64))>>32).to(tl.uint32)
    high=unsigned_high+au*bh+ah*bu
    return (high.to(tl.uint64)<<32)|low.to(tl.uint64)
