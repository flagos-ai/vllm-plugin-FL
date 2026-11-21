"""r1293 Group64 publication primitives shared by the probe and future reader."""
import hashlib
from pathlib import Path
import re

import triton
import triton.language as tl
from triton.language import core

from triton_reader_metadata_qscale_guard_r1073 import metadata_i32, page_i32

IDENTITY = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def builtin(name, values, builder, out=None):
    args = [core._to_tensor(x, builder) for x in values]
    dtype = args[1].type if out is None else core.block_type(out, args[0].type.shape)
    return core.tensor(builder.create_extern_elementwise(
        '', '', name, [x.handle for x in args], dtype.to_ir(builder), False), dtype)


@core.builtin
def publish_k(p, offset, owner, valid, rotate, _builder=None):
    return builtin('__m128_publish_k', (p, offset, owner, valid, rotate), _builder)


@core.builtin
def publish_v(p, offset, owner, valid, pos, length, _builder=None):
    return builtin('__m128_publish_v', (p, offset, owner, valid, pos, length), _builder)


@core.builtin
def shared_k(word, _builder=None):
    return builtin('__m128_shared_k', (word,), _builder, core.int32)


@core.builtin
def shared_v(packet, _builder=None):
    return builtin('__m128_shared_v', (packet,), _builder, core.uint64)


def lower_publication(ir):
    for name in ('publish_k', 'publish_v', 'shared_k', 'shared_v'):
        ir, n = re.subn(r'(?m)^declare[^\n]*@__m128_' + name + r'\([^\n]*\)[^\n]*$', '', ir)
        assert n == 1, name
    return ir + '''
@__m128_k = internal addrspace(3) global [8192 x i8] undef, align 16
@__m128_v = internal addrspace(3) global [16384 x i8] undef, align 16
define internal i32 @__m128_publish_k(ptr addrspace(1) %p,i32 %offset,i32 %owner,i32 %valid,i32 %rotate) alwaysinline nounwind convergent {
entry:
  %owns = icmp ne i32 %owner,0
  br i1 %owns,label %owned,label %done
owned:
  %take = icmp ne i32 %valid,0
  br i1 %take,label %read,label %store
read:
  %packet = load <4 x i32>,ptr addrspace(1) %p,align 16
  br label %store
store:
  %value = phi <4 x i32> [zeroinitializer,%owned],[%packet,%read]
  %rotated = shufflevector <4 x i32> %value,<4 x i32> poison,<4 x i32> <i32 2,i32 3,i32 0,i32 1>
  %upper = icmp ne i32 %rotate,0
  %arranged = select i1 %upper,<4 x i32> %rotated,<4 x i32> %value
  %dst = getelementptr inbounds i8,ptr addrspace(3) @__m128_k,i32 %offset
  store <4 x i32> %arranged,ptr addrspace(3) %dst,align 16
  br label %done
done:
  ret i32 %offset
}
define internal i32 @__m128_publish_v(ptr addrspace(1) %p,i32 %offset,i32 %owner,i32 %valid,i32 %pos,i32 %length) alwaysinline nounwind convergent {
entry:
  %owns = icmp ne i32 %owner,0
  br i1 %owns,label %owned,label %done
owned:
  %take = icmp ne i32 %valid,0
  br i1 %take,label %read,label %store
read:
  %packet = load <4 x i32>,ptr addrspace(1) %p,align 16
  %last = add i32 %pos,3
  %tail = icmp sge i32 %last,%length
  br i1 %tail,label %masked,label %store
masked:
  %p1 = add i32 %pos,1
  %p2 = add i32 %pos,2
  %low0 = icmp slt i32 %pos,%length
  %high0 = icmp slt i32 %p1,%length
  %low1 = icmp slt i32 %p2,%length
  %high1 = icmp slt i32 %last,%length
  %a = select i1 %low0,i32 65535,i32 0
  %b = select i1 %high0,i32 -65536,i32 0
  %c = select i1 %low1,i32 65535,i32 0
  %d = select i1 %high1,i32 -65536,i32 0
  %m0 = or i32 %a,%b
  %m1 = or i32 %c,%d
  %m2 = insertelement <2 x i32> poison,i32 %m0,i32 0
  %m3 = insertelement <2 x i32> %m2,i32 %m1,i32 1
  %mask = shufflevector <2 x i32> %m3,<2 x i32> poison,<4 x i32> <i32 0,i32 1,i32 0,i32 1>
  %clipped = and <4 x i32> %packet,%mask
  br label %store
store:
  %value = phi <4 x i32> [zeroinitializer,%owned],[%packet,%read],[%clipped,%masked]
  %dst = getelementptr inbounds i8,ptr addrspace(3) @__m128_v,i32 %offset
  store <4 x i32> %value,ptr addrspace(3) %dst,align 16
  br label %done
done:
  ret i32 %offset
}
define internal i32 @__m128_shared_k(i32 %word) alwaysinline nounwind convergent {
  %p = getelementptr inbounds i32,ptr addrspace(3) @__m128_k,i32 %word
  %v = load i32,ptr addrspace(3) %p,align 4
  ret i32 %v
}
define internal i64 @__m128_shared_v(i32 %packet) alwaysinline nounwind convergent {
  %p = getelementptr inbounds i64,ptr addrspace(3) @__m128_v,i32 %packet
  %v = load i64,ptr addrspace(3) %p,align 8
  ret i64 %v
}
'''


@triton.jit
def publish_page(K,V,physical,slot,tile,length,owner,lane,head,FULL:tl.constexpr):
    for half in tl.static_range(2):
        producer=lane+half*64
        row=producer//8
        col=(producer%8)*16
        valid=tl.full((256,),1,tl.int32) if FULL else (tile*16+row<length).to(tl.int32)
        publish_k(K+((physical.to(tl.int64)*16+row)*2+head)*32+col//4,
                  slot*2048+row*128+(col^((row&7)<<4)),owner,valid,(row>=8).to(tl.int32))
    for i in tl.static_range(4):
        packet=lane+i*64
        quad=(packet%4)^(((packet//4*2)>>3)&3)
        pos=tile*16+quad*4
        # Full groups reach the same packet operation with the tail predicate false.
        p=tl.full((256,),0,tl.int32) if FULL else pos
        valid=tl.full((256,),1,tl.int32) if FULL else (pos<length).to(tl.int32)
        publish_v(V+(physical.to(tl.int64)*2+head)*1024+packet*4,
                  slot*4096+packet*16,owner,valid,p,4 if FULL else length)


@triton.jit
def publish_group(K,V,Table,request,block,length,Width,tid,head):
    wave=tid//64
    lane=tid%64
    base=Table+request*Width+block*4
    if (block+1)*64<=length:
        page0=metadata_i32(base+0)
        page1=metadata_i32(base+1)
        page2=metadata_i32(base+2)
        page3=metadata_i32(base+3)
        owned_page=metadata_i32(base+wave)
        publish_page(K,V,owned_page,wave,block*4+wave,length,tl.full((256,),1,tl.int32),lane,head,True)
    else:
        tile=block*4+0
        page0=page_i32(base+0,tile*16<length)
        publish_page(K,V,page0,tl.full((256,),0,tl.int32),tile,length,(wave==0).to(tl.int32),lane,head,False)
        tile=block*4+1
        page1=page_i32(base+1,tile*16<length)
        publish_page(K,V,page1,tl.full((256,),1,tl.int32),tile,length,(wave==1).to(tl.int32),lane,head,False)
        tile=block*4+2
        page2=page_i32(base+2,tile*16<length)
        publish_page(K,V,page2,tl.full((256,),2,tl.int32),tile,length,(wave==2).to(tl.int32),lane,head,False)
        tile=block*4+3
        page3=page_i32(base+3,tile*16<length)
        publish_page(K,V,page3,tl.full((256,),3,tl.int32),tile,length,(wave==3).to(tl.int32),lane,head,False)
    tl.debug_barrier()
    return page0,page1,page2,page3
