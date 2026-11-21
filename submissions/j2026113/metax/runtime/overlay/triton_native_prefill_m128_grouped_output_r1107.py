"""Experimental r1293 M128 reader; never selected by model dispatch."""
import hashlib
from pathlib import Path
import re

import triton
import triton.language as tl
from triton.language import core

from triton_c500_mma_r881 import mma_i8, mma_bf16, lower_mma
from triton_c500_storage_r704b import physical_thread_id, install_storage_lowering
from triton_m128_publication_r1087 import publish_group, shared_k, shared_v, lower_publication
from triton_reader_math_r899c import scale_pair, native_exp2, native_bpermute, lower_math
from triton_reader_metadata_qscale_guard_r1073 import (
    metadata_i32, scale_u16, page_i32, query_word, query_scale,
    lower_metadata, lower_query_word, lower_query_scale,
)


def unary_float(name, value, builder):
    t=core._to_tensor(value,builder)
    return core.tensor(builder.create_extern_elementwise(
        '', '', name, [t.handle], t.type.to_ir(builder), True),t.type)


@core.builtin
def reciprocal(x,_builder=None):
    return unary_float('llvm.mxc.rcp.f32',x,_builder)


@core.builtin
def maxnum(a,b,_builder=None):
    x=core._to_tensor(a,_builder)
    y=core._to_tensor(b,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', 'llvm.maxnum.f32', [x.handle,y.handle], x.type.to_ir(_builder), True),x.type)


@core.builtin
def write_packet(p,active,norm,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,_builder=None):
    args=[core._to_tensor(x,_builder) for x in (p,active,norm,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31)]
    args[1]=args[1].to(core.int32,_builder=_builder)
    dtype=args[1].type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__m128_write_packet', [x.handle for x in args],
        dtype.to_ir(_builder), False),dtype)


@triton.jit
def b16(x):
    bits=x.to(tl.uint32,bitcast=True)
    return (((bits+0x7fff+((bits>>16)&1))>>16)&0xffff)


@triton.jit
def wave_xor(x,lane,offset:tl.constexpr):
    return native_bpermute(((lane^offset)<<2).to(tl.int32),
                           x.to(tl.int32,bitcast=True)).to(tl.float32,bitcast=True)


@triton.jit(do_not_specialize=[11,12])
def m128_reader(Q,K,V,QS,KS,Visible,Table,Owners,Bases,Counts,Out,Capacity,Width):
    g=tl.program_id(0)
    task=Capacity-1-g//16
    qhead=g%16
    head=qhead//8
    count=metadata_i32(Counts+task)
    if count>0:
        request=metadata_i32(Owners+task)
        qbase=metadata_i32(Bases+task)
        visible=metadata_i32(Visible+task)
        tid=physical_thread_id(tl.arange(0,256))
        row=tid%16
        wave=tid//64
        lane=tid%64
        kg=lane//16
        length=visible+count-1
        offset0=wave*32+0+row
        query0=qbase+offset0
        active0=offset0<count
        qidx0=query0.to(tl.int64)*16+qhead
        qr0_0=query_word(Q+qidx0*32+kg,active0,0)
        qr0_1=query_word(Q+qidx0*32+kg,active0,4)
        qr0_2=query_word(Q+qidx0*32+kg,active0,8)
        qr0_3=query_word(Q+qidx0*32+kg,active0,12)
        qr0_4=query_word(Q+qidx0*32+kg,active0,16)
        qr0_5=query_word(Q+qidx0*32+kg,active0,20)
        qr0_6=query_word(Q+qidx0*32+kg,active0,24)
        qr0_7=query_word(Q+qidx0*32+kg,active0,28)
        qs0=query_scale(QS+query0*16+qhead,active0).to(tl.float32,bitcast=True)
        acc0_0_0=tl.full((256,),0.,tl.float32)
        acc0_0_1=tl.full((256,),0.,tl.float32)
        acc0_0_2=tl.full((256,),0.,tl.float32)
        acc0_0_3=tl.full((256,),0.,tl.float32)
        acc0_1_0=tl.full((256,),0.,tl.float32)
        acc0_1_1=tl.full((256,),0.,tl.float32)
        acc0_1_2=tl.full((256,),0.,tl.float32)
        acc0_1_3=tl.full((256,),0.,tl.float32)
        acc0_2_0=tl.full((256,),0.,tl.float32)
        acc0_2_1=tl.full((256,),0.,tl.float32)
        acc0_2_2=tl.full((256,),0.,tl.float32)
        acc0_2_3=tl.full((256,),0.,tl.float32)
        acc0_3_0=tl.full((256,),0.,tl.float32)
        acc0_3_1=tl.full((256,),0.,tl.float32)
        acc0_3_2=tl.full((256,),0.,tl.float32)
        acc0_3_3=tl.full((256,),0.,tl.float32)
        acc0_4_0=tl.full((256,),0.,tl.float32)
        acc0_4_1=tl.full((256,),0.,tl.float32)
        acc0_4_2=tl.full((256,),0.,tl.float32)
        acc0_4_3=tl.full((256,),0.,tl.float32)
        acc0_5_0=tl.full((256,),0.,tl.float32)
        acc0_5_1=tl.full((256,),0.,tl.float32)
        acc0_5_2=tl.full((256,),0.,tl.float32)
        acc0_5_3=tl.full((256,),0.,tl.float32)
        acc0_6_0=tl.full((256,),0.,tl.float32)
        acc0_6_1=tl.full((256,),0.,tl.float32)
        acc0_6_2=tl.full((256,),0.,tl.float32)
        acc0_6_3=tl.full((256,),0.,tl.float32)
        acc0_7_0=tl.full((256,),0.,tl.float32)
        acc0_7_1=tl.full((256,),0.,tl.float32)
        acc0_7_2=tl.full((256,),0.,tl.float32)
        acc0_7_3=tl.full((256,),0.,tl.float32)
        max0=tl.full((256,),float("-inf"),tl.float32)
        norm0=tl.full((256,),0.,tl.float32)
        offset1=wave*32+16+row
        query1=qbase+offset1
        active1=offset1<count
        qidx1=query1.to(tl.int64)*16+qhead
        qr1_0=query_word(Q+qidx1*32+kg,active1,0)
        qr1_1=query_word(Q+qidx1*32+kg,active1,4)
        qr1_2=query_word(Q+qidx1*32+kg,active1,8)
        qr1_3=query_word(Q+qidx1*32+kg,active1,12)
        qr1_4=query_word(Q+qidx1*32+kg,active1,16)
        qr1_5=query_word(Q+qidx1*32+kg,active1,20)
        qr1_6=query_word(Q+qidx1*32+kg,active1,24)
        qr1_7=query_word(Q+qidx1*32+kg,active1,28)
        qs1=query_scale(QS+query1*16+qhead,active1).to(tl.float32,bitcast=True)
        acc1_0_0=tl.full((256,),0.,tl.float32)
        acc1_0_1=tl.full((256,),0.,tl.float32)
        acc1_0_2=tl.full((256,),0.,tl.float32)
        acc1_0_3=tl.full((256,),0.,tl.float32)
        acc1_1_0=tl.full((256,),0.,tl.float32)
        acc1_1_1=tl.full((256,),0.,tl.float32)
        acc1_1_2=tl.full((256,),0.,tl.float32)
        acc1_1_3=tl.full((256,),0.,tl.float32)
        acc1_2_0=tl.full((256,),0.,tl.float32)
        acc1_2_1=tl.full((256,),0.,tl.float32)
        acc1_2_2=tl.full((256,),0.,tl.float32)
        acc1_2_3=tl.full((256,),0.,tl.float32)
        acc1_3_0=tl.full((256,),0.,tl.float32)
        acc1_3_1=tl.full((256,),0.,tl.float32)
        acc1_3_2=tl.full((256,),0.,tl.float32)
        acc1_3_3=tl.full((256,),0.,tl.float32)
        acc1_4_0=tl.full((256,),0.,tl.float32)
        acc1_4_1=tl.full((256,),0.,tl.float32)
        acc1_4_2=tl.full((256,),0.,tl.float32)
        acc1_4_3=tl.full((256,),0.,tl.float32)
        acc1_5_0=tl.full((256,),0.,tl.float32)
        acc1_5_1=tl.full((256,),0.,tl.float32)
        acc1_5_2=tl.full((256,),0.,tl.float32)
        acc1_5_3=tl.full((256,),0.,tl.float32)
        acc1_6_0=tl.full((256,),0.,tl.float32)
        acc1_6_1=tl.full((256,),0.,tl.float32)
        acc1_6_2=tl.full((256,),0.,tl.float32)
        acc1_6_3=tl.full((256,),0.,tl.float32)
        acc1_7_0=tl.full((256,),0.,tl.float32)
        acc1_7_1=tl.full((256,),0.,tl.float32)
        acc1_7_2=tl.full((256,),0.,tl.float32)
        acc1_7_3=tl.full((256,),0.,tl.float32)
        max1=tl.full((256,),float("-inf"),tl.float32)
        norm1=tl.full((256,),0.,tl.float32)
        for block in range(tl.cdiv(length,64)):
            page0,page1,page2,page3=publish_group(K,V,Table,request,block,length,Width,tid,head)
            pos=block*64+0
            c0_0=tl.full((256,),0,tl.int32)
            c0_1=tl.full((256,),0,tl.int32)
            c0_2=tl.full((256,),0,tl.int32)
            c0_3=tl.full((256,),0,tl.int32)
            c1_0=tl.full((256,),0,tl.int32)
            c1_1=tl.full((256,),0,tl.int32)
            c1_2=tl.full((256,),0,tl.int32)
            c1_3=tl.full((256,),0,tl.int32)
            kword=shared_k(0+row*32+((0+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_0,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_0,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(0+row*32+((4+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_1,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_1,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(0+row*32+((8+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_2,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_2,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(0+row*32+((12+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_3,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_3,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(0+row*32+((16+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_4,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_4,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(0+row*32+((20+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_5,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_5,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(0+row*32+((24+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_6,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_6,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(0+row*32+((28+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_7,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_7,kword,c1_0,c1_1,c1_2,c1_3)
            kscale=scale_u16(KS+page0.to(tl.int64)*2+head).to(tl.bfloat16,bitcast=True).to(tl.float32)
            scale=qs0*kscale
            s0_0_0=tl.where(active0&(pos+kg*4+0<visible+offset0),(c0_0.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_0_1=tl.where(active0&(pos+kg*4+1<visible+offset0),(c0_1.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_0_2=tl.where(active0&(pos+kg*4+2<visible+offset0),(c0_2.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_0_3=tl.where(active0&(pos+kg*4+3<visible+offset0),(c0_3.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            scale=qs1*kscale
            s1_0_0=tl.where(active1&(pos+kg*4+0<visible+offset1),(c1_0.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_0_1=tl.where(active1&(pos+kg*4+1<visible+offset1),(c1_1.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_0_2=tl.where(active1&(pos+kg*4+2<visible+offset1),(c1_2.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_0_3=tl.where(active1&(pos+kg*4+3<visible+offset1),(c1_3.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            pos=block*64+16
            c0_0=tl.full((256,),0,tl.int32)
            c0_1=tl.full((256,),0,tl.int32)
            c0_2=tl.full((256,),0,tl.int32)
            c0_3=tl.full((256,),0,tl.int32)
            c1_0=tl.full((256,),0,tl.int32)
            c1_1=tl.full((256,),0,tl.int32)
            c1_2=tl.full((256,),0,tl.int32)
            c1_3=tl.full((256,),0,tl.int32)
            kword=shared_k(512+row*32+((0+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_0,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_0,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(512+row*32+((4+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_1,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_1,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(512+row*32+((8+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_2,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_2,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(512+row*32+((12+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_3,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_3,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(512+row*32+((16+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_4,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_4,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(512+row*32+((20+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_5,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_5,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(512+row*32+((24+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_6,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_6,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(512+row*32+((28+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_7,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_7,kword,c1_0,c1_1,c1_2,c1_3)
            kscale=scale_u16(KS+page1.to(tl.int64)*2+head).to(tl.bfloat16,bitcast=True).to(tl.float32)
            scale=qs0*kscale
            s0_1_0=tl.where(active0&(pos+kg*4+0<visible+offset0),(c0_0.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_1_1=tl.where(active0&(pos+kg*4+1<visible+offset0),(c0_1.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_1_2=tl.where(active0&(pos+kg*4+2<visible+offset0),(c0_2.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_1_3=tl.where(active0&(pos+kg*4+3<visible+offset0),(c0_3.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            scale=qs1*kscale
            s1_1_0=tl.where(active1&(pos+kg*4+0<visible+offset1),(c1_0.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_1_1=tl.where(active1&(pos+kg*4+1<visible+offset1),(c1_1.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_1_2=tl.where(active1&(pos+kg*4+2<visible+offset1),(c1_2.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_1_3=tl.where(active1&(pos+kg*4+3<visible+offset1),(c1_3.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            pos=block*64+32
            c0_0=tl.full((256,),0,tl.int32)
            c0_1=tl.full((256,),0,tl.int32)
            c0_2=tl.full((256,),0,tl.int32)
            c0_3=tl.full((256,),0,tl.int32)
            c1_0=tl.full((256,),0,tl.int32)
            c1_1=tl.full((256,),0,tl.int32)
            c1_2=tl.full((256,),0,tl.int32)
            c1_3=tl.full((256,),0,tl.int32)
            kword=shared_k(1024+row*32+((0+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_0,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_0,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1024+row*32+((4+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_1,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_1,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1024+row*32+((8+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_2,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_2,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1024+row*32+((12+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_3,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_3,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1024+row*32+((16+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_4,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_4,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1024+row*32+((20+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_5,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_5,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1024+row*32+((24+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_6,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_6,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1024+row*32+((28+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_7,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_7,kword,c1_0,c1_1,c1_2,c1_3)
            kscale=scale_u16(KS+page2.to(tl.int64)*2+head).to(tl.bfloat16,bitcast=True).to(tl.float32)
            scale=qs0*kscale
            s0_2_0=tl.where(active0&(pos+kg*4+0<visible+offset0),(c0_0.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_2_1=tl.where(active0&(pos+kg*4+1<visible+offset0),(c0_1.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_2_2=tl.where(active0&(pos+kg*4+2<visible+offset0),(c0_2.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_2_3=tl.where(active0&(pos+kg*4+3<visible+offset0),(c0_3.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            scale=qs1*kscale
            s1_2_0=tl.where(active1&(pos+kg*4+0<visible+offset1),(c1_0.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_2_1=tl.where(active1&(pos+kg*4+1<visible+offset1),(c1_1.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_2_2=tl.where(active1&(pos+kg*4+2<visible+offset1),(c1_2.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_2_3=tl.where(active1&(pos+kg*4+3<visible+offset1),(c1_3.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            pos=block*64+48
            c0_0=tl.full((256,),0,tl.int32)
            c0_1=tl.full((256,),0,tl.int32)
            c0_2=tl.full((256,),0,tl.int32)
            c0_3=tl.full((256,),0,tl.int32)
            c1_0=tl.full((256,),0,tl.int32)
            c1_1=tl.full((256,),0,tl.int32)
            c1_2=tl.full((256,),0,tl.int32)
            c1_3=tl.full((256,),0,tl.int32)
            kword=shared_k(1536+row*32+((0+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_0,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_0,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1536+row*32+((4+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_1,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_1,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1536+row*32+((8+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_2,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_2,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1536+row*32+((12+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_3,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_3,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1536+row*32+((16+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_4,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_4,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1536+row*32+((20+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_5,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_5,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1536+row*32+((24+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_6,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_6,kword,c1_0,c1_1,c1_2,c1_3)
            kword=shared_k(1536+row*32+((28+kg)^((row&7)<<2)^((row>>3)<<1)))
            c0_0,c0_1,c0_2,c0_3=mma_i8(qr0_7,kword,c0_0,c0_1,c0_2,c0_3)
            c1_0,c1_1,c1_2,c1_3=mma_i8(qr1_7,kword,c1_0,c1_1,c1_2,c1_3)
            kscale=scale_u16(KS+page3.to(tl.int64)*2+head).to(tl.bfloat16,bitcast=True).to(tl.float32)
            scale=qs0*kscale
            s0_3_0=tl.where(active0&(pos+kg*4+0<visible+offset0),(c0_0.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_3_1=tl.where(active0&(pos+kg*4+1<visible+offset0),(c0_1.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_3_2=tl.where(active0&(pos+kg*4+2<visible+offset0),(c0_2.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s0_3_3=tl.where(active0&(pos+kg*4+3<visible+offset0),(c0_3.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            scale=qs1*kscale
            s1_3_0=tl.where(active1&(pos+kg*4+0<visible+offset1),(c1_0.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_3_1=tl.where(active1&(pos+kg*4+1<visible+offset1),(c1_1.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_3_2=tl.where(active1&(pos+kg*4+2<visible+offset1),(c1_2.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            s1_3_3=tl.where(active1&(pos+kg*4+3<visible+offset1),(c1_3.to(tl.float32)*scale)*0.12751743084247367,float("-inf"))
            m=tl.full((256,),float("-inf"),tl.float32)
            m=maxnum(m,s0_0_0)
            m=maxnum(m,s0_0_1)
            m=maxnum(m,s0_0_2)
            m=maxnum(m,s0_0_3)
            m=maxnum(m,s0_1_0)
            m=maxnum(m,s0_1_1)
            m=maxnum(m,s0_1_2)
            m=maxnum(m,s0_1_3)
            m=maxnum(m,s0_2_0)
            m=maxnum(m,s0_2_1)
            m=maxnum(m,s0_2_2)
            m=maxnum(m,s0_2_3)
            m=maxnum(m,s0_3_0)
            m=maxnum(m,s0_3_1)
            m=maxnum(m,s0_3_2)
            m=maxnum(m,s0_3_3)
            m=maxnum(m,wave_xor(m,lane,16))
            m=maxnum(m,wave_xor(m,lane,32))
            next0=maxnum(max0,m)
            next0=tl.where(next0==float("-inf"),0.,next0)
            alpha0=tl.where(max0==float("-inf"),0.,native_exp2(max0-next0))
            sum0=tl.full((256,),0.,tl.float32)
            acc0_0_0,acc0_0_1=scale_pair(acc0_0_0,acc0_0_1,alpha0)
            acc0_0_2,acc0_0_3=scale_pair(acc0_0_2,acc0_0_3,alpha0)
            acc0_1_0,acc0_1_1=scale_pair(acc0_1_0,acc0_1_1,alpha0)
            acc0_1_2,acc0_1_3=scale_pair(acc0_1_2,acc0_1_3,alpha0)
            acc0_2_0,acc0_2_1=scale_pair(acc0_2_0,acc0_2_1,alpha0)
            acc0_2_2,acc0_2_3=scale_pair(acc0_2_2,acc0_2_3,alpha0)
            acc0_3_0,acc0_3_1=scale_pair(acc0_3_0,acc0_3_1,alpha0)
            acc0_3_2,acc0_3_3=scale_pair(acc0_3_2,acc0_3_3,alpha0)
            acc0_4_0,acc0_4_1=scale_pair(acc0_4_0,acc0_4_1,alpha0)
            acc0_4_2,acc0_4_3=scale_pair(acc0_4_2,acc0_4_3,alpha0)
            acc0_5_0,acc0_5_1=scale_pair(acc0_5_0,acc0_5_1,alpha0)
            acc0_5_2,acc0_5_3=scale_pair(acc0_5_2,acc0_5_3,alpha0)
            acc0_6_0,acc0_6_1=scale_pair(acc0_6_0,acc0_6_1,alpha0)
            acc0_6_2,acc0_6_3=scale_pair(acc0_6_2,acc0_6_3,alpha0)
            acc0_7_0,acc0_7_1=scale_pair(acc0_7_0,acc0_7_1,alpha0)
            acc0_7_2,acc0_7_3=scale_pair(acc0_7_2,acc0_7_3,alpha0)
            m=tl.full((256,),float("-inf"),tl.float32)
            m=maxnum(m,s1_0_0)
            m=maxnum(m,s1_0_1)
            m=maxnum(m,s1_0_2)
            m=maxnum(m,s1_0_3)
            m=maxnum(m,s1_1_0)
            m=maxnum(m,s1_1_1)
            m=maxnum(m,s1_1_2)
            m=maxnum(m,s1_1_3)
            m=maxnum(m,s1_2_0)
            m=maxnum(m,s1_2_1)
            m=maxnum(m,s1_2_2)
            m=maxnum(m,s1_2_3)
            m=maxnum(m,s1_3_0)
            m=maxnum(m,s1_3_1)
            m=maxnum(m,s1_3_2)
            m=maxnum(m,s1_3_3)
            m=maxnum(m,wave_xor(m,lane,16))
            m=maxnum(m,wave_xor(m,lane,32))
            next1=maxnum(max1,m)
            next1=tl.where(next1==float("-inf"),0.,next1)
            alpha1=tl.where(max1==float("-inf"),0.,native_exp2(max1-next1))
            sum1=tl.full((256,),0.,tl.float32)
            acc1_0_0,acc1_0_1=scale_pair(acc1_0_0,acc1_0_1,alpha1)
            acc1_0_2,acc1_0_3=scale_pair(acc1_0_2,acc1_0_3,alpha1)
            acc1_1_0,acc1_1_1=scale_pair(acc1_1_0,acc1_1_1,alpha1)
            acc1_1_2,acc1_1_3=scale_pair(acc1_1_2,acc1_1_3,alpha1)
            acc1_2_0,acc1_2_1=scale_pair(acc1_2_0,acc1_2_1,alpha1)
            acc1_2_2,acc1_2_3=scale_pair(acc1_2_2,acc1_2_3,alpha1)
            acc1_3_0,acc1_3_1=scale_pair(acc1_3_0,acc1_3_1,alpha1)
            acc1_3_2,acc1_3_3=scale_pair(acc1_3_2,acc1_3_3,alpha1)
            acc1_4_0,acc1_4_1=scale_pair(acc1_4_0,acc1_4_1,alpha1)
            acc1_4_2,acc1_4_3=scale_pair(acc1_4_2,acc1_4_3,alpha1)
            acc1_5_0,acc1_5_1=scale_pair(acc1_5_0,acc1_5_1,alpha1)
            acc1_5_2,acc1_5_3=scale_pair(acc1_5_2,acc1_5_3,alpha1)
            acc1_6_0,acc1_6_1=scale_pair(acc1_6_0,acc1_6_1,alpha1)
            acc1_6_2,acc1_6_3=scale_pair(acc1_6_2,acc1_6_3,alpha1)
            acc1_7_0,acc1_7_1=scale_pair(acc1_7_0,acc1_7_1,alpha1)
            acc1_7_2,acc1_7_3=scale_pair(acc1_7_2,acc1_7_3,alpha1)
            p0=native_exp2(s0_0_0-next0)
            p1=native_exp2(s0_0_1-next0)
            p2=native_exp2(s0_0_2-next0)
            p3=native_exp2(s0_0_3-next0)
            sum0=sum0+((p0+p1)+(p2+p3))
            p0_01=(b16(p0)|(b16(p1)<<16)).to(tl.int32,bitcast=True)
            p0_23=(b16(p2)|(b16(p3)<<16)).to(tl.int32,bitcast=True)
            p0=native_exp2(s1_0_0-next1)
            p1=native_exp2(s1_0_1-next1)
            p2=native_exp2(s1_0_2-next1)
            p3=native_exp2(s1_0_3-next1)
            sum1=sum1+((p0+p1)+(p2+p3))
            p1_01=(b16(p0)|(b16(p1)<<16)).to(tl.int32,bitcast=True)
            p1_23=(b16(p2)|(b16(p3)<<16)).to(tl.int32,bitcast=True)
            channel=row+0
            v=shared_v(0+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3)
            acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3)
            channel=row+16
            v=shared_v(0+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3)
            acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3)
            channel=row+32
            v=shared_v(0+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3)
            acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3)
            channel=row+48
            v=shared_v(0+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3)
            acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3)
            channel=row+64
            v=shared_v(0+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3)
            acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3)
            channel=row+80
            v=shared_v(0+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3)
            acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3)
            channel=row+96
            v=shared_v(0+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3)
            acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3)
            channel=row+112
            v=shared_v(0+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3)
            acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3)
            p0=native_exp2(s0_1_0-next0)
            p1=native_exp2(s0_1_1-next0)
            p2=native_exp2(s0_1_2-next0)
            p3=native_exp2(s0_1_3-next0)
            sum0=sum0+((p0+p1)+(p2+p3))
            p0_01=(b16(p0)|(b16(p1)<<16)).to(tl.int32,bitcast=True)
            p0_23=(b16(p2)|(b16(p3)<<16)).to(tl.int32,bitcast=True)
            p0=native_exp2(s1_1_0-next1)
            p1=native_exp2(s1_1_1-next1)
            p2=native_exp2(s1_1_2-next1)
            p3=native_exp2(s1_1_3-next1)
            sum1=sum1+((p0+p1)+(p2+p3))
            p1_01=(b16(p0)|(b16(p1)<<16)).to(tl.int32,bitcast=True)
            p1_23=(b16(p2)|(b16(p3)<<16)).to(tl.int32,bitcast=True)
            channel=row+0
            v=shared_v(512+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3)
            acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3)
            channel=row+16
            v=shared_v(512+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3)
            acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3)
            channel=row+32
            v=shared_v(512+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3)
            acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3)
            channel=row+48
            v=shared_v(512+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3)
            acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3)
            channel=row+64
            v=shared_v(512+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3)
            acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3)
            channel=row+80
            v=shared_v(512+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3)
            acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3)
            channel=row+96
            v=shared_v(512+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3)
            acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3)
            channel=row+112
            v=shared_v(512+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3)
            acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3)
            p0=native_exp2(s0_2_0-next0)
            p1=native_exp2(s0_2_1-next0)
            p2=native_exp2(s0_2_2-next0)
            p3=native_exp2(s0_2_3-next0)
            sum0=sum0+((p0+p1)+(p2+p3))
            p0_01=(b16(p0)|(b16(p1)<<16)).to(tl.int32,bitcast=True)
            p0_23=(b16(p2)|(b16(p3)<<16)).to(tl.int32,bitcast=True)
            p0=native_exp2(s1_2_0-next1)
            p1=native_exp2(s1_2_1-next1)
            p2=native_exp2(s1_2_2-next1)
            p3=native_exp2(s1_2_3-next1)
            sum1=sum1+((p0+p1)+(p2+p3))
            p1_01=(b16(p0)|(b16(p1)<<16)).to(tl.int32,bitcast=True)
            p1_23=(b16(p2)|(b16(p3)<<16)).to(tl.int32,bitcast=True)
            channel=row+0
            v=shared_v(1024+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3)
            acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3)
            channel=row+16
            v=shared_v(1024+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3)
            acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3)
            channel=row+32
            v=shared_v(1024+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3)
            acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3)
            channel=row+48
            v=shared_v(1024+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3)
            acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3)
            channel=row+64
            v=shared_v(1024+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3)
            acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3)
            channel=row+80
            v=shared_v(1024+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3)
            acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3)
            channel=row+96
            v=shared_v(1024+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3)
            acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3)
            channel=row+112
            v=shared_v(1024+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3)
            acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3)
            p0=native_exp2(s0_3_0-next0)
            p1=native_exp2(s0_3_1-next0)
            p2=native_exp2(s0_3_2-next0)
            p3=native_exp2(s0_3_3-next0)
            sum0=sum0+((p0+p1)+(p2+p3))
            p0_01=(b16(p0)|(b16(p1)<<16)).to(tl.int32,bitcast=True)
            p0_23=(b16(p2)|(b16(p3)<<16)).to(tl.int32,bitcast=True)
            p0=native_exp2(s1_3_0-next1)
            p1=native_exp2(s1_3_1-next1)
            p2=native_exp2(s1_3_2-next1)
            p3=native_exp2(s1_3_3-next1)
            sum1=sum1+((p0+p1)+(p2+p3))
            p1_01=(b16(p0)|(b16(p1)<<16)).to(tl.int32,bitcast=True)
            p1_23=(b16(p2)|(b16(p3)<<16)).to(tl.int32,bitcast=True)
            channel=row+0
            v=shared_v(1536+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3)
            acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3)
            channel=row+16
            v=shared_v(1536+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3)
            acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3)
            channel=row+32
            v=shared_v(1536+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3)
            acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3)
            channel=row+48
            v=shared_v(1536+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3)
            acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3)
            channel=row+64
            v=shared_v(1536+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3)
            acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3)
            channel=row+80
            v=shared_v(1536+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3)
            acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3)
            channel=row+96
            v=shared_v(1536+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3)
            acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3)
            channel=row+112
            v=shared_v(1536+(channel>>1)*8+(kg^((channel>>3)&3))*2+(channel&1))
            vlo=v.to(tl.uint32).to(tl.int32,bitcast=True)
            vhi=(v>>32).to(tl.uint32).to(tl.int32,bitcast=True)
            acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3=mma_bf16(p0_01,p0_23,vlo,vhi,acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3)
            acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3=mma_bf16(p1_01,p1_23,vlo,vhi,acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3)
            tl.debug_barrier()
            sum0=sum0+wave_xor(sum0,lane,16)
            sum0=sum0+wave_xor(sum0,lane,32)
            norm0=tl.fma(norm0,alpha0,sum0)
            max0=next0
            sum1=sum1+wave_xor(sum1,lane,16)
            sum1=sum1+wave_xor(sum1,lane,32)
            norm1=tl.fma(norm1,alpha1,sum1)
            max1=next1
        output_base=Out+(qbase.to(tl.int64)*16+qhead)*64
        output_lane=(wave*32+row)*1024+kg*2
        write_packet(output_base+output_lane+0,active0,norm0,acc0_0_0,acc0_0_1,acc0_0_2,acc0_0_3,acc0_1_0,acc0_1_1,acc0_1_2,acc0_1_3,acc0_2_0,acc0_2_1,acc0_2_2,acc0_2_3,acc0_3_0,acc0_3_1,acc0_3_2,acc0_3_3,acc0_4_0,acc0_4_1,acc0_4_2,acc0_4_3,acc0_5_0,acc0_5_1,acc0_5_2,acc0_5_3,acc0_6_0,acc0_6_1,acc0_6_2,acc0_6_3,acc0_7_0,acc0_7_1,acc0_7_2,acc0_7_3)
        write_packet(output_base+output_lane+16384,active1,norm1,acc1_0_0,acc1_0_1,acc1_0_2,acc1_0_3,acc1_1_0,acc1_1_1,acc1_1_2,acc1_1_3,acc1_2_0,acc1_2_1,acc1_2_2,acc1_2_3,acc1_3_0,acc1_3_1,acc1_3_2,acc1_3_3,acc1_4_0,acc1_4_1,acc1_4_2,acc1_4_3,acc1_5_0,acc1_5_1,acc1_5_2,acc1_5_3,acc1_6_0,acc1_6_1,acc1_6_2,acc1_6_3,acc1_7_0,acc1_7_1,acc1_7_2,acc1_7_3)


def lower_reader(ir):
    anchor='"metaxgpu-flat-work-group-size"="1, 256"'
    assert ir.count(anchor)==1
    ir=ir.replace(anchor,anchor+' "metaxgpu-max-block-size"="256"')
    ir,n=re.subn(r'(?m)^declare[^\n]*@__m128_write_packet\([^\n]*\)[^\n]*$','',ir)
    assert n==1
    ir += """
declare float @llvm.mxc.rcp.f32(float)
define internal i32 @__m128_write_packet(ptr addrspace(1) %p,i32 %active,float %norm,float %x0,float %x1,float %x2,float %x3,float %x4,float %x5,float %x6,float %x7,float %x8,float %x9,float %x10,float %x11,float %x12,float %x13,float %x14,float %x15,float %x16,float %x17,float %x18,float %x19,float %x20,float %x21,float %x22,float %x23,float %x24,float %x25,float %x26,float %x27,float %x28,float %x29,float %x30,float %x31) alwaysinline nounwind convergent {
entry:
  %take = icmp ne i32 %active,0
  br i1 %take,label %write,label %done
write:
  %positive = fcmp ogt float %norm,0.0
  %denom = select i1 %positive,float %norm,float 1.0
  %rcp = call float @llvm.mxc.rcp.f32(float %denom)
  %scaled0 = fmul float %x0,%rcp
  %raw0 = bitcast float %scaled0 to i32
  %high0 = lshr i32 %raw0,16
  %tie0 = and i32 %high0,1
  %biased0 = add i32 %raw0,32767
  %rounded0 = add i32 %biased0,%tie0
  %bf0 = lshr i32 %rounded0,16
  %wide0 = zext i32 %bf0 to i64
  %bits0 = shl i64 %wide0,0
  %scaled1 = fmul float %x1,%rcp
  %raw1 = bitcast float %scaled1 to i32
  %high1 = lshr i32 %raw1,16
  %tie1 = and i32 %high1,1
  %biased1 = add i32 %raw1,32767
  %rounded1 = add i32 %biased1,%tie1
  %bf1 = lshr i32 %rounded1,16
  %wide1 = zext i32 %bf1 to i64
  %bits1 = shl i64 %wide1,16
  %scaled2 = fmul float %x2,%rcp
  %raw2 = bitcast float %scaled2 to i32
  %high2 = lshr i32 %raw2,16
  %tie2 = and i32 %high2,1
  %biased2 = add i32 %raw2,32767
  %rounded2 = add i32 %biased2,%tie2
  %bf2 = lshr i32 %rounded2,16
  %wide2 = zext i32 %bf2 to i64
  %bits2 = shl i64 %wide2,32
  %scaled3 = fmul float %x3,%rcp
  %raw3 = bitcast float %scaled3 to i32
  %high3 = lshr i32 %raw3,16
  %tie3 = and i32 %high3,1
  %biased3 = add i32 %raw3,32767
  %rounded3 = add i32 %biased3,%tie3
  %bf3 = lshr i32 %rounded3,16
  %wide3 = zext i32 %bf3 to i64
  %bits3 = shl i64 %wide3,48
  %a0 = or i64 %bits0,%bits1
  %b0 = or i64 %bits2,%bits3
  %pack0 = or i64 %a0,%b0
  %dst0 = getelementptr i32,ptr addrspace(1) %p,i32 0
  store i64 %pack0,ptr addrspace(1) %dst0,align 8
  %scaled4 = fmul float %x4,%rcp
  %raw4 = bitcast float %scaled4 to i32
  %high4 = lshr i32 %raw4,16
  %tie4 = and i32 %high4,1
  %biased4 = add i32 %raw4,32767
  %rounded4 = add i32 %biased4,%tie4
  %bf4 = lshr i32 %rounded4,16
  %wide4 = zext i32 %bf4 to i64
  %bits4 = shl i64 %wide4,0
  %scaled5 = fmul float %x5,%rcp
  %raw5 = bitcast float %scaled5 to i32
  %high5 = lshr i32 %raw5,16
  %tie5 = and i32 %high5,1
  %biased5 = add i32 %raw5,32767
  %rounded5 = add i32 %biased5,%tie5
  %bf5 = lshr i32 %rounded5,16
  %wide5 = zext i32 %bf5 to i64
  %bits5 = shl i64 %wide5,16
  %scaled6 = fmul float %x6,%rcp
  %raw6 = bitcast float %scaled6 to i32
  %high6 = lshr i32 %raw6,16
  %tie6 = and i32 %high6,1
  %biased6 = add i32 %raw6,32767
  %rounded6 = add i32 %biased6,%tie6
  %bf6 = lshr i32 %rounded6,16
  %wide6 = zext i32 %bf6 to i64
  %bits6 = shl i64 %wide6,32
  %scaled7 = fmul float %x7,%rcp
  %raw7 = bitcast float %scaled7 to i32
  %high7 = lshr i32 %raw7,16
  %tie7 = and i32 %high7,1
  %biased7 = add i32 %raw7,32767
  %rounded7 = add i32 %biased7,%tie7
  %bf7 = lshr i32 %rounded7,16
  %wide7 = zext i32 %bf7 to i64
  %bits7 = shl i64 %wide7,48
  %a1 = or i64 %bits4,%bits5
  %b1 = or i64 %bits6,%bits7
  %pack1 = or i64 %a1,%b1
  %dst1 = getelementptr i32,ptr addrspace(1) %p,i32 8
  store i64 %pack1,ptr addrspace(1) %dst1,align 8
  %scaled8 = fmul float %x8,%rcp
  %raw8 = bitcast float %scaled8 to i32
  %high8 = lshr i32 %raw8,16
  %tie8 = and i32 %high8,1
  %biased8 = add i32 %raw8,32767
  %rounded8 = add i32 %biased8,%tie8
  %bf8 = lshr i32 %rounded8,16
  %wide8 = zext i32 %bf8 to i64
  %bits8 = shl i64 %wide8,0
  %scaled9 = fmul float %x9,%rcp
  %raw9 = bitcast float %scaled9 to i32
  %high9 = lshr i32 %raw9,16
  %tie9 = and i32 %high9,1
  %biased9 = add i32 %raw9,32767
  %rounded9 = add i32 %biased9,%tie9
  %bf9 = lshr i32 %rounded9,16
  %wide9 = zext i32 %bf9 to i64
  %bits9 = shl i64 %wide9,16
  %scaled10 = fmul float %x10,%rcp
  %raw10 = bitcast float %scaled10 to i32
  %high10 = lshr i32 %raw10,16
  %tie10 = and i32 %high10,1
  %biased10 = add i32 %raw10,32767
  %rounded10 = add i32 %biased10,%tie10
  %bf10 = lshr i32 %rounded10,16
  %wide10 = zext i32 %bf10 to i64
  %bits10 = shl i64 %wide10,32
  %scaled11 = fmul float %x11,%rcp
  %raw11 = bitcast float %scaled11 to i32
  %high11 = lshr i32 %raw11,16
  %tie11 = and i32 %high11,1
  %biased11 = add i32 %raw11,32767
  %rounded11 = add i32 %biased11,%tie11
  %bf11 = lshr i32 %rounded11,16
  %wide11 = zext i32 %bf11 to i64
  %bits11 = shl i64 %wide11,48
  %a2 = or i64 %bits8,%bits9
  %b2 = or i64 %bits10,%bits11
  %pack2 = or i64 %a2,%b2
  %dst2 = getelementptr i32,ptr addrspace(1) %p,i32 16
  store i64 %pack2,ptr addrspace(1) %dst2,align 8
  %scaled12 = fmul float %x12,%rcp
  %raw12 = bitcast float %scaled12 to i32
  %high12 = lshr i32 %raw12,16
  %tie12 = and i32 %high12,1
  %biased12 = add i32 %raw12,32767
  %rounded12 = add i32 %biased12,%tie12
  %bf12 = lshr i32 %rounded12,16
  %wide12 = zext i32 %bf12 to i64
  %bits12 = shl i64 %wide12,0
  %scaled13 = fmul float %x13,%rcp
  %raw13 = bitcast float %scaled13 to i32
  %high13 = lshr i32 %raw13,16
  %tie13 = and i32 %high13,1
  %biased13 = add i32 %raw13,32767
  %rounded13 = add i32 %biased13,%tie13
  %bf13 = lshr i32 %rounded13,16
  %wide13 = zext i32 %bf13 to i64
  %bits13 = shl i64 %wide13,16
  %scaled14 = fmul float %x14,%rcp
  %raw14 = bitcast float %scaled14 to i32
  %high14 = lshr i32 %raw14,16
  %tie14 = and i32 %high14,1
  %biased14 = add i32 %raw14,32767
  %rounded14 = add i32 %biased14,%tie14
  %bf14 = lshr i32 %rounded14,16
  %wide14 = zext i32 %bf14 to i64
  %bits14 = shl i64 %wide14,32
  %scaled15 = fmul float %x15,%rcp
  %raw15 = bitcast float %scaled15 to i32
  %high15 = lshr i32 %raw15,16
  %tie15 = and i32 %high15,1
  %biased15 = add i32 %raw15,32767
  %rounded15 = add i32 %biased15,%tie15
  %bf15 = lshr i32 %rounded15,16
  %wide15 = zext i32 %bf15 to i64
  %bits15 = shl i64 %wide15,48
  %a3 = or i64 %bits12,%bits13
  %b3 = or i64 %bits14,%bits15
  %pack3 = or i64 %a3,%b3
  %dst3 = getelementptr i32,ptr addrspace(1) %p,i32 24
  store i64 %pack3,ptr addrspace(1) %dst3,align 8
  %scaled16 = fmul float %x16,%rcp
  %raw16 = bitcast float %scaled16 to i32
  %high16 = lshr i32 %raw16,16
  %tie16 = and i32 %high16,1
  %biased16 = add i32 %raw16,32767
  %rounded16 = add i32 %biased16,%tie16
  %bf16 = lshr i32 %rounded16,16
  %wide16 = zext i32 %bf16 to i64
  %bits16 = shl i64 %wide16,0
  %scaled17 = fmul float %x17,%rcp
  %raw17 = bitcast float %scaled17 to i32
  %high17 = lshr i32 %raw17,16
  %tie17 = and i32 %high17,1
  %biased17 = add i32 %raw17,32767
  %rounded17 = add i32 %biased17,%tie17
  %bf17 = lshr i32 %rounded17,16
  %wide17 = zext i32 %bf17 to i64
  %bits17 = shl i64 %wide17,16
  %scaled18 = fmul float %x18,%rcp
  %raw18 = bitcast float %scaled18 to i32
  %high18 = lshr i32 %raw18,16
  %tie18 = and i32 %high18,1
  %biased18 = add i32 %raw18,32767
  %rounded18 = add i32 %biased18,%tie18
  %bf18 = lshr i32 %rounded18,16
  %wide18 = zext i32 %bf18 to i64
  %bits18 = shl i64 %wide18,32
  %scaled19 = fmul float %x19,%rcp
  %raw19 = bitcast float %scaled19 to i32
  %high19 = lshr i32 %raw19,16
  %tie19 = and i32 %high19,1
  %biased19 = add i32 %raw19,32767
  %rounded19 = add i32 %biased19,%tie19
  %bf19 = lshr i32 %rounded19,16
  %wide19 = zext i32 %bf19 to i64
  %bits19 = shl i64 %wide19,48
  %a4 = or i64 %bits16,%bits17
  %b4 = or i64 %bits18,%bits19
  %pack4 = or i64 %a4,%b4
  %dst4 = getelementptr i32,ptr addrspace(1) %p,i32 32
  store i64 %pack4,ptr addrspace(1) %dst4,align 8
  %scaled20 = fmul float %x20,%rcp
  %raw20 = bitcast float %scaled20 to i32
  %high20 = lshr i32 %raw20,16
  %tie20 = and i32 %high20,1
  %biased20 = add i32 %raw20,32767
  %rounded20 = add i32 %biased20,%tie20
  %bf20 = lshr i32 %rounded20,16
  %wide20 = zext i32 %bf20 to i64
  %bits20 = shl i64 %wide20,0
  %scaled21 = fmul float %x21,%rcp
  %raw21 = bitcast float %scaled21 to i32
  %high21 = lshr i32 %raw21,16
  %tie21 = and i32 %high21,1
  %biased21 = add i32 %raw21,32767
  %rounded21 = add i32 %biased21,%tie21
  %bf21 = lshr i32 %rounded21,16
  %wide21 = zext i32 %bf21 to i64
  %bits21 = shl i64 %wide21,16
  %scaled22 = fmul float %x22,%rcp
  %raw22 = bitcast float %scaled22 to i32
  %high22 = lshr i32 %raw22,16
  %tie22 = and i32 %high22,1
  %biased22 = add i32 %raw22,32767
  %rounded22 = add i32 %biased22,%tie22
  %bf22 = lshr i32 %rounded22,16
  %wide22 = zext i32 %bf22 to i64
  %bits22 = shl i64 %wide22,32
  %scaled23 = fmul float %x23,%rcp
  %raw23 = bitcast float %scaled23 to i32
  %high23 = lshr i32 %raw23,16
  %tie23 = and i32 %high23,1
  %biased23 = add i32 %raw23,32767
  %rounded23 = add i32 %biased23,%tie23
  %bf23 = lshr i32 %rounded23,16
  %wide23 = zext i32 %bf23 to i64
  %bits23 = shl i64 %wide23,48
  %a5 = or i64 %bits20,%bits21
  %b5 = or i64 %bits22,%bits23
  %pack5 = or i64 %a5,%b5
  %dst5 = getelementptr i32,ptr addrspace(1) %p,i32 40
  store i64 %pack5,ptr addrspace(1) %dst5,align 8
  %scaled24 = fmul float %x24,%rcp
  %raw24 = bitcast float %scaled24 to i32
  %high24 = lshr i32 %raw24,16
  %tie24 = and i32 %high24,1
  %biased24 = add i32 %raw24,32767
  %rounded24 = add i32 %biased24,%tie24
  %bf24 = lshr i32 %rounded24,16
  %wide24 = zext i32 %bf24 to i64
  %bits24 = shl i64 %wide24,0
  %scaled25 = fmul float %x25,%rcp
  %raw25 = bitcast float %scaled25 to i32
  %high25 = lshr i32 %raw25,16
  %tie25 = and i32 %high25,1
  %biased25 = add i32 %raw25,32767
  %rounded25 = add i32 %biased25,%tie25
  %bf25 = lshr i32 %rounded25,16
  %wide25 = zext i32 %bf25 to i64
  %bits25 = shl i64 %wide25,16
  %scaled26 = fmul float %x26,%rcp
  %raw26 = bitcast float %scaled26 to i32
  %high26 = lshr i32 %raw26,16
  %tie26 = and i32 %high26,1
  %biased26 = add i32 %raw26,32767
  %rounded26 = add i32 %biased26,%tie26
  %bf26 = lshr i32 %rounded26,16
  %wide26 = zext i32 %bf26 to i64
  %bits26 = shl i64 %wide26,32
  %scaled27 = fmul float %x27,%rcp
  %raw27 = bitcast float %scaled27 to i32
  %high27 = lshr i32 %raw27,16
  %tie27 = and i32 %high27,1
  %biased27 = add i32 %raw27,32767
  %rounded27 = add i32 %biased27,%tie27
  %bf27 = lshr i32 %rounded27,16
  %wide27 = zext i32 %bf27 to i64
  %bits27 = shl i64 %wide27,48
  %a6 = or i64 %bits24,%bits25
  %b6 = or i64 %bits26,%bits27
  %pack6 = or i64 %a6,%b6
  %dst6 = getelementptr i32,ptr addrspace(1) %p,i32 48
  store i64 %pack6,ptr addrspace(1) %dst6,align 8
  %scaled28 = fmul float %x28,%rcp
  %raw28 = bitcast float %scaled28 to i32
  %high28 = lshr i32 %raw28,16
  %tie28 = and i32 %high28,1
  %biased28 = add i32 %raw28,32767
  %rounded28 = add i32 %biased28,%tie28
  %bf28 = lshr i32 %rounded28,16
  %wide28 = zext i32 %bf28 to i64
  %bits28 = shl i64 %wide28,0
  %scaled29 = fmul float %x29,%rcp
  %raw29 = bitcast float %scaled29 to i32
  %high29 = lshr i32 %raw29,16
  %tie29 = and i32 %high29,1
  %biased29 = add i32 %raw29,32767
  %rounded29 = add i32 %biased29,%tie29
  %bf29 = lshr i32 %rounded29,16
  %wide29 = zext i32 %bf29 to i64
  %bits29 = shl i64 %wide29,16
  %scaled30 = fmul float %x30,%rcp
  %raw30 = bitcast float %scaled30 to i32
  %high30 = lshr i32 %raw30,16
  %tie30 = and i32 %high30,1
  %biased30 = add i32 %raw30,32767
  %rounded30 = add i32 %biased30,%tie30
  %bf30 = lshr i32 %rounded30,16
  %wide30 = zext i32 %bf30 to i64
  %bits30 = shl i64 %wide30,32
  %scaled31 = fmul float %x31,%rcp
  %raw31 = bitcast float %scaled31 to i32
  %high31 = lshr i32 %raw31,16
  %tie31 = and i32 %high31,1
  %biased31 = add i32 %raw31,32767
  %rounded31 = add i32 %biased31,%tie31
  %bf31 = lshr i32 %rounded31,16
  %wide31 = zext i32 %bf31 to i64
  %bits31 = shl i64 %wide31,48
  %a7 = or i64 %bits28,%bits29
  %b7 = or i64 %bits30,%bits31
  %pack7 = or i64 %a7,%b7
  %dst7 = getelementptr i32,ptr addrspace(1) %p,i32 56
  store i64 %pack7,ptr addrspace(1) %dst7,align 8
  br label %done
done:
  ret i32 %active
}
"""
    return lower_publication(lower_query_scale(lower_query_word(lower_metadata(lower_math(lower_mma(ir))))))


def dependency_identity():
    path=Path(__file__).parent
    sources=('triton_m128_publication_r1087.py','triton_c500_mma_r881.py',
             'triton_reader_math_r899c.py','triton_reader_metadata_qscale_guard_r1073.py')
    return hashlib.sha256(b''.join((path/n).read_bytes() for n in sources)).hexdigest()


install_storage_lowering(m128_reader,extra_lowering=lower_reader,extra_identity=dependency_identity())
