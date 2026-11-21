"""Original reader memory primitives: asyncK128, predicatedV128,6KiBshared.

ABI and intrinsic names from actual native reader885. This module implements
ordinary memory/vector packing and the native per-lane V tail branch.
Page, lane and swizzle decisions remain in Triton.
"""
import hashlib
from pathlib import Path
import re
from triton.language import core

IDENTITY=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load4(name,pointer,valid,builder,*extra):
    args=[core._to_tensor(x,builder) for x in (pointer,valid,*extra)]
    args[1]=args[1].to(core.int32,_builder=builder)
    dtype=core.block_type(core.int32,args[1].type.shape) if isinstance(args[1].type,core.block_type) else core.int32
    return tuple(core.tensor(builder.create_extern_elementwise('','',name+str(i),
        [x.handle for x in args],dtype.to_ir(builder),False),dtype) for i in range(4))


@core.builtin
def k_packet(pointer,valid,_builder=None):
    p=core._to_tensor(pointer,_builder)
    v=core._to_tensor(valid,_builder).to(core.int32,_builder=_builder)
    dtype=core.block_type(core.uint64,v.type.shape)
    return tuple(core.tensor(_builder.create_extern_elementwise(
        '', '', '__reader_k_packet_'+str(i), [p.handle,v.handle], dtype.to_ir(_builder), False),dtype)
        for i in range(2))


@core.builtin
def shared_store_k(byte_offset,lo,hi,upper,_builder=None):
    args=[core._to_tensor(x,_builder) for x in (byte_offset,lo,hi,upper)]
    args[3]=args[3].to(core.int32,_builder=_builder)
    dtype=args[0].type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__reader_shared_store_k', [x.handle for x in args], dtype.to_ir(_builder), False),dtype)


@core.builtin
def v_packet(pointer,valid,pos,length,_builder=None):
    return _load4('__reader_v_packet_',pointer,valid,_builder,pos,length)


@core.builtin
def arrive64(reference,_builder=None):
    x=core._to_tensor(reference,_builder)
    return core.tensor(_builder.create_extern_elementwise('','','__reader_arrive64',[x.handle],x.type.to_ir(_builder),False),x.type)


@core.builtin
def shared_store4(byte_offset,A,B,C,D,_builder=None):
    args=[core._to_tensor(x,_builder) for x in [byte_offset,A,B,C,D]];dtype=args[0].type
    return core.tensor(_builder.create_extern_elementwise('','','__reader_shared_store4',
        [x.handle for x in args],dtype.to_ir(_builder),False),dtype)


@core.builtin
def shared_load32(word_index,_builder=None):
    x=core._to_tensor(word_index,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__reader_shared_load32', [x.handle], x.type.to_ir(_builder), False),x.type)


@core.builtin
def shared_load64(packet_index,_builder=None):
    x=core._to_tensor(packet_index,_builder)
    dtype=core.block_type(core.uint64,x.type.shape) if isinstance(x.type,core.block_type) else core.uint64
    return core.tensor(_builder.create_extern_elementwise('','','__reader_shared_load64',[x.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def shared_load32_parts(row_offset,fragment_offset,_builder=None):
    args=[core._to_tensor(v,_builder) for v in (row_offset,fragment_offset)]
    dtype=args[0].type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__reader_shared_load32_parts', [a.handle for a in args], dtype.to_ir(_builder), False),dtype)


def lower_reader_memory(module):
    for name in ['__reader_shared_store_k','__reader_arrive64','__reader_shared_store4','__reader_shared_load32','__reader_shared_load32_parts','__reader_shared_load64']:
        module,n=re.subn(r'(?m)^declare[^\n]*@'+name+r'\([^\n]*\)[^\n]*$','',module)
        assert n in [0,1]
    for kind in ['k','v']:
        count, scalar = (2, 'i64') if kind == 'k' else (4, 'i32')
        vector = f'<{count} x {scalar}>'
        ndecl=0
        for i in range(count):
            module,n=re.subn(r'(?m)^declare[^\n]*@__reader_'+kind+r'_packet_'+str(i)+r'\([^\n]*\)[^\n]*$','',module);ndecl+=n
        if not ndecl:continue
        assert ndecl==count
        groups={};seen={}
        pattern=r'(?m)^( *)(%[-a-zA-Z$._0-9]+) = (?:tail )?call '+scalar+' @__reader_'+kind+r'_packet_(\d)\(([^\n]*)\)[^\n]*$'
        def replace(match):
            indent,result,index,args=match.groups();prefix=''
            if args not in groups:
                name='%reader_'+kind+'_pack_'+str(len(groups));groups[args]=name;seen[args]=set()
                prefix=indent+name+' = call '+vector+' @__reader_'+kind+'_packet('+args+')\n'
            assert index not in seen[args];seen[args].add(index)
            return prefix+indent+result+' = extractelement '+vector+' '+groups[args]+', i32 '+index
        module,n=re.subn(pattern,replace,module)
        assert n==len(groups)*count and all(v==set(map(str,range(count))) for v in seen.values())
    # Arrays exactly match native K2KiB plus V4KiB. The combined address base
    # makes offsets explicit and leaves swizzle/indexing in the Triton source.
    definitions='''
@__reader_shared = internal addrspace(3) global [6144 x i8] undef, align 16
declare <4 x i32> @llvm.mxc.load.global.async.v4i32(ptr addrspace(1))
declare <4 x i32> @llvm.mxc.ldg.predicator.v4i32(ptr addrspace(1),i32 immarg,i64,i1 immarg,i1 immarg,i1 immarg,i1 immarg)
declare void @llvm.mxc.arrive(i32 immarg)
define internal <2 x i64> @__reader_k_packet(ptr addrspace(1) %p,i32 %valid) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %valid,0
  br i1 %active,label %read,label %done
read:
  %v = call <4 x i32> @llvm.mxc.load.global.async.v4i32(ptr addrspace(1) %p)
  %packed = bitcast <4 x i32> %v to <2 x i64>
  br label %done
done:
  %result = phi <2 x i64> [%packed,%read],[zeroinitializer,%entry]
  ret <2 x i64> %result
}
define internal <4 x i32> @__reader_v_packet(ptr addrspace(1) %p,i32 %valid,i32 %pos,i32 %length) alwaysinline nounwind convergent {
entry:
  %mask = call i64 @llvm.mxc.icmp.i64.i32(i32 %valid,i32 1,i32 32) nomerge
  %v = call <4 x i32> @llvm.mxc.ldg.predicator.v4i32(ptr addrspace(1) %p,i32 0,i64 %mask,i1 true,i1 true,i1 false,i1 false)
  %last = add i32 %pos,3
  %tail = icmp sge i32 %last,%length
  br i1 %tail,label %masked,label %done
masked:
  %t1 = add i32 %pos,1
  %t2 = add i32 %pos,2
  %lo0 = icmp slt i32 %pos,%length
  %hi0 = icmp slt i32 %t1,%length
  %lo1 = icmp slt i32 %t2,%length
  %hi1 = icmp slt i32 %last,%length
  %ml0 = select i1 %lo0,i32 65535,i32 0
  %mh0 = select i1 %hi0,i32 -65536,i32 0
  %ml1 = select i1 %lo1,i32 65535,i32 0
  %mh1 = select i1 %hi1,i32 -65536,i32 0
  %m0 = or i32 %ml0,%mh0
  %m1 = or i32 %ml1,%mh1
  %a = insertelement <4 x i32> poison,i32 %m0,i32 0
  %b = insertelement <4 x i32> %a,i32 %m1,i32 1
  %c = insertelement <4 x i32> %b,i32 %m0,i32 2
  %m = insertelement <4 x i32> %c,i32 %m1,i32 3
  %tail_value = and <4 x i32> %v,%m
  br label %done
done:
  %result = phi <4 x i32> [%v,%entry],[%tail_value,%masked]
  ret <4 x i32> %result
}
define internal i32 @__reader_arrive64(i32 %reference) alwaysinline nounwind convergent {
  call void @llvm.mxc.arrive(i32 64)
  ret i32 %reference
}
define internal i32 @__reader_shared_store_k(i32 %offset,i64 %lo,i64 %hi,i32 %upper) alwaysinline nounwind convergent {
entry:
  %rotate = icmp ne i32 %upper,0
  br i1 %rotate,label %upper_store,label %publish
upper_store:
  %lo64 = call i64 asm sideeffect "shl_b64 $0, 0, $1", "=r,r,~{memory}"(i64 %lo)
  %hi64 = call i64 asm sideeffect "shl_b64 $0, 0, $1", "=r,r,~{memory}"(i64 %hi)
  br label %publish
publish:
  %a0 = phi i64 [%lo,%entry],[%lo64,%upper_store]
  %b0 = phi i64 [%hi,%entry],[%hi64,%upper_store]
  %p = getelementptr i8,ptr addrspace(3) @__reader_shared,i32 %offset
  %a = insertelement <2 x i64> poison,i64 %a0,i32 0
  %b = insertelement <2 x i64> %a,i64 %b0,i32 1
  store <2 x i64> %b,ptr addrspace(3) %p,align 16
  ret i32 %offset
}
define internal i32 @__reader_shared_store4(i32 %offset,i32 %a,i32 %b,i32 %c,i32 %d) alwaysinline nounwind convergent {
  %p = getelementptr i8,ptr addrspace(3) @__reader_shared,i32 %offset
  %v0 = insertelement <4 x i32> undef,i32 %a,i32 0
  %v1 = insertelement <4 x i32> %v0,i32 %b,i32 1
  %v2 = insertelement <4 x i32> %v1,i32 %c,i32 2
  %v3 = insertelement <4 x i32> %v2,i32 %d,i32 3
  store <4 x i32> %v3,ptr addrspace(3) %p,align 16
  ret i32 %offset
}
define internal i32 @__reader_shared_load32(i32 %word) alwaysinline nounwind convergent {
  %p = getelementptr inbounds i32,ptr addrspace(3) @__reader_shared,i32 %word
  %v = load i32,ptr addrspace(3) %p,align 4
  ret i32 %v
}
define internal i32 @__reader_shared_load32_parts(i32 %row,i32 %fragment) alwaysinline nounwind convergent {
  %base = getelementptr inbounds i8,ptr addrspace(3) @__reader_shared,i32 %row
  %p = getelementptr inbounds i8,ptr addrspace(3) %base,i32 %fragment
  %v = load i32,ptr addrspace(3) %p,align 4
  ret i32 %v
}
define internal i64 @__reader_shared_load64(i32 %packet) alwaysinline nounwind convergent {
  %tile = getelementptr inbounds i8,ptr addrspace(3) @__reader_shared,i32 2048
  %p = getelementptr inbounds i64,ptr addrspace(3) %tile,i32 %packet
  %v = load i64,ptr addrspace(3) %p,align 8
  ret i64 %v
}
'''
    if not re.search(r'(?m)^declare[^\n]*@llvm\.mxc\.icmp\.i64\.i32\(',module):
        definitions+='\ndeclare i64 @llvm.mxc.icmp.i64.i32(i32,i32,i32 immarg) convergent nounwind\n'
    return module+'\n'+definitions


