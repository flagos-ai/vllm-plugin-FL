"""Project-local Triton lowering for missing native shared/warp primitives.

No installed compiler or backend is modified. The pinned Python compiler
driver is copied into a private namespace and one LLIR-stage hook defines
the storage primitives emitted by this kernel's Triton builtins. The kernel
algorithm remains Triton. This is an instruction-reproduction experiment,
not an externally compiled kernel or a custom .so call.
"""
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import re

import triton
from triton.language import core


@core.builtin
def shared_load32(byte_offset, _builder=None):
    p=core._to_tensor(byte_offset,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_shared_load32',[p.handle],p.type.to_ir(_builder),False),p.type)


@core.builtin
def shared_store32(byte_offset,value,mask,_builder=None):
    p=core._to_tensor(byte_offset,_builder)
    v=core._to_tensor(value,_builder)
    m=core._to_tensor(mask,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_shared_store32',[p.handle,v.handle,m.handle],p.type.to_ir(_builder),False),p.type)


@core.builtin
def warp_sync(token,_builder=None):
    x=core._to_tensor(token,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_warp_sync32',[x.handle],x.type.to_ir(_builder),False),x.type)


@core.builtin
def shared_load_plain32(byte_offset,_builder=None):
    p=core._to_tensor(byte_offset,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_shared_load_plain32',[p.handle],p.type.to_ir(_builder),False),p.type)


@core.builtin
def shared_store_plain32(byte_offset,value,mask,_builder=None):
    p=core._to_tensor(byte_offset,_builder)
    v=core._to_tensor(value,_builder)
    m=core._to_tensor(mask,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_shared_store_plain32',[p.handle,v.handle,m.handle],p.type.to_ir(_builder),False),p.type)


@core.builtin
def native_lane_id(reference,_builder=None):
    reference=core._to_tensor(reference,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_lane_id32',[reference.handle],reference.type.to_ir(_builder),False),reference.type)


@core.builtin
def global_atomic_read_u32(pointer,_builder=None):
    p=core._to_tensor(pointer,_builder)
    dtype=core.block_type(core.uint32,p.type.shape) if isinstance(p.type,core.block_type) else core.uint32
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_atomic_read_u32',[p.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def global_store_u32(pointer,value,_builder=None):
    p=core._to_tensor(pointer,_builder);v=core._to_tensor(value,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_global_store_u32',[p.handle,v.handle],v.type.to_ir(_builder),False),v.type)


@core.builtin
def claimed_load(reference,_builder=None):
    reference=core._to_tensor(reference,_builder)
    dtype=reference.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_claimed_load',[reference.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def claimed_store(value,mask,_builder=None):
    value=core._to_tensor(value,_builder)
    mask=core._to_tensor(mask,_builder)
    dtype=value.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_claimed_store',[value.handle,mask.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def prefix_load(index,_builder=None):
    index=core._to_tensor(index,_builder)
    dtype=index.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_prefix_load',[index.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def prefix_store(index,value,mask,_builder=None):
    index=core._to_tensor(index,_builder)
    value=core._to_tensor(value,_builder)
    mask=core._to_tensor(mask,_builder)
    dtype=index.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_prefix_store',[index.handle,value.handle,mask.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def scatter_load(index,_builder=None):
    index=core._to_tensor(index,_builder)
    dtype=index.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_scatter_load',[index.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def scatter_store(index,value,mask,_builder=None):
    index=core._to_tensor(index,_builder)
    value=core._to_tensor(value,_builder)
    mask=core._to_tensor(mask,_builder)
    dtype=index.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_scatter_store',[index.handle,value.handle,mask.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def global_atomic_add_u32(pointer,value,mask,_builder=None):
    pointer=core._to_tensor(pointer,_builder)
    value=core._to_tensor(value,_builder)
    mask=core._to_tensor(mask,_builder)
    dtype=value.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_atomic_add_u32',[pointer.handle,value.handle,mask.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def global_atomic_xchg_u32(pointer,value,mask,_builder=None):
    pointer=core._to_tensor(pointer,_builder)
    value=core._to_tensor(value,_builder)
    mask=core._to_tensor(mask,_builder)
    dtype=value.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_atomic_xchg_u32',[pointer.handle,value.handle,mask.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def global_load_u32(pointer,reference,_builder=None):
    pointer=core._to_tensor(pointer,_builder)
    reference=core._to_tensor(reference,_builder)
    dtype=reference.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__c500_global_load_u32',[pointer.handle,reference.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def global_masked_load32(pointer,mask,other,_builder=None):
    p=core._to_tensor(pointer,_builder);m=core._to_tensor(mask,_builder);v=core._to_tensor(other,_builder)
    dtype=m.type
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_global_masked_load32',
        [p.handle,m.handle,v.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def global_masked_store32(pointer,value,mask,_builder=None):
    p=core._to_tensor(pointer,_builder);v=core._to_tensor(value,_builder);m=core._to_tensor(mask,_builder)
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_global_masked_store32',
        [p.handle,v.handle,m.handle],v.type.to_ir(_builder),False),v.type)


@core.builtin
def shared_load32_masked(index,mask,_builder=None):
    p=core._to_tensor(index,_builder);m=core._to_tensor(mask,_builder)
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_shared_load32_masked',
        [p.handle,m.handle],p.type.to_ir(_builder),False),p.type)


@core.builtin
def prefix_load_masked(index,mask,_builder=None):
    p=core._to_tensor(index,_builder);m=core._to_tensor(mask,_builder)
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_prefix_load_masked',
        [p.handle,m.handle],p.type.to_ir(_builder),False),p.type)


@core.builtin
def scatter_load_masked(index,mask,_builder=None):
    p=core._to_tensor(index,_builder);m=core._to_tensor(mask,_builder)
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_scatter_load_masked',
        [p.handle,m.handle],p.type.to_ir(_builder),False),p.type)


@core.builtin
def input_pair32(raw_pointer,payload_pointer,mask,_builder=None):
    raw=core._to_tensor(raw_pointer,_builder);payload=core._to_tensor(payload_pointer,_builder);mask=core._to_tensor(mask,_builder)
    dtype=core.block_type(core.uint64,mask.type.shape)
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_input_pair32',
        [raw.handle,payload.handle,mask.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def output_pair32(raw_pointer,payload_pointer,raw,payload,mask,_builder=None):
    args=[core._to_tensor(v,_builder) for v in [raw_pointer,payload_pointer,raw,payload,mask]]
    dtype=args[2].type
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_output_pair32',
        [v.handle for v in args],dtype.to_ir(_builder),False),dtype)


@core.builtin
def claim_and_publish(pointer,mask,_builder=None):
    p=core._to_tensor(pointer,_builder);m=core._to_tensor(mask,_builder)
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_claim_and_publish',
        [p.handle,m.handle],m.type.to_ir(_builder),False),m.type)


@core.builtin
def physical_thread_id(reference,_builder=None):
    x=core._to_tensor(reference,_builder)
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_thread_id32',
        [x.handle],x.type.to_ir(_builder),False),x.type)


STORAGE_LLVM=r'''
; Project-local lowering: native Rank temporary has 32B scan +512B aliased counters.
@__c500_claimed_tile = internal addrspace(3) global i32 undef, align 4
@__c500_rank_temporary = internal addrspace(3) global [544 x i8] undef, align 16
@__c500_scatter_base = internal addrspace(3) global [16 x i32] undef, align 4
@__c500_local_bin_prefix = internal addrspace(3) global [17 x i32] undef, align 4

define internal i32 @__c500_shared_load32(i32 %offset) alwaysinline nounwind convergent {
entry:
  %p = getelementptr i8, ptr addrspace(3) @__c500_rank_temporary, i32 %offset
  %v = load volatile i32, ptr addrspace(3) %p, align 4
  ret i32 %v
}

define internal i32 @__c500_shared_store32(i32 %offset, i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
  %p = getelementptr i8, ptr addrspace(3) @__c500_rank_temporary, i32 %offset
  store volatile i32 %value, ptr addrspace(3) %p, align 4
  br label %done
done:
  ret i32 %value
}

define internal i32 @__c500_warp_sync32(i32 %token) alwaysinline nounwind convergent {
entry:
  fence syncscope("warp") release
  call void @llvm.mxc.barrier.warp()
  fence syncscope("warp") acquire
  ret i32 %token
}
'''


STORAGE_LLVM += r"""
define internal i32 @__c500_shared_load_plain32(i32 %offset) alwaysinline nounwind convergent {
entry:
  %p = getelementptr i8, ptr addrspace(3) @__c500_rank_temporary, i32 %offset
  %v = load i32, ptr addrspace(3) %p, align 4
  ret i32 %v
}
define internal i32 @__c500_shared_store_plain32(i32 %offset, i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
  %p = getelementptr i8, ptr addrspace(3) @__c500_rank_temporary, i32 %offset
  store i32 %value, ptr addrspace(3) %p, align 4
  br label %done
done:
  ret i32 %value
}
"""

STORAGE_LLVM += r"""
declare i32 @llvm.mxc.mbcnt.lo(i32, i32)
declare i32 @llvm.mxc.mbcnt.hi(i32, i32)
define internal i32 @__c500_lane_id32(i32 %unused) alwaysinline nounwind convergent {
entry:
  %lo = call i32 @llvm.mxc.mbcnt.lo(i32 -1, i32 0) nomerge
  %hi = call i32 @llvm.mxc.mbcnt.hi(i32 -1, i32 %lo) nomerge
  ret i32 %hi
}
"""

STORAGE_LLVM += r"""
define internal i32 @__c500_atomic_read_u32(ptr addrspace(1) %p) alwaysinline nounwind convergent {
entry:
  %v = atomicrmw or ptr addrspace(1) %p, i32 0 syncscope("device-one-as") monotonic, align 4
  ret i32 %v
}
define internal i32 @__c500_global_store_u32(ptr addrspace(1) %p, i32 %value) alwaysinline nounwind convergent {
entry:
  store i32 %value, ptr addrspace(1) %p, align 4
  ret i32 %value
}
"""

STORAGE_LLVM += r"""
define internal i32 @__c500_claimed_load(i32 %unused) alwaysinline nounwind convergent {
entry:
  %v = load i32, ptr addrspace(3) @__c500_claimed_tile, align 4
  ret i32 %v
}
define internal i32 @__c500_claimed_store(i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
  store i32 %value, ptr addrspace(3) @__c500_claimed_tile, align 4
  br label %done
done:
  ret i32 %value
}
define internal i32 @__c500_prefix_load(i32 %index) alwaysinline nounwind convergent {
entry:
  %p = getelementptr i32, ptr addrspace(3) @__c500_local_bin_prefix, i32 %index
  %v = load i32, ptr addrspace(3) %p, align 4
  ret i32 %v
}
define internal i32 @__c500_prefix_store(i32 %index, i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
  %p = getelementptr i32, ptr addrspace(3) @__c500_local_bin_prefix, i32 %index
  store i32 %value, ptr addrspace(3) %p, align 4
  br label %done
done:
  ret i32 %value
}
define internal i32 @__c500_scatter_load(i32 %index) alwaysinline nounwind convergent {
entry:
  %p = getelementptr i32, ptr addrspace(3) @__c500_scatter_base, i32 %index
  %v = load i32, ptr addrspace(3) %p, align 4
  ret i32 %v
}
define internal i32 @__c500_scatter_store(i32 %index, i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
  %p = getelementptr i32, ptr addrspace(3) @__c500_scatter_base, i32 %index
  store i32 %value, ptr addrspace(3) %p, align 4
  br label %done
done:
  ret i32 %value
}
define internal i32 @__c500_atomic_add_u32(ptr addrspace(1) %p, i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
  %v = atomicrmw add ptr addrspace(1) %p, i32 %value syncscope("device-one-as") monotonic, align 4
  br label %done
done:
  %ret = phi i32 [ %v, %write ], [ 0, %entry ]
  ret i32 %ret
}
define internal i32 @__c500_atomic_xchg_u32(ptr addrspace(1) %p, i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
  %v = atomicrmw xchg ptr addrspace(1) %p, i32 %value syncscope("device-one-as") monotonic, align 4
  br label %done
done:
  %ret = phi i32 [ %v, %write ], [ 0, %entry ]
  ret i32 %ret
}
define internal i32 @__c500_global_load_u32(ptr addrspace(1) %p, i32 %unused) alwaysinline nounwind convergent {
entry:
  %v = load i32, ptr addrspace(1) %p, align 4
  ret i32 %v
}
"""

STORAGE_LLVM += r"""
define internal i32 @__c500_global_masked_load32(ptr addrspace(1) %p, i32 %mask, i32 %other) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %read, label %done
read:
  %v = load i32, ptr addrspace(1) %p, align 4
  br label %done
done:
  %ret = phi i32 [ %v, %read ], [ undef, %entry ]
  ret i32 %ret
}
define internal i32 @__c500_global_masked_store32(ptr addrspace(1) %p, i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
  store i32 %value, ptr addrspace(1) %p, align 4
  br label %done
done:
  ret i32 %value
}
"""

STORAGE_LLVM += r"""
define internal i32 @__c500_shared_load32_masked(i32 %index, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %read, label %done
read:
  %p = getelementptr i8, ptr addrspace(3) @__c500_rank_temporary, i32 %index
  %v = load volatile i32, ptr addrspace(3) %p, align 4
  br label %done
done:
  %ret = phi i32 [ %v, %read ], [ undef, %entry ]
  ret i32 %ret
}
define internal i32 @__c500_prefix_load_masked(i32 %index, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %read, label %done
read:
  %p = getelementptr i32, ptr addrspace(3) @__c500_local_bin_prefix, i32 %index
  %v = load i32, ptr addrspace(3) %p, align 4
  br label %done
done:
  %ret = phi i32 [ %v, %read ], [ undef, %entry ]
  ret i32 %ret
}
define internal i32 @__c500_scatter_load_masked(i32 %index, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %read, label %done
read:
  %p = getelementptr i32, ptr addrspace(3) @__c500_scatter_base, i32 %index
  %v = load i32, ptr addrspace(3) %p, align 4
  br label %done
done:
  %ret = phi i32 [ %v, %read ], [ undef, %entry ]
  ret i32 %ret
}
"""

STORAGE_LLVM += r"""
define internal i64 @__c500_input_pair32(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i32 %mask) alwaysinline nounwind convergent {
entry:
  %valid = icmp ne i32 %mask, 0
  br i1 %valid, label %read, label %done
read:
  %raw = load i32, ptr addrspace(1) %p0, align 4
  %payload = load i32, ptr addrspace(1) %p1, align 4
  %lo = zext i32 %raw to i64
  %hi = zext i32 %payload to i64
  %upper = shl i64 %hi, 32
  %pair = or i64 %upper, %lo
  br label %done
done:
  %ret = phi i64 [ %pair, %read ], [ undef, %entry ]
  ret i64 %ret
}
define internal i32 @__c500_output_pair32(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i32 %raw, i32 %payload, i32 %mask) alwaysinline nounwind convergent {
entry:
  %valid = icmp ne i32 %mask, 0
  br i1 %valid, label %write, label %done
write:
  store i32 %raw, ptr addrspace(1) %p0, align 4
  store i32 %payload, ptr addrspace(1) %p1, align 4
  br label %done
done:
  ret i32 %raw
}
define internal i32 @__c500_claim_and_publish(ptr addrspace(1) %p, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %claim, label %done
claim:
  %tile = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope("device-one-as") monotonic, align 4
  store i32 %tile, ptr addrspace(3) @__c500_claimed_tile, align 4
  br label %done
done:
  ret i32 0
}
"""

STORAGE_LLVM += r"""
declare i32 @llvm.mxc.thread.id.x()
define internal i32 @__c500_thread_id32(i32 %unused) alwaysinline nounwind {
entry:
  %tid = call i32 @llvm.mxc.thread.id.x()
  ret i32 %tid
}
"""

def lower_storage(module):
    assert isinstance(module,str)
    for symbol in ['__c500_shared_load32','__c500_shared_store32','__c500_warp_sync32','__c500_shared_load_plain32','__c500_shared_store_plain32','__c500_lane_id32','__c500_atomic_read_u32','__c500_global_store_u32','__c500_claimed_load','__c500_claimed_store','__c500_prefix_load','__c500_prefix_store','__c500_scatter_load','__c500_scatter_store','__c500_atomic_add_u32','__c500_atomic_xchg_u32','__c500_global_load_u32','__c500_global_masked_load32','__c500_global_masked_store32','__c500_shared_load32_masked','__c500_prefix_load_masked','__c500_scatter_load_masked','__c500_input_pair32','__c500_output_pair32','__c500_claim_and_publish','__c500_thread_id32']:
        pattern=r'(?m)^declare[^\n]*@'+symbol+r'\([^\n]*\)[^\n]*$'
        module,n=re.subn(pattern,'',module)
        assert n in [0,1],('Unexpected primitive declarations',symbol,n)
    if 'declare void @llvm.mxc.barrier.warp()' not in module:
        module+='\ndeclare void @llvm.mxc.barrier.warp() convergent nounwind\n'
    definitions=STORAGE_LLVM
    if re.search(r'(?m)^declare[^\n]*@llvm\.mxc\.thread\.id\.x\(',module):
        definitions=definitions.replace('declare i32 @llvm.mxc.thread.id.x()\n','')
    return module+'\n'+definitions


def install_storage_lowering(kernel,extra_lowering=None,extra_identity=""):
    compiler=importlib.import_module('triton.compiler.compiler')
    source=inspect.getsource(compiler.compile)
    assert hashlib.sha256(source.encode()).hexdigest()=='3c8da5f7a2efdfa83e466bdd4c5cf0db426244d5a3f3c428728dbcd1eda67cc0'
    assert hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest()=='7ac81ba646e01c1345cd0be1b655df72dcbb34d64d05c15d5578ace80a25a39c'
    anchor='        next_module = compile_ir(module, metadata)'
    assert source.count(anchor)==1
    derived=source.replace('def compile(','def compile_storage_impl(',1).replace(anchor,
        anchor+"\n        if ext == 'llir':\n            next_module = _project_lower_storage(next_module)")
    namespace=dict(compiler.__dict__)
    namespace['_project_lower_storage']=(lambda text:extra_lowering(lower_storage(text))) if extra_lowering else lower_storage
    exec(compile(derived,str(Path(__file__))+'::private_compile','exec'),namespace)
    driver=namespace['compile_storage_impl']
    paths={Path(__file__).resolve(),Path(inspect.getsourcefile(kernel.fn)).resolve()}
    # The pinned dependency finder treats names starting with 'triton' as
    # builtins. Include every project helper source explicitly in this key.
    for value in kernel.fn.__globals__.values():
        function=value.fn if isinstance(value,triton.runtime.JITFunction) else value
        if callable(function):
            try:
                path=inspect.getsourcefile(inspect.unwrap(function))
            except (TypeError,ValueError):
                continue
            if path and Path(path).parent==Path(__file__).parent:
                paths.add(Path(path).resolve())
    identity=hashlib.sha256(b''.join(p.name.encode()+b'\0'+p.read_bytes() for p in sorted(paths))+extra_identity.encode()).hexdigest()
    def compile_one(src,target=None,options=None):
        assert isinstance(src,compiler.ASTSource)
        original_hash=src.hash();local=copy.copy(src)
        local.hash=lambda:hashlib.sha256((original_hash+'\n'+identity).encode()).hexdigest()
        return driver(local,target=target,options=options)
    if kernel.binder is None:
        kernel.create_binder()
    kernel.compile=compile_one
    return kernel
