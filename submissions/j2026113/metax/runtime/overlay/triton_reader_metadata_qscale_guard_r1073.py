"""Native scalar metadata loads; no host reads, synchronization or alternate path."""
import hashlib
from pathlib import Path
import re
from triton.language import core

IDENTITY=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@core.builtin
def metadata_i32(pointer,_builder=None):
    p=core._to_tensor(pointer,_builder)
    dtype=core.block_type(core.int32,p.type.shape) if isinstance(p.type,core.block_type) else core.int32
    return core.tensor(_builder.create_extern_elementwise('','','__reader_metadata_i32',
        [p.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def scale_u16(pointer,_builder=None):
    p=core._to_tensor(pointer,_builder)
    dtype=core.block_type(core.uint16,p.type.shape) if isinstance(p.type,core.block_type) else core.uint16
    return core.tensor(_builder.create_extern_elementwise('','','__reader_scale_u16',
        [p.handle],dtype.to_ir(_builder),False),dtype)


@core.builtin
def page_i32(pointer,valid,_builder=None):
    p=core._to_tensor(pointer,_builder)
    v=core._to_tensor(valid,_builder).to(core.int32,_builder=_builder)
    dtype=v.type
    return core.tensor(_builder.create_extern_elementwise('','','__reader_page_i32',
        [p.handle,v.handle],dtype.to_ir(_builder),False),dtype)


def lower_metadata(module):
    for name in ['__reader_metadata_i32','__reader_page_i32','__reader_scale_u16']:
        module,n=re.subn(r'(?m)^declare[^\n]*@'+name+r'\([^\n]*\)[^\n]*$','',module)
        assert n==1
    return module+'''
define internal i32 @__reader_metadata_i32(ptr addrspace(1) %p) alwaysinline nounwind {
  %value = load i32,ptr addrspace(1) %p,align 4
  ret i32 %value
}
define internal i16 @__reader_scale_u16(ptr addrspace(1) %p) alwaysinline nounwind {
  %value = load i16,ptr addrspace(1) %p,align 2
  ret i16 %value
}
define internal i32 @__reader_page_i32(ptr addrspace(1) %p,i32 %valid) alwaysinline nounwind {
entry:
  %active = icmp ne i32 %valid,0
  br i1 %active,label %read,label %done
read:
  %value = load i32,ptr addrspace(1) %p,align 4
  br label %done
done:
  %result = phi i32 [%value,%read],[0,%entry]
  ret i32 %result
}
'''



@core.builtin
def query_word(pointer,active,offset,_builder=None):
    p=core._to_tensor(pointer,_builder)
    a=core._to_tensor(active,_builder).to(core.int32,_builder=_builder)
    offset=core._to_tensor(offset,_builder).to(core.int32,_builder=_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__reader_query_word', [p.handle,a.handle,offset.handle],
        a.type.to_ir(_builder),False),a.type)


def lower_query_word(module):
    module,n=re.subn(r'(?m)^declare[^\n]*@__reader_query_word\([^\n]*\)[^\n]*$','',module)
    assert n==1
    return module+'\n' + """define internal i32 @__reader_query_word(ptr addrspace(1) %p,i32 %active,i32 %offset) alwaysinline nounwind convergent {
entry:
  %take = icmp ne i32 %active,0
  br i1 %take,label %load,label %zero
load:
  %word = getelementptr i32, ptr addrspace(1) %p, i32 %offset
  %v = load i32, ptr addrspace(1) %word, align 4
  ret i32 %v
zero:
  ret i32 0
}
"""



@core.builtin
def query_scale(pointer,active,_builder=None):
    p=core._to_tensor(pointer,_builder)
    a=core._to_tensor(active,_builder).to(core.int32,_builder=_builder)
    dtype=a.type
    return core.tensor(_builder.create_extern_elementwise(
        '', '', '__reader_query_scale', [p.handle,a.handle],
        dtype.to_ir(_builder),False),dtype)


def lower_query_scale(module):
    module,n=re.subn(r'(?m)^declare[^\n]*@__reader_query_scale\([^\n]*\)[^\n]*$','',module)
    assert n==1
    return module+"""
define internal i32 @__reader_query_scale(ptr addrspace(1) %p,i32 %active) alwaysinline nounwind convergent {
entry:
  %take = icmp ne i32 %active,0
  br i1 %take,label %load,label %one
load:
  %raw = load i16,ptr addrspace(1) %p,align 2
  %bits = zext i16 %raw to i32
  %value = shl i32 %bits,16
  ret i32 %value
one:
  ret i32 1065353216
}
"""
