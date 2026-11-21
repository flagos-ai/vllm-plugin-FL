import triton
import triton.language as tl
from triton.language import core

@core.builtin
def ballot64(predicate, _builder=None):
    predicate=core._to_tensor(predicate,_builder)
    zero=core._to_tensor(0,_builder)
    ne=core._to_tensor(33,_builder)
    dtype=core.block_type(core.uint64,predicate.type.shape)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', 'llvm.mxc.icmp.i64.i32',
        [predicate.handle,zero.handle,ne.handle],dtype.to_ir(_builder),False),dtype)

@core.builtin
def popcount64(value, _builder=None):
    value=core._to_tensor(value,_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', 'llvm.ctpop.i64',[value.handle],value.type.to_ir(_builder),True),value.type)

@core.builtin
def warp_exchange(counter, source_lane, _builder=None):
    counter=core._to_tensor(counter,_builder)
    source_lane=core._to_tensor(source_lane,_builder)
    byte_offset=source_lane.__lshift__(2,_builder=_builder)
    return core.tensor(_builder.create_extern_elementwise(
        '', '', 'llvm.mxc.bsm.bpermute',
        [byte_offset.handle,counter.handle],counter.type.to_ir(_builder),False),counter.type)

