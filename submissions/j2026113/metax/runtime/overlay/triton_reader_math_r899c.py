"""Native Float2 scale ABI; scalar exp2/shuffle use installed Triton intrinsic API.

This adds no reader algorithm. Only the native two-float ABI needs a vector
wrapper; scalar bpermute/exp2 are emitted directly by the installed frontend.
"""
import hashlib
from pathlib import Path
import re
from triton.language import core

IDENTITY = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@core.builtin
def scale_pair(x0, x1, alpha, _builder=None):
    args = [core._to_tensor(x, _builder) for x in [x0, x1, alpha]]
    dtype = args[0].type
    return tuple(core.tensor(_builder.create_extern_elementwise(
        '', '', '__reader_scale_pair_' + str(i), [v.handle for v in args],
        dtype.to_ir(_builder), False), dtype) for i in range(2))


def lower_math(module):
    declarations = 0
    for i in range(2):
        module, n = re.subn(r'(?m)^declare[^\n]*@__reader_scale_pair_' + str(i) + r'\([^\n]*\)[^\n]*$', '', module)
        declarations += n
    if not declarations:
        return module
    assert declarations == 2
    groups = {}
    seen = {}
    pattern = r'(?m)^( *)(%[-a-zA-Z$._0-9]+) = (?:tail )?call float @__reader_scale_pair_(\d)\(([^\n]*)\)[^\n]*$'

    def replace(match):
        indent, result, index, args = match.groups()
        prefix = ''
        if args not in groups:
            name = '%reader_scale_pack_' + str(len(groups))
            groups[args] = name
            seen[args] = set()
            prefix = indent + name + ' = call <2 x float> @__reader_scale_pair(' + args + ')\n'
        assert index not in seen[args]
        seen[args].add(index)
        return prefix + indent + result + ' = extractelement <2 x float> ' + groups[args] + ', i32 ' + index

    module, n = re.subn(pattern, replace, module)
    assert n == 2 * len(groups) and all(v == {'0', '1'} for v in seen.values())
    return module + '''
declare <2 x float> @llvm.mxc.pk.fma.f32(<2 x float>,<2 x float>,<2 x float>)
define internal <2 x float> @__reader_scale_pair(float %x0,float %x1,float %alpha) alwaysinline nounwind {
  %x_0 = insertelement <2 x float> undef,float %x0,i32 0
  %x = insertelement <2 x float> %x_0,float %x1,i32 1
  %a_0 = insertelement <2 x float> undef,float %alpha,i32 0
  %a = insertelement <2 x float> %a_0,float %alpha,i32 1
  %r = call <2 x float> @llvm.mxc.pk.fma.f32(<2 x float> %x,<2 x float> %a,<2 x float> zeroinitializer)
  ret <2 x float> %r
}
'''


@core.builtin
def native_bpermute(address,value,_builder=None):
    a=core._to_tensor(address,_builder)
    v=core._to_tensor(value,_builder)
    return core.tensor(_builder.create_extern_elementwise('','','llvm.mxc.bsm.bpermute',
        [a.handle,v.handle],v.type.to_ir(_builder),False),v.type)


@core.builtin
def native_exp2(value,_builder=None):
    v=core._to_tensor(value,_builder)
    return core.tensor(_builder.create_extern_elementwise('','','llvm.exp2.f32',
        [v.handle],v.type.to_ir(_builder),True),v.type)
