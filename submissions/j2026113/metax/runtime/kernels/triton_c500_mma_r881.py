"""Expose installed native MMA register interfaces as project Triton builtins.

Interfaces are copied from actual pinned tl.dot LLVM880 and SDK headers869.
Only ABI vector packing/extraction surrounds one intrinsic. The reader's
algorithm is not implemented here. No alternate execution path or native.so.
"""
import hashlib
from pathlib import Path
import re
from triton.language import core

IDENTITY=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _fields(name,values,builder):
    args=[core._to_tensor(v,builder) for v in values]
    dtype=args[-1].type
    return tuple(core.tensor(builder.create_extern_elementwise('','',name+str(i),
        [v.handle for v in args],dtype.to_ir(builder),False),dtype) for i in range(4))


@core.builtin
def mma_i8(A,B,C0,C1,C2,C3,_builder=None):
    return _fields('__c500_mma_i8_',[A,B,C0,C1,C2,C3],_builder)


@core.builtin
def mma_bf16(A0,A1,B0,B1,C0,C1,C2,C3,_builder=None):
    return _fields('__c500_mma_bf16_',[A0,A1,B0,B1,C0,C1,C2,C3],_builder)


def lower_mma(module):
    for kind,scalar in [('i8','i32'),('bf16','float')]:
        declarations=0
        for index in range(4):
            module,n=re.subn(r'(?m)^declare[^\n]*@__c500_mma_'+kind+'_'+str(index)+r'\([^\n]*\)[^\n]*$','',module)
            declarations+=n
        if declarations==0:continue
        assert declarations==4
        pattern=r'(?m)^( *)(%[-a-zA-Z$._0-9]+) = (?:tail )?call '+scalar+r' @__c500_mma_'+kind+r'_(\d)\(([^\n]*)\)[^\n]*$'
        groups={};seen={}
        vector='<4 x '+scalar+'>'
        def replace(match):
            indent,result,index,args=match.groups()
            prefix=''
            if args not in groups:
                name='%c500_mma_'+kind+'_pack_'+str(len(groups));groups[args]=name;seen[args]=set()
                prefix=indent+name+' = call '+vector+' @__c500_mma_'+kind+'('+args+')\n'
            assert index not in seen[args];seen[args].add(index)
            return prefix+indent+result+' = extractelement '+vector+' '+groups[args]+', i32 '+index
        module,n=re.subn(pattern,replace,module)
        assert n==4*len(groups) and all(v=={'0','1','2','3'} for v in seen.values()),(kind,n,len(groups))
        if kind=='i8':
            function='''
declare <4 x i32> @llvm.mxc.mma.i32.16x16x16i8(i32,i32,<4 x i32>) convergent nounwind
define internal <4 x i32> @__c500_mma_i8(i32 %a,i32 %b,i32 %c0,i32 %c1,i32 %c2,i32 %c3) alwaysinline nounwind convergent {
  %c_0 = insertelement <4 x i32> undef, i32 %c0, i32 0
  %c_1 = insertelement <4 x i32> %c_0, i32 %c1, i32 1
  %c_2 = insertelement <4 x i32> %c_1, i32 %c2, i32 2
  %c_3 = insertelement <4 x i32> %c_2, i32 %c3, i32 3
  %result = call <4 x i32> @llvm.mxc.mma.i32.16x16x16i8(i32 %b,i32 %a,<4 x i32> %c_3)
  ret <4 x i32> %result
}
'''
        else:
            function='''
declare <4 x float> @llvm.mxc.mma.f32.16x16x2bf16(<4 x half>,<4 x half>,<4 x float>) convergent nounwind
define internal <4 x float> @__c500_mma_bf16(i32 %a0,i32 %a1,i32 %b0,i32 %b1,float %c0,float %c1,float %c2,float %c3) alwaysinline nounwind convergent {
  %a_0 = insertelement <2 x i32> undef, i32 %a0, i32 0
  %a_1 = insertelement <2 x i32> %a_0, i32 %a1, i32 1
  %b_0 = insertelement <2 x i32> undef, i32 %b0, i32 0
  %b_1 = insertelement <2 x i32> %b_0, i32 %b1, i32 1
  %a = bitcast <2 x i32> %a_1 to <4 x half>
  %b = bitcast <2 x i32> %b_1 to <4 x half>
  %c_0 = insertelement <4 x float> undef, float %c0, i32 0
  %c_1 = insertelement <4 x float> %c_0, float %c1, i32 1
  %c_2 = insertelement <4 x float> %c_1, float %c2, i32 2
  %c_3 = insertelement <4 x float> %c_2, float %c3, i32 3
  %result = call <4 x float> @llvm.mxc.mma.f32.16x16x2bf16(<4 x half> %b,<4 x half> %a,<4 x float> %c_3)
  ret <4 x float> %result
}
'''
        module+='\n'+function
    assert not re.search(r'@__c500_mma_(?:i8|bf16)_\d\(',module)
    return module
