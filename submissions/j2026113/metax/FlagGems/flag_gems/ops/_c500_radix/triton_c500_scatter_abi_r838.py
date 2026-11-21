"""ABI-only adaptation of original scalar scatter compiled from Triton."""
import hashlib
from pathlib import Path
import re
from triton.language import core

HELPER=Path(__file__).with_name('scalar_scatter_helper_r837.ll').read_text()
IDENTITY=hashlib.sha256(HELPER.encode()).hexdigest()


def inject_scatter(module):
    module,count=re.subn(r'(?m)^declare[^\n]*@__c500_Triton_scatter\([^\n]*\)[^\n]*$','',module)
    assert count==1
    match=re.search(r'(?m)^define[^\n]*@scalar_scatter_phase\(([^\n]*)\)[^\n]*\{',HELPER)
    assert match
    end=HELPER.index('\n}',match.end())+2
    function=HELPER[match.start():end]
    function='define internal i32 @__c500_Triton_scatter('+match.group(1)+') alwaysinline nounwind convergent {'+function[function.index('\n'):]
    assert function.count('ret void')==1 and 'atomicrmw' not in function and 'barrier' not in function
    function=function.replace('ret void','ret i32 0')
    # The scalar scatter contains neither loop nor alias metadata; no remapping.
    assert not re.search(r'!\d+',function)
    return module+'\n'+function+'\n'


@core.builtin
def scatter_per_lane(Out,OutIndices,Row,N,Tile,Logical,Raw,Payload,Digit,Rank,_builder=None):
    args=[core._to_tensor(v,_builder) for v in [Out,OutIndices,Row,N,Tile,Logical,Raw,Payload,Digit,Rank]]
    dtype=args[5].type
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_Triton_scatter',
        [v.handle for v in args],dtype.to_ir(_builder),False),dtype)
