"""Project-local ABI adaptation of scalar Triton-produced publication loop."""
from pathlib import Path
import hashlib
import re
from triton.language import core

HELPER=Path(__file__).with_name('scalar_publish_helper_r680c.ll').read_text()
IDENTITY=hashlib.sha256(HELPER.encode()).hexdigest()


def inject_publish(module):
    module,n=re.subn(r'(?m)^declare[^\n]*@__c500_Triton_publish\([^\n]*\)[^\n]*$','',module)
    assert n==1
    match=re.search(r'(?m)^define[^\n]*@scalar_publish_phase\(([^\n]*)\)[^\n]*\{',HELPER);assert match
    end=HELPER.index('\n}',match.end())+2
    function=HELPER[match.start():end]
    function='define internal i32 @__c500_Triton_publish('+match.group(1)+') alwaysinline nounwind convergent {'+function[function.index('\n'):]
    assert function.count('ret void')==1
    function=function.replace('ret void','ret i32 0')
    # Bodies are identical except return/ABI, all algorithm statements from Triton.
    assert function.count('atomicrmw or')==2 and function.count('atomicrmw xchg')==1 and 'barrier' not in function
    offset=max([int(x) for x in re.findall(r'!(\d+)',module)]+[-1])+1
    metadata='\n'.join(re.findall(r'(?m)^!\d+\s*=.*$',HELPER))
    numbered=set(map(int,re.findall(r'(?m)^!(\d+)\s*=',metadata)))
    assert set(map(int,re.findall(r'!(\d+)',function)))<=numbered
    remap=lambda text:re.sub(r'!(\d+)',lambda m:'!'+str(int(m.group(1))+offset),text)
    if 'declare i32 @llvm.smax.i32(' not in module:
        module+='\ndeclare i32 @llvm.smax.i32(i32, i32)\n'
    return module+'\n'+remap(function)+'\n'+remap(metadata)+'\n'


@core.builtin
def publish_per_lane(Status,GlobalPrefix,Row,Bin,Tile,Tiles,Passes,PassId,N,_builder=None):
    args=[core._to_tensor(x,_builder) for x in [Status,GlobalPrefix,Row,Bin,Tile,Tiles,Passes,PassId,N]]
    dtype=args[3].type
    return core.tensor(_builder.create_extern_elementwise('', '', '__c500_Triton_publish',
        [x.handle for x in args],dtype.to_ir(_builder),False),dtype)
