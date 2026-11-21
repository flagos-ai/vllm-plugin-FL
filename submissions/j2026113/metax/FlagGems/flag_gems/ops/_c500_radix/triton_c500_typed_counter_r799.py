"""Native SDK counter pointer [16][3], ordinary volatile memory primitives.

SDK provenance: native.actual.ll RankKeys uses generic in-bounds GEP
[16 x [3 x i32]], starting32bytes into the same544byte AS3 temporary.
The original radix4 kernel guarantees digit0..15,warp0..1. No extra states,
collectives, rank arithmetic, stores or fallback paths are introduced.
"""
import hashlib
from pathlib import Path
import re
from triton.language import core

IDENTITY=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@core.builtin
def counter_load(digit,warp,_builder=None):
    args=[core._to_tensor(v,_builder) for v in [digit,warp]];dtype=args[0].type
    return core.tensor(_builder.create_extern_elementwise('','','__c500_typed_counter_load',
                       [v.handle for v in args],dtype.to_ir(_builder),False),dtype)


@core.builtin
def counter_store(digit,warp,value,mask,_builder=None):
    args=[core._to_tensor(v,_builder) for v in [digit,warp,value,mask]];dtype=args[0].type
    return core.tensor(_builder.create_extern_elementwise('','','__c500_typed_counter_store',
                       [v.handle for v in args],dtype.to_ir(_builder),False),dtype)


def inject_typed_counter(module):
    for name in ['load','store']:
        module,count=re.subn(r'(?m)^declare[^\n]*@__c500_typed_counter_'+name+r'\([^\n]*\)[^\n]*$','',module)
        assert count==1,name
    pointer='''
  %digit64 = zext i32 %digit to i64
  %warp64 = zext i32 %warp to i64
  %p = getelementptr inbounds [16 x [3 x i32]], ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @__c500_rank_temporary to ptr), i64 32), i64 0, i64 %digit64, i64 %warp64
'''
    return module+'''
define internal i32 @__c500_typed_counter_load(i32 %digit, i32 %warp) alwaysinline nounwind convergent {
entry:
'''+pointer+'''
  %value = load volatile i32, ptr %p, align 4
  ret i32 %value
}
define internal i32 @__c500_typed_counter_store(i32 %digit, i32 %warp, i32 %value, i32 %mask) alwaysinline nounwind convergent {
entry:
  %active = icmp ne i32 %mask, 0
  br i1 %active, label %write, label %done
write:
'''+pointer+'''
  store volatile i32 %value, ptr %p, align 4
  br label %done
done:
  ret i32 %value
}
'''
