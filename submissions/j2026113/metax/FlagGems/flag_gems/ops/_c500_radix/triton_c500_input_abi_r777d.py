"""Register-result ABI for the scalar conditional compiled from Triton.

The helper algorithm comes entirely from r775b's compiled Triton source.
Its three artificial output stores become struct return fields. Three TTIR
result markers are combined into one LLVM call before native optimization;
they do not produce three device calls, duplicate loads or temporary memory.
"""
import hashlib
from pathlib import Path
import re
from triton.language import core

HELPER=Path(__file__).with_name('scalar_input_helper_r776d.ll').read_text()
IDENTITY=hashlib.sha256(HELPER.encode()+Path(__file__).read_bytes()).hexdigest()
FIELDS={'raw':0,'payload':1,'ordered':2}


def inject_input(module):
    for field in FIELDS:
        module,count=re.subn(r'(?m)^declare[^\n]*@__c500_Triton_input_'+field+r'\([^\n]*\)[^\n]*$','',module)
        assert count==1,field
    match=re.search(r'(?m)^define[^\n]*@scalar_input_phase\(([^\n]*)\)[^\n]*\{',HELPER)
    assert match
    end=HELPER.index('\n}',match.end())+2
    function=HELPER[match.start():end]
    args=match.group(1).split(', ')
    assert len(args)==11
    # Name all anonymous LLVM locals before removing three parameters.
    function=re.sub(r'%(\d+)\b',lambda m:'%input_'+m.group(1),function)
    function=re.sub(r'(?m)^(\d+):',lambda m:'input_'+m.group(1)+':',function)
    signature=','.join(args[:2]+args[5:])
    signature=re.sub(r'%(\d+)\b',lambda m:'%input_'+m.group(1),signature)
    # Naming the former numeric arguments also requires naming the implicit
    # first block, which originally followed the eleven anonymous parameters.
    function='define internal {i32,i32,i32} @__c500_Triton_input('+signature+') alwaysinline nounwind convergent {\ninput_11:'+function[function.index('\n'):]
    values=[]
    for index in [2,3,4]:
        pattern=r'(?m)^\s*store i32 ([^,\n]+), ptr addrspace\(1\) %input_'+str(index)+r', align 4[^\n]*$'
        stores=re.findall(pattern,function);assert len(stores)==1,(index,stores)
        values.append(stores[0]);function,count=re.subn(pattern,'',function);assert count==1
    assert function.count('ret void')==1
    registers=('  %input_result_0 = insertvalue {i32,i32,i32} undef, i32 '+values[0]+', 0\n'
               '  %input_result_1 = insertvalue {i32,i32,i32} %input_result_0, i32 '+values[1]+', 1\n'
               '  %input_result_2 = insertvalue {i32,i32,i32} %input_result_1, i32 '+values[2]+', 2\n'
               '  ret {i32,i32,i32} %input_result_2')
    function=function.replace('ret void',registers)
    assert not re.search(r'!\d+',function) and 'store ' not in function and 'alloca' not in function
    pattern=r'(?m)^( *)(%[-a-zA-Z$._0-9]+) = (?:tail )?call i32 @__c500_Triton_input_(raw|payload|ordered)\(([^\n]*)\)[^\n]*$'
    groups={};seen={};calls=0
    def replace(match):
        nonlocal calls
        indent,result,field,args=match.groups();calls+=1
        prefix=''
        if args not in groups:
            name='%c500_input_pack_'+str(len(groups));groups[args]=name;seen[args]=set()
            prefix=indent+name+' = call {i32,i32,i32} @__c500_Triton_input('+args+')\n'
        assert field not in seen[args]
        seen[args].add(field)
        return prefix+indent+result+' = extractvalue {i32,i32,i32} '+groups[args]+', '+str(FIELDS[field])
    module,count=re.subn(pattern,replace,module)
    assert calls==count==48 and len(groups)==16 and all(s==set(FIELDS) for s in seen.values()),(count,len(groups))
    return module+'\n'+function+'\n'


@core.builtin
def input_per_lane(Values,Indices,Row,N,Tile,Logical,Offset,Desc,_builder=None):
    args=[core._to_tensor(v,_builder) for v in [Values,Indices,Row,N,Tile,Logical,Offset,Desc]]
    dtype=args[5].type
    out=[]
    for field in FIELDS:
        handle=_builder.create_extern_elementwise('','','__c500_Triton_input_'+field,
            [v.handle for v in args],dtype.to_ir(_builder),False)
        out.append(core.tensor(handle,dtype))
    return tuple(out)
