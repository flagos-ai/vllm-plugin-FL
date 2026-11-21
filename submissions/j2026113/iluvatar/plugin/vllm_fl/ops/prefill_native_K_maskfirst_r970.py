"""Prepare tail mask scratch before the native K allocation, no copy changes."""
import re

from .prefill_native_K_pages_r968 import rewrite as native_rewrite


def rewrite(text):
    changed,receipt=native_rewrite(text)
    lines=changed.splitlines(keepends=True)
    alloc=[i for i,s in enumerate(lines) if re.search(r'= ttg.local_alloc : \(\) -> !ttg.memdesc<128x64xbf16',s)]
    assert len(alloc)==1
    alloc=alloc[0]
    conditions=[i for i,s in enumerate(lines) if '%kp_width =' in s or '%kp_end =' in s or '%kp_full =' in s]
    assert len(conditions)==3 and min(conditions)>alloc
    begin=next(i for i,s in enumerate(lines) if '%kp_commit = scf.if %kp_full' in s)
    tail_begin=next(i for i in range(begin+1,len(lines)) if lines[i].strip()=='} else {')
    tail_end=next(i for i in range(tail_begin+1,len(lines)) if '%kp_tail_commit = ' in lines[i])
    segment=lines[tail_begin+1:tail_end]
    preparation=[];keep=[]
    for line in segment:
        if '"ttg.async_copy_global_to_local"' in line or '%kp_col_pointer' in line:keep.append(line)
        else:preparation.append(line)
    assert len(keep)==8
    assert len([s for s in preparation if '%kp_col_predicate' in s])==4
    assert all(not any(op in s for op in ['tt.load ','ttg.local_alloc','ttg.local_load','ttg.async_copy']) for s in preparation)
    indent=re.match(r'\s*',lines[alloc]).group()
    predicate='tensor<128x16xi1, #kp_col_sme>'
    # Full pages do not require mask scratch. Tail-only conversion still uses
    # the same exact source DAG/ownership as the preceding numerical candidate.
    block=[lines[i][len(indent):].rstrip() for i in conditions]
    block += [f'%kp_prepared_masks:4 = scf.if %kp_full -> ({", ".join([predicate]*4)}) {{',
              f'  %kp_prepared_alltrue = arith.constant dense<true> : {predicate}',
              '  scf.yield '+', '.join(['%kp_prepared_alltrue']*4)+' : '+', '.join([predicate]*4),
              '} else {']
    tail_indent=len(re.match(r'\s*',preparation[0]).group())
    block += ['  '+s[tail_indent:].rstrip() for s in preparation]
    block += ['  scf.yield '+', '.join(f'%kp_col_predicate{i}' for i in range(4))+' : '+', '.join([predicate]*4), '}']
    # Keep constant zero values outside the tail-preparation region as operands
    # of the existing masked copies. Their values/layouts are unchanged.
    for i in range(4):block.append(f'%kp_prepared_zero{i} = arith.constant dense<0.000000e+00> : tensor<128x16xbf16, #kp_col_sme>')
    for i in conditions:lines[i]=''
    for i in range(tail_begin+1,tail_end):
        if lines[i] in preparation:lines[i]=''
        elif '"ttg.async_copy_global_to_local"' in lines[i]:
            for page in range(4):
                lines[i]=lines[i].replace(f'%kp_col_predicate{page}, %kp_col_zero{page}',f'%kp_prepared_masks#{page}, %kp_prepared_zero{page}')
    lines[alloc:alloc]=[indent+s+'\n' for s in block]
    final=''.join(lines)
    assert final.count('%kp_full =')==1
    assert final.index('%kp_prepared_masks:4')<final.index(' = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16')
    receipt.update(K_mask_preparation_before_allocation=True,
                   page_copy_addresses_and_types_unchanged=True,
                   same_native_K_allocation_and_QK_consumer=True)
    return final,receipt

