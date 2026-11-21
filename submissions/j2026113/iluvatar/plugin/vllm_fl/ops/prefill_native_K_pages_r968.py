"""Write physical K pages directly into views of the native K allocation."""
import re

from .prefill_paged_K_SME_r963 import rewrite as page_rewrite

SSA=r'%[A-Za-z0-9_.$]+'


def rewrite(text):
    changed,receipt=page_rewrite(text)
    lines=changed.splitlines(keepends=True)
    original=receipt['original_K_shared_type']
    m=re.match(r'!ttg.memdesc<128x64xbf16, (#[A-Za-z0-9_]+), (#[A-Za-z0-9_]+)>',original)
    assert m
    shared,space=m.groups()
    full=f'!ttg.memdesc<128x64xbf16, {shared}, {space}, mutable>'
    split=f'!ttg.memdesc<128x4x16xbf16, #kp_native_split_shared, {space}, mutable>'
    frames=f'!ttg.memdesc<4x128x16xbf16, #kp_frames_shared, {space}, mutable>'
    sub=f'!ttg.memdesc<128x16xbf16, {shared}, {space}, mutable>'
    old_alloc=[i for i,s in enumerate(lines) if '%kp_frames = ttg.local_alloc' in s]
    assert len(old_alloc)==1
    i=old_alloc[0];assert '%kp_buffer = ttg.memdesc_reshape' in lines[i+1]
    desc=re.match(rf'\s*({SSA}) = ttg.memdesc_trans %kp_buffer ',lines[i+2]);assert desc
    indent=re.match(r'\s*',lines[i]).group()
    lines[i:i+3]=[
        indent+f'{desc[1]} = ttg.local_alloc : () -> {full}\n',
        indent+f'%kp_native_split = ttg.memdesc_reshape {desc[1]} : {full} -> {split}\n',
        indent+f'%kp_frames = ttg.memdesc_trans %kp_native_split {{order = array<i32: 1, 0, 2>}} : {split} -> {frames}\n']
    old_frames=re.findall(r'!ttg.memdesc<4x16x128xbf16, #kp_frames_shared, #[A-Za-z0-9_]+, mutable>',changed)
    assert old_frames
    # V-shaped pointer generation is retained and then transposed into the
    # official column-SME ownership. The destination uses native K layout.
    rowptr='tensor<16x128x!tt.ptr<bf16>, #kp_sme>'
    colptr='tensor<128x16x!tt.ptr<bf16>, #kp_col_sme>'
    rowmask='tensor<16x128xi1, #kp_sme>'
    colmask='tensor<128x16xi1, #kp_col_sme>'
    colvalue='tensor<128x16xbf16, #kp_col_sme>'
    result=[]
    for line in lines:
        if line.startswith('#kp_frames_shared = '):
            line=('#kp_frames_shared = #ttg.shared_linear<{offset = [[0, 1, 0], [0, 0, 1], [0, 0, 2], [0, 8, 0], [0, 16, 0], [0, 2, 0], [0, 4, 0], [0, 8, 4], [0, 16, 8], [0, 32, 0], [0, 64, 0], [1, 0, 0], [2, 0, 0]]}, alignment = 16>\n')
        if re.search(r'%kp_dst\d+ = ttg.memdesc_index',line):
            line=line.replace(old_frames[0],frames)
            line=re.sub(r'!ttg.memdesc<16x128xbf16, #[A-Za-z0-9_]+, '+re.escape(space)+r', mutable>',sub,line)
        if '"ttg.async_copy_global_to_local"' in line and '%kp_' in line:
            match=re.search(r'%kp_(?:full|tail)_token(\d+) = ',line);assert match
            index=match[1];lead=re.match(r'\s*',line).group()
            result.append(lead+f'%kp_col_pointer{index}_{"tail" if "tail_token" in line else "full"} = tt.trans %kp_pointer{index} {{order = array<i32: 1, 0>}} : {rowptr} -> {colptr}\n')
            pointer=f'%kp_col_pointer{index}_'+('tail' if 'tail_token' in line else 'full')
            line=line.replace(f'(%kp_pointer{index},',f'({pointer},').replace(rowptr,colptr)
            line=re.sub(r'!ttg.memdesc<16x128xbf16, #[A-Za-z0-9_]+, '+re.escape(space)+r', mutable>',sub,line)
            if 'tail_token' in line:
                result.append(lead+f'%kp_col_predicate{index} = tt.trans %kp_tail_predicate{index} {{order = array<i32: 1, 0>}} : {rowmask} -> {colmask}\n')
                result.append(lead+f'%kp_col_zero{index} = arith.constant dense<0.000000e+00> : {colvalue}\n')
                line=line.replace(f'%kp_tail_predicate{index}, %kp_tail_zero{index}',f'%kp_col_predicate{index}, %kp_col_zero{index}')
                line=line.replace(rowmask,colmask).replace('tensor<16x128xbf16, #kp_sme>',colvalue)
        result.append(line)
    definitions=[
        '#kp_native_split_shared = #ttg.swizzled_shared<{vec = 32, perPhase = 1, maxPhase = 1, order = [0, 2, 1], useTcu = true}>\n',
        '#kp_col_sme = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [0, 1], isSme = true, smeMask = false, smeWarpsPerCTA = [4, 1]}>\n']
    final=''.join(definitions+result)
    assert f'{desc[1]} = ttg.local_alloc : () -> {full}' in final
    assert final.count('ttg.async_copy_global_to_local')==changed.count('ttg.async_copy_global_to_local')
    receipt.update(shared_K_layout_identical=True,native_K_allocation_with_page_subviews=True,
                   column_SME_physical_page_copy=True,no_register_K_layout_conversion=True)
    return final,receipt

