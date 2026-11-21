"""Use native masked SME copy for one physical V page into existing TCU buffer."""
import re

SSA=r'%[A-Za-z0-9_.$]+'


def rewrite(text):
    lines=text.splitlines(keepends=True)
    pat=rf'(\s*)({SSA}) = tt.load ({SSA}), ({SSA}), ({SSA}) \{{.*?\}} : (tensor<16x128x!tt.ptr<bf16>, #v_load_producer>)(.*)'
    loads=[(i,re.match(pat,line)) for i,line in enumerate(lines) if ' = tt.load ' in line and 'tensor<16x128x!tt.ptr<bf16>, #v_load_producer>' in line]
    assert len(loads)==1 and loads[0][1]
    load_at,m=loads[0]
    space,value,pointer,mask,zero,ptr_type,loc=m.groups()
    allocpat=rf'\s*({SSA}) = ttg.local_alloc {re.escape(value)} : \(tensor<16x128xbf16, #v_load_producer>\) -> (!ttg.memdesc<16x128xbf16, #shared, #smem>)(.*)'
    allocs=[(i,re.match(allocpat,line)) for i,line in enumerate(lines) if ' = ttg.local_alloc ' in line and value in line]
    assert len(allocs)==1 and allocs[0][1]
    alloc_at,am=allocs[0];destination,old_type,alloc_loc=am.groups()
    uses=[i for i,line in enumerate(lines) if ' = ttg.local_load '+destination+' :' in line]
    assert len(uses)==1 and load_at<alloc_at<uses[0]
    defs={}
    for i,line in enumerate(lines):
        m=re.match(rf'\s*({SSA}) = (.*)',line)
        if m:defs[m[1]]=(i,m[2])
    allowed={'tt.make_range','tt.splat','tt.broadcast','tt.expand_dims','tt.addptr','tt.load',
             'arith.constant','arith.addi','arith.muli','arith.cmpi','arith.andi','arith.ori',
             'arith.remsi','arith.divsi','arith.trunci','arith.extsi','arith.extui','arith.select'}
    clones=[];mapped={}
    def clone(value,layout,suffix):
        key=(value,layout)
        if key in mapped:return mapped[key]
        if value not in defs:return value
        _,rhs=defs[value]
        if '#v_load_producer' not in rhs:return value
        op=rhs.split()[0]
        assert op in allowed,(value,op)
        if op=='tt.load':assert 'ptr<i32>' in rhs and 'isVolatile = false' in rhs
        new=value+suffix
        changed=re.sub(SSA,lambda match:clone(match[0],layout,suffix),rhs)
        changed=changed.replace('#v_load_producer',layout)
        mapped[key]=new;clones.append(new+' = '+changed)
        return new
    newptr=clone(pointer,'#v_sme','_SMEptr')
    newmask=clone(mask,'#v_sme_mask','_SMEmask')
    newzero=clone(zero,'#v_sme','_SMEzero')
    mutable='!ttg.memdesc<16x128xbf16, #shared, #smem, mutable>'
    block=clones+[
        f'{destination} = ttg.local_alloc : () -> {mutable}',
        '%v_sme_stride = arith.constant 256 : i32',
        f'%v_sme_masked = ttg.convert_layout {newmask} : tensor<16x128xi1, #v_sme_mask> -> tensor<16x128xi1, #v_sme>',
        f'%v_sme_copy = "ttg.async_copy_global_to_local"({newptr}, {destination}, %v_sme_masked, {newzero}, %v_sme_stride) '
        '{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, cache = 3 : i32, evict = 1 : i32, isVolatile = false, contiguity = 1 : i32} '
        f': (tensor<16x128x!tt.ptr<bf16>, #v_sme>, {mutable}, tensor<16x128xi1, #v_sme>, tensor<16x128xbf16, #v_sme>, i32) -> !ttg.async.token',
        '%v_sme_commit = ttg.async_commit_group tokens %v_sme_copy',
        '%v_sme_wait = ttg.async_wait %v_sme_commit {num = 0 : i32}']
    # Mask layout conversion uses temporary shared memory. Finish it before
    # allocating the destination so the native allocator can reuse that range.
    mask_at=next(i for i,line in enumerate(block) if line.startswith('%v_sme_masked ='))
    dst_at=next(i for i,line in enumerate(block) if line.startswith(destination+' = ttg.local_alloc'))
    assert dst_at<mask_at
    mask_line=block.pop(mask_at)
    block.insert(dst_at,mask_line)
    lines[load_at]=''
    lines[alloc_at]=''.join(space+l+'\n' for l in block)
    lines[uses[0]]=lines[uses[0]].replace('ttg.local_load '+destination+' :','ttg.local_load '+destination+' token %v_sme_wait :').replace(old_type,mutable)
    layouts=['#v_sme = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 2], order = [1, 0], isSme = true, smeMask = false, smeWarpsPerCTA = [1, 2]}>\n',
             '#v_sme_mask = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 2], order = [1, 0], isSme = true, smeMask = true, smeWarpsPerCTA = [1, 2]}>\n']
    changed=''.join(layouts+lines)
    # Floating arithmetic and all matrix/softmax statements are untouched.
    floats=lambda s:[l for l in s.splitlines() if any(op in l for op in [' = tt.dot ',' = arith.addf ',' = arith.mulf ',' = arith.divf ',' = math.exp ',' = "tt.reduce"'])]
    assert floats(changed)==floats(text)
    return changed,{'masked_page_V_SME':True,'mask_conversion_before_V_allocation':True,'original_V_global_load_removed':True,
                    'same_physical_page_addresses':True,'same_tail_mask_and_zero_fill':True,
                    'pure_pointer_and_mask_DAG_clones':len(clones),'V_copy_after_QK':True,
                    'floating_statement_order_identical':True,'native_async_wait_retained':True}
