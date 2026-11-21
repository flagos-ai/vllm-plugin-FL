"""Page-aware V shared producer for independent BF16 prefill attention copies."""
import hashlib
import json
import re
from pathlib import Path

SSA=r'%[A-Za-z0-9_.$]+'


def patch_source(source):
    old='''            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
                cache_modifier=(".cg" if DIRECT_DECODE_SEQ and IS_3D else ""),
            )'''
    assert source.count(old)==1
    new='''            if not IS_3D and not DIRECT_DECODE_SEQ and BLOCK_M == 128 and TILE_SIZE == 64 and HEAD_SIZE == 128 and BLOCK_SIZE == 16 and KV_QUANT_MODE == 0:
                page_rows = tl.arange(0, 16)
                page_columns = tl.arange(0, 128)
                page_table_base = block_tables_ptr + block_table_offset + j * 4
                page_id0 = tl.load(page_table_base).to(tl.int64)
                page_id1 = tl.load(page_table_base + 1).to(tl.int64)
                page_id2 = tl.load(page_table_base + 2).to(tl.int64)
                page_id3 = tl.load(page_table_base + 3).to(tl.int64)
                page_inner = page_rows[:, None] * stride_v_cache_1 + kv_head_idx * stride_v_cache_2 + page_columns[None, :] * stride_v_cache_3
                page_v0 = tl.load(value_cache_ptr + page_id0 * stride_v_cache_0 + page_inner, mask=(j * 64 + page_rows)[:, None] < max_seq_prefix_len, other=0.)
                page_v1 = tl.load(value_cache_ptr + page_id1 * stride_v_cache_0 + page_inner, mask=(j * 64 + page_rows + 16)[:, None] < max_seq_prefix_len, other=0.)
                page_v2 = tl.load(value_cache_ptr + page_id2 * stride_v_cache_0 + page_inner, mask=(j * 64 + page_rows + 32)[:, None] < max_seq_prefix_len, other=0.)
                page_v3 = tl.load(value_cache_ptr + page_id3 * stride_v_cache_0 + page_inner, mask=(j * 64 + page_rows + 48)[:, None] < max_seq_prefix_len, other=0.)
                page_low = tl.reshape(tl.permute(tl.join(page_v0, page_v1), (2, 0, 1)), (32, 128))
                page_high = tl.reshape(tl.permute(tl.join(page_v2, page_v3), (2, 0, 1)), (32, 128))
                V_load = tl.reshape(tl.permute(tl.join(page_low, page_high), (2, 0, 1)), (64, 128))
            else:
                V_load = tl.load(
                    value_cache_ptr + v_offset,
                    mask=dim_mask[None, :] & tile_mask[:, None],
                    other=0.0,
                    cache_modifier=(".cg" if DIRECT_DECODE_SEQ and IS_3D else ""),
                )'''
    return source.replace(old,new)


def rewrite(text):
    all_lines=text.splitlines(keepends=True)
    funcs=[i for i,l in enumerate(all_lines) if l.startswith('  tt.func ')]
    selected=[]
    for ordinal,begin in enumerate(funcs):
        end=funcs[ordinal+1] if ordinal+1<len(funcs) else next(i for i in range(begin+1,len(all_lines)) if all_lines[i].startswith('} loc('))
        body=''.join(all_lines[begin:end])
        if 'tensor<128x128xf32' in body and ' = scf.for ' in body:selected.append((begin,end))
    assert len(selected)==1
    begin,end=selected[0];lines=all_lines[begin:end]
    pattern=rf'(\s*)({SSA}) = tt.load ({SSA}), ({SSA}), ({SSA}) \{{.*?\}} : (tensor<16x128x!tt.ptr<bf16>, (#[A-Za-z0-9_]+)>)'
    loads=[re.match(pattern,line) for line in lines if ' = tt.load ' in line and 'tensor<16x128x!tt.ptr<bf16>' in line]
    assert len(loads)==4 and all(loads),len(loads)
    allocations=[(i,re.match(rf'(\s*)({SSA}) = ttg.local_alloc ({SSA}) : \(tensor<64x128xbf16, (#[A-Za-z0-9_]+)>\) -> (!ttg.memdesc<64x128xbf16, (#[A-Za-z0-9_]+), (#[A-Za-z0-9_]+)>)',line))
                 for i,line in enumerate(lines) if ' = ttg.local_alloc ' in line and '(tensor<64x128xbf16,' in line]
    direct_operand=False
    if allocations:
        assert len(allocations)==1 and allocations[0][1]
        at,alloc=allocations[0];indent,descriptor,old_value,value_layout,old_type,shared,space=alloc.groups()
    else:
        operands=[(i,re.match(rf'(\s*)({SSA}) = tt.reshape .* -> (tensor<64x128xbf16, #ttg.dot_op<.*?>>)',line))
                  for i,line in enumerate(lines) if ' = tt.reshape ' in line and '-> tensor<64x128xbf16, #ttg.dot_op<' in line]
        assert len(operands)==1 and operands[0][1],operands
        at,operand=operands[0];indent,operand_value,operand_type=operand.groups()
        shared_defs=[re.match(r'(#[A-Za-z0-9_]+) = ',line)[1] for line in all_lines if '#ttg.swizzled_shared<' in line and 'order = [1, 0]' in line and 'useTcu = true' in line]
        spaces=[re.match(r'(#[A-Za-z0-9_]+) = ',line)[1] for line in all_lines if '#ttg.shared_memory' in line]
        assert len(shared_defs)==len(spaces)==1
        shared=shared_defs[0];space=spaces[0];descriptor='%paged_buffer';old_type=None;direct_operand=True
    definitions={}
    for i,line in enumerate(lines):
        match=re.match(rf'\s*({SSA}) = (.*)',line)
        if match:definitions[match[1]]=(i,match[2])
    loops=[m[1] for line in lines if (m:=re.search(r'scf.for ('+SSA+r') = ',line))]
    prefixes=set(re.findall(r'tt.splat ('+SSA+r') : i32 ->', ''.join(l for l in lines if 'max_seq_prefix_len' in l)))
    prefixes={v for v in prefixes if v.startswith('%max_seq_prefix_len')}
    assert len(loops)==len(prefixes)==1,(loops,prefixes)
    max_prefix=next(iter(prefixes));loop=loops[0]
    assert '%stride_v_cache_1' in lines[0]
    aliases={}
    for line in all_lines:
        m=re.match(r'(#[A-Za-z0-9_]+) = #ttg.blocked<',line)
        if m and 'isSme = false' in line:aliases[m[1]]='#paged_mask_'+m[1][1:]
    for line in all_lines:
        m=re.match(r'(#[A-Za-z0-9_]+) = #ttg.slice<.*?parent = (#[A-Za-z0-9_]+)',line)
        if m and m[2] in aliases:aliases[m[1]]='#paged_mask_'+m[1][1:]
    alias_pattern=re.compile('|'.join(re.escape(v)+r'(?![A-Za-z0-9_])' for v in sorted(aliases,key=len,reverse=True)))
    mask_defs=[]
    for line in all_lines:
        m=re.match(r'(#[A-Za-z0-9_]+) = ',line)
        if m and m[1] in aliases:
            if '#ttg.blocked<' in line:
                mask_defs.append(aliases[m[1]]+' = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [1, 0], isSme = true, smeMask = true, smeWarpsPerCTA = [1, 4]}>\n')
            else:mask_defs.append(alias_pattern.sub(lambda m:aliases[m[0]],line))
    cloned={};clones=[]
    allowed={'tt.make_range','tt.splat','tt.broadcast','tt.expand_dims','arith.constant','arith.addi','arith.muli','arith.cmpi','arith.andi','arith.ori','arith.trunci','arith.extsi','arith.extui'}
    def clone_mask(value):
        if value in cloned:return cloned[value]
        if value not in definitions:return value
        _,rhs=definitions[value]
        if not alias_pattern.search(rhs):return value
        assert rhs.split()[0] in allowed,(value,rhs)
        name=value+'_paged_mask';changed=re.sub(SSA,lambda m:clone_mask(m[0]),rhs)
        changed=alias_pattern.sub(lambda m:aliases[m[0]],changed)
        cloned[value]=name;clones.append(name+' = '+changed);return name
    mask_values=[clone_mask(match[4]) for match in loads]
    full=f'!ttg.memdesc<64x128xbf16, {shared}, {space}, mutable>'
    sub=f'!ttg.memdesc<16x128xbf16, {shared}, {space}, mutable>'
    frames=f'!ttg.memdesc<4x16x128xbf16, #paged_shared_frames, {space}, mutable>'
    block=['%paged_frames = ttg.local_alloc : () -> '+frames,
           descriptor+' = ttg.memdesc_reshape %paged_frames : '+frames+' -> '+full,
           '%paged_stride = arith.trunci %stride_v_cache_1 : i64 to i32']
    tail=list(clones);copies=[]
    for index,match in enumerate(loads):
        _,_,pointer,mask,other,pointer_type,layout=match.groups()
        assert layout in aliases,layout
        ptr='tensor<16x128x!tt.ptr<bf16>, #paged_sme>'
        mtype='tensor<16x128xi1, #paged_sme>';vtype='tensor<16x128xbf16, #paged_sme>'
        block += [f'%paged_ptr{index} = ttg.convert_layout {pointer} : {pointer_type} -> {ptr}',
                  f'%paged_index{index} = arith.constant {index} : i32',
                  f'%paged_dst{index} = ttg.memdesc_index %paged_frames[%paged_index{index}] : {frames} -> {sub}']
        tail += [f'%paged_mask{index} = ttg.convert_layout {mask_values[index]} : tensor<16x128xi1, {aliases[layout]}> -> {mtype}',
                 f'%paged_other{index} = ttg.convert_layout {other} : tensor<16x128xbf16, {layout}> -> {vtype}']
        copies.append((
            f'%paged_token{index}_full = "ttg.async_copy_global_to_local"(%paged_ptr{index}, %paged_dst{index}, %paged_stride) '
            '{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, contiguity = 1 : i32} '
            f': ({ptr}, {sub}, i32) -> !ttg.async.token',
            f'%paged_token{index}_tail = "ttg.async_copy_global_to_local"(%paged_ptr{index}, %paged_dst{index}, %paged_mask{index}, %paged_other{index}, %paged_stride) '
            '{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, contiguity = 1 : i32} '
            f': ({ptr}, {sub}, {mtype}, {vtype}, i32) -> !ttg.async.token'))
    block += ['%paged_width = arith.constant 64 : i32',
              '%paged_begin = arith.muli '+loop+', %paged_width : i32',
              '%paged_end = arith.addi %paged_begin, %paged_width : i32',
              '%paged_full = arith.cmpi sle, %paged_end, '+max_prefix+' : i32',
              '%paged_commit = scf.if %paged_full -> (!ttg.async.token) {']
    block += ['  '+a for a,b in copies]
    block += ['  %paged_commit_full = ttg.async_commit_group tokens '+', '.join('%paged_token'+str(i)+'_full' for i in range(4)),
              '  scf.yield %paged_commit_full : !ttg.async.token','} else {']
    block += ['  '+line for line in tail]+['  '+b for a,b in copies]
    block += ['  %paged_commit_tail = ttg.async_commit_group tokens '+', '.join('%paged_token'+str(i)+'_tail' for i in range(4)),
              '  scf.yield %paged_commit_tail : !ttg.async.token','}']
    if direct_operand:
        consumers=[i for i,line in enumerate(lines) if ' = tt.dot ' in line and ', '+operand_value+', ' in line]
        assert len(consumers)==1 and consumers[0]>at
        consumer=consumers[0]
        lines[consumer:consumer]=[indent+'%paged_wait = ttg.async_wait %paged_commit {num = 0 : i32}\n',
            indent+operand_value+' = ttg.local_load '+descriptor+' token %paged_wait : '+full+' -> '+operand_type+'\n']
    else:
        consumers=[i for i,line in enumerate(lines) if re.search(r'ttg.local_load '+re.escape(descriptor)+r'(?![A-Za-z0-9_.$])',line)]
        assert len(consumers)==1 and consumers[0]>at
        consumer=consumers[0]
        lines[consumer]=lines[consumer].replace('ttg.local_load '+descriptor+' :','ttg.local_load '+descriptor+' token %paged_wait :').replace(old_type,full)
        lines[consumer:consumer]=[indent+'%paged_wait = ttg.async_wait %paged_commit {num = 0 : i32}\n']
    lines[at:at+1]=[indent+line+'\n' for line in block]
    extra=['#paged_sme = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [1, 0], isSme = true, smeMask = false, smeWarpsPerCTA = [1, 4]}>\n',
           '#paged_shared_frames = #ttg.swizzled_shared<{vec = 32, perPhase = 1, maxPhase = 1, order = [2, 1, 0], useTcu = true}>\n']
    return ''.join(extra+mask_defs+all_lines[:begin]+lines+all_lines[end:]),{'long_private_functions_changed':1,
        'per_page_V_loads_replaced':4,'wait_at_V_consumer':True,'short_query_branch_unchanged':True,
        'full_tail_condition':'loop_j*64+64 <= original GPU max_seq_prefix_len','original_V_allocation_type':old_type,'original_V_producer':'register_concat' if direct_operand else 'shared_alloc'}



def delay_long_K_staging(text):
    from collections import Counter
    lines=text.splitlines(keepends=True);functions=[i for i,l in enumerate(lines) if l.startswith('  tt.func ')]
    candidates=[]
    for ordinal,begin in enumerate(functions):
        end=functions[ordinal+1] if ordinal+1<len(functions) else next(i for i in range(begin+1,len(lines)) if lines[i].startswith('} loc('))
        body=''.join(lines[begin:end])
        if 'tensor<128x128xf32' in body and ' = scf.for ' in body:candidates.append((begin,end))
    assert len(candidates)==1
    begin,end=candidates[0];body=lines[begin:end]
    allocations=[(i,re.match(rf'\s*({SSA}) = ttg.local_alloc ({SSA}) : \(tensor<128x64xbf16, ',line)) for i,line in enumerate(body) if ' = ttg.local_alloc ' in line and '(tensor<128x64xbf16,' in line]
    assert len(allocations)==1 and allocations[0][1]
    alloc,match=allocations[0];descriptor,loaded=match.groups();load=alloc-1
    assert re.match(rf'\s*{re.escape(loaded)} = tt.load ',body[load])
    consumers=[i for i,line in enumerate(body) if re.search(r'ttg.local_load '+re.escape(descriptor)+r'(?![A-Za-z0-9_.$])',line)]
    assert len(consumers)==1 and consumers[0]>alloc
    consumer=consumers[0];crossed=''.join(body[alloc+1:consumer])
    assert 'ttg.async_commit_group' in crossed and 'tt.store ' not in crossed
    for value in [descriptor,loaded]:assert not re.search(re.escape(value)+r'(?![A-Za-z0-9_.$])',crossed)
    changed=body[:load]+body[alloc+1:consumer]+body[load:alloc+1]+body[consumer:]
    assert Counter(changed)==Counter(body)
    floating=lambda seq:[line for line in seq if any(op in line for op in ['tt.dot ','arith.addf ','arith.mulf ','math.exp ','tt.reduce '])]
    assert floating(changed)==floating(body)
    return ''.join(lines[:begin]+changed+lines[end:]),{'moved_K_global_load_and_shared_alloc':2,'statement_multiset_identical':True,'floating_statement_order_identical':True,'K_allocation_after_V_tail_mask_scratch':True}

def install(module,root):
    kernel=module.kernel_unified_attention;previous=kernel.compile
    closed={key:cell.cell_contents for key,cell in zip(previous.__code__.co_freevars,previous.__closure__)}
    backend_compile=closed['original_compile'];root=Path(root);root.mkdir(parents=True,exist_ok=False);receipts=[]
    def compile_local(src,target=None,options=None,**kwargs):
        constants={src.fn.arg_names[key[0]]:value for key,value in src.constants.items() if len(key)==1}
        ordinary=previous(src,target=target,options=options,**kwargs)
        eligible=(not constants.get('DIRECT_DECODE_SEQ') and not constants.get('IS_3D') and constants.get('BLOCK_M')==128
            and constants.get('TILE_SIZE')==64 and constants.get('HEAD_SIZE')==128 and constants.get('BLOCK_SIZE')==16
            and constants.get('KV_QUANT_MODE')==0 and src.signature.get('query_ptr')==src.signature.get('value_cache_ptr')=='*bf16')
        if not eligible:return ordinary
        assert target.backend=='corex' and target.arch==71 and target.warp_size==64
        assert ordinary.metadata.num_warps==8 and ordinary.metadata.num_stages==1
        assert not (options or {}).get('ir_override')
        (root/('ordinary_'+src.hash()+'.ttgir')).write_text(ordinary.asm['ttgir'])
        changed,receipt=rewrite(ordinary.asm['ttgir'])
        digest=hashlib.sha256(changed.encode()).hexdigest();path=root/(digest+'.ttgir');path.write_text(changed)
        from triton._C.libtriton import ir,passes
        from triton.backends import backends
        context=ir.context();ir.load_dialects(context);backends['iluvatar'].compiler(target).load_dialects(context)
        parsed=ir.parse_mlir_module(str(path),context);parsed.context=context
        pm=ir.pass_manager(context);passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.common.add_canonicalizer(pm);passes.common.add_cse(pm)
        pm.run(parsed,'Attention_per_page_V_SME');changed=str(parsed)
        changed,k_schedule=delay_long_K_staging(changed);receipt['K_staging_schedule']=k_schedule;path.write_text(changed)
        assert not any(' = tt.load ' in line and 'tensor<16x128x!tt.ptr<bf16>' in line for line in changed.splitlines())
        optimized=backend_compile(src,target=target,options=dict(options or {},ir_override=str(path)),**kwargs)
        asm=optimized.asm['asm'];(root/(digest+'.asm')).write_text(asm)
        sme=[line for line in asm.splitlines() if re.match(r'\s*sl_sme_load',line)];assert len(sme)>=8
        receipt.update(kernel_hash=optimized.hash,shared_before=ordinary.metadata.shared,shared_after=optimized.metadata.shared,
            VGPR=int(re.findall(r'workitem_vgpr_count = (\d+)',asm)[-1]),
            private_bytes=int(re.findall(r'workitem_private_segment_byte_size = (\d+)',asm)[-1]),
            SME_instructions=sme,ir_path=str(path),source_hash=src.hash())
        receipts.append(receipt);(root/'receipts.json').write_text(json.dumps(receipts,indent=2));return optimized
    kernel.compile=compile_local;return receipts
