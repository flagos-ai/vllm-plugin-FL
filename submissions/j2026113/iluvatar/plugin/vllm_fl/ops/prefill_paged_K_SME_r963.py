"""Experiment: four physical K pages into the original transposed shared view.

Each physical page is copied in (token, head-dimension) order. Four page
descriptors form (64,128), then memdesc_trans exposes the exact (128,64) K
consumer view. No FP arithmetic or attention recurrence is changed.
"""
import hashlib
import json
import re
from pathlib import Path

SSA=r'%[A-Za-z0-9_.$]+'


def rewrite(text):
    all_lines=text.splitlines(keepends=True)
    starts=[i for i,s in enumerate(all_lines) if s.startswith('  tt.func ')]
    choices=[]
    for ordinal,begin in enumerate(starts):
        end=starts[ordinal+1] if ordinal+1<len(starts) else next(i for i in range(begin+1,len(all_lines)) if all_lines[i].startswith('} loc('))
        body=''.join(all_lines[begin:end])
        if 'tensor<128x128xf32' in body and ' = scf.for ' in body:choices.append((begin,end))
    assert len(choices)==1
    begin,end=choices[0];lines=all_lines[begin:end]
    allocs=[(i,re.match(rf'(\s*)({SSA}) = ttg.local_alloc ({SSA}) : \(tensor<128x64xbf16, (#[A-Za-z0-9_]+)>\) -> (!ttg.memdesc<128x64xbf16, (#[A-Za-z0-9_]+), (#[A-Za-z0-9_]+)>)',s))
            for i,s in enumerate(lines) if ' = ttg.local_alloc ' in s and '(tensor<128x64xbf16,' in s]
    assert len(allocs)==1 and allocs[0][1]
    alloc_at,m=allocs[0];indent,desc,loaded,layout,old_type,k_shared,space=m.groups()
    load_at=alloc_at-1
    assert re.match(rf'\s*{re.escape(loaded)} = tt.load ',lines[load_at])
    consumers=[i for i,s in enumerate(lines) if ' = ttg.local_load '+desc+' :' in s]
    assert len(consumers)==1 and consumers[0]>alloc_at
    pageids=[]
    for index in range(4):
        matches=[re.match(rf'\s*({SSA}) = arith.extsi {SSA} : i32 to i64',s)
                 for s in lines[:load_at] if re.match(rf'\s*%page_id{index}(?:_[0-9]+)? = arith.extsi ',s)]
        assert len(matches)==1 and matches[0]
        pageids.append(matches[0][1])
    assert '%kv_head_idx' in ''.join(lines[:load_at]) and '%stride_k_cache_1:' in lines[0]
    seq_offsets=[re.match(rf'\s*(%seq_offset) = arith.muli {SSA}, {SSA} : i32',s) for s in lines[:load_at] if re.match(r'\s*%seq_offset = arith.muli ',s)]
    assert len(seq_offsets)==1 and seq_offsets[0]
    prefixes=set(re.findall(r'tt.splat (%max_seq_prefix_len[A-Za-z0-9_.$]*) : i32', ''.join(lines[:load_at])))
    assert len(prefixes)==1
    prefix=next(iter(prefixes))
    row_shared=[re.match(r'(#[A-Za-z0-9_]+) = ',s)[1] for s in all_lines
                if '#ttg.swizzled_shared<' in s and 'order = [1, 0]' in s and 'useTcu = true' in s]
    assert len(row_shared)==1
    v_shared=row_shared[0]
    full_v=f'!ttg.memdesc<64x128xbf16, {v_shared}, {space}, mutable>'
    full_k=f'!ttg.memdesc<128x64xbf16, {k_shared}, {space}, mutable>'
    sub=f'!ttg.memdesc<16x128xbf16, {v_shared}, {space}, mutable>'
    frames=f'!ttg.memdesc<4x16x128xbf16, #kp_frames_shared, {space}, mutable>'
    rows='tensor<16xi32, #ttg.slice<{dim = 1, parent = #kp_sme}>>'
    rows64='tensor<16xi64, #ttg.slice<{dim = 1, parent = #kp_sme}>>'
    cols='tensor<128xi32, #ttg.slice<{dim = 0, parent = #kp_sme}>>'
    cols64='tensor<128xi64, #ttg.slice<{dim = 0, parent = #kp_sme}>>'
    one_col='tensor<16x1xi64, #kp_sme>';one_row='tensor<1x128xi64, #kp_sme>';matrix='tensor<16x128xi64, #kp_sme>'
    ptr='tensor<16x128x!tt.ptr<bf16>, #kp_sme>';mask='tensor<16x128xi1, #kp_sme>';value='tensor<16x128xbf16, #kp_sme>'
    block=[f'%kp_rows = tt.make_range {{end = 16 : i32, start = 0 : i32}} : {rows}',
           f'%kp_cols = tt.make_range {{end = 128 : i32, start = 0 : i32}} : {cols}',
           f'%kp_rows64 = arith.extsi %kp_rows : {rows} to {rows64}',
           f'%kp_cols64 = arith.extsi %kp_cols : {cols} to {cols64}',
           f'%kp_stride_rows = tt.splat %stride_k_cache_1 : i64 -> {rows64}',
           f'%kp_row_offset = arith.muli %kp_rows64, %kp_stride_rows : {rows64}',
           '%kp_head64 = arith.extsi %kv_head_idx : i32 to i64',
           '%kp_head_offset = arith.muli %kp_head64, %stride_k_cache_2 : i64',
           f'%kp_heads = tt.splat %kp_head_offset : i64 -> {rows64}',
           f'%kp_row_head = arith.addi %kp_row_offset, %kp_heads : {rows64}',
           f'%kp_vertical = tt.expand_dims %kp_row_head {{axis = 1 : i32}} : {rows64} -> {one_col}',
           f'%kp_horizontal = tt.expand_dims %kp_cols64 {{axis = 0 : i32}} : {cols64} -> {one_row}',
           f'%kp_vfull = tt.broadcast %kp_vertical : {one_col} -> {matrix}',
           f'%kp_hfull = tt.broadcast %kp_horizontal : {one_row} -> {matrix}',
           f'%kp_inner = arith.addi %kp_vfull, %kp_hfull : {matrix}',
           f'%kp_frames = ttg.local_alloc : () -> {frames}',
           f'%kp_buffer = ttg.memdesc_reshape %kp_frames : {frames} -> {full_v}',
           f'{desc} = ttg.memdesc_trans %kp_buffer {{order = array<i32: 1, 0>}} : {full_v} -> {full_k}',
           '%kp_stride = arith.trunci %stride_k_cache_1 : i64 to i32',
           '%kp_width = arith.constant 64 : i32',
           '%kp_end = arith.addi %seq_offset, %kp_width : i32',
           f'%kp_full = arith.cmpi sle, %kp_end, {prefix} : i32']
    fullcopies=[];tail=[];tailcopies=[]
    mask_rows='tensor<16xi32, #ttg.slice<{dim = 1, parent = #kp_mask}>>'
    mask_bools='tensor<16xi1, #ttg.slice<{dim = 1, parent = #kp_mask}>>'
    for i,page in enumerate(pageids):
        block += [f'%kp_offset{i} = arith.muli {page}, %stride_k_cache_0 : i64',
                  f'%kp_base{i} = tt.addptr %key_cache_ptr, %kp_offset{i} : !tt.ptr<bf16>, i64',
                  f'%kp_splat{i} = tt.splat %kp_base{i} : !tt.ptr<bf16> -> {ptr}',
                  f'%kp_pointer{i} = tt.addptr %kp_splat{i}, %kp_inner : {ptr}, {matrix}',
                  f'%kp_index{i} = arith.constant {i} : i32',
                  f'%kp_dst{i} = ttg.memdesc_index %kp_frames[%kp_index{i}] : {frames} -> {sub}']
        attributes='{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, contiguity = 1 : i32}'
        fullcopies.append(f'%kp_full_token{i} = "ttg.async_copy_global_to_local"(%kp_pointer{i}, %kp_dst{i}, %kp_stride) {attributes} : ({ptr}, {sub}, i32) -> !ttg.async.token')
        tail += [f'%kp_tail_rows{i} = tt.make_range {{end = 16 : i32, start = 0 : i32}} : {mask_rows}',
                 f'%kp_tail_offset{i} = arith.constant {i*16} : i32',
                 f'%kp_tail_begin{i} = arith.addi %seq_offset, %kp_tail_offset{i} : i32',
                 f'%kp_tail_base{i} = tt.splat %kp_tail_begin{i} : i32 -> {mask_rows}',
                 f'%kp_tail_position{i} = arith.addi %kp_tail_rows{i}, %kp_tail_base{i} : {mask_rows}',
                 f'%kp_tail_limit{i} = tt.splat {prefix} : i32 -> {mask_rows}',
                 f'%kp_tail_valid{i} = arith.cmpi slt, %kp_tail_position{i}, %kp_tail_limit{i} : {mask_rows}',
                 f'%kp_tail_column{i} = tt.expand_dims %kp_tail_valid{i} {{axis = 1 : i32}} : {mask_bools} -> tensor<16x1xi1, #kp_mask>',
                 f'%kp_tail_broadcast{i} = tt.broadcast %kp_tail_column{i} : tensor<16x1xi1, #kp_mask> -> tensor<16x128xi1, #kp_mask>',
                 f'%kp_tail_predicate{i} = ttg.convert_layout %kp_tail_broadcast{i} : tensor<16x128xi1, #kp_mask> -> {mask}',
                 f'%kp_tail_zero{i} = arith.constant dense<0.000000e+00> : {value}']
        attributes='{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, contiguity = 1 : i32}'
        tailcopies.append(f'%kp_tail_token{i} = "ttg.async_copy_global_to_local"(%kp_pointer{i}, %kp_dst{i}, %kp_tail_predicate{i}, %kp_tail_zero{i}, %kp_stride) {attributes} : ({ptr}, {sub}, {mask}, {value}, i32) -> !ttg.async.token')
    block+=['%kp_commit = scf.if %kp_full -> (!ttg.async.token) {']
    block+=['  '+s for s in fullcopies]+['  %kp_full_commit = ttg.async_commit_group tokens '+', '.join(f'%kp_full_token{i}' for i in range(4)),
        '  scf.yield %kp_full_commit : !ttg.async.token','} else {']
    block+=['  '+s for s in tail+tailcopies]+['  %kp_tail_commit = ttg.async_commit_group tokens '+', '.join(f'%kp_tail_token{i}' for i in range(4)),
        '  scf.yield %kp_tail_commit : !ttg.async.token','}', '%kp_wait = ttg.async_wait %kp_commit {num = 0 : i32}']
    at=consumers[0]
    lines[at]=lines[at].replace('ttg.local_load '+desc+' :','ttg.local_load '+desc+' token %kp_wait :').replace(old_type,full_k)
    lines[load_at:alloc_at+1]=[indent+s+'\n' for s in block]
    definitions=[
        '#kp_sme = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [1, 0], isSme = true, smeMask = false, smeWarpsPerCTA = [1, 4]}>\n',
        '#kp_mask = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [1, 0], isSme = true, smeMask = true, smeWarpsPerCTA = [1, 4]}>\n',
        '#kp_frames_shared = #ttg.swizzled_shared<{vec = 32, perPhase = 1, maxPhase = 1, order = [2, 1, 0], useTcu = true}>\n']
    changed=''.join(definitions+all_lines[:begin]+lines+all_lines[end:])
    floats=lambda s:[l for l in s.splitlines() if any(t in l for t in [' = tt.dot ',' = arith.addf ',' = arith.mulf ',' = arith.divf ',' = math.exp ',' = "tt.reduce"'])]
    assert floats(changed)==floats(text)
    return changed,{'four_scalar_physical_pages':pageids,'K_shared_transpose_view':True,'floating_statement_order_identical':True,
        'original_K_register_read_and_shared_store_replaced':True,'short_query_branch_unchanged':True,'original_K_shared_type':old_type}

