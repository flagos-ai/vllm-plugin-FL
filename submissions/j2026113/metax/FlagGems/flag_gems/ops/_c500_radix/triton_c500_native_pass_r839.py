"""Native SDK warp-counter phase expressed in Triton, storage lowering local.

This isolates RankKeys before block scan, not a new ranking algorithm.
Temporary: native 32-byte scan prefix +128x4-byte aliased raking storage.
Warp counters: offset32+(digit*3+warp)*4, including native third-column pad.
"""
import triton
import triton.language as tl
from .primitives import ballot64,popcount64,warp_exchange
from triton.language import core
from .gems_builtin_discovery_r856 import install_project_builtin_discovery
from .triton_c500_row_product_r825b import original_row_product
from .triton_c500_typed_counter_r799 import counter_load,counter_store,inject_typed_counter,IDENTITY as COUNTER_IDENTITY
from .triton_c500_storage_r704b import native_lane_id,shared_load32,shared_store32,shared_load_plain32,shared_store_plain32,warp_sync,install_storage_lowering


@triton.jit
def native_counter_strip(digit,warp,lower):
    peer_mask=tl.full((128,),0xffffffffffffffff,tl.uint64)
    for bit in tl.static_range(4):
        predicate=(digit&(1<<bit)).to(tl.int32)
        mask=ballot64(predicate)
        peer_mask=peer_mask&tl.where(predicate!=0,mask,~mask)
    previous=counter_load(digit,warp)
    previous=warp_sync(previous)
    lower_count=popcount64(peer_mask&lower).to(tl.int32)
    count=popcount64(peer_mask).to(tl.int32)
    updated=counter_store(digit,warp,previous+count,(lower_count==0).to(tl.int32))
    warp_sync(updated)
    return previous+lower_count


from .triton_c500_storage_r704b import claimed_load,claimed_store,prefix_load,prefix_store,scatter_load,global_atomic_add_u32,global_atomic_xchg_u32,global_load_u32
from .triton_c500_storage_r704b import global_masked_load32,global_masked_store32
from .triton_c500_storage_r704b import shared_load32_masked,prefix_load_masked,scatter_load_masked
from .triton_c500_storage_r704b import input_pair32,output_pair32,claim_and_publish,physical_thread_id
from .triton_c500_publish_r682 import publish_per_lane,inject_publish,IDENTITY
from .triton_c500_input_abi_r777d import input_per_lane,inject_input,IDENTITY as INPUT_IDENTITY
from .triton_c500_scatter_abi_r838 import scatter_per_lane,inject_scatter,IDENTITY as SCATTER_IDENTITY


@triton.jit
def input_strip(Values,Indices,row64,N,tile,logical,OFFSET,DESC):
    raw,payload,ordered=input_per_lane(Values,Indices,original_row_product(row64.to(tl.int32),N),N,tile,logical,OFFSET,DESC.to(tl.int32))
    digit=((ordered.to(tl.uint32,bitcast=True)>>OFFSET)&15).to(tl.int32)
    return raw.to(tl.uint32,bitcast=True),payload,digit


@triton.jit
def output_strip(Out,OutIndices,row64,N,tile,logical,raw,payload,digit,rank):
    scatter_per_lane(Out,OutIndices,original_row_product(row64.to(tl.int32),N),N,tile,logical,
                     raw.to(tl.int32,bitcast=True),payload,digit,rank)


@triton.jit(do_not_specialize=[9,10,11,12,13,14,15,16])
def native_digit_pass(Values,Indices,Out,OutIndices,GlobalPrefix,Status,Tickets,Flags,Errors,
                      M,N,Tiles,Passes,PassId,Offset,Descending,Check):
    row=tl.program_id(1)
    row64=row.to(tl.uint64)
    run=True
    if Check:run=global_load_u32(Flags+row,0)!=0
    if run:
        tid=physical_thread_id(tl.arange(0,128))
        lane=tid%64
        warp=tid//64
        claim_and_publish(Tickets+row+tid*0,(tid==0).to(tl.int32))
        tl.debug_barrier()
        tile=claimed_load(tid)
        raw_0,payload_0,digit_0=input_strip(Values,Indices,row64,N,tile,tid+warp*960+0*64,Offset,Descending)
        raw_1,payload_1,digit_1=input_strip(Values,Indices,row64,N,tile,tid+warp*960+1*64,Offset,Descending)
        raw_2,payload_2,digit_2=input_strip(Values,Indices,row64,N,tile,tid+warp*960+2*64,Offset,Descending)
        raw_3,payload_3,digit_3=input_strip(Values,Indices,row64,N,tile,tid+warp*960+3*64,Offset,Descending)
        raw_4,payload_4,digit_4=input_strip(Values,Indices,row64,N,tile,tid+warp*960+4*64,Offset,Descending)
        raw_5,payload_5,digit_5=input_strip(Values,Indices,row64,N,tile,tid+warp*960+5*64,Offset,Descending)
        raw_6,payload_6,digit_6=input_strip(Values,Indices,row64,N,tile,tid+warp*960+6*64,Offset,Descending)
        raw_7,payload_7,digit_7=input_strip(Values,Indices,row64,N,tile,tid+warp*960+7*64,Offset,Descending)
        raw_8,payload_8,digit_8=input_strip(Values,Indices,row64,N,tile,tid+warp*960+8*64,Offset,Descending)
        raw_9,payload_9,digit_9=input_strip(Values,Indices,row64,N,tile,tid+warp*960+9*64,Offset,Descending)
        raw_10,payload_10,digit_10=input_strip(Values,Indices,row64,N,tile,tid+warp*960+10*64,Offset,Descending)
        raw_11,payload_11,digit_11=input_strip(Values,Indices,row64,N,tile,tid+warp*960+11*64,Offset,Descending)
        raw_12,payload_12,digit_12=input_strip(Values,Indices,row64,N,tile,tid+warp*960+12*64,Offset,Descending)
        raw_13,payload_13,digit_13=input_strip(Values,Indices,row64,N,tile,tid+warp*960+13*64,Offset,Descending)
        raw_14,payload_14,digit_14=input_strip(Values,Indices,row64,N,tile,tid+warp*960+14*64,Offset,Descending)
        raw_15,payload_15,digit_15=input_strip(Values,Indices,row64,N,tile,tid+warp*960+15*64,Offset,Descending)
        shared_store_plain32(32+tid*4,0,1)
        tl.debug_barrier()
        rank_lane=native_lane_id(tid)
        lower=(tl.full((128,),1,tl.uint64)<<rank_lane.to(tl.uint64))-1
        rank_0=native_counter_strip(digit_0,warp,lower)
        rank_1=native_counter_strip(digit_1,warp,lower)
        rank_2=native_counter_strip(digit_2,warp,lower)
        rank_3=native_counter_strip(digit_3,warp,lower)
        rank_4=native_counter_strip(digit_4,warp,lower)
        rank_5=native_counter_strip(digit_5,warp,lower)
        rank_6=native_counter_strip(digit_6,warp,lower)
        rank_7=native_counter_strip(digit_7,warp,lower)
        rank_8=native_counter_strip(digit_8,warp,lower)
        rank_9=native_counter_strip(digit_9,warp,lower)
        rank_10=native_counter_strip(digit_10,warp,lower)
        rank_11=native_counter_strip(digit_11,warp,lower)
        rank_12=native_counter_strip(digit_12,warp,lower)
        rank_13=native_counter_strip(digit_13,warp,lower)
        rank_14=native_counter_strip(digit_14,warp,lower)
        rank_15=native_counter_strip(digit_15,warp,lower)
        tl.debug_barrier()
        original=shared_load_plain32(32+tid*4)
        aggregate_lane=native_lane_id(tid)
        scan_lane=native_lane_id(tid)
        inclusive=original
        for step in tl.static_range(6):
            distance=1<<step
            source_lane=native_lane_id(tid)
            previous=source_lane-distance
            group_start=source_lane&-64
            previous=tl.where(previous<group_start,source_lane,previous)
            moved=warp_exchange(inclusive,previous)
            inclusive+=tl.where(scan_lane<distance,0,moved)
        exclusive=inclusive-original
        shared_store_plain32(warp*4,inclusive,(aggregate_lane==63).to(tl.int32))
        tl.debug_barrier()
        warp0_total=shared_load_plain32(0)
        prefix=tl.where(tid<64,0,warp0_total)+tl.where(aggregate_lane==0,0,exclusive)
        shared_store_plain32(32+tid*4,prefix,1)
        tl.debug_barrier()
        rank_0+=counter_load(digit_0,warp)
        rank_1+=counter_load(digit_1,warp)
        rank_2+=counter_load(digit_2,warp)
        rank_3+=counter_load(digit_3,warp)
        rank_4+=counter_load(digit_4,warp)
        rank_5+=counter_load(digit_5,warp)
        rank_6+=counter_load(digit_6,warp)
        rank_7+=counter_load(digit_7,warp)
        rank_8+=counter_load(digit_8,warp)
        rank_9+=counter_load(digit_9,warp)
        rank_10+=counter_load(digit_10,warp)
        rank_11+=counter_load(digit_11,warp)
        rank_12+=counter_load(digit_12,warp)
        rank_13+=counter_load(digit_13,warp)
        rank_14+=counter_load(digit_14,warp)
        rank_15+=counter_load(digit_15,warp)
        # Same lower16 prefix reads followed by separate 17word shared array.
        bin_prefix=shared_load32_masked(32+tid*3*4,(tid<16).to(tl.int32))
        prefix_store(tid,bin_prefix,(tid<16).to(tl.int32))
        prefix_store(16+tid*0,tid*0+2048,(tid==0).to(tl.int32))
        tl.debug_barrier()
        lo=prefix_load_masked(tid,(tid<16).to(tl.int32)).to(tl.uint32)
        hi=prefix_load_masked(tid+1,(tid<16).to(tl.int32)).to(tl.uint32)
        padding=tl.maximum(0,(tile+1)*2048-N)
        count=hi-lo-tl.where(tid==15,padding,0).to(tl.uint32)
        cell=(row64*16+tid.to(tl.uint64))*Tiles.to(tl.uint64)+tile.to(tl.uint64)
        global_atomic_xchg_u32(Status+cell,(0x40000000|count).to(tl.int32),(tid<16).to(tl.int32))
        tl.debug_barrier()
        publish_per_lane(Status,GlobalPrefix,row,tid,tile,Tiles,Passes,PassId,N)
        tl.debug_barrier()
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+0*64,raw_0,payload_0,digit_0,rank_0)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+1*64,raw_1,payload_1,digit_1,rank_1)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+2*64,raw_2,payload_2,digit_2,rank_2)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+3*64,raw_3,payload_3,digit_3,rank_3)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+4*64,raw_4,payload_4,digit_4,rank_4)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+5*64,raw_5,payload_5,digit_5,rank_5)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+6*64,raw_6,payload_6,digit_6,rank_6)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+7*64,raw_7,payload_7,digit_7,rank_7)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+8*64,raw_8,payload_8,digit_8,rank_8)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+9*64,raw_9,payload_9,digit_9,rank_9)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+10*64,raw_10,payload_10,digit_10,rank_10)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+11*64,raw_11,payload_11,digit_11,rank_11)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+12*64,raw_12,payload_12,digit_12,rank_12)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+13*64,raw_13,payload_13,digit_13,rank_13)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+14*64,raw_14,payload_14,digit_14,rank_14)
        output_strip(Out,OutIndices,row64,N,tile,tid+warp*960+15*64,raw_15,payload_15,digit_15,rank_15)


def inject_native_phases(module):
    return inject_typed_counter(inject_input(inject_scatter(inject_publish(module))))


install_storage_lowering(native_digit_pass,extra_lowering=inject_native_phases,extra_identity=IDENTITY+SCATTER_IDENTITY+INPUT_IDENTITY+COUNTER_IDENTITY)


def launch_digit(values,indices,out,out_indices,prefix,status,tickets,flags,
                 *,pass_id,offset,errors,descending=False,check_low16=False):
    m,n=values.shape
    if n<1 or n>=2**30 or not 0<m<=65535 or offset!=pass_id*4 or not 0<=pass_id<8:
        raise ValueError('Original native radix4 contract')
    return native_digit_pass[(triton.cdiv(n,2048),m)](values,indices,out,out_indices,prefix,
        status,tickets,flags,errors,m,n,triton.cdiv(n,2048),8,pass_id,offset,descending,check_low16,
        num_warps=2,num_stages=1)


install_project_builtin_discovery(globals())
