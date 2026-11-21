"""Instance-runtime prefill scope with optional matched-input observation."""
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
from vllm.v1.kv_cache_interface import KVQuantMode

from .attention_prefill_paged_V_r675 import unified_attention as candidate

_parent=None
_cache_ordinals={}
_counts={}
_records=[]


def bind_parent_attention(parent):
    global _parent
    assert _parent is None
    path=Path(inspect.getfile(parent));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=='a61956351b8d807bb0e5ffa2cad859d13179c4203eafea78a461be4f7590a29f'
    _parent=parent
    return {'parent_file':str(path),'parent_SHA256':digest,
        'prefill_source':inspect.getfile(candidate),
        'prefill_source_SHA256':hashlib.sha256(Path(inspect.getfile(candidate)).read_bytes()).hexdigest(),
        'scope':'BF16causalprefillmaxQ>1,Q16/KV2/head128,page16/commonKVstrides8192/256/128/1/aligned64bytebases,noextras. Prior r581 attention retained for all other inputs. Same K64/softmax ordering, Q reuse, nativeMMA and GPU metadata; V producer only.',
        'operator_qualification':'r672 seven fixtures + r673 six same-graph ownership/page/length/NaN states. Short33tokenoperatoroverheadremains; no universal gain claim.'}


def run_candidate(kw):
    root=os.environ.get('TIANSHU_PREFILL_PAGED_V_VERIFY')
    selected=False;ordinal=None;call=None
    if root:
        phase=Path(root)/'phase.txt'
        if phase.exists() and phase.read_text().strip()=='wave':
            key=kw['k'].data_ptr()
            if key not in _cache_ordinals:_cache_ordinals[key]=len(_cache_ordinals)
            ordinal=_cache_ordinals[key];_counts[ordinal]=_counts.get(ordinal,0)+1;call=_counts[ordinal]
            selected=ordinal in [0,20,41] and call<=2
    if selected:
        assert not torch.cuda.is_current_stream_capturing()
        before_q=kw['q'].clone()
        before_meta={name:kw[name].clone() for name in ['cu_seqlens_q','seqused_k','block_table']}
        before_rng=torch.cuda.get_rng_state().clone()
    result=candidate(**kw)
    if selected:
        reference=torch.empty_like(kw['out']);old=dict(kw,out=reference)
        _parent(**old);torch.cuda.synchronize()
        out=kw['out'];checks={'attention_BF16_parent_exact':bool(torch.equal(out,reference)),
            'output_finite':bool(torch.isfinite(out).all()),'Q_bits_unchanged':bool(torch.equal(kw['q'].view(torch.int16),before_q.view(torch.int16))),
            'metadata_unchanged':all(torch.equal(kw[name],value) for name,value in before_meta.items()),
            'RNG_state_unchanged':bool(torch.equal(before_rng,torch.cuda.get_rng_state()))}
        cpu=out.detach().contiguous().cpu()
        row={'cache_first_encounter_ordinal':ordinal,'prefill_call_for_cache':call,'shape':list(kw['q'].shape),
            'Q_stride':list(kw['q'].stride()),'K_stride':list(kw['k'].stride()),'V_stride':list(kw['v'].stride()),
            'max_seqlen_q':kw['max_seqlen_q'],'checks':checks,
            'output_SHA256':hashlib.sha256(cpu.view(torch.int16).numpy().tobytes()).hexdigest(),
            'different_BF16_elements':int((out.view(torch.int16)!=reference.view(torch.int16)).sum())}
        _records.append(row);directory=Path(root);directory.mkdir(exist_ok=True)
        state={'complete':all(all(record['checks'].values()) for record in _records),'diagnostic_only':True,
            'model_returns_candidate_output':True,'verified_calls':_records,'unique_cache_pointers_seen':len(_cache_ordinals),
            'scope':'Matched old/new actual model Q/K/V/metadata, selected cache ordinals0/20/41 for first two prefill calls. Extra old execution/copies/sync changes latency; not a throughput or quality score. Full KV mutation checks are from r673 operator test, not copies of whole live cache.'}
        if not all(checks.values()):
            torch.save({'q':before_q.cpu(),'parent_out':reference.cpu(),'candidate_out':cpu,
                'cu':before_meta['cu_seqlens_q'].cpu(),'lens':before_meta['seqused_k'].cpu(),
                'table':before_meta['block_table'].cpu()},directory/'failed_actual_prefill.pt')
        (directory/'state.json').write_text(json.dumps(state,indent=2));assert all(checks.values()),row
    return result


def unified_attention(q,k,v,out,cu_seqlens_q,max_seqlen_q,seqused_k,max_seqlen_k,softmax_scale,causal,window_size,block_table,softcap,q_descale,k_descale,v_descale,seq_threshold_3D=None,num_par_softmax_segments=None,softmax_segm_output=None,softmax_segm_max=None,softmax_segm_expsum=None,alibi_slopes=None,output_scale=None,qq_bias=None,sinks=None,mm_prefix_range=None,use_alibi_sqrt=False,kv_quant_mode:KVQuantMode=KVQuantMode.NONE,k_scale_cache=None,v_scale_cache=None,chunk_lookback=-1,use_td:bool=False):
    if _parent is None:raise RuntimeError('Bind prior attention before using paged prefill')
    kw=dict(q=q,k=k,v=v,out=out,cu_seqlens_q=cu_seqlens_q,max_seqlen_q=max_seqlen_q,seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,softmax_scale=softmax_scale,causal=causal,window_size=window_size,
        block_table=block_table,softcap=softcap,q_descale=q_descale,k_descale=k_descale,v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,softmax_segm_max=softmax_segm_max,softmax_segm_expsum=softmax_segm_expsum,
        alibi_slopes=alibi_slopes,output_scale=output_scale,qq_bias=qq_bias,sinks=sinks,mm_prefix_range=mm_prefix_range,
        use_alibi_sqrt=use_alibi_sqrt,kv_quant_mode=kv_quant_mode,k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,chunk_lookback=chunk_lookback,use_td=use_td)
    eligible=(max_seqlen_q>1 and q.is_cuda and q.dtype==k.dtype==v.dtype==out.dtype==torch.bfloat16
        and q.shape[1:]==(16,128) and k.shape[1:]==v.shape[1:]==(16,2,128)
        and k.stride()==v.stride()==(8192,256,128,1) and q.stride(1)==128 and q.stride(2)==1
        and k.data_ptr()%64==0 and v.data_ptr()%64==0 and cu_seqlens_q.shape[0]-1<=64
        and kv_quant_mode==KVQuantMode.NONE and causal and window_size==(-1,-1) and softcap==0.
        and alibi_slopes is None and output_scale is None and qq_bias is None and sinks is None
        and mm_prefix_range is None and not use_alibi_sqrt and q_descale is None
        and k_scale_cache is None and v_scale_cache is None and chunk_lookback==-1 and not use_td)
    if eligible:return run_candidate(kw)
    return _parent(**kw)
