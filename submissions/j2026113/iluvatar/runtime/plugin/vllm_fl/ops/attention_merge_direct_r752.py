"""Scope a direct-sequence merge to the existing pure-decode-only module."""
import hashlib
import inspect
import json
import os
from pathlib import Path


def install(module, kernel):
    import torch
    path=Path(inspect.getfile(module))
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='a61956351b8d807bb0e5ffa2cad859d13179c4203eafea78a461be4f7590a29f'
    # This module calls its merge ONLY in _specialized_unified_attention, whose
    # existing outer eligibility requires max_seqlen_q==1. If total query count
    # equals sequence count, every sequence has one query (zero slots excluded).
    assert 'max_seqlen_q == 1' in inspect.getsource(module.unified_attention)
    parent=module.reduce_segments
    assert not getattr(parent,'_direct_merge_r752',False)
    stats={'selected_launches':0,'retained_launches':0,'records':[]}

    def eligible(grid,kw):
        out=kw.get('output_ptr');cu=kw.get('query_start_len_ptr');n=kw.get('num_seqs')
        return (isinstance(n,int) and 16<=n<=64 and out is not None and cu is not None
                and out.shape==(n,16,128) and out.dtype==torch.bfloat16
                and cu.shape==(n+1,) and tuple(grid)==(n,16)
                and kw.get('BLOCK_Q')==1 and kw.get('TILE_SIZE')==16
                and kw.get('NUM_SEGMENTS_PER_SEQ')==16 and kw.get('num_query_heads')==16
                and kw.get('HEAD_SIZE')==kw.get('HEAD_SIZE_PADDED')==128
                and kw.get('USE_FP8') is False)

    class Selected:
        _direct_merge_r752=True
        def __getitem__(self,grid):
            def launch(*args,**kw):
                if args or not eligible(grid,kw):
                    stats['retained_launches']+=1
                    return parent[grid](*args,**kw)
                options=dict(kw)
                assert options.get('num_warps',4)==4
                options['num_warps']=4
                result=kernel[grid](**options)
                stats['selected_launches']+=1
                if not any(r['queries']==kw['num_seqs'] for r in stats['records']):
                    record={'queries':kw['num_seqs'],'kernel_hash':result.hash,'shared_bytes':result.metadata.shared,
                            'num_warps':4,'native_module_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'scope':'One-query-per-sequence established by pure-only native caller plus totalqueries==numseq. Capture/launch receipt, not replay frequency.'}
                    stats['records'].append(record)
                    root=os.environ.get('TIANSHU_DIRECT_MERGE_RECEIPTS')
                    if root:
                        directory=Path(root);directory.mkdir(parents=True,exist_ok=True)
                        (directory/f'C{kw["num_seqs"]}.json').write_text(json.dumps(record,indent=2))
                return result
            return launch

    module.reduce_segments=Selected()
    return {'stats':stats,'eligible':eligible,'parent':parent,'selected':module.reduce_segments}


def install_on_runner(cls):
    import functools
    original=cls.load_model
    @functools.wraps(original)
    def load(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        from . import attention_decode_native_KSME_r581 as module
        from .attention_merge_direct_kernel_r752 import reduce_segments
        self._direct_merge_r752=install(module,reduce_segments)
        return result
    cls.load_model=load
