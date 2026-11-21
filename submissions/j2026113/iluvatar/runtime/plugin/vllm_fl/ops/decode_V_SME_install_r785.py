"""Select qualified V-SME kernel inside the current native decode scope."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path


def install(module,candidate):
    import torch
    assert hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()=='a61956351b8d807bb0e5ffa2cad859d13179c4203eafea78a461be4f7590a29f'
    parent=module.kernel_unified_attention
    assert not getattr(parent,'_V_SME_r785',False)
    stats={'selected_launches':0,'retained_launches':0,'receipts':[]}
    def eligible(grid,kw):
        n=kw.get('num_seqs');q=kw.get('query_ptr');k=kw.get('key_cache_ptr');v=kw.get('value_cache_ptr')
        return (isinstance(n,int) and 24<=n<=64 and kw.get('IS_3D') is True
                and kw.get('DIRECT_DECODE_SEQ') is True and tuple(grid)==(16,2,n)
                and kw.get('BLOCK_M')==8 and kw.get('TILE_SIZE')==kw.get('BLOCK_SIZE')==16
                and kw.get('HEAD_SIZE')==kw.get('HEAD_SIZE_PADDED')==128
                and kw.get('NUM_SEGMENTS_PER_SEQ')==16 and kw.get('KV_QUANT_MODE')==0
                and q is not None and k is not None and v is not None
                and q.dtype==k.dtype==v.dtype==torch.bfloat16
                and k.stride()==v.stride()==(8192,256,128,1)
                and q.stride(1)==128 and q.stride(2)==1)
    class Kernel:
        _V_SME_r785=True
        def __getitem__(self,grid):
            def launch(*args,**kw):
                if args or not eligible(grid,kw):
                    stats['retained_launches']+=1
                    return parent[grid](*args,**kw)
                value=candidate.kernel_unified_attention[grid](**kw)
                assert value.metadata.shared==4096
                stats['selected_launches']+=1
                if not any(v['num_seqs']==kw['num_seqs'] for v in stats['receipts']):
                    row={'num_seqs':kw['num_seqs'],'query_rows':kw['query_ptr'].shape[0],
                         'kernel_hash':value.hash,'shared_bytes':value.metadata.shared,
                         'scope':'Host launch/capture receipt, not per-replay frequency. Native caller eligibility unchanged; only24..64seq V-SME kernel choice.'}
                    stats['receipts'].append(row)
                    directory=os.environ.get('TIANSHU_V_SME_RECEIPTS')
                    if directory:
                        p=Path(directory);p.mkdir(parents=True,exist_ok=True)
                        (p/f'C{kw["num_seqs"]}.json').write_text(json.dumps(row,indent=2))
                return value
            return launch
    selected=Kernel();module.kernel_unified_attention=selected
    return {'parent':parent,'selected':selected,'eligible':eligible,'stats':stats}


def install_on_runner(cls):
    original=cls.load_model
    @functools.wraps(original)
    def load(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        from . import attention_decode_native_KSME_r581 as parent
        from . import attention_decode_V_SME_kernel_r784 as candidate
        from .decode_frontend_install_r766 import install as frontend
        self._V_SME_frontend_r785=frontend(candidate,Path(os.environ['TIANSHU_Q_OPERAND_CACHE'])/'V_SME_native_frontend')
        self._V_SME_r785=install(parent,candidate)
        return result
    cls.load_model=load
