import importlib
import importlib.util
import json

import torch
import triton
import vllm

import flaggems_vllm as gems
from baseline import mhc_pre as workload
from flaggems_vllm.utils import libentry,libtuner
from flaggems_vllm.utils.libentry import LibTuner
from flaggems_vllm.utils.triton_version_utils import HAS_TLE

m=importlib.import_module('flaggems_vllm.ops.mhc.mhc_pre')
spec=importlib.util.spec_from_file_location('oracle','/candidate/tests/test_mhc_with_norm.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
entry=m.mhc_pre_packed_reader_kernel
tuner=entry.fn
print('PREFLIGHT',json.dumps({'is_hopper':m._IS_HOPPER,'tuner_type':type(tuner).__name__,'is_libtuner':isinstance(tuner,LibTuner)}),flush=True)
assert m._IS_HOPPER and isinstance(tuner,LibTuner)
configs=list(tuner.configs)
assert len(configs)==15
caps=[c for c in configs if c.maxnreg==128]
assert not caps
assert all(c.num_stages==1 for c in configs if c.kwargs['BLOCK_M']>=64)
receipt={'source':m.__file__,'torch':torch.__version__,'triton':triton.__version__,'vllm':vllm.__version__,'tle':HAS_TLE,'arch_guard':m._IS_HOPPER,'tuner':type(tuner).__name__,'config_count':len(configs),'note':'per-config forcing is validation only; final gate must use actual automatic source','repeats_per_config':3,'rows':[]}
print('RUNTIME',json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)
for binding in workload.grid():
    tokens,hidden=binding['shape']
    r,fn,s,b,n=ref._inputs(tokens=tokens,hidden=hidden)
    x=torch.randn(tokens,hidden,dtype=torch.bfloat16,device='cuda')
    p=torch.randn(tokens,4,1,device='cuda')*.3
    c=torch.randn(tokens,4,4,device='cuda')*.2
    r=gems.mhc_post(x,r,p,c)
    expected=ref._pre_reference(r,fn,s,b,repeats=10)
    for index,config in enumerate(configs):
        one=libentry()(libtuner(configs=[config],key=['M','stride_am'])(tuner.fn))
        m.mhc_pre_packed_reader_kernel=one
        errors=[]
        with torch.inference_mode():
            for repeat in range(3):
                actual=gems.mhc_pre(r,fn,s,b,1e-6,1e-6,1e-6,1.,10)
                for a,e in zip(actual[:2],expected[:2]):torch.testing.assert_close(a,e,rtol=2e-5,atol=2e-5)
                torch.testing.assert_close(actual[2],expected[2],rtol=.01,atol=.01)
                errors.append([(a-e).abs().max().item() for a,e in zip(actual[:2],expected[:2])])
        compiled=next(iter(one.kernel_cache[0].values()))[0]
        if config.maxnreg is not None:
            assert compiled.metadata.maxnreg==config.maxnreg
            assert compiled.n_regs<=config.maxnreg
        row={'shape':binding['shape'],'config_index':index,'config':str(config),'split':m._mhc_pre_reader_s4(tokens,64,128,hidden*4),'correct':True,'max_abs':errors,'n_regs':compiled.n_regs,'n_spills':compiled.n_spills,'metadata_maxnreg':compiled.metadata.maxnreg}
        receipt['rows'].append(row);print('RESULT',json.dumps(row),flush=True)
    with open('/probe/hopper-source-strong15.json','w') as f:json.dump(receipt,f,indent=2)
m.mhc_pre_packed_reader_kernel=entry
print('ALL_PASS',len(receipt['rows']),flush=True)
