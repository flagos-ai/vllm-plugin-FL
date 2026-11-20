import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch
import triton
import triton.testing
import vllm

import flaggems_vllm as gems
from baseline import mhc_pre as workload
from flaggems_vllm.utils.libentry import LibTuner
from flaggems_vllm.utils.triton_version_utils import HAS_TLE

label=sys.argv[1]
reverse='reverse' in label
critical='critical' in label
diagnostic='nosplit' in label
m=importlib.import_module('flaggems_vllm.ops.mhc.mhc_pre')
entry=m.mhc_pre_packed_reader_kernel
tuner=entry.fn
assert m._IS_HOPPER and isinstance(tuner,LibTuner)
assert len(tuner.configs)==15 and all(c.maxnreg is None for c in tuner.configs)
assert torch.__version__.startswith('2.13') and triton.__version__=='3.6.0' and vllm.__version__.startswith('0.32') and HAS_TLE
spec=importlib.util.spec_from_file_location('oracle','/candidate/tests/test_mhc_with_norm.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
native=workload.native()
receipt={'label':label,'torch':torch.__version__,'triton':triton.__version__,'triton_path':triton.__file__,'vllm':vllm.__version__,'tle':HAS_TLE,'source_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),'yaml_sha256':hashlib.sha256(Path('/candidate/src/flaggems_vllm/runtime/backend/_nvidia/hopper/tune_configs.yaml').read_bytes()).hexdigest(),'native':native.__module__+'.'+native.__name__,'tuner':type(tuner).__name__,'protocol':{'scope':'three original failing shapes, source ablation diagnostic' if diagnostic else 'original12 pinned shapes; critical reverse rerun independent process','timing':'repository kernel do_bench','warmup_ms':30,'rep_ms':80,'paired_repeats':3,'order':'candidate,native' if reverse else 'native,candidate','coefficient_rtol':2e-5,'coefficient_atol':2e-5,'native_rtol':.01,'native_atol':.01,'weights':'normal version-tracked fn created outside inference mode','versionless_limit':'repacking cost is outside frozen benchmark inputs; no performance claim for versionless weights','production_state':'no function/config/cache injection; automatic actual source tuning'},'rows':[]}
print('RUNTIME',json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)
for binding in workload.grid():
    if critical and binding['shape']!=[8192,1280]:continue
    if diagnostic and tuple(binding['shape']) not in {(2048,2560),(2048,4096),(8192,1280)}:continue
    args,kwargs=workload.build_inputs(binding,torch.bfloat16,'cuda')
    assert not args[1].is_inference()
    expected=ref._pre_reference(args[0],args[1],args[2],args[3],repeats=args[8])
    before=set(entry.kernel_cache[0])
    with torch.inference_mode():
        actual=gems.mhc_pre(*args,**kwargs)
        baseline=native(*args,**kwargs)
        for a,e in zip(actual[:2],expected[:2]):torch.testing.assert_close(a,e,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(actual[2],expected[2],rtol=.01,atol=.01)
        for a,e in zip(actual,baseline):torch.testing.assert_close(a,e,rtol=.01,atol=.01)
        added=set(entry.kernel_cache[0])-before
        assert len(added)==1
        compiled,consts,*_=entry.kernel_cache[0][added.pop()]
        ns=[];cs=[]
        for repeat in range(3):
            if reverse:
                cs.append(triton.testing.do_bench(lambda:gems.mhc_pre(*args,**kwargs),warmup=30,rep=80,return_mode='median')*1000)
                ns.append(triton.testing.do_bench(lambda:native(*args,**kwargs),warmup=30,rep=80,return_mode='median')*1000)
            else:
                ns.append(triton.testing.do_bench(lambda:native(*args,**kwargs),warmup=30,rep=80,return_mode='median')*1000)
                cs.append(triton.testing.do_bench(lambda:gems.mhc_pre(*args,**kwargs),warmup=30,rep=80,return_mode='median')*1000)
    row={'shape':binding['shape'],'split':m._mhc_pre_reader_s4(args[0].shape[0],64,128,args[0].shape[-1]*4),'candidate_us':statistics.median(cs),'candidate_samples_us':cs,'native_us':statistics.median(ns),'native_samples_us':ns,'speedup':statistics.median(ns)/statistics.median(cs),'selected_config':str(tuner.best_config),'launch_consts':consts,'metadata_maxnreg':compiled.metadata.maxnreg,'n_regs':compiled.n_regs,'n_spills':compiled.n_spills,'shared':compiled.metadata.shared,'accuracy':'passed'}
    if compiled.metadata.maxnreg is not None:assert compiled.n_regs<=compiled.metadata.maxnreg
    receipt['rows'].append(row)
    print('RESULT',json.dumps(row),flush=True)
    with open('/probe/hopper-gate-'+label+'.json','w') as f:json.dump(receipt,f,indent=2)
receipt['summary']={'passed':sum(r['speedup']>=.9 for r in receipt['rows']),'total':len(receipt['rows']),'minimum':min(r['speedup'] for r in receipt['rows'])}
with open('/probe/hopper-gate-'+label+'.json','w') as f:json.dump(receipt,f,indent=2)
print('SUMMARY',json.dumps(receipt['summary']),flush=True)
