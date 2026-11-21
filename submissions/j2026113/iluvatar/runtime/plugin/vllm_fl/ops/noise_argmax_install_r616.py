"""Experiment-local sampling suffix fusion; original RNG/softmax prefixes kept."""
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import triton

try:
    from .fused_noise_argmax_r614 import noise_argmax,reduce_noise_argmax
except ImportError:
    from fused_noise_argmax_r614 import noise_argmax,reduce_noise_argmax


def eligible(probs,noise):
    return (probs.ndim==noise.ndim==2 and probs.shape==noise.shape
        and 8<=probs.shape[0]<=64 and probs.shape[1]==130560
        and probs.dtype==noise.dtype==torch.float32 and probs.is_cuda and noise.is_cuda
        and probs.device==noise.device and probs.stride(1)==noise.stride(1)==1
        and probs.stride(0)>=130560 and noise.stride(0)>=130560
        and probs.untyped_storage().data_ptr()!=noise.untyped_storage().data_ptr())


def install(module):
    assert not getattr(module,'_ts_noise_argmax_installed',False)
    assert torch.cuda.get_device_capability()==(7,1)
    old_helper=module.sample_with_exponential_noise
    old_compiled=module.compiled_random_sample
    old_source=inspect.getsource(old_compiled)
    assert 'probs = logits.softmax(dim=-1, dtype=torch.float32)' in old_source
    assert 'q = torch.empty_like(probs)' in old_source and 'q.exponential_()' in old_source
    assert 'return probs.div(q).argmax(dim=-1).view(-1)' in old_source
    seen=set();verified=set();stats={'fused_inplace':0,'fused_out_of_place':0,'retained_helper':0,'retained_compiled':0}
    receipt={'kernel_SHA256':hashlib.sha256(Path(noise_argmax.__code__.co_filename).read_bytes()).hexdigest() if hasattr(noise_argmax,'__code__') else hashlib.sha256(Path(noise_argmax.fn.__code__.co_filename).read_bytes()).hexdigest(),
        'original_helper_file':old_helper.__code__.co_filename,
        'original_compiled_file':old_compiled.__code__.co_filename,
        'scope':'CoreX71 FP32M8..64,N130560,unitinnerstrides,separatestorage;norm/positiveexponentialnoise producer unchanged. Other shapes/dtypes retain original functions.',
        'config':{'BLOCK':4096,'warps':4,'splits':4},
        'RNG_prefix':'Original softmax -> empty_like -> exponential_ sequence copied byte-for-byte in compiled wrapper. Seeded random_sample caller unchanged; helper only fuses after noise exists.',
        'inplace_contract':'Helper still overwrites probs with bitidenticalscores; noise unchanged. Compiled caller uses no score store, as original probabilities were not modified.'}
    def fused(probs,noise,inplace):
        m=probs.shape[0]
        verify_dir=os.environ.get('TIANSHU_NOISE_ARGMAX_VERIFY')
        selected=False
        if verify_dir and m==64 and inplace not in verified:
            phase=Path(verify_dir)/'phase.txt'
            selected=phase.exists() and phase.read_text().strip()=='wave'
        if selected:
            assert not torch.cuda.is_current_stream_capturing()
            verified.add(inplace)
            before_p=probs.clone();before_q=noise.clone()
            before_rng=torch.cuda.get_rng_state().clone()
        answer=torch.empty((m,),device=probs.device,dtype=torch.int64)
        partial_values=torch.empty((m,4),device=probs.device,dtype=torch.float32)
        partial_indexes=torch.empty((m,4),device=probs.device,dtype=torch.int64)
        first=noise_argmax[(m,4)](probs,noise,probs,answer,partial_values,partial_indexes,
            130560,probs.stride(0),noise.stride(0),4096,4,inplace,num_warps=4)
        second=reduce_noise_argmax[(m,)](partial_values,partial_indexes,answer,4,4,num_warps=4)
        if selected:
            expected_scores=before_p.clone()
            expected=old_helper(expected_scores,before_q.clone())
            torch.cuda.synchronize()
            effects=expected_scores if inplace else before_p
            checks={'ids_exact':bool(torch.equal(answer,expected)),
                'probability_side_effect_bitexact':bool(torch.equal(probs.view(torch.int32),effects.view(torch.int32))),
                'noise_unchanged':bool(torch.equal(noise.view(torch.int32),before_q.view(torch.int32))),
                'global_RNG_state_unchanged':bool(torch.equal(before_rng,torch.cuda.get_rng_state()))}
            root=Path(verify_dir);root.mkdir(exist_ok=True)
            snapshot=root/f'actual_probability_noise_{int(inplace)}.pt'
            torch.save({'probabilities':before_p.cpu(),'noise':before_q.cpu(),
                        'original_ids':expected.cpu(),'candidate_ids':answer.cpu(),
                        'inplace':inplace},snapshot)
            row={'complete':all(checks.values()),'diagnostic_only':True,'M':m,'N':130560,
                'inplace':inplace,'checks':checks,'snapshot':str(snapshot),
                'snapshot_SHA256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                'model_uses_candidate_ids':True,'RNG_operations_added':0,
                'scope':'OneactualC64suffixverifiedagainstoriginalhelperonidenticalprobs/noise. Extraoldhelper/copies/onesyncperturbtiming;not a performance score.'}
            (root/'state.json').write_text(json.dumps(row,indent=2))
            assert all(checks.values()),row
        stats['fused_inplace' if inplace else 'fused_out_of_place']+=1
        key=(inplace,m)
        directory=os.environ.get('TIANSHU_NOISE_ARGMAX_RECEIPTS')
        if directory and key not in seen:
            seen.add(key);root=Path(directory);root.mkdir(exist_ok=True)
            row={'M':m,'inplace':inplace,'kernel_hashes':[first.hash,second.hash],
                 'shared_bytes':[first.metadata.shared,second.metadata.shared],
                 'partial_bytes':m*4*(4+8),'RNG_operations_inside_candidate':0,
                 'original_prefix_retained':True}
            (root/f'{m}_{int(inplace)}.json').write_text(json.dumps(row,indent=2))
        return answer.view(-1)
    def helper(probs,q):
        if eligible(probs,q):return fused(probs,q,True)
        stats['retained_helper']+=1
        return old_helper(probs,q)
    def compiled(logits):
        if not(logits.ndim==2 and logits.dtype==torch.float32 and logits.is_cuda
               and 8<=logits.shape[0]<=64 and logits.shape[1]==130560):
            stats['retained_compiled']+=1
            return old_compiled(logits)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        return fused(probs,q,False)
    module.sample_with_exponential_noise=helper
    module.compiled_random_sample=compiled
    module._ts_noise_argmax_installed=True
    return {'receipt':receipt,'stats':stats,'original_helper':old_helper,
        'original_compiled':old_compiled,'helper':helper,'compiled':compiled}
