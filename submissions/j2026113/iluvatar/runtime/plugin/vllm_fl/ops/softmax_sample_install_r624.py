"""Instance-scoped native sampler fusion, preserving filtering and RNG draws."""
import hashlib
import inspect
import json
import os
import types
from pathlib import Path

import torch
import triton

try:
    from .fused_softmax_sample_r622 import row_stats_kernel,output_sample
    from .fused_noise_argmax_r614 import reduce_noise_argmax
except ImportError:
    from fused_softmax_sample_r622 import row_stats_kernel,output_sample
    from fused_noise_argmax_r614 import reduce_noise_argmax


def make_noise(logits,generators,module):
    # Qualified logits and native softmax output have identical FP32 contiguous
    # shape/layout. Moving the draw before deterministic softmax changes no
    # ordering among RNG operations. Paired noise/state tests are mandatory.
    q=module.empty_exponential_noise_like(logits,False)
    if len(generators)!=logits.shape[0]:q.exponential_()
    if generators:
        for i,generator in generators.items():q[i].exponential_(generator=generator)
    return q


def launch(logits,noise,debug_prob=None,debug_score=None):
    m,n=logits.shape
    result=torch.empty((m,),device=logits.device,dtype=torch.int64)
    stats=torch.empty((m,2),device=logits.device,dtype=torch.float32)
    values=torch.empty((m,4),device=logits.device,dtype=torch.float32)
    indexes=torch.empty((m,4),device=logits.device,dtype=torch.int64)
    debug=debug_prob is not None
    prob_pointer=debug_prob if debug else logits
    score_pointer=debug_score if debug else logits
    first=row_stats_kernel[(m,)](logits,stats,m,n,4096,num_warps=16,num_stages=3)
    second=output_sample[(m,4)](logits,noise,stats,values,indexes,prob_pointer,score_pointer,n,4096,4,debug,num_warps=4)
    third=reduce_noise_argmax[(m,)](values,indexes,result,4,4,num_warps=4)
    return result,(first,second,third)


def eligible(instance,logits):
    return (not instance.use_fp64_gumbel and logits.ndim==2 and logits.dtype==torch.float32
        and logits.is_cuda and logits.is_contiguous() and 8<=logits.shape[0]<=64
        and logits.shape[1]==130560)


def install(sampler):
    import flag_gems
    from vllm.v1.sample.ops import topk_topp_sampler as module
    assert torch.cuda.get_device_capability()==(7,1)
    seen=set();verified=set();bound=[]
    stats={'fused_calls':0,'retained_calls':0}
    originals=[]
    for name,instance in sampler.named_modules():
        if not isinstance(instance,module.TopKTopPSampler):continue
        old=instance.forward
        if getattr(old,'__func__',None) is not module.TopKTopPSampler.forward_native:continue
        assert not getattr(instance,'_ts_softmax_sample_installed',False)
        originals.append((instance,old))
        def forward(self,logits,generators,k,p,_old=old,_name=name):
            if not eligible(self,logits):
                stats['retained_calls']+=1
                return _old(logits,generators,k,p)
            filtered=module.apply_top_k_top_p(logits,k,p)
            logits_to_return=None
            if self.logprobs_mode=='processed_logits':logits_to_return=filtered
            elif self.logprobs_mode=='processed_logprobs':logits_to_return=filtered.log_softmax(dim=-1,dtype=torch.float32)
            noise=make_noise(filtered,generators,module)
            verify_root=os.environ.get('TIANSHU_SOFTMAX_SAMPLE_VERIFY')
            selected=False
            if verify_root and filtered.shape[0]==64 and _name not in verified:
                phase=Path(verify_root)/'phase.txt'
                selected=phase.exists() and phase.read_text().strip()=='wave'
            if selected:
                verified.add(_name)
                assert not torch.cuda.is_current_stream_capturing()
                before_logits=filtered.clone();before_noise=noise.clone()
                rng_after_draw=torch.cuda.get_rng_state().clone()
                per_generator={i:g.get_state().clone() for i,g in generators.items()}
            answer,kernels=launch(filtered,noise)
            stats['fused_calls']+=1
            receipts=os.environ.get('TIANSHU_SOFTMAX_SAMPLE_RECEIPTS')
            key=(_name,filtered.shape[0])
            if receipts and key not in seen:
                seen.add(key);root=Path(receipts);root.mkdir(exist_ok=True)
                row={'sampler':_name,'M':filtered.shape[0],'kernel_hashes':[v.hash for v in kernels],
                    'shared_bytes':[v.metadata.shared for v in kernels],
                    'no_probability_or_score_matrix_store':True,'extra_workspace_bytes':filtered.shape[0]*(2*4+4*(4+8)),
                    'noise_draw_moved_before_deterministic_softmax':True,'RNG_draw_order_among_draws_unchanged':True}
                (root/f'M{filtered.shape[0]}.json').write_text(json.dumps(row,indent=2))
            if selected:
                reference_prob=filtered.softmax(dim=-1,dtype=torch.float32)
                reference_scores=reference_prob.clone()
                expected=module.sample_with_exponential_noise(reference_scores,noise.clone())
                debug_p=torch.empty_like(filtered);debug_s=torch.empty_like(filtered)
                debug_ids,_=launch(filtered,noise,debug_p,debug_s)
                torch.cuda.synchronize()
                checks={'ids_exact':bool(torch.equal(answer,expected)),
                    'debug_ids_exact':bool(torch.equal(debug_ids,expected)),
                    'probability_FP32_bits_exact':bool(torch.equal(debug_p.view(torch.int32),reference_prob.view(torch.int32))),
                    'score_FP32_bits_exact':bool(torch.equal(debug_s.view(torch.int32),reference_scores.view(torch.int32))),
                    'filtered_logits_unchanged':bool(torch.equal(filtered.view(torch.int32),before_logits.view(torch.int32))),
                    'noise_unchanged':bool(torch.equal(noise.view(torch.int32),before_noise.view(torch.int32))),
                    'global_RNG_unchanged_during_kernel_ops':bool(torch.equal(rng_after_draw,torch.cuda.get_rng_state())),
                    'request_generators_unchanged_during_kernel_ops':all(torch.equal(g.get_state(),per_generator[i]) for i,g in generators.items())}
                root=Path(verify_root);root.mkdir(exist_ok=True)
                path=root/'actual_filtered_logits_noise.pt'
                torch.save({'filtered_logits':before_logits.cpu(),'noise':before_noise.cpu(),
                    'original_ids':expected.cpu(),'candidate_ids':answer.cpu()},path)
                row={'complete':all(checks.values()),'diagnostic_only':True,'shape':list(filtered.shape),
                     'checks':checks,'model_returns_candidate_ids':True,'snapshot':str(path),
                     'snapshot_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'scope':'OneactualC64samplerfusioncall comparedonnativeprobability/scorebitsandidenticalnoise. Extraoldsoftmax/helper/debuglaunch/copies+syncperturblatency;not a performance score.'}
                (root/'state.json').write_text(json.dumps(row,indent=2))
                assert all(checks.values()),row
            return answer,logits_to_return
        instance.forward=types.MethodType(forward,instance)
        instance._ts_softmax_sample_installed=True
        bound.append({'name':name,'original_forward_file':old.__code__.co_filename,
            'original_forward_name':old.__name__,'use_fp64_gumbel':instance.use_fp64_gumbel,
            'logprobs_mode':instance.logprobs_mode})
    assert len(bound)==1,bound
    source=Path(inspect.getfile(flag_gems.softmax))
    receipt={'bindings':bound,'native_softmax_source':str(source),
        'native_softmax_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'native_recurrence':'4096/W16/stages3,FP32onlinevectorM/Zandreductionunchanged',
        'scope':'Onlycurrentlyselectedforward_nativeinstance,FP32contiguousM8..64/N130560/non-FP64Gumbel;originalforward retainedoutside. Existingfilter/logprobs/seededgeneratorsequencepreserved.',
        'observable_contract':'forward returnsIDs/logitsOrLogprobs only;ephemeralprobabilitiesandmutatedscoresarenotreturned,sofullmatrixstoreseliminated. HelperAPI not redefined.',
        'RNG_note':'Noise creation moved before deterministic softmax, samecontiguousshape/dtype/drawcountandper-generatororder. Pairednoisebyte/statechecksrequired;notliteraloperationorderidentity.'}
    return {'receipt':receipt,'stats':stats,'originals':originals}
