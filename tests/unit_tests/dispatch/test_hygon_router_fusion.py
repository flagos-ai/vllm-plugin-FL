# Copyright (c) 2026 BAAI. All rights reserved.
import pytest
import torch
import flag_gems

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Hygon GPU required",
)
from vllm_fl.dispatch.backends.vendor.hygon.impl.native_ops.router_fusion import try_router_fusion
from vllm_fl.dispatch.backends.vendor.hygon.impl.native_ops.topk_softplus_sqrt import topk_softplus_sqrt_hygon


_GEMS_EXCLUDE=[]

@pytest.fixture(autouse=True,params=['all_gems','worker'])
def dispatch_profile(request):
    global _GEMS_EXCLUDE
    if request.param=='worker':
        from vllm_fl.utils import get_flag_gems_whitelist_blacklist
        _,_GEMS_EXCLUDE=get_flag_gems_whitelist_blacklist()
        assert 'sum_dim' in _GEMS_EXCLUDE
    else:
        _GEMS_EXCLUDE=[]


def gems_context():
    return flag_gems.use_gems(exclude=_GEMS_EXCLUDE)


def reference(x,b,ids,table,renorm=True,scale=1.5):
    raw=torch.sqrt(torch.nn.functional.softplus(x.float()))
    score=raw+b.float() if b is not None else raw
    chosen=table[ids.long()] if table is not None else torch.topk(score,6,dim=-1).indices
    w=raw.gather(1,chosen.long())
    if renorm:w=w/w.sum(-1,keepdim=True).clamp(min=1e-20)
    return w*scale,chosen


def tensors(m=64):
    x=torch.randn(m,256,device='cuda');b=torch.randn(256,device='cuda')*.1
    w=torch.empty(m,6,device='cuda');i=torch.empty(m,6,device='cuda',dtype=torch.int32)
    return x,b,w,i


@pytest.mark.parametrize('case',['random','std8','tiny','ties','partial_ties','near_ties','large_bias','negative','negative100','positive','no_bias','hash','hash_duplicates','hash_negative_token','no_renorm','zero_scale'])
def test_exact_expert_order_and_weights(case):
    torch.manual_seed(921);x,b,w,i=tensors();ids=table=None;renorm=True;scale=1.5
    if case=='std8':x.mul_(8.)
    elif case=='tiny':x.mul_(1e-7);b.zero_()
    elif case=='ties':x.zero_();b.zero_()
    elif case=='partial_ties':x.copy_(torch.randint(0,4,x.shape,device='cuda'));b.zero_()
    elif case=='near_ties':x.mul_(1e-6).add_(1.);b.zero_()
    elif case=='large_bias':b.fill_(1e8)
    elif case=='negative':x.fill_(-30.);b.zero_()
    elif case=='negative100':x.fill_(-100.);b.zero_()
    elif case=='positive':x.fill_(1000.);b.zero_()
    elif case=='no_bias':b=None
    elif case=='no_renorm':renorm=False
    elif case=='zero_scale':scale=0.
    elif case.startswith('hash'):
        ids=torch.arange(64,device='cuda',dtype=torch.int32);table=torch.randint(0,256,(128,6),device='cuda',dtype=torch.int64)
        if case=='hash_duplicates':table[:,1:].copy_(table[:,:1].expand(-1,5))
        if case=='hash_negative_token':ids.sub_(128)
    te=torch.full_like(i,-123)
    with gems_context():
        rw,ri=reference(x,b,ids,table,renorm,scale)
        assert try_router_fusion(w,i,x,renorm,scale,b,ids,table)
        assert torch.equal(i,ri)
        torch.testing.assert_close(w,rw,atol=0,rtol=0)
        topk_softplus_sqrt_hygon(w,i,te,x,renorm,scale,b,ids,table)
        assert torch.equal(i,ri) and torch.all(te==-123)
        torch.testing.assert_close(w,rw,atol=0,rtol=0)


@pytest.mark.parametrize('m',[1,8,32,64,128,256])
@pytest.mark.parametrize('hashed',[False,True])
def test_changed_routes_graph(m,hashed):
    x,b,w,i=tensors(m)
    ids=torch.arange(m,device='cuda',dtype=torch.int32) if hashed else None
    table=torch.randint(0,256,(256,6),device='cuda',dtype=torch.int32) if hashed else None
    te=torch.full_like(i,-456)
    with gems_context():
        topk_softplus_sqrt_hygon(w,i,te,x,True,1.5,b,ids,table)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):topk_softplus_sqrt_hygon(w,i,te,x,True,1.5,b,ids,table)
        for std in [.001,1.,8.]:
            x.normal_(std=std);b.normal_(std=.1)
            if hashed:ids.random_(0,256);table.random_(0,256)
            graph.replay();rw,ri=reference(x,b,ids,table)
            assert torch.equal(i,ri) and torch.all(te==-456)
            torch.testing.assert_close(w,rw,atol=0,rtol=0)


@pytest.mark.parametrize('case',['prefill','bf16','stride','alias','topk8','off','bias_dtype','ids_stride','table_stride','grad','plain_torch'])
def test_guarded_fallback(case,monkeypatch):
    x,b,w,i=tensors(257 if case=='prefill' else 64);ids=table=None
    if case=='bf16':x=x.bfloat16()
    elif case=='stride':x=torch.randn(64,512,device='cuda')[:,::2]
    elif case=='alias':w=x.reshape(-1)[:64*6].reshape(64,6)
    elif case=='topk8':w=torch.empty(64,8,device='cuda');i=torch.empty(64,8,device='cuda',dtype=torch.int32)
    elif case=='off':monkeypatch.setenv('VLLM_FL_HYGON_ROUTER_FUSION','0')
    elif case=='bias_dtype':b=b.bfloat16()
    elif case in ['ids_stride','table_stride']:
        ids=torch.arange(128,device='cuda',dtype=torch.int32)[::2];table=torch.randint(0,256,(128,6),device='cuda',dtype=torch.int32)
        if case=='table_stride':ids=ids.contiguous();table=torch.randint(0,256,(128,12),device='cuda',dtype=torch.int32)[:,::2]
    elif case=='grad':x.requires_grad_(True)
    if case=='plain_torch':
        assert not try_router_fusion(w,i,x,True,1.5,b)
        rw,ri=reference(x,b,None,None)
        topk_softplus_sqrt_hygon(w,i,torch.empty_like(i),x,True,1.5,b)
        assert torch.equal(i,ri);torch.testing.assert_close(w,rw,atol=0,rtol=0)
    else:
        with gems_context():assert not try_router_fusion(w,i,x,True,1.5,b,ids,table)


def test_hash_plain_torch_preserves_subnormal_softplus():
    x,b,w,i=tensors();x.normal_(mean=-30.,std=20.)
    ids=torch.arange(64,device='cuda',dtype=torch.int32)
    table=torch.randint(0,256,(128,6),device='cuda',dtype=torch.int32)
    assert try_router_fusion(w,i,x,True,1.5,b,ids,table)
    rw,ri=reference(x,b,ids,table)
    assert torch.equal(i,ri)
    torch.testing.assert_close(w,rw,atol=0,rtol=0)
