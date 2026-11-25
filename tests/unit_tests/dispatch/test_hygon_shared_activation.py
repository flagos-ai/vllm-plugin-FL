# Copyright (c) 2026 BAAI. All rights reserved.
import pytest
import torch
import flag_gems

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Hygon GPU required",
)

from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm_fl.dispatch.backends.vendor.hygon.patches.shared_activation import patch_shared_swiglu
from vllm_fl.dispatch.backends.vendor.hygon.impl.native_ops.shared_activation import try_shared_swiglu


@pytest.fixture(autouse=True)
def vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        yield


def reference(x, limit=10., alpha=1., beta=0.):
    d=x.shape[-1]//2
    gate=x[..., :d].clamp(max=limit)
    up=x[..., d:].clamp(-limit,limit)
    return gate*torch.sigmoid(alpha*gate)*(up+beta)


@pytest.mark.parametrize('m',[1,8,32,64,128,256])
def test_actual_custom_op_changed_input_graph(m):
    patch_shared_swiglu()
    op=SiluAndMulWithClamp(10.,compile_native=False)
    x=torch.randn(m,512,device='cuda',dtype=torch.bfloat16)
    op(x)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):y=op(x)
    for std in [.001,1.,8.]:
        x.normal_(std=std)
        x[0,:8]=torch.tensor([-10.0625,-10.,-9.9375,-0.,0.,9.9375,10.,10.0625],device='cuda',dtype=x.dtype)
        g.replay()
        with flag_gems.use_gems():expected=reference(x)
        torch.testing.assert_close(y,expected,atol=0,rtol=0)
        assert getattr(op._forward_method,'_vllm_fl_hygon_shared_swiglu',False)


def test_all_bf16_gate_values():
    x=torch.empty(256,512,device='cuda',dtype=torch.bfloat16)
    x[:,:256]=torch.arange(65536,device='cuda',dtype=torch.int32).short().view(torch.bfloat16).reshape(256,256)
    x[:,256:]=torch.linspace(-12,12,256,device='cuda').bfloat16()
    y=try_shared_swiglu(x,10.,1.,0.)
    with flag_gems.use_gems():ref=reference(x)
    torch.testing.assert_close(y,ref,atol=0,rtol=0,equal_nan=True)


@pytest.mark.parametrize('case',['prefill','fp32','stride','shape','3d','grad','limit','alpha','beta','disabled','cpu'])
def test_original_fallback(case,monkeypatch):
    patch_shared_swiglu()
    kw=dict(swiglu_limit=10.,alpha=1.,beta=0.,compile_native=False)
    x=torch.randn(64,512,device='cuda',dtype=torch.bfloat16)
    if case=='prefill':x=torch.randn(257,512,device='cuda',dtype=x.dtype)
    elif case=='fp32':x=x.float()
    elif case=='stride':x=torch.randn(64,1024,device='cuda',dtype=x.dtype)[:,::2]
    elif case=='shape':x=x[:,:256].contiguous()
    elif case=='3d':x=x.reshape(2,32,512)
    elif case=='grad':x.requires_grad_(True)
    elif case=='limit':kw['swiglu_limit']=7.
    elif case=='alpha':kw['alpha']=1.702
    elif case=='beta':kw['beta']=1.
    elif case=='disabled':monkeypatch.setenv('VLLM_FL_HYGON_SHARED_SWIGLU','0')
    elif case=='cpu':x=x.cpu()
    assert try_shared_swiglu(x,kw['swiglu_limit'],kw['alpha'],kw['beta']) is None
    op=SiluAndMulWithClamp(**kw)
    with flag_gems.use_gems():
        y=op(x);ref=reference(x,kw['swiglu_limit'],kw['alpha'],kw['beta'])
    torch.testing.assert_close(y,ref,atol=0,rtol=0)
    if case=='grad':y.sum().backward();assert x.grad is not None


def test_patch_idempotent():
    patch_shared_swiglu();first=SiluAndMulWithClamp.forward_native
    patch_shared_swiglu();assert SiluAndMulWithClamp.forward_native is first
