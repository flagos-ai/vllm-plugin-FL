"""Extra coverage for flaggems mla/fused_moe/quantization helpers."""
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
from vllm_fl.dispatch.backends.flaggems.impl import fused_moe, mla, quantization

class _ConcreteMLAFLImpl(mla.MLAFLImpl):
    """Concrete stand-in: MLAFLImpl keeps forward_mqa abstract upstream."""

    def forward_mqa(self, *args, **kwargs):
        raise NotImplementedError('forward_mqa is not exercised by these tests')
_ConcreteMLAFLImpl.__abstractmethods__ = frozenset()

def _mla_shell():
    return _ConcreteMLAFLImpl.__new__(_ConcreteMLAFLImpl)

def test_mla_flash_attn_unpacks_tuple_without_lse(monkeypatch):
    impl = _mla_shell()
    impl._pad_v = False
    monkeypatch.setattr(mla, 'flash_attn_varlen_func', lambda **kw: ('attn-out', 'attn-lse'))
    q = torch.zeros((2, 4, 8))
    v = torch.zeros((2, 4, 4))
    assert impl._flash_attn_varlen_diff_headdims(q, q, v) == 'attn-out'

def test_mla_flash_attn_pads_v_for_lse(monkeypatch):
    impl = _mla_shell()
    impl._pad_v = True
    seen = {}

    def flash(**kw):
        seen.update(kw)
        return ('attn-out', 'attn-lse')
    monkeypatch.setattr(mla, 'flash_attn_varlen_func', flash)
    q = torch.zeros((2, 4, 8))
    v = torch.zeros((2, 4, 4))
    (out, lse) = impl._flash_attn_varlen_diff_headdims(q, q, v, return_softmax_lse=True, softmax_scale=0.5)
    assert (out, lse) == ('attn-out', 'attn-lse')
    assert seen['v'].shape[-1] == 8
    assert seen['softmax_scale'] == 0.5

def test_mla_forward_decode_asserts_nonempty_cache():
    impl = _mla_shell()
    impl.kv_cache_dtype = 'auto'
    metadata = SimpleNamespace(decode=SimpleNamespace())
    with pytest.raises(AssertionError):
        impl._forward_decode(torch.zeros((1, 1, 4)), torch.zeros(0), metadata, None)

def test_mla_forward_decode_asserts_decode_metadata():
    impl = _mla_shell()
    impl.kv_cache_dtype = 'auto'
    metadata = SimpleNamespace(decode=None)
    with pytest.raises(AssertionError):
        impl._forward_decode(torch.zeros((1, 1, 4)), torch.zeros((2, 2, 4)), metadata, None)

def test_topk_softmax_typeerror_fallback_does_not_normalize(monkeypatch):
    state = {'calls': 0}

    def flaky(*args):
        state['calls'] += 1
        if state['calls'] == 1:
            raise TypeError('signature changed')
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', SimpleNamespace(topk_softmax=flaky))
    weights = torch.tensor([[2.0, 4.0]])
    indices = torch.zeros((1, 2), dtype=torch.int32)
    token_indices = torch.zeros((1, 2), dtype=torch.int32)
    gating = torch.zeros((1, 3))
    (out_w, _) = fused_moe.topk_softmax_flaggems(weights, indices, token_indices, gating, renormalize=False)
    assert state['calls'] == 2
    torch.testing.assert_close(out_w, torch.tensor([[2.0, 4.0]]))

def test_moe_sum_import_error_uses_metax_fallback(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', None)
    monkeypatch.setitem(sys.modules, 'flag_gems', SimpleNamespace(vendor_name='metax', moe_sum=lambda inp, out: calls.append('metax')))
    inp = torch.zeros((2, 2))
    out = torch.zeros((2,))
    fused_moe.moe_sum_flaggems(inp, out)
    assert calls == ['metax']

def test_moe_sum_import_error_propagates_for_non_metax(monkeypatch):
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', None)
    monkeypatch.setitem(sys.modules, 'flag_gems', SimpleNamespace(vendor_name='ascend'))
    with pytest.raises(ImportError):
        fused_moe.moe_sum_flaggems(torch.zeros((2, 2)), torch.zeros((2,)))

def test_quant_triton_accepts_non_contiguous_input(monkeypatch):
    launched = []

    class FakeKernel:

        def __getitem__(self, grid):

            def run(*args, **kwargs):
                launched.append((grid, kwargs))
            return run
    monkeypatch.setattr(quantization, '_dynamic_per_token_quant_int8_kernel', FakeKernel())
    monkeypatch.setattr(quantization, 'torch_device_fn', SimpleNamespace(device=lambda device: nullcontext()))
    x = torch.randn(5, 3).t()
    (out, scale) = quantization.dynamic_per_token_quant_int8_flaggems_triton(x)
    assert out.shape == (3, 5) and out.dtype == torch.int8
    assert scale.shape == (3, 1)
    assert launched[0][0] == (3,)
    assert launched[0][1]['BLOCK_SIZE'] == 8