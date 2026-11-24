"""Coverage tests for the FlagGems backend surface and impl helpers."""
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
import vllm_fl.dispatch.backends.flaggems.flaggems as fg
from vllm_fl.dispatch.backends.flaggems.impl import activation as fg_activation, fused_moe, mla, musa_ops, normalization as fg_norm, quantization, rotary as fg_rotary

def test_backend_name_and_availability(monkeypatch):
    backend = fg.FlagGemsBackend()
    assert backend.name == 'flagos'
    monkeypatch.setattr(fg.FlagGemsBackend, '_available', True)
    assert backend.is_available() is True
    monkeypatch.setattr(fg.FlagGemsBackend, '_available', False)
    assert backend.is_available() is False
    monkeypatch.setattr(fg.FlagGemsBackend, '_available', None)
    real_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **k):
        if name == 'flag_gems':
            raise ImportError('missing')
        return real_import(name, *a, **k)
    monkeypatch.setattr('builtins.__import__', fake_import)
    assert backend.is_available() is False

def _patch_impl(monkeypatch, module, name, value):
    monkeypatch.setattr(module, name, value)

def test_attention_backend_selection(monkeypatch):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    backend = fg.FlagGemsBackend()
    monkeypatch.setattr(fg.torch.cuda, 'is_available', lambda : False)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        backend.attention_backend()
    monkeypatch.setattr(fg.torch.cuda, 'is_available', lambda : True)
    with pytest.raises(NotImplementedError, match='NOT support mla'):
        backend.attention_backend(use_mla=True)
    with pytest.raises(ValueError, match='use_sparse=True requires use_mla'):
        backend.attention_backend(use_sparse=True)
    monkeypatch.delenv('VLLM_FL_USE_FLAGGEMS_ATTN', raising=False)
    assert backend.attention_backend() == AttentionBackendEnum.TRITON_ATTN.get_path()
    monkeypatch.setenv('VLLM_FL_USE_FLAGGEMS_ATTN', 'yes')
    assert backend.attention_backend().endswith('attention.AttentionFLBackend')

def test_musa_ops_kernel_boundaries(monkeypatch):
    calls = []
    norm_mod = SimpleNamespace(gems_rms_forward=lambda x, r, w, e: calls.append(('rms', x, r, w, e)) or torch.zeros_like(x))
    silu_mod = SimpleNamespace(silu_and_mul_out=lambda a, b, out: calls.append(('silu', a, b, out)))
    rope_mod = SimpleNamespace(gems_rope_forward=lambda q, k, c, s, **kw: calls.append(('rope', kw)) or (q, k))
    for (name, mod) in {'flag_gems.modules.normalization': norm_mod, 'flag_gems.modules.rotary_embedding': rope_mod, 'flag_gems.fused.silu_and_mul': silu_mod}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    x = torch.zeros((2, 4))
    assert musa_ops._rms_norm(x, torch.ones(4), 1e-06).shape == x.shape
    assert musa_ops._rms_norm_fake(x, torch.ones(4), 1e-06).shape == x.shape
    assert musa_ops._fused_add_rms_norm(x, x.clone(), torch.ones(4), 1e-06) is None
    assert musa_ops._fused_add_rms_norm_fake(x, x, torch.ones(4), 1e-06) is None
    half = torch.zeros((2, 8))
    out = torch.zeros((2, 4))
    musa_ops._silu_and_mul(out, half)
    assert calls[-1][0] == 'silu'
    assert musa_ops._silu_and_mul_fake(out, half) is None
    q = torch.zeros((2, 3, 4))
    rope_ret = musa_ops._rotary_embedding(q, q, q, q, None, False)
    assert rope_ret[0].shape == q.shape
    fake_ret = musa_ops._rotary_embedding_fake(q, q, q, q, None, False)
    assert fake_ret[0].shape == q.shape
    assert musa_ops._rotary_embedding_inplace(q, q, q, q, None, True) is None
    assert musa_ops._rotary_embedding_inplace_fake(q, q, q, q, None, True) is None

def test_musa_ops_public_wrappers(monkeypatch):
    ops_calls = []

    class _Ops:

        def rms_norm(self, *a):
            ops_calls.append(('rms', a))
            return 'rms-out'

        def fused_add_rms_norm(self, *a):
            ops_calls.append(('fused', a))

        def silu_and_mul(self, out, x):
            ops_calls.append(('silu', out, x))

        def rotary_embedding_inplace(self, *a):
            ops_calls.append(('rope_inplace', a))

        def rotary_embedding(self, *a):
            ops_calls.append(('rope', a))
            return ('q', 'k')
    real_torch = torch
    monkeypatch.setattr(musa_ops, 'torch', SimpleNamespace(ops=SimpleNamespace(vllm_fl_musa=_Ops()), empty_like=real_torch.empty_like))
    x = real_torch.zeros((2, 4))
    assert musa_ops.rms_norm(x, None, real_torch.ones(4), 1e-06) == 'rms-out'
    (a, b) = musa_ops.rms_norm(x, real_torch.zeros((2, 4)), real_torch.ones(4), 1e-06)
    assert a.shape == b.shape == (2, 4)
    assert musa_ops.silu_and_mul(real_torch.zeros((2, 8))).shape == (2, 4)
    q = real_torch.zeros((2, 3, 4))
    assert musa_ops.rotary_embedding(q, q, q, q, None, False, True)[0].shape == q.shape
    assert musa_ops.rotary_embedding(q, q, q, q, None, False, False) == ('q', 'k')
    assert len(ops_calls) == 5

def test_mla_backend_metadata():
    assert mla.MLAFLBackend.get_name() == 'MLAFL'
    assert mla.MLAFLBackend.get_impl_cls() is mla.MLAFLImpl

def _mla_impl(monkeypatch, **overrides):

    def base_init(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing, **kw):
        self.kv_cache_dtype = kv_cache_dtype
    monkeypatch.setattr(mla.MLACommonImpl, '__init__', base_init)
    params = dict(num_heads=4, head_size=8, scale=0.1, num_kv_heads=4, alibi_slopes=None, sliding_window=None, kv_cache_dtype='auto', logits_soft_cap=None, attn_type=mla.AttentionType.DECODER, kv_sharing_target_layer_name=None)
    params.update(overrides)
    return mla.MLAFLImpl(**params)

def _fg_module(**funcs):
    return SimpleNamespace(**funcs)

def test_moe_align_block_size(monkeypatch):

    def triton_stub(topk_ids, num_experts, block_size, sorted_ids, expert_ids, ntpp):
        expert_ids.fill_(0)
        ntpp.fill_(int(topk_ids.numel()))
        sorted_ids.fill_(0)
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', _fg_module(moe_align_block_size_triton=triton_stub))
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    (sorted_ids, expert_ids, ntpp) = fused_moe.moe_align_block_size_flaggems(ids, 2, 4, pad_sorted_ids=True)
    assert sorted_ids.numel() % 2 == 0
    assert ntpp.item() == 4
    small = torch.tensor([[0]], dtype=torch.int32)
    fused_moe.moe_align_block_size_flaggems(small, 2, 4)
    boosted = _fg_module(moe_align_block_size_triton=triton_stub)
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', boosted)
    (_, mapped, _) = fused_moe.moe_align_block_size_flaggems(ids, 2, 4, expert_map=torch.tensor([7, 8], dtype=torch.int32))
    assert mapped.max().item() == 7

def test_invoke_and_grouped(monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', _fg_module(invoke_fused_moe_triton_kernel=lambda *a, **k: seen.append((a, k)), grouped_topk=lambda *a: 'grouped'))
    a = [0] * 18
    fused_moe.invoke_fused_moe_triton_kernel_flaggems(*a)
    assert seen and seen[0][1]['block_shape'] is None
    assert fused_moe.grouped_topk_flaggems(None, 1, 1, 1, False, 1.0, None) == 'grouped'

def test_moe_sum_paths(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', _fg_module(moe_sum=lambda i, o: calls.append('vllm')))
    (inp, out) = (torch.zeros((2, 2)), torch.zeros((2,)))
    fused_moe.moe_sum_flaggems(inp, out)
    assert calls == ['vllm']

    def boom(i, o):
        raise RuntimeError('no maca kernel')
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', _fg_module(moe_sum=boom))
    monkeypatch.setitem(sys.modules, 'flag_gems', _fg_module(vendor_name='metax', moe_sum=lambda i, o: calls.append('metax')))
    fused_moe.moe_sum_flaggems(inp, out)
    assert calls[-1] == 'metax'
    monkeypatch.setitem(sys.modules, 'flag_gems', _fg_module(vendor_name='ascend'))
    with pytest.raises(RuntimeError, match='no maca kernel'):
        fused_moe.moe_sum_flaggems(inp, out)

def test_quant_vllm_delegates(monkeypatch):
    monkeypatch.setitem(sys.modules, 'flaggems_vllm', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'flaggems_vllm.ops', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'flaggems_vllm.ops.scaled_int8_quant', SimpleNamespace(scaled_int8_quant=lambda x, **k: ('out', 'scale', None)))
    assert quantization.dynamic_per_token_quant_int8_flaggems_vllm(torch.zeros((2, 4))) == ('out', 'scale')

def test_quant_triton_validations_and_launch(monkeypatch):
    with pytest.raises(ValueError, match='2D'):
        quantization.dynamic_per_token_quant_int8_flaggems_triton(torch.zeros(4))
    with pytest.raises(TypeError, match='floating point'):
        quantization.dynamic_per_token_quant_int8_flaggems_triton(torch.zeros((2, 4), dtype=torch.int8))
    with pytest.raises(ValueError, match='hidden_size must be positive'):
        quantization.dynamic_per_token_quant_int8_flaggems_triton(torch.zeros((2, 0)))
    launched = []

    class FakeKernel:

        def __getitem__(self, grid):

            def launch(*a, **k):
                launched.append((grid, a, k))
            return launch
    monkeypatch.setattr(quantization, '_dynamic_per_token_quant_int8_kernel', FakeKernel())
    monkeypatch.setattr(quantization, 'torch_device_fn', SimpleNamespace(device=lambda d: nullcontext()))
    (out, scale) = quantization.dynamic_per_token_quant_int8_flaggems_triton(torch.randn(3, 5))
    assert out.dtype == torch.int8 and out.shape == (3, 5)
    assert scale.shape == (3, 1)
    assert launched[0][0] == (3,)
    assert launched[0][2]['BLOCK_SIZE'] == 8