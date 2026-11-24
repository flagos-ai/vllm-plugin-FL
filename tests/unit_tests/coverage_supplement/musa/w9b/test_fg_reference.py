"""Coverage tests for the PyTorch reference backend."""
import sys
from types import SimpleNamespace
import pytest
import torch
from vllm_fl.dispatch.backends.reference import reference as ref
from vllm_fl.dispatch.backends.reference import register_ops as ref_register
from vllm_fl.dispatch.backends.reference.impl import activation as ref_activation, fused_moe as ref_fused_moe, normalization as ref_norm, quantization as ref_quant, rotary as ref_rotary
from vllm_fl.dispatch.types import BackendImplKind

def test_topk_softmax_reference():
    backend = ref.ReferenceBackend()
    for renormalize in (False, True):
        w = torch.zeros((2, 2), dtype=torch.float32)
        idx = torch.zeros((2, 2), dtype=torch.int32)
        tei = torch.zeros((2, 2), dtype=torch.int32)
        gating = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
        (rw, ri) = backend.topk_softmax(w, idx, tei, gating, renormalize=renormalize)
        assert rw is w and ri is idx
        assert torch.allclose(w.sum(-1), torch.ones(2)) if renormalize else True
        assert tei.dtype == torch.int32

def test_moe_align_block_size_reference_variants():
    backend = ref.ReferenceBackend()
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    (sorted_ids, expert_ids, ntpp) = backend.moe_align_block_size(ids, 2, 4)
    assert sorted_ids.dtype == torch.int32
    assert sorted_ids.numel() % 2 == 0
    assert ntpp.numel() == 1
    ids_bad = torch.tensor([[0, -1], [3, 9]], dtype=torch.int32)
    (_, _, ntpp2) = backend.moe_align_block_size(ids_bad, 2, 4, pad_sorted_ids=True)
    assert ntpp2.item() >= 0
    emap = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    backend.moe_align_block_size(ids, 2, 4, expert_map=emap, ignore_invalid_experts=True)
    out = backend.moe_align_block_size(ids, 2, 4, expert_map=emap, ignore_invalid_experts=False)
    assert out[1].dtype == torch.int32

def test_moe_sum_and_metadata():
    backend = ref.ReferenceBackend()
    inp = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out = torch.zeros((2, 4))
    backend.moe_sum(inp, out)
    torch.testing.assert_close(out, inp.sum(dim=1))
    assert backend.name == 'reference'
    monkeypatch_avail = ref.ReferenceBackend._available
    try:
        ref.ReferenceBackend._available = None
        assert backend.is_available() is True
        ref.ReferenceBackend._available = False
        assert backend.is_available() is False
    finally:
        ref.ReferenceBackend._available = monkeypatch_avail

def test_reference_backend_delegation(monkeypatch):
    backend = ref.ReferenceBackend()
    x = torch.zeros((2, 4))
    monkeypatch.setattr(ref_quant, 'dynamic_per_token_quant_int8_torch', lambda v: ('q', v))
    monkeypatch.setattr(ref_activation, 'silu_and_mul_torch', lambda o, v: 's')
    monkeypatch.setattr(ref_activation, 'gelu_and_mul_torch', lambda o, v: 'g')
    monkeypatch.setattr(ref_norm, 'rms_norm_torch', lambda o, v, r: 'n')
    monkeypatch.setattr(ref_rotary, 'rotary_embedding_torch', lambda *a, **k: 'r')
    monkeypatch.setattr(ref_fused_moe, 'topk_softmax', lambda *a: ('w', 'i'))
    assert backend.dynamic_per_token_quant_int8(x)[0] == 'q'
    assert backend.silu_and_mul(None, x) == 's'
    assert backend.gelu_and_mul(None, x) == 'g'
    assert backend.rms_norm(None, x, None) == 'n'
    assert backend.rotary_embedding(None, x, x, x, x, x) == 'r'
    assert backend.topk_softmax(x, x, x, x) == ('w', 'i')
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    assert backend.attention_backend(use_mla=True, use_sparse=True) == AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
    assert backend.attention_backend(use_mla=True) == AttentionBackendEnum.FLASHMLA.get_path()
    assert backend.attention_backend() == AttentionBackendEnum.FLASH_ATTN.get_path()

def test_invoke_fused_moe_delegation(monkeypatch):
    import vllm.model_executor.layers.fused_moe.fused_moe as upstream
    seen = []
    monkeypatch.setattr(upstream, 'invoke_fused_moe_triton_kernel', lambda *a, **k: seen.append(k))
    ref.ReferenceBackend().invoke_fused_moe_triton_kernel(None, None, None, None, None, None, None, None, None, False, 2, None, None, False, False, False, False, False, block_shape=(1, 1), B_bias='b')
    assert seen[0]['block_shape'] == (1, 1) and seen[0]['B_bias'] == 'b'

class _Registry:

    def __init__(self):
        self.impls = []

    def register_many(self, impls):
        self.impls.extend(impls)

def test_reference_register_builtins():
    registry = _Registry()
    ref_register.register_builtins(registry)
    ids = {impl.op_name: impl.impl_id for impl in registry.impls}
    assert ids['rms_norm'] == 'reference.torch'
    assert ids['w8a8_moe_experts'] == 'reference.torch'
    assert all((i.kind == BackendImplKind.REFERENCE for i in registry.impls))
    assert all((callable(i.fn._is_available) for i in registry.impls))

def test_reference_w8a8_helper(monkeypatch):
    import vllm_fl.quantization.w8a8.moe_experts as moe_experts
    monkeypatch.setattr(moe_experts, '_native_w8a8_fused_experts', lambda **k: ('native', k))
    with pytest.raises(NotImplementedError, match='SwiGLU'):
        ref_register._reference_w8a8_moe_experts(activation='gelu', global_num_experts=8)
    assert ref_register._reference_w8a8_moe_experts(activation='silu', global_num_experts=8, hidden_states='h') == ('native', {'hidden_states': 'h'})

def test_activation_impls():
    x = torch.randn(2, 8)
    ref_activation.silu_and_mul_torch(None, x)
    obj = SimpleNamespace(approximate='tanh')
    assert ref_activation.gelu_and_mul_torch(obj, x).shape == (2, 4)
    assert ref_activation.gelu_and_mul_torch(None, x).shape == (2, 4)

def test_normalization_impls():

    class Obj:

        def __init__(self, dtype=torch.float32):
            self.weight = torch.ones(4, dtype=dtype)
            self.variance_epsilon = 1e-06
    x = torch.randn(2, 4, dtype=torch.float16)
    out = ref_norm.rms_norm_torch(Obj(), x)
    assert out.shape == x.shape and out.dtype == x.dtype
    (out2, res2) = ref_norm.rms_norm_torch(Obj(), x, residual=torch.ones_like(x))
    assert res2.dtype == x.dtype and out2.shape == x.shape
    obj = Obj()
    obj.variance_size_override = 2
    obj.pass_weight = False
    obj.pass_weight_add = False
    assert ref_norm.rms_norm_torch(obj, x).shape == x.shape
    assert ref_norm.rms_norm_torch(obj, x, residual=torch.zeros_like(x))[0].shape == x.shape

def test_reference_fused_moe_topk():
    gating = torch.tensor([[1.0, 2.0, 3.0], [3.0, 0.0, 1.0]])
    w = torch.zeros((2, 2))
    idx = torch.zeros((2, 2), dtype=torch.int32)
    tei = torch.zeros((2, 2), dtype=torch.int32)
    ref_fused_moe.topk_softmax(w, idx, tei, gating, renormalize=True)
    torch.testing.assert_close(w.sum(-1), torch.ones(2))
    bias = torch.tensor([0.0, 10.0, -10.0])
    w2 = torch.zeros((2, 2))
    idx2 = torch.zeros((2, 2), dtype=torch.int32)
    tei2 = torch.zeros((2, 2), dtype=torch.int32)
    ref_fused_moe.topk_softmax(w2, idx2, tei2, gating, e_score_correction_bias=bias)
    assert idx2[0, 0].item() == 1

def test_reference_quantization_delegates(monkeypatch):
    import vllm_fl.quantization.w8a8.reference as w8a8_ref
    monkeypatch.setattr(w8a8_ref, 'dynamic_per_token_quant_int8', lambda x: ('o', 's'))
    assert ref_quant.dynamic_per_token_quant_int8_torch(torch.zeros(2)) == ('o', 's')

def _rope_inputs(dim=4, tokens=2, heads=2):
    q = torch.randn(tokens, heads, dim)
    k = torch.randn(tokens, heads, dim)
    cos = torch.randn(tokens, dim)
    sin = torch.randn(tokens, dim)
    pos = torch.arange(tokens, dtype=torch.int64)
    return (q, k, cos, sin, pos)