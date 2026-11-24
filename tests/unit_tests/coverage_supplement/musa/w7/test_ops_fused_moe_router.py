import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from vllm_fl.ops.fused_moe import router as router_mod

def test_is_ascend_npu_tensor_detection():
    assert router_mod._is_ascend_npu_tensor(torch.zeros(1)) is False
    fake = SimpleNamespace(device=SimpleNamespace(type='npu'))
    assert router_mod._is_ascend_npu_tensor(fake) is True

def test_has_native_moe_topk_probe(monkeypatch):
    monkeypatch.setattr(router_mod, 'torch', SimpleNamespace(ops=SimpleNamespace(_moe_C=None)))
    assert router_mod._has_native_moe_topk() is False
    monkeypatch.setattr(router_mod, 'torch', SimpleNamespace(ops=SimpleNamespace(_moe_C=SimpleNamespace(topk_softmax=1))))
    assert router_mod._has_native_moe_topk() is True
    monkeypatch.setattr(router_mod, 'torch', SimpleNamespace(ops=SimpleNamespace(_moe_C=object())))
    assert router_mod._has_native_moe_topk() is False

def test_sqrtsoftplus_without_bias_or_hash():
    logits = torch.tensor([[0.0, 1.0, -1.0, 3.0]], dtype=torch.float32)
    (weights, ids) = router_mod._sqrtsoftplus_topk(gating_output=logits, e_score_correction_bias=None, topk=2, renormalize=False, indices_type=None, input_tokens=None, hash_indices_table=None, routed_scaling_factor=0.5)
    scores = torch.sqrt(F.softplus(logits))
    expected_ids = torch.topk(scores, k=2, dim=-1, sorted=False).indices
    expected_weights = scores.gather(1, expected_ids) * 0.5
    assert ids.dtype == torch.int32
    assert torch.equal(ids, expected_ids)
    assert torch.allclose(weights, expected_weights)

def test_sqrtsoftplus_with_bias_renormalize_and_nan():
    logits = torch.tensor([[float('nan'), 1.0, 2.0]], dtype=torch.float32)
    bias = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float32)
    (weights, ids) = router_mod._sqrtsoftplus_topk(gating_output=logits, e_score_correction_bias=bias, topk=2, renormalize=True, indices_type=torch.int64, input_tokens=None, hash_indices_table=None, routed_scaling_factor=2.0)
    assert ids.dtype == torch.int64
    assert torch.isfinite(weights).all()
    expected_sum = torch.full((1,), 2.0, dtype=torch.float32)
    assert torch.allclose(weights.sum(dim=-1), expected_sum)

def test_sqrtsoftplus_hash_table_and_dtype_cast():
    hash_table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
    (weights, ids) = router_mod._sqrtsoftplus_topk(gating_output=torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32), e_score_correction_bias=torch.zeros(3), topk=2, renormalize=False, indices_type=torch.int32, input_tokens=torch.tensor([1], dtype=torch.int32), hash_indices_table=hash_table, routed_scaling_factor=1.0)
    assert torch.equal(ids, torch.tensor([[1, 2]], dtype=torch.int32))
    expected = torch.sqrt(F.softplus(torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32))).gather(1, torch.tensor([[1, 2]]))
    assert torch.allclose(weights, expected)

def _stub_ascend_fused_topk(monkeypatch, recorder):
    base = 'vllm_fl.dispatch.backends.vendor.ascend'
    pkg = ModuleType(base)
    pkg.__path__ = []
    impl = ModuleType(f'{base}.impl')
    impl.__path__ = []
    fm = ModuleType(f'{base}.impl.fused_moe')
    fm.fused_topk = recorder
    monkeypatch.setitem(sys.modules, base, pkg)
    monkeypatch.setitem(sys.modules, f'{base}.impl', impl)
    monkeypatch.setitem(sys.modules, f'{base}.impl.fused_moe', fm)

def test_fused_topk_token_mismatch():
    with pytest.raises(AssertionError):
        router_mod.fused_topk(hidden_states=torch.zeros(2, 4), gating_output=torch.zeros(3, 4), topk=1, renormalize=False)

def test_fused_topk_ascend_dispatch(monkeypatch):
    captured = {}

    def ascend(gating_output, topk, renormalize, indices_type, scoring_func, mask):
        captured.update(gating_output=gating_output, topk=topk, renormalize=renormalize, indices_type=indices_type, scoring_func=scoring_func, mask=mask)
        return ('w', 'i', 't')
    _stub_ascend_fused_topk(monkeypatch, ascend)
    mask = torch.tensor([False, True])
    monkeypatch.setattr(router_mod, '_is_ascend_npu_tensor', lambda value: True)
    monkeypatch.setattr(router_mod, '_get_padding_mask', lambda n: mask)
    result = router_mod.fused_topk(hidden_states=torch.zeros(2, 4), gating_output=torch.zeros(2, 3), topk=2, renormalize=True, indices_type=torch.int32, scoring_func='sigmoid')
    assert result == ('w', 'i', 't')
    assert captured['mask'] is mask
    assert captured['scoring_func'] == 'sigmoid'

@pytest.mark.parametrize(('scoring_func', 'native', 'mask'), [('sigmoid', False, None), ('softmax', True, torch.tensor([False, True]))])
def test_fused_topk_defers_to_upstream(monkeypatch, scoring_func, native, mask):
    captured = {}
    monkeypatch.setattr(router_mod, '_is_ascend_npu_tensor', lambda value: False)
    monkeypatch.setattr(router_mod, '_has_native_moe_topk', lambda : native)
    monkeypatch.setattr(router_mod, '_get_padding_mask', lambda n: mask)

    def upstream(**kwargs):
        captured.update(kwargs)
        return ('w', 'i', 't')
    monkeypatch.setattr(router_mod, 'upstream_fused_topk', upstream)
    result = router_mod.fused_topk(hidden_states=torch.zeros(2, 4), gating_output=torch.zeros(2, 3), topk=2, renormalize=False, scoring_func=scoring_func)
    assert result == ('w', 'i', 't')
    assert captured['scoring_func'] == scoring_func

def test_fused_topk_uses_fl_kernel_and_zeroes_padding(monkeypatch):

    def fake_topk(weights, ids, token_expert_indices, gating_output, renormalize):
        weights.fill_(1.0)
        ids.fill_(0)
        return (weights, ids)
    monkeypatch.setattr(router_mod, '_is_ascend_npu_tensor', lambda value: False)
    monkeypatch.setattr(router_mod, '_has_native_moe_topk', lambda : False)
    monkeypatch.setattr(router_mod, '_topk_softmax', fake_topk)
    monkeypatch.setattr(router_mod, '_get_padding_mask', lambda n: torch.tensor([False, True])[:n])
    monkeypatch.setattr(router_mod, 'upstream_fused_topk', lambda **kwargs: pytest.fail('upstream must not be used'))
    (weights, ids, token_expert_indices) = router_mod.fused_topk(hidden_states=torch.zeros(2, 4), gating_output=torch.zeros(2, 3), topk=2, renormalize=False)
    assert torch.equal(weights, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert ids.shape == (2, 2)
    assert token_expert_indices.shape == (2, 2)

def test_fused_topk_without_padding_mask(monkeypatch):

    def fake_topk(weights, ids, token_expert_indices, gating_output, renormalize):
        weights.fill_(0.5)
        ids.fill_(1)
        return (weights, ids)
    monkeypatch.setattr(router_mod, '_is_ascend_npu_tensor', lambda value: False)
    monkeypatch.setattr(router_mod, '_has_native_moe_topk', lambda : False)
    monkeypatch.setattr(router_mod, '_topk_softmax', fake_topk)
    monkeypatch.setattr(router_mod, '_get_padding_mask', lambda n: None)
    (weights, _, _) = router_mod.fused_topk(hidden_states=torch.zeros(2, 4), gating_output=torch.zeros(2, 3), topk=2, renormalize=False)
    assert torch.equal(weights, torch.full((2, 2), 0.5))

def test_fused_topk_router_compute_routing(monkeypatch):
    monkeypatch.setattr(router_mod, 'fused_topk', lambda **kwargs: (torch.ones(1, 2), torch.zeros(1, 2), None))
    router = SimpleNamespace(top_k=2, renormalize=True, scoring_func='softmax')
    (weights, ids) = router_mod.FusedTopKRouterFL._compute_routing(router, torch.zeros(1, 4), torch.zeros(1, 3), torch.int32)
    assert weights.shape == (1, 2)
    assert ids.shape == (1, 2)

@pytest.mark.parametrize('scoring_func', ['softmax', 'sigmoid'])
def test_fl_grouped_topk_without_bias(scoring_func):
    gating = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    (weights, ids) = router_mod._fl_grouped_topk(hidden_states=torch.zeros(2, 4), gating_output=gating, topk=2, renormalize=True, num_expert_group=2, topk_group=1, scoring_func=scoring_func, routed_scaling_factor=2.0)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert weights.shape == (2, 2)
    assert torch.allclose(weights.sum(-1), torch.full((2,), 2.0), atol=1e-05)

def test_fl_grouped_topk_unsupported_without_bias():
    with pytest.raises(ValueError, match='Unsupported scoring function'):
        router_mod._fl_grouped_topk(hidden_states=torch.zeros(1, 4), gating_output=torch.zeros(1, 4), topk=1, renormalize=False, num_expert_group=2, topk_group=1, scoring_func='linear')

@pytest.mark.parametrize('scoring_func', ['sigmoid', 'softmax'])
def test_fl_grouped_topk_with_bias_delegates(monkeypatch, scoring_func):
    captured = {}

    def fake_grouped(scores, num_expert_group, topk_group, topk, renormalize, scaling, bias, flag):
        captured.update(scores=scores.clone(), num_expert_group=num_expert_group, topk_group=topk_group, bias=bias, flag=flag)
        return (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32))
    monkeypatch.setattr(router_mod, '_grouped_topk', fake_grouped)
    gating = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    bias = torch.zeros(4)
    (weights, ids) = router_mod._fl_grouped_topk(hidden_states=torch.zeros(1, 4), gating_output=gating, topk=2, renormalize=True, num_expert_group=2, topk_group=1, scoring_func=scoring_func, routed_scaling_factor=1.5, e_score_correction_bias=bias)
    assert weights.shape == (1, 2)
    assert captured['flag'] == (1 if scoring_func == 'sigmoid' else 0)
    if scoring_func == 'sigmoid':
        assert torch.equal(captured['scores'], gating)
    else:
        assert torch.allclose(captured['scores'], torch.softmax(gating, dim=-1))

def test_fl_grouped_topk_unsupported_with_bias():
    with pytest.raises(ValueError, match='Unsupported scoring function'):
        router_mod._fl_grouped_topk(hidden_states=torch.zeros(1, 4), gating_output=torch.zeros(1, 4), topk=1, renormalize=False, num_expert_group=2, topk_group=1, scoring_func='linear', e_score_correction_bias=torch.zeros(4))

def _grouped_router(**overrides):
    values = {'num_expert_group': 2, 'topk_group': 1, 'top_k': 2, 'renormalize': True, 'scoring_func': 'softmax', 'routed_scaling_factor': 1.0, 'e_score_correction_bias': None, 'num_fused_shared_experts': 0, 'global_num_experts': 4}
    values.update(overrides)
    return SimpleNamespace(**values)

def test_grouped_router_invalid_grouping_without_bias(monkeypatch):
    monkeypatch.setattr(router_mod, 'fused_topk', lambda **kwargs: (torch.ones(1, 1), torch.zeros(1, 1), None))
    router = _grouped_router(num_expert_group=4)
    (weights, ids) = router_mod.GroupedTopKRouterFL._compute_routing(router, torch.zeros(1, 4), torch.zeros(1, 3), torch.int32)
    assert weights.shape == (1, 1)

def test_grouped_router_aiter_asserts_shared_experts(monkeypatch):
    monkeypatch.setattr(router_mod, 'rocm_aiter_ops', SimpleNamespace(is_fused_moe_enabled=lambda : True, is_fusion_moe_shared_experts_enabled=lambda : False))
    router = _grouped_router(num_fused_shared_experts=1)
    with pytest.raises(AssertionError):
        router_mod.GroupedTopKRouterFL._compute_routing(router, torch.zeros(1, 4), torch.zeros(1, 4), torch.int32)

def test_grouped_router_uses_aiter_impl(monkeypatch):
    captured = {}
    monkeypatch.setattr(router_mod, 'rocm_aiter_ops', SimpleNamespace(is_fused_moe_enabled=lambda : True, is_fusion_moe_shared_experts_enabled=lambda : True))

    def fake_aiter(**kwargs):
        captured.update(kwargs)
        return (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32))
    monkeypatch.setattr(router_mod, 'rocm_aiter_grouped_topk', fake_aiter)
    router = _grouped_router(num_fused_shared_experts=2)
    (weights, ids) = router_mod.GroupedTopKRouterFL._compute_routing(router, torch.zeros(1, 4), torch.zeros(1, 4), torch.int32)
    assert captured['num_fused_shared_experts'] == 2
    assert captured['topk'] == 2

def test_grouped_router_uses_fl_grouped_topk(monkeypatch):
    captured = {}

    def fake_fl(**kwargs):
        captured.update(kwargs)
        return (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32))
    monkeypatch.setattr(router_mod, 'rocm_aiter_ops', SimpleNamespace(is_fused_moe_enabled=lambda : False, is_fusion_moe_shared_experts_enabled=lambda : False))
    monkeypatch.setattr(router_mod, '_fl_grouped_topk', fake_fl)
    router = _grouped_router()
    router_mod.GroupedTopKRouterFL._compute_routing(router, torch.zeros(1, 4), torch.zeros(1, 4), torch.int32)
    assert captured['num_expert_group'] == 2
    assert captured['topk_group'] == 1

def test_bias_router_sqrtsoftplus_on_ascend(monkeypatch):
    monkeypatch.setattr(router_mod, '_is_ascend_npu_tensor', lambda value: True)
    captured = {}

    def fake_sqrt(**kwargs):
        captured.update(kwargs)
        return (torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32))
    monkeypatch.setattr(router_mod, '_sqrtsoftplus_topk', fake_sqrt)
    router = SimpleNamespace(e_score_correction_bias=torch.nn.Parameter(torch.zeros(4)), top_k=1, renormalize=True, scoring_func='sqrtsoftplus', routed_scaling_factor=1.0, _hash_indices_table=None, num_fused_shared_experts=0, bias_vl=None, image_sentinel_lo=0)
    (weights, ids) = router_mod.FusedTopKBiasRouterFL._compute_routing(router, torch.zeros(1, 4), torch.zeros(1, 3), torch.int32)
    assert captured['routed_scaling_factor'] == 1.0
    assert weights.shape == (1, 1)

def test_bias_router_delegates_and_appends_shared_experts(monkeypatch):
    captured = {}
    routed_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    routed_ids = torch.tensor([[2, 4]], dtype=torch.int32)

    def fake_bias(**kwargs):
        captured.update(kwargs)
        return (routed_weights.clone(), routed_ids.clone())
    monkeypatch.setattr(router_mod, '_is_ascend_npu_tensor', lambda value: False)
    monkeypatch.setattr(router_mod, 'fused_topk_bias', fake_bias)
    hash_table = torch.tensor([[2, 4]], dtype=torch.int32)
    router = SimpleNamespace(e_score_correction_bias=torch.nn.Parameter(torch.zeros(6)), bias_vl=None, image_sentinel_lo=3, top_k=2, renormalize=True, scoring_func='sqrtsoftplus', routed_scaling_factor=0.5, _hash_indices_table=hash_table, num_fused_shared_experts=2, shared_expert_weight=1.25, global_num_experts=6)
    (weights, ids) = router_mod.FusedTopKBiasRouterFL._compute_routing(router, torch.zeros(1, 4), torch.zeros(1, 6), torch.int32, input_ids=torch.tensor([0], dtype=torch.int32))
    assert captured['bias_vl'] is None
    assert captured['image_sentinel_lo'] == 3
    assert torch.equal(ids, torch.tensor([[2, 4, 6, 7]], dtype=torch.int32))
    assert torch.equal(weights, torch.tensor([[0.25, 0.75, 1.25, 1.25]]))

def test_bias_router_no_shared_experts(monkeypatch):
    monkeypatch.setattr(router_mod, '_is_ascend_npu_tensor', lambda value: False)
    monkeypatch.setattr(router_mod, 'fused_topk_bias', lambda **kwargs: (torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32)))
    router = SimpleNamespace(e_score_correction_bias=None, bias_vl=None, image_sentinel_lo=0, top_k=1, renormalize=True, scoring_func='softmax', routed_scaling_factor=1.0, _hash_indices_table=None, num_fused_shared_experts=0)
    (weights, ids) = router_mod.FusedTopKBiasRouterFL._compute_routing(router, torch.zeros(1, 4), torch.zeros(1, 3), torch.int32)
    assert weights.shape == (1, 1)

def test_replace_router_with_fl_patches_all_routers(monkeypatch):
    monkeypatch.setattr(router_mod.FusedTopKRouter, '_compute_routing', lambda self, *a, **k: None)
    monkeypatch.setattr(router_mod.GroupedTopKRouter, '_compute_routing', lambda self, *a, **k: None)
    monkeypatch.setattr(router_mod.FusedTopKBiasRouter, '_compute_routing', lambda self, *a, **k: None)
    router_mod.replace_router_with_fl()
    assert router_mod.FusedTopKRouter._compute_routing is router_mod.FusedTopKRouterFL._compute_routing
    assert router_mod.GroupedTopKRouter._compute_routing is router_mod.GroupedTopKRouterFL._compute_routing
    assert router_mod.FusedTopKBiasRouter._compute_routing is router_mod.FusedTopKBiasRouterFL._compute_routing