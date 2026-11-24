import sys
from types import SimpleNamespace
import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm_fl.quantization.w8a8 import moe_experts

def _quant_config(**overrides):
    config = SimpleNamespace(use_int8_w8a8=True, per_act_token_quant=True, block_shape=None, w1_scale=torch.ones((2, 8, 1), dtype=torch.float32), w2_scale=torch.ones((2, 4, 1), dtype=torch.float32), w1_bias=None, w2_bias=None)
    for (key, value) in overrides.items():
        setattr(config, key, value)
    return config

def _arguments(**overrides):
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    values = {'output': torch.empty_like(hidden_states), 'hidden_states': hidden_states, 'w1': torch.ones((2, 8, 4), dtype=torch.int8), 'w2': torch.ones((2, 4, 4), dtype=torch.int8), 'topk_weights': torch.ones((2, 1), dtype=torch.float32), 'topk_ids': torch.zeros((2, 1), dtype=torch.int64), 'activation': MoEActivation.SILU, 'global_num_experts': 2, 'expert_map': None, 'a1q_scale': None, 'a2_scale': None, 'workspace13': torch.empty(0), 'workspace2': torch.empty(0), 'expert_tokens_meta': None, 'apply_router_weight_on_input': False}
    values.update(overrides)
    return values

def _experts_instance(quant_config=None, *, clamp_limit=None, lora=None):
    return SimpleNamespace(quant_config=_quant_config() if quant_config is None else quant_config, activation_config=SimpleNamespace(clamp_limit=clamp_limit), _lora_context=lora)

def test_dynamic_int8_qdq_preserves_dtype_and_handles_zero_rows():
    value = torch.tensor([[3.0, -6.0, 1.5], [0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    result = moe_experts._dynamic_int8_qdq(value)
    assert result.dtype == torch.bfloat16
    assert torch.count_nonzero(result[1]) == 0
    assert torch.isfinite(result).all()
    assert result[0, 0] > 0
    assert result[0, 1] < 0

@pytest.mark.parametrize('clamp_limit', [None, 0.0])
def test_w8a8_swiglu_without_clamp(clamp_limit):
    gate_up = torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.float32)
    expected = torch.nn.functional.silu(gate_up[:, :2]) * gate_up[:, 2:]
    assert torch.allclose(moe_experts._w8a8_swiglu(gate_up, clamp_limit), expected)

def test_w8a8_swiglu_with_clamp():
    gate_up = torch.tensor([[5.0, -5.0, 10.0, -10.0]], dtype=torch.float32)
    (gate, up) = gate_up.float().chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate.clamp(max=2.0)) * up.clamp(-2.0, 2.0)).to(gate_up.dtype)
    assert torch.allclose(moe_experts._w8a8_swiglu(gate_up, 2.0), expected)

def test_local_expert_ids_handles_none_and_negative_routes():
    topk_ids = torch.tensor([[2, -1], [0, 1]], dtype=torch.int32)
    direct = moe_experts._local_expert_ids(topk_ids, None)
    assert direct.dtype == torch.int64
    assert torch.equal(direct, topk_ids.long())
    expert_map = torch.tensor([1, -1, 0], dtype=torch.int64)
    mapped = moe_experts._local_expert_ids(topk_ids, expert_map)
    assert torch.equal(mapped, torch.tensor([[0, -1], [1, -1]]))

def test_native_w8a8_fused_experts_applies_router_weights_and_skips_invalid():
    hidden = torch.tensor([[1.0, -1.0, 0.5, 2.0], [0.5, 0.25, -0.75, 1.0]])
    w1 = torch.arange(1, 33, dtype=torch.int8).reshape(1, 8, 4)
    w2 = torch.arange(1, 17, dtype=torch.int8).reshape(1, 4, 4)
    w1_scale = torch.full((1, 8, 1), 0.01, dtype=torch.float32)
    w2_scale = torch.full((1, 4, 1), 0.02, dtype=torch.float32)
    topk_ids = torch.zeros((2, 1), dtype=torch.int64)
    weights = torch.tensor([[0.75], [0.5]], dtype=torch.float32)
    unweighted = moe_experts._native_w8a8_fused_experts(hidden, w1, w2, weights, topk_ids, w1_scale, w2_scale, None, None, None, True, clamp_limit=None)
    weighted = moe_experts._native_w8a8_fused_experts(hidden, w1, w2, weights, topk_ids, w1_scale, w2_scale, None, None, None, False, clamp_limit=None)
    assert unweighted.shape == hidden.shape
    assert torch.allclose(weighted, unweighted * weights, rtol=0.1, atol=0.1)
    expert_map = torch.tensor([0, -1], dtype=torch.int64)
    skipped = moe_experts._native_w8a8_fused_experts(hidden, w1, w2, weights, torch.tensor([[0], [1]], dtype=torch.int64), w1_scale, w2_scale, None, None, expert_map, False, clamp_limit=None)
    assert torch.count_nonzero(skipped[1]) == 0

def _contract_args():
    return {'hidden_states': torch.ones((2, 4), dtype=torch.bfloat16), 'w1': torch.ones((2, 8, 4), dtype=torch.int8), 'w2': torch.ones((2, 4, 4), dtype=torch.int8), 'topk_weights': torch.ones((2, 1), dtype=torch.float32), 'topk_ids': torch.zeros((2, 1), dtype=torch.int64), 'activation': MoEActivation.SILU, 'w1_scale': torch.ones((2, 8, 1), dtype=torch.float32), 'w2_scale': torch.ones((2, 4, 1), dtype=torch.float32), 'w1_bias': None, 'w2_bias': None}

def test_validate_w8a8_contract_accepts_canonical_contract():
    assert moe_experts._validate_w8a8_contract(**_contract_args()) is None

def _mutated(args, **updates):
    values = dict(args)
    values.update(updates)
    return values

@pytest.mark.parametrize(('mutate', 'error'), [(lambda a: {**a, 'activation': MoEActivation.GELU}, NotImplementedError), (lambda a: {**a, 'hidden_states': torch.ones((2, 4), dtype=torch.bfloat16).t()}, ValueError), (lambda a: {**a, 'w1': torch.ones((8, 4), dtype=torch.int8)}, ValueError), (lambda a: {**a, 'w2': torch.ones((4, 4), dtype=torch.int8)}, ValueError), (lambda a: {**a, 'w1': torch.ones((2, 8, 4), dtype=torch.float32)}, TypeError)])
def test_validate_w8a8_contract_rejects_bad_shapes_and_dtypes(mutate, error):
    with pytest.raises(error):
        moe_experts._validate_w8a8_contract(**mutate(_contract_args()))

def test_validate_w8a8_contract_accepts_matching_bias():
    base = _contract_args()
    assert moe_experts._validate_w8a8_contract(**{**base, 'w1_bias': torch.zeros((2, 8), dtype=torch.float32), 'w2_bias': torch.zeros((2, 4), dtype=torch.float32)}) is None

@pytest.mark.parametrize(('mutate', 'error'), [(lambda c: _quant_config(use_int8_w8a8=False), ValueError), (lambda c: _quant_config(per_act_token_quant=False), ValueError), (lambda c: _quant_config(block_shape=[128, 128]), ValueError)])
def test_functional_and_flaggems_experts_reject_bad_quant_config(monkeypatch, mutate, error):
    monkeypatch.setattr(moe_experts, '_flaggems_fused_experts_impl', lambda **kwargs: pytest.fail('must not run'))
    monkeypatch.setattr(moe_experts, '_vllm_fused_experts', lambda **kwargs: pytest.fail('must not run'))
    config = mutate(None)
    for cls in (moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts):
        with pytest.raises(error):
            cls.apply(_experts_instance(config), **_arguments())

def test_experts_reject_lora_and_prequantized_input(monkeypatch):
    monkeypatch.setattr(moe_experts, '_flaggems_fused_experts_impl', lambda **kwargs: pytest.fail('must not run'))
    for cls in (moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts):
        with pytest.raises(NotImplementedError, match='LoRA'):
            cls.apply(_experts_instance(lora=object()), **_arguments())
        with pytest.raises(ValueError, match='quantized before'):
            cls.apply(_experts_instance(), **_arguments(hidden_states=torch.ones((2, 4), dtype=torch.int8), a1q_scale=torch.ones((2, 1))))

def test_experts_require_float_dtype(monkeypatch):
    monkeypatch.setattr(moe_experts, '_flaggems_fused_experts_impl', lambda **kwargs: pytest.fail('must not run'))
    for cls in (moe_experts.FlagGemsW8A8Experts, moe_experts.VllmFunctionalW8A8Experts):
        with pytest.raises(TypeError, match='floating-point'):
            cls.apply(_experts_instance(), **_arguments(hidden_states=torch.ones((2, 4), dtype=torch.int64)))

def test_functional_experts_dispatch_to_native_vllm(monkeypatch):
    calls = []
    monkeypatch.setattr(moe_experts, '_vllm_fused_experts', lambda **kwargs: calls.append(kwargs) or torch.full_like(kwargs['hidden_states'], 2))
    config = _quant_config()
    args = _arguments()
    moe_experts.VllmFunctionalW8A8Experts.apply(SimpleNamespace(quant_config=config, _lora_context=None), **args)
    assert calls[0]['quant_config'] is config
    assert torch.equal(args['output'], torch.full_like(args['output'], 2))

def test_flaggems_experts_dispatch_non_npu(monkeypatch):
    calls = []
    monkeypatch.setattr(moe_experts, '_flaggems_fused_experts_impl', lambda **kwargs: calls.append(kwargs) or torch.full_like(kwargs['hidden_states'], 3))
    args = _arguments()
    moe_experts.FlagGemsW8A8Experts.apply(_experts_instance(), **args)
    assert calls[0]['use_int8_w8a8'] is True
    assert calls[0]['per_channel_quant'] is True
    assert torch.equal(args['output'], torch.full_like(args['output'], 3))

def test_flaggems_experts_dispatch_npu(monkeypatch):
    import vllm_fl.dispatch as dispatch
    seen = {}

    def fake_call_op(op_name, **kwargs):
        seen['op'] = op_name
        seen['kwargs'] = kwargs
        return torch.full_like(kwargs['hidden_states'], 4)
    monkeypatch.setattr(moe_experts, '_is_ascend_npu_tensor', lambda value: True)
    monkeypatch.setattr(moe_experts, '_flaggems_fused_experts_impl', lambda **kwargs: pytest.fail('NPU must not run FlagGems'))
    monkeypatch.setattr(dispatch, 'call_op', fake_call_op)
    args = _arguments()
    moe_experts.FlagGemsW8A8Experts.apply(_experts_instance(clamp_limit=10.0), **args)
    assert seen['op'] == 'w8a8_moe_experts'
    assert seen['kwargs']['clamp_limit'] == 10.0
    assert torch.equal(args['output'], torch.full_like(args['output'], 4))

def _fake_torch_npu(monkeypatch, **functions):
    module = SimpleNamespace(**functions)
    monkeypatch.setitem(sys.modules, 'torch_npu', module)
    return module

def test_ascend_grouped_rejects_unsupported_dtype(monkeypatch):
    _fake_torch_npu(monkeypatch)
    with pytest.raises(TypeError, match='bfloat16 or float16'):
        moe_experts._ascend_w8a8_grouped_experts(torch.ones((1, 4), dtype=torch.float32), torch.ones((1, 4, 4), dtype=torch.int8), torch.ones((1, 4, 4), dtype=torch.int8), torch.ones((1, 1)), torch.zeros((1, 1), dtype=torch.int64), torch.ones((1, 4, 1)), torch.ones((1, 4, 1)), None, None, None, False)

def test_ascend_grouped_handles_empty_inputs(monkeypatch):
    _fake_torch_npu(monkeypatch)
    w1 = torch.ones((2, 8, 4), dtype=torch.int8)
    w2 = torch.ones((2, 4, 4), dtype=torch.int8)
    empty = moe_experts._ascend_w8a8_grouped_experts(torch.ones((0, 4), dtype=torch.bfloat16), w1, w2, torch.ones((0, 1)), torch.zeros((0, 1), dtype=torch.int64), torch.ones((2, 8, 1)), torch.ones((2, 4, 1)), None, None, None, False)
    assert empty.shape == (0, 4)
    all_remote = moe_experts._ascend_w8a8_grouped_experts(torch.ones((2, 4), dtype=torch.bfloat16), w1, w2, torch.ones((2, 1)), torch.zeros((2, 1), dtype=torch.int64), torch.ones((2, 8, 1)), torch.ones((2, 4, 1)), None, None, torch.full((2,), -1, dtype=torch.int64), False)
    assert all_remote.shape == (2, 4)
    assert torch.count_nonzero(all_remote) == 0