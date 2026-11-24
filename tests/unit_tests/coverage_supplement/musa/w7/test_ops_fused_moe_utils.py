import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch
from vllm_fl.ops.fused_moe import fused_moe_utils as mu
Backend = mu.UnquantizedMoeBackend
Fmt = mu.mk.FusedMoEActivationFormat

def _platform(**overrides):
    values = {'is_tpu': lambda : False, 'is_rocm': lambda : False, 'is_cuda': lambda : False, 'is_xpu': lambda : False, 'is_cpu': lambda : False, 'is_out_of_tree': lambda : False, 'is_device_capability_family': lambda cap: False, 'device_type': 'cuda', 'vendor_name': 'nvidia'}
    values.update(overrides)
    return SimpleNamespace(**values)

def _moe_config(**overrides):
    values = {'moe_parallel_config': SimpleNamespace(dp_size=1, use_ep=False, use_batched_activation_format=False), 'activation': mu.MoEActivation.SILU, 'moe_backend': 'auto', 'is_lora_enabled': False}
    values.update(overrides)
    return SimpleNamespace(**values)

def _kernel_cls(supported, reason=None):

    class Kernel:

        def is_supported_config(self, config, a, b, activation_format):
            return (supported, reason)
    return Kernel

def test_should_use_fl_triton_experts_policy(monkeypatch):
    platform = _platform(is_cuda=lambda : True)
    monkeypatch.setattr(mu, 'current_platform', platform)
    monkeypatch.setattr(mu, 'use_flaggems', lambda : True)
    monkeypatch.setattr(mu, 'get_oot_blacklist', lambda : [])
    assert mu._should_use_fl_triton_experts() is True
    monkeypatch.setattr(mu, 'get_oot_blacklist', lambda : ['fused_moe'])
    assert mu._should_use_fl_triton_experts() is False
    monkeypatch.setattr(mu, 'get_oot_blacklist', lambda : [])
    monkeypatch.setattr(mu, 'use_flaggems', lambda : False)
    assert mu._should_use_fl_triton_experts() is False
    monkeypatch.setattr(mu, 'use_flaggems', lambda : True)
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cpu=lambda : True))
    assert mu._should_use_fl_triton_experts() is False

def test_priority_backends_rocm_xpu_cpu_iluvatar(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_rocm=lambda : True))
    assert mu._get_priority_backends(_moe_config()) == [Backend.AITER, Backend.TRITON, Backend.BATCHED_TRITON]
    monkeypatch.setattr(mu, 'current_platform', _platform(is_xpu=lambda : True))
    assert mu._get_priority_backends(_moe_config()) == [Backend.XPU]
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cpu=lambda : True))
    assert mu._get_priority_backends(_moe_config()) == [Backend.CPU]
    monkeypatch.setattr(mu, 'current_platform', _platform(vendor_name='iluvatar'))
    assert mu._get_priority_backends(_moe_config()) == [Backend.TRITON, Backend.BATCHED_TRITON]

def test_priority_backends_cuda_ordering(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True))
    assert mu._get_priority_backends(_moe_config()) == [Backend.FLASHINFER_TRTLLM, Backend.FLASHINFER_CUTLASS, Backend.TRITON, Backend.BATCHED_TRITON]
    hopper = _platform(is_cuda=lambda : True, is_device_capability_family=lambda cap: cap == 90)
    monkeypatch.setattr(mu, 'current_platform', hopper)
    assert mu._get_priority_backends(_moe_config()) == [Backend.TRITON, Backend.BATCHED_TRITON, Backend.FLASHINFER_TRTLLM, Backend.FLASHINFER_CUTLASS]
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True))
    config = _moe_config()
    config.moe_parallel_config.dp_size = 2
    assert mu._get_priority_backends(config) == [Backend.FLASHINFER_TRTLLM, Backend.TRITON, Backend.BATCHED_TRITON, Backend.FLASHINFER_CUTLASS]
    config = _moe_config(activation=mu.MoEActivation.SWIGLUOAI)
    assert mu._get_priority_backends(config) == [Backend.TRITON, Backend.BATCHED_TRITON, Backend.FLASHINFER_TRTLLM, Backend.FLASHINFER_CUTLASS]

def test_select_backend_tpu(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_tpu=lambda : True))
    assert mu.select_unquantized_moe_backend_oot(_moe_config()) == (Backend.TPU, None)

def _npu_platform(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True, device_type='npu'))

def test_select_backend_npu_accepts_auto(monkeypatch):
    _npu_platform(monkeypatch)
    (backend, experts) = mu.select_unquantized_moe_backend_oot(_moe_config())
    assert backend == Backend.TRITON
    assert experts is mu.TritonExpertsFL

@pytest.mark.parametrize('override', [{'is_lora_enabled': True}, {'moe_backend': 'flashinfer'}])
def test_select_backend_npu_rejects_unsupported(monkeypatch, override):
    _npu_platform(monkeypatch)
    with pytest.raises(NotImplementedError):
        mu.select_unquantized_moe_backend_oot(_moe_config(**override))

@pytest.mark.parametrize('parallel', [{'use_ep': True}, {'dp_size': 2}, {'use_batched_activation_format': True}])
def test_select_backend_npu_rejects_parallel_modes(monkeypatch, parallel):
    _npu_platform(monkeypatch)
    config = _moe_config()
    for (key, value) in parallel.items():
        setattr(config.moe_parallel_config, key, value)
    with pytest.raises(NotImplementedError):
        mu.select_unquantized_moe_backend_oot(config)

def test_select_backend_fl_triton(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True, device_type='cuda'))
    monkeypatch.setattr(mu, '_should_use_fl_triton_experts', lambda : True)
    (backend, experts) = mu.select_unquantized_moe_backend_oot(_moe_config())
    assert backend == Backend.TRITON
    assert experts is mu.TritonExpertsFL

def test_select_backend_lora_trtllm(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True, device_type='cuda'))
    monkeypatch.setattr(mu, '_should_use_fl_triton_experts', lambda : False)
    monkeypatch.setattr(mu, '_trtllm_bf16_lora_supported', lambda config: True)
    sentinel = object()
    module = ModuleType('vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe')
    module.TrtLlmBf16LoRAExperts = sentinel
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe', module)
    (backend, experts) = mu.select_unquantized_moe_backend_oot(_moe_config(is_lora_enabled=True))
    assert backend == Backend.FLASHINFER_TRTLLM
    assert experts is sentinel

def test_select_backend_lora_triton(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True, device_type='cuda'))
    monkeypatch.setattr(mu, '_should_use_fl_triton_experts', lambda : False)
    monkeypatch.setattr(mu, '_trtllm_bf16_lora_supported', lambda config: False)
    sentinel = object()
    monkeypatch.setattr(mu, 'backend_to_kernel_cls', lambda backend: [sentinel])
    (backend, experts) = mu.select_unquantized_moe_backend_oot(_moe_config(is_lora_enabled=True))
    assert backend == Backend.TRITON
    assert experts is sentinel

def test_select_backend_explicit_mapping_and_batched(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True, device_type='cuda'))
    monkeypatch.setattr(mu, '_should_use_fl_triton_experts', lambda : False)
    monkeypatch.setattr(mu, 'map_unquantized_backend', lambda name: Backend.TRITON)
    kernel = _kernel_cls(True)
    monkeypatch.setattr(mu, 'backend_to_kernel_cls', lambda backend: [kernel])
    (backend, experts) = mu.select_unquantized_moe_backend_oot(_moe_config(moe_backend='custom'))
    assert backend == Backend.TRITON
    assert experts is kernel
    config = _moe_config(moe_backend='custom')
    config.moe_parallel_config.use_batched_activation_format = True
    (backend, experts) = mu.select_unquantized_moe_backend_oot(config)
    assert backend == Backend.BATCHED_TRITON
    assert experts is kernel

def test_select_backend_explicit_unsupported_raises(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True, device_type='cuda'))
    monkeypatch.setattr(mu, '_should_use_fl_triton_experts', lambda : False)
    monkeypatch.setattr(mu, 'map_unquantized_backend', lambda name: Backend.TRITON)
    monkeypatch.setattr(mu, 'backend_to_kernel_cls', lambda backend: [_kernel_cls(False, 'bad shape')])
    with pytest.raises(ValueError, match='does not support'):
        mu.select_unquantized_moe_backend_oot(_moe_config(moe_backend='custom'))
    monkeypatch.setattr(mu, 'backend_to_kernel_cls', lambda backend: [_kernel_cls(False, None)])
    with pytest.raises(ValueError):
        mu.select_unquantized_moe_backend_oot(_moe_config(moe_backend='custom'))

def test_select_backend_auto_success_and_failure(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True, device_type='cuda'))
    monkeypatch.setattr(mu, '_should_use_fl_triton_experts', lambda : False)
    monkeypatch.setattr(mu, '_get_priority_backends', lambda config: [Backend.TRITON])
    monkeypatch.setattr(mu.envs, 'is_set', lambda name: False)
    kernel = _kernel_cls(True)
    monkeypatch.setattr(mu, 'backend_to_kernel_cls', lambda backend: [kernel])
    assert mu.select_unquantized_moe_backend_oot(_moe_config()) == (Backend.TRITON, kernel)
    monkeypatch.setattr(mu, 'backend_to_kernel_cls', lambda backend: [_kernel_cls(False, 'no')])
    with pytest.raises(NotImplementedError):
        mu.select_unquantized_moe_backend_oot(_moe_config())

def test_select_backend_aiter_skip_and_select(monkeypatch):
    monkeypatch.setattr(mu, 'current_platform', _platform(is_cuda=lambda : True, device_type='cuda'))
    monkeypatch.setattr(mu, '_should_use_fl_triton_experts', lambda : False)
    monkeypatch.setattr(mu.envs, 'is_set', lambda name: True)
    monkeypatch.setattr(mu, 'backend_to_kernel_cls', lambda backend: [_kernel_cls(True)])
    monkeypatch.setattr(mu, '_get_priority_backends', lambda config: [Backend.AITER, Backend.TRITON])
    monkeypatch.setattr(mu.envs, 'VLLM_ROCM_USE_AITER', False, raising=False)
    monkeypatch.setattr(mu.envs, 'VLLM_ROCM_USE_AITER_MOE', True, raising=False)
    monkeypatch.setattr(mu.rocm_aiter_ops, 'is_rdna_aiter_enabled', lambda : False)
    kernel = _kernel_cls(True)
    monkeypatch.setattr(mu, 'backend_to_kernel_cls', lambda backend: [kernel])
    assert mu.select_unquantized_moe_backend_oot(_moe_config()) == (Backend.TRITON, kernel)
    monkeypatch.setattr(mu.envs, 'VLLM_ROCM_USE_AITER', True, raising=False)
    monkeypatch.setattr(mu.envs, 'VLLM_ROCM_USE_AITER_MOE', True, raising=False)
    monkeypatch.setattr(mu.rocm_aiter_ops, 'is_rdna_aiter_enabled', lambda : True)
    (backend, _) = mu.select_unquantized_moe_backend_oot(_moe_config())
    assert backend in (Backend.TRITON, Backend.BATCHED_TRITON)
    monkeypatch.setattr(mu.rocm_aiter_ops, 'is_rdna_aiter_enabled', lambda : False)
    (backend, _) = mu.select_unquantized_moe_backend_oot(_moe_config())
    assert backend == Backend.AITER

def test_prepare_expert_assignment_naive_path():
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    (sorted_ids, flat_ids, padded) = mu._prepare_expert_assignment(topk_ids, {'BLOCK_SIZE_M': 16}, num_tokens=2, top_k_num=2, global_num_experts=100, expert_map=None)
    assert sorted_ids is None
    assert torch.equal(flat_ids, topk_ids.view(-1))
    assert padded.dtype == torch.int32
    assert padded.tolist() == [4 * 16]

@pytest.mark.parametrize('kwargs', [{'expert_map': torch.arange(4)}, {'num_tokens': 100, 'global_num_experts': 10}, {'use_int8_w8a16': True, 'block_shape': [128, 128]}])
def test_prepare_expert_assignment_align_path(monkeypatch, kwargs):
    captured = {}
    monkeypatch.setattr(mu, '_moe_align_block_size', lambda *args, **call_kwargs: captured.update(call_kwargs) or ('sorted', 'expert', 'padded'))
    base = {'num_tokens': 2, 'top_k_num': 2, 'global_num_experts': 100, 'expert_map': None}
    base.update(kwargs)
    result = mu._prepare_expert_assignment(torch.zeros((2, 2), dtype=torch.int32), {'BLOCK_SIZE_M': 8}, **base)
    assert result == ('sorted', 'expert', 'padded')
    assert captured['ignore_invalid_experts'] is False

def test_workspace_shapes_expands_on_npu(monkeypatch):
    monkeypatch.setattr(mu.TritonExperts, 'workspace_shapes', lambda self, M, N, K, topk, g, l, meta, act: ((M, 2), (M, 3), (M, 4)))
    instance = object.__new__(mu.TritonExpertsFL)
    instance.moe_config = SimpleNamespace(max_num_tokens=10)
    monkeypatch.setattr(mu, 'current_platform', _platform(device_type='npu'))
    (workspace13, workspace2, output) = mu.TritonExpertsFL.workspace_shapes(instance, 5, 8, 4, 1, 2, 2, None, mu.MoEActivation.SILU)
    assert workspace13 == (10, 2)
    assert workspace2 == (10, 3)
    assert output == (5, 4)
    monkeypatch.setattr(mu, 'current_platform', _platform(device_type='cuda'))
    (workspace13, workspace2, output) = mu.TritonExpertsFL.workspace_shapes(instance, 5, 8, 4, 1, 2, 2, None, mu.MoEActivation.SILU)
    assert workspace13 == (5, 2)
    assert workspace2 == (5, 3)

def _apply_args(hidden=None):
    if hidden is None:
        hidden = torch.zeros((2, 4), dtype=torch.float32)
    return {'output': torch.zeros((2, 4), dtype=torch.float32), 'hidden_states': hidden, 'w1': torch.zeros((2, 8, 4), dtype=torch.float32), 'w2': torch.zeros((2, 4, 4), dtype=torch.float32), 'topk_weights': torch.ones((2, 1), dtype=torch.float32), 'topk_ids': torch.zeros((2, 1), dtype=torch.int64), 'activation': mu.MoEActivation.SILU, 'global_num_experts': 2, 'expert_map': None, 'a1q_scale': None, 'a2_scale': None, 'workspace13': torch.empty(0), 'workspace2': torch.empty(0), 'expert_tokens_meta': None, 'apply_router_weight_on_input': False}

def _quant_config(**overrides):
    values = {'quant_dtype': None, 'weight_quant_dtype': None, 'use_fp8_w8a8': False, 'use_int8_w8a8': False, 'use_int8_w8a16': False, 'use_int4_w4a16': False, 'config_name': lambda dtype: 'cfg'}
    values.update(overrides)
    return SimpleNamespace(**values)

def _experts(**overrides):
    values = {'_lora_context': None, 'quant_config': _quant_config(), 'per_act_token_quant': False, 'block_shape': None, 'w1_scale': None, 'w2_scale': None, 'w1_bias': None, 'w2_bias': None, 'quant_dtype': None, 'quantization_emulation': False, 'activation_config': SimpleNamespace(clamp_limit=None)}
    values.update(overrides)
    return SimpleNamespace(**values)

def test_apply_npu_uses_grouped_experts(monkeypatch):
    captured = {}
    monkeypatch.setattr(mu, 'call_op', lambda name, *args, **kwargs: captured.update(name=name, args=args, kwargs=kwargs) or torch.full((2, 4), 5.0))
    instance = _experts()
    args = _apply_args()
    dummy_hidden = SimpleNamespace(device=SimpleNamespace(type='npu'))
    args['hidden_states'] = dummy_hidden
    mu.TritonExpertsFL.apply(instance, **args)
    assert captured['name'] == 'grouped_moe_experts'
    assert torch.equal(args['output'], torch.full((2, 4), 5.0))

def test_apply_kunlunxin_uses_vendor_impl(monkeypatch):
    sentinel = torch.full((2, 4), 3.0)
    captured = {}

    def klx_impl(*args, **kwargs):
        captured.update(kwargs)
        return sentinel
    base = 'vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe'
    outer = ModuleType('vllm_fl.dispatch.backends.vendor.kunlunxin')
    outer.__path__ = []
    impl = ModuleType('vllm_fl.dispatch.backends.vendor.kunlunxin.impl')
    impl.__path__ = []
    package = ModuleType(base)
    package.__path__ = []
    module = ModuleType(f'{base}.fused_moe')
    module.fused_experts_impl = klx_impl
    monkeypatch.setitem(sys.modules, outer.__name__, outer)
    monkeypatch.setitem(sys.modules, impl.__name__, impl)
    monkeypatch.setitem(sys.modules, base, package)
    monkeypatch.setitem(sys.modules, f'{base}.fused_moe', module)
    monkeypatch.setattr(mu, 'current_platform', _platform(vendor_name='kunlunxin'))
    instance = _experts(quant_config=_quant_config(use_fp8_w8a8=True, use_int8_w8a8=True))
    args = _apply_args()
    mu.TritonExpertsFL.apply(instance, **args)
    assert torch.equal(args['output'], sentinel)
    assert captured['use_fp8_w8a8'] is True
    assert captured['use_int8_w8a8'] is True

@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_apply_lora_pipeline(monkeypatch, dtype):
    monkeypatch.setattr(mu, 'try_get_optimal_moe_config', lambda *args, **kwargs: {'BLOCK_SIZE_M': 16})
    monkeypatch.setattr(mu, '_resize_cache', lambda buffer, shape: torch.empty(shape, dtype=dtype))
    monkeypatch.setattr(mu, '_prepare_expert_assignment', lambda *args, **kwargs: (None, None, None))
    kernel_calls = []
    monkeypatch.setattr(mu, '_invoke_fused_moe_triton_kernel', lambda *args, **kwargs: kernel_calls.append((args, kwargs)))
    activation_calls = []
    monkeypatch.setattr(mu, 'apply_moe_activation', lambda *args, **kwargs: activation_calls.append((args, kwargs)))
    monkeypatch.setattr(mu, 'moe_kernel_quantize_input', lambda A, *args, **kwargs: (A, None))
    lora_calls = {'w13': [], 'w2': []}
    instance = _experts(_lora_context=object())
    instance.moe_problem_size = lambda hidden, w1, w2, ids: (2, 2, 8, 4, 1)
    instance.adjust_N_for_activation = lambda N, activation: N // 2
    instance.moe_sum = lambda tensor, output: None
    instance.apply_w13_lora = lambda *args, **kwargs: lora_calls['w13'].append(kwargs) or ('s', 'e', 'p', 'map')
    instance.apply_w2_lora = lambda *args, **kwargs: lora_calls['w2'].append(kwargs)
    args = _apply_args(hidden=torch.zeros((2, 4), dtype=dtype))
    args['output'] = torch.zeros((2, 4), dtype=dtype)
    args['w1'] = torch.zeros((2, 8, 4), dtype=dtype)
    args['w2'] = torch.zeros((2, 4, 4), dtype=dtype)
    mu.TritonExpertsFL.apply(instance, **args)
    assert len(kernel_calls) == 2
    assert kernel_calls[0][0][9] is False
    assert kernel_calls[1][0][9] is True
    assert len(activation_calls) == 1
    assert len(lora_calls['w13']) == 1
    assert len(lora_calls['w2']) == 1

def test_apply_lora_pipeline_respects_router_weight_flag(monkeypatch):
    monkeypatch.setattr(mu, 'try_get_optimal_moe_config', lambda *args, **kwargs: {'BLOCK_SIZE_M': 16})
    monkeypatch.setattr(mu, '_resize_cache', lambda buffer, shape: torch.empty(shape))
    monkeypatch.setattr(mu, '_prepare_expert_assignment', lambda *args, **kwargs: (None, None, None))
    kernel_calls = []
    monkeypatch.setattr(mu, '_invoke_fused_moe_triton_kernel', lambda *args, **kwargs: kernel_calls.append((args, kwargs)))
    monkeypatch.setattr(mu, 'apply_moe_activation', lambda *args, **kwargs: None)
    monkeypatch.setattr(mu, 'moe_kernel_quantize_input', lambda A, *args, **kwargs: (A, None))
    instance = _experts(_lora_context=object())
    instance.moe_problem_size = lambda hidden, w1, w2, ids: (2, 2, 8, 4, 1)
    instance.adjust_N_for_activation = lambda N, activation: N // 2
    instance.moe_sum = lambda tensor, output: None
    instance.apply_w13_lora = lambda *args, **kwargs: (None, None, None, None)
    instance.apply_w2_lora = lambda *args, **kwargs: None
    args = _apply_args()
    args['apply_router_weight_on_input'] = True
    mu.TritonExpertsFL.apply(instance, **args)
    assert kernel_calls[1][0][9] is False

def test_apply_lora_pipeline_asserts(monkeypatch):
    instance = _experts(_lora_context=object())
    args = _apply_args()
    args['hidden_states'] = torch.zeros((2, 2, 4))
    with pytest.raises(AssertionError):
        mu.TritonExpertsFL.apply(instance, **args)
    args = _apply_args()
    args['hidden_states'] = torch.zeros((4, 2)).t()
    with pytest.raises(AssertionError):
        mu.TritonExpertsFL.apply(instance, **args)
    args = _apply_args()
    args['w1'] = torch.zeros((2, 4, 8)).transpose(1, 2)
    with pytest.raises(AssertionError):
        mu.TritonExpertsFL.apply(instance, **args)

def test_moe_sum_dispatches(monkeypatch):
    captured = {}
    monkeypatch.setattr(mu, '_moe_sum', lambda tensor, output: captured.update(output=output))
    output = torch.zeros(2, 4)
    mu.TritonExpertsFL.moe_sum(SimpleNamespace(), torch.zeros(2, 1, 4), output)
    assert captured['output'] is output