"""CPU contracts for PlatformFL helpers (no accelerator initialization).

Vendor-specific branches are reached by injecting lightweight module doubles
and by swapping ``torch.cuda`` / ``torch.musa`` / ``torch.gcu`` attributes.
Heavy constructors are bypassed with ``object.__new__``.
"""
import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from vllm.platforms import Platform
import vllm_fl.platform as platform_mod
from vllm_fl.platform import PlatformFL
_counter = {'n': 0}

def _sub(**attrs):
    _counter['n'] += 1
    return type(f"ProbePlatform{_counter['n']}", (PlatformFL,), attrs)

def _inst(**attrs):
    obj = object.__new__(PlatformFL)
    for (key, value) in attrs.items():
        setattr(obj, key, value)
    return obj

def _install_fake_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__path__ = []
    for (key, value) in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    (parent_name, _, child) = name.rpartition('.')
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child, module, raising=False)
    return module

@pytest.mark.parametrize('vendor,device_type,expected', [('iluvatar', 'cuda', False), ('mthreads', 'musa', True), ('hygon', 'cuda', False), ('gcu', 'gcu', True), ('enflame', 'gcu', True), ('nvidia', 'cuda', True), ('ascend', 'npu', False)])
def test_is_cuda_alike(vendor, device_type, expected):
    obj = _inst(vendor_name=vendor, device_type=device_type)
    assert PlatformFL.is_cuda_alike(obj) is expected

@pytest.mark.parametrize('vendor,device_type,expected', [('nvidia', 'cuda', True), ('amd', 'cuda', False), ('ascend', 'npu', False)])
def test_is_cuda(vendor, device_type, expected):
    obj = _inst(vendor_name=vendor, device_type=device_type)
    assert PlatformFL.is_cuda(obj) is expected

@pytest.mark.parametrize('vendor,expected', [('amd', True), ('nvidia', False)])
def test_is_rocm(vendor, expected):
    assert PlatformFL.is_rocm(_inst(vendor_name=vendor)) is expected

@pytest.mark.parametrize('vendor,expected', [('nvidia', True), ('amd', False)])
def test_is_sleep_mode_available(vendor, expected):
    assert PlatformFL.is_sleep_mode_available(_inst(vendor_name=vendor)) is expected

@pytest.mark.parametrize('present,available,expected', [(True, True, True), (True, False, False), (False, None, False)])
def test_is_musa(monkeypatch, present, available, expected):
    if present:
        monkeypatch.setattr(torch, 'musa', SimpleNamespace(is_available=Mock(return_value=available)), raising=False)
    else:
        monkeypatch.delattr(torch, 'musa', raising=False)
    assert PlatformFL.is_musa(_inst()) is expected

@pytest.mark.parametrize('present,available,expected', [(True, True, True), (True, False, False), (False, None, False)])
def test_is_gcu(monkeypatch, present, available, expected):
    if present:
        monkeypatch.setattr(torch, 'gcu', SimpleNamespace(is_available=Mock(return_value=available)), raising=False)
    else:
        monkeypatch.delattr(torch, 'gcu', raising=False)
    assert PlatformFL.is_gcu(_inst()) is expected

def test_supported_dtypes_property():
    assert PlatformFL.supported_dtypes.fget(_inst()) == [torch.bfloat16, torch.float16, torch.float32]

def test_check_if_supports_dtype_raises_for_low_capability():
    cls = _sub(vendor_name='nvidia', device_name='RTX', has_device_capability=staticmethod(lambda cc: False), get_device_capability=classmethod(lambda cls, device_id=0: SimpleNamespace(as_version_str=lambda : '7.5')))
    with pytest.raises(ValueError, match='at least 8.0') as exc:
        cls.check_if_supports_dtype(torch.bfloat16)
    assert 'compute capability 7.5' in str(exc.value)

def test_check_if_supports_dtype_handles_missing_capability():
    cls = _sub(vendor_name='nvidia', device_name='RTX', has_device_capability=staticmethod(lambda cc: False), get_device_capability=classmethod(lambda cls, device_id=0: None))
    with pytest.raises(ValueError, match='does not have a compute capability'):
        cls.check_if_supports_dtype(torch.bfloat16)

def test_check_if_supports_dtype_accepts_supported():
    cls = _sub(vendor_name='nvidia', has_device_capability=staticmethod(lambda cc: True))
    assert cls.check_if_supports_dtype(torch.bfloat16) is None

def test_check_if_supports_dtype_ignores_other_vendors():
    cls = _sub(vendor_name='mthreads')
    assert cls.check_if_supports_dtype(torch.bfloat16) is None

def test_get_current_memory_usage():
    fn = SimpleNamespace(empty_cache=Mock(), reset_peak_memory_stats=Mock(), max_memory_allocated=Mock(return_value=1234))
    cls = _sub(torch_device_fn=fn)
    assert cls.get_current_memory_usage('cpu') == 1234
    fn.empty_cache.assert_called_once_with()
    fn.reset_peak_memory_stats.assert_called_once_with('cpu')
    fn.max_memory_allocated.assert_called_once_with('cpu')

def test_set_device_nvidia_warms_context(monkeypatch):
    fn = SimpleNamespace(set_device=Mock())
    zeros = Mock(return_value='tensor')
    monkeypatch.setattr(torch, 'zeros', zeros)
    cls = _sub(vendor_name='nvidia', torch_device_fn=fn)
    cls.set_device('cuda:0')
    fn.set_device.assert_called_once_with('cuda:0')
    zeros.assert_called_once()

def test_set_device_non_nvidia_skips_warmup(monkeypatch):
    fn = SimpleNamespace(set_device=Mock())
    zeros = Mock()
    monkeypatch.setattr(torch, 'zeros', zeros)
    cls = _sub(vendor_name='mthreads', torch_device_fn=fn)
    cls.set_device('musa:0')
    fn.set_device.assert_called_once_with('musa:0')
    zeros.assert_not_called()

def test_empty_cache():
    fn = SimpleNamespace(empty_cache=Mock())
    _sub(torch_device_fn=fn).empty_cache()
    fn.empty_cache.assert_called_once_with()

def test_get_device_name_returns_configured_name():
    assert _sub(device_name='musa').get_device_name(3) == 'musa'

@pytest.mark.parametrize('device_type,expected', [('cuda', True), ('xpu', True), ('npu', True), ('musa', True), ('gcu', True), ('cpu', False)])
def test_is_pin_memory_available(device_type, expected):
    assert _sub(device_type=device_type).is_pin_memory_available() is expected

def test_import_kernels_metax_imports_all_vendor_modules(monkeypatch):
    _install_fake_module(monkeypatch, 'mcoplib._C')
    _install_fake_module(monkeypatch, 'mcoplib._moe_C')
    _install_fake_module(monkeypatch, 'vllm_fl.dispatch.backends.vendor.metax.patches')
    cls = _sub(vendor_name='metax', device_type='cuda')
    cls.import_kernels()

def test_import_kernels_metax_swallows_missing_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mcoplib._C', None)
    monkeypatch.setitem(sys.modules, 'mcoplib._moe_C', None)
    monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch.backends.vendor.metax.patches', None)
    cls = _sub(vendor_name='metax', device_type='cuda')
    cls.import_kernels()

def test_import_kernels_musa_applies_patches(monkeypatch):
    applied = Mock()
    _install_fake_module(monkeypatch, 'vllm_fl.dispatch.backends.vendor.musa.patch', apply_musa_patches=applied)
    monkeypatch.setattr(Platform, 'import_kernels', classmethod(lambda cls: None), raising=False)
    cls = _sub(vendor_name='mthreads', device_type='musa')
    cls.import_kernels()
    applied.assert_called_once_with()

def test_import_kernels_musa_swallows_failure(monkeypatch):

    def boom():
        raise RuntimeError('patch failed')
    _install_fake_module(monkeypatch, 'vllm_fl.dispatch.backends.vendor.musa.patch', apply_musa_patches=boom)
    monkeypatch.setattr(Platform, 'import_kernels', classmethod(lambda cls: None), raising=False)
    _sub(vendor_name='mthreads', device_type='musa').import_kernels()

def test_import_kernels_other_vendor_defers_to_super(monkeypatch):
    base = Mock()
    monkeypatch.setattr(Platform, 'import_kernels', classmethod(lambda cls: base()), raising=False)
    _sub(vendor_name='nvidia', device_type='cuda').import_kernels()
    base.assert_called_once_with()

def test_import_ir_kernels(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.kernels')
    assert _sub().import_ir_kernels() is None

def test_get_punica_wrapper():
    assert _sub().get_punica_wrapper() == 'vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU'

@pytest.mark.parametrize('dist_backend,device_type,vendor,expected', [('flagcx', 'cuda', 'iluvatar', 'vllm_fl.dispatch.backends.vendor.iluvatar.communicator.IluvatarCommunicatorFL'), ('flagcx', 'cuda', 'amd', 'vllm_fl.distributed.cuda_flagcx_communicator.CudaFlagcxCommunicatorFL'), ('flagcx', 'cuda', 'nvidia', 'vllm_fl.distributed.communicator.CommunicatorFL'), ('nccl', 'cuda', 'nvidia', 'vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator')])
def test_get_device_communicator_cls(dist_backend, device_type, vendor, expected):
    cls = _sub(dist_backend=dist_backend, device_type=device_type, vendor_name=vendor)
    assert cls.get_device_communicator_cls() == expected

def test_get_static_graph_wrapper_cls_musa():
    assert _sub(device_type='musa').get_static_graph_wrapper_cls() == 'vllm_fl.dispatch.backends.vendor.musa.graph.MUSAGraphWrapper'

def test_get_static_graph_wrapper_cls_amd_v2(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.config', get_current_vllm_config_or_none=lambda : SimpleNamespace(use_v2_model_runner=True))
    cls = _sub(device_type='cuda', vendor_name='amd')
    assert cls.get_static_graph_wrapper_cls() == 'vllm.compilation.cuda_graph.CUDAGraphWrapper'

def test_get_static_graph_wrapper_cls_amd_v1(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.config', get_current_vllm_config_or_none=lambda : SimpleNamespace(use_v2_model_runner=False))
    cls = _sub(device_type='cuda', vendor_name='amd')
    assert cls.get_static_graph_wrapper_cls() == 'vllm_fl.compilation.graph.GraphWrapper'

def test_get_static_graph_wrapper_cls_amd_no_config(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.config', get_current_vllm_config_or_none=lambda : None)
    cls = _sub(device_type='cuda', vendor_name='amd')
    assert cls.get_static_graph_wrapper_cls() == 'vllm_fl.compilation.graph.GraphWrapper'

@pytest.mark.parametrize('device_type', ['npu', 'musa'])
def test_get_compile_backend_eager_for_graph_less(device_type):
    assert _sub(device_type=device_type).get_compile_backend() == 'eager'

def test_get_compile_backend_defers_to_super(monkeypatch):
    monkeypatch.setattr(Platform, 'get_compile_backend', classmethod(lambda cls: 'base-backend'), raising=False)
    assert _sub(device_type='cuda').get_compile_backend() == 'base-backend'

@pytest.mark.parametrize('vendor,expected', [('nvidia', True), ('ascend', True), ('metax', True), ('hygon', True), ('mthreads', True), ('iluvatar', True), ('thead', True), ('gcu', True), ('enflame', True), ('kunlunxin', True), ('amd', True), ('unknown', False)])
def test_support_static_graph_mode(vendor, expected):
    assert _sub(vendor_name=vendor).support_static_graph_mode() is expected

def test_insert_blocks_to_device_copies_selected_blocks():
    src = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    dst = torch.zeros(2, 3, 2)
    PlatformFL.insert_blocks_to_device(src, dst, torch.tensor([2, 0]), torch.tensor([1, 2]))
    assert torch.equal(dst[:, 1], src[:, 2])
    assert torch.equal(dst[:, 2], src[:, 0])

def test_swap_out_blocks_to_host_copies_selected_blocks():
    src = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    dst = torch.zeros(2, 3, 2)
    PlatformFL.swap_out_blocks_to_host(src, dst, torch.tensor([1]), torch.tensor([0]))
    assert torch.equal(dst[:, 0], src[:, 1])

def test_align_hybrid_block_size_ascend_dense(monkeypatch):
    align = Mock()
    _install_fake_module(monkeypatch, 'vllm_fl.kv_cache.ascend.hybrid', align_hybrid_block_size=align, uses_dense_hybrid_cache=lambda model_config: True)
    base = Mock()
    monkeypatch.setattr(Platform, '_align_hybrid_block_size', classmethod(lambda cls, c, b: base()), raising=False)
    cfg = SimpleNamespace(model_config=object())
    _sub(vendor_name='ascend')._align_hybrid_block_size(cfg, 'backend')
    align.assert_called_once_with(cfg)
    base.assert_not_called()

def test_align_hybrid_block_size_other_vendor(monkeypatch):
    base = Mock()
    monkeypatch.setattr(Platform, '_align_hybrid_block_size', classmethod(lambda cls, c, b: base()), raising=False)
    _sub(vendor_name='nvidia')._align_hybrid_block_size(SimpleNamespace(), 'b')
    base.assert_called_once()

def test_support_and_opaque_flags():
    cls = _sub()
    assert cls.support_hybrid_kv_cache() is True
    assert cls.opaque_attention_op() is True

@pytest.mark.parametrize('vendor,dist_backend,expected', [('hygon', 'nccl', False), ('amd', 'nccl', False), ('nvidia', 'nccl', True), ('nvidia', 'flagcx', False)])
def test_use_custom_allreduce(vendor, dist_backend, expected):
    cls = _sub(vendor_name=vendor, dist_backend=dist_backend)
    assert cls.use_custom_allreduce() is expected

def test_pre_register_npu_forces_breakable_off(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm_fl.dispatch.backends.vendor.ascend')
    monkeypatch.setenv('VLLM_USE_BREAKABLE_CUDAGRAPH', '1')
    _sub(device_name='npu', vendor_name='mthreads').pre_register_and_update()
    import os
    assert os.environ['VLLM_USE_BREAKABLE_CUDAGRAPH'] == '0'

def test_pre_register_npu_sets_default(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm_fl.dispatch.backends.vendor.ascend')
    monkeypatch.delenv('VLLM_USE_BREAKABLE_CUDAGRAPH', raising=False)
    _sub(device_name='npu', vendor_name='mthreads').pre_register_and_update()
    import os
    assert os.environ['VLLM_USE_BREAKABLE_CUDAGRAPH'] == '0'

def test_pre_register_iluvatar_applies_patches(monkeypatch):
    chained = Mock()
    router = Mock()
    _install_fake_module(monkeypatch, 'vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar', patch_triton_chained_or_for_iluvatar=chained)
    _install_fake_module(monkeypatch, 'vllm_fl.dispatch.backends.vendor.iluvatar.moe_router', patch_moe_router=router)
    _sub(device_name='cuda', vendor_name='iluvatar').pre_register_and_update()
    chained.assert_called_once_with()
    router.assert_called_once_with()

def test_pre_register_amd_disables_native_kernels():
    disable = Mock()
    cls = _sub(device_name='cuda', vendor_name='amd', _disable_rocm_native_kernels=staticmethod(disable))
    cls.pre_register_and_update()
    disable.assert_called_once_with()

def test_disable_rocm_native_kernels_present(monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object())
    import os
    monkeypatch.delenv('VLLM_ROCM_USE_SKINNY_GEMM', raising=False)
    _sub(vendor_name='amd')._disable_rocm_native_kernels()
    assert 'VLLM_ROCM_USE_SKINNY_GEMM' not in os.environ

def test_disable_rocm_native_kernels_absent_sets_default(monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    import os
    monkeypatch.delenv('VLLM_ROCM_USE_SKINNY_GEMM', raising=False)
    _sub(vendor_name='amd')._disable_rocm_native_kernels()
    assert os.environ['VLLM_ROCM_USE_SKINNY_GEMM'] == '0'

def test_disable_rocm_native_kernels_respects_override(monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    import os
    monkeypatch.setenv('VLLM_ROCM_USE_SKINNY_GEMM', '1')
    _sub(vendor_name='amd')._disable_rocm_native_kernels()
    assert os.environ['VLLM_ROCM_USE_SKINNY_GEMM'] == '1'

@pytest.mark.parametrize('vendor,capability,expected', [('nvidia', True, True), ('nvidia', False, False), ('mthreads', True, False)])
def test_supports_fp8(vendor, capability, expected):
    cls = _sub(vendor_name=vendor, has_device_capability=staticmethod(lambda cc: capability))
    assert cls.supports_fp8() is expected

def test_get_device_uuid_iluvatar_synthetic():
    cls = _sub(device_type='cuda', vendor_name='iluvatar')
    assert cls.get_device_uuid(3) == 'ILUVATAR-3'

def test_get_device_uuid_cuda_uses_pynvml(monkeypatch):
    pynvml = ModuleType('pynvml')
    pynvml.nvmlInit = Mock()
    pynvml.nvmlShutdown = Mock()
    pynvml.nvmlDeviceGetHandleByIndex = Mock(return_value='handle')
    pynvml.nvmlDeviceGetUUID = Mock(return_value='GPU-uuid')
    monkeypatch.setitem(sys.modules, 'pynvml', pynvml)
    cls = _sub(device_type='cuda', vendor_name='nvidia', visible_device_id_to_physical_device_id=classmethod(lambda cls, device_id: 5))
    assert cls.get_device_uuid(2) == 'GPU-uuid'
    pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(5)
    pynvml.nvmlShutdown.assert_called_once_with()

def test_get_device_uuid_npu_with_env(monkeypatch):
    import os
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '3,4')
    assert _sub(device_type='npu', vendor_name='ascend').get_device_uuid(1) == 'NPU-4'

def test_get_device_uuid_npu_without_env(monkeypatch):
    import os
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    assert _sub(device_type='npu', vendor_name='ascend').get_device_uuid(1) == 'NPU-1'

def test_get_device_uuid_other_device_type():
    assert _sub(device_type='musa', vendor_name='mthreads').get_device_uuid(2) == 'musa-2'

def test_get_device_total_memory():
    fn = SimpleNamespace(get_device_properties=lambda device_id: SimpleNamespace(total_memory=4096))
    assert _sub(torch_device_fn=fn).get_device_total_memory(0) == 4096

@pytest.mark.parametrize('device_type,vendor,dist_backend,expected', [('txda', 'tsingmicro', 'tccl', True), ('cuda', 'nvidia', 'nccl', True), ('cuda', 'thead', 'nccl', True), ('cuda', 'iluvatar', 'nccl', True), ('musa', 'mthreads', 'mccl', True), ('musa', 'mthreads', 'other', False), ('cuda', 'amd', 'flagcx', True), ('cuda', 'amd', 'rccl', False), ('npu', 'ascend', 'hccl', False)])
def test_use_custom_op_collectives(device_type, vendor, dist_backend, expected):
    cls = _sub(device_type=device_type, vendor_name=vendor, dist_backend=dist_backend)
    assert cls.use_custom_op_collectives() is expected

def test_num_compute_units():
    fn = SimpleNamespace(get_device_properties=lambda device_id: SimpleNamespace(multi_processor_count=132))
    assert _sub(torch_device_fn=fn).num_compute_units(1) == 132

@pytest.mark.parametrize('device_type', ['npu', 'ptpu', 'txda'])
def test_get_device_capability_returns_none(device_type):
    assert _sub(device_type=device_type).get_device_capability(0) is None

def test_get_device_capability_musa(monkeypatch):
    monkeypatch.setattr(torch, 'musa', SimpleNamespace(get_device_capability=lambda device_id: (8, 0)), raising=False)
    cap = _sub(device_type='musa').get_device_capability(0)
    assert (cap.major, cap.minor) == (8, 0)

def test_get_device_capability_gcu_none(monkeypatch):
    monkeypatch.delattr(torch, 'gcu', raising=False)
    assert _sub(device_type='gcu').get_device_capability(0) is None

def test_get_device_capability_gcu(monkeypatch):
    monkeypatch.setattr(torch, 'gcu', SimpleNamespace(get_device_capability=lambda device_id: (7, 5)), raising=False)
    cap = _sub(device_type='gcu').get_device_capability(0)
    assert (cap.major, cap.minor) == (7, 5)

def test_get_device_capability_default_device_fn():
    fn = SimpleNamespace(get_device_capability=lambda device_id: (9, 0))
    cap = _sub(device_type='cuda', torch_device_fn=fn).get_device_capability(0)
    assert (cap.major, cap.minor) == (9, 0)

def test_check_runner_kv_caches_multi_layer():
    assert _sub().check_runner_kv_caches_multi_layer() is None

@pytest.mark.parametrize('device_type,vendor,cap90,family,expected', [('cuda', 'nvidia', True, False, True), ('cuda', 'nvidia', False, True, True), ('cuda', 'nvidia', False, False, False), ('musa', 'mthreads', True, True, False)])
def test_support_deep_gemm(device_type, vendor, cap90, family, expected):
    cls = _sub(device_type=device_type, vendor_name=vendor, is_device_capability=staticmethod(lambda cc: cap90), is_device_capability_family=staticmethod(lambda cc: family))
    assert cls.support_deep_gemm() is expected

class _FakeCompilationMode:
    NONE = 0
    VLLM_COMPILE = 3

class _FakeIrOpPriorityConfig:

    @classmethod
    def with_default(cls, default, **kwargs):
        return {'default': list(default), 'kwargs': kwargs}

def test_get_default_ir_op_priority_inductor(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.config.compilation', CompilationMode=_FakeCompilationMode)
    _install_fake_module(monkeypatch, 'vllm.config.kernel', IrOpPriorityConfig=_FakeIrOpPriorityConfig)
    cls = _sub(device_type='cuda', vendor_name='nvidia')
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(backend='inductor', mode=3))
    result = cls.get_default_ir_op_priority(cfg)
    assert result['default'] == ['native']

def test_get_default_ir_op_priority_npu_skips_vllm_c_probe(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.config.compilation', CompilationMode=_FakeCompilationMode)
    _install_fake_module(monkeypatch, 'vllm.config.kernel', IrOpPriorityConfig=_FakeIrOpPriorityConfig)
    probe = Mock(return_value=True)
    monkeypatch.setattr(platform_mod, '_vllm_c_available', probe)
    cls = _sub(device_type='npu', vendor_name='ascend')
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(backend='eager', mode=0))
    assert cls.get_default_ir_op_priority(cfg)['default'] == ['native']
    probe.assert_not_called()

def test_get_default_ir_op_priority_amd_registered(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.config.compilation', CompilationMode=_FakeCompilationMode)
    _install_fake_module(monkeypatch, 'vllm.config.kernel', IrOpPriorityConfig=_FakeIrOpPriorityConfig)
    cls = _sub(device_type='cuda', vendor_name='amd', _vllm_c_is_registered_provider=staticmethod(lambda : True))
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(backend='eager', mode=0))
    assert cls.get_default_ir_op_priority(cfg)['default'] == ['vllm_c', 'native']

def test_get_default_ir_op_priority_cuda_without_vllm_c(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.config.compilation', CompilationMode=_FakeCompilationMode)
    _install_fake_module(monkeypatch, 'vllm.config.kernel', IrOpPriorityConfig=_FakeIrOpPriorityConfig)
    monkeypatch.setattr(platform_mod, '_vllm_c_available', lambda : False)
    cls = _sub(device_type='cuda', vendor_name='nvidia')
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(backend='eager', mode=0))
    assert cls.get_default_ir_op_priority(cfg)['default'] == ['native']

def test_get_default_ir_op_priority_cuda_with_vllm_c(monkeypatch):
    _install_fake_module(monkeypatch, 'vllm.config.compilation', CompilationMode=_FakeCompilationMode)
    _install_fake_module(monkeypatch, 'vllm.config.kernel', IrOpPriorityConfig=_FakeIrOpPriorityConfig)
    monkeypatch.setattr(platform_mod, '_vllm_c_available', lambda : True)
    cls = _sub(device_type='cuda', vendor_name='nvidia')
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(backend='eager', mode=0))
    assert cls.get_default_ir_op_priority(cfg)['default'] == ['vllm_c', 'native']

def test__safe_device_type_uses_detected():
    info = SimpleNamespace(device_type='musa')
    assert platform_mod._safe_device_type(info, 'mthreads') == 'musa'

def test__safe_device_type_empty_without_vendor():
    info = SimpleNamespace(device_type='')
    assert platform_mod._safe_device_type(info, '') == ''

def test__safe_device_type_falls_back_to_map(monkeypatch):
    info = SimpleNamespace(device_type='')
    monkeypatch.setattr(platform_mod, 'get_device_type', lambda vendor: 'npu')
    assert platform_mod._safe_device_type(info, 'ascend') == 'npu'

def test__safe_device_type_unknown_vendor(monkeypatch):
    info = SimpleNamespace(device_type='')

    def boom(vendor):
        raise ValueError('unknown')
    monkeypatch.setattr(platform_mod, 'get_device_type', boom)
    assert platform_mod._safe_device_type(info, 'bogus') == ''

@pytest.mark.parametrize('found,expected', [(True, True), (False, False)])
def test__vllm_c_available(monkeypatch, found, expected):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object() if found else None)
    assert platform_mod._vllm_c_available() is expected

@pytest.mark.parametrize('exc', [ImportError, ValueError])
def test__vllm_c_available_handles_probe_errors(monkeypatch, exc):

    def boom(name):
        raise exc('probe')
    monkeypatch.setattr(importlib.util, 'find_spec', boom)
    assert platform_mod._vllm_c_available() is False

def _install_ir_modules(monkeypatch, *, ops, op_class=None):

    class IrOp:

        def __init__(self, impls=None):
            self.impls = impls if impls is not None else {'vllm_c'}
    if op_class is not None:
        IrOp = op_class

    @dataclass
    class IrOpPriorityConfig:
        rms_norm: object = None
        fused_add_rms_norm: object = None
    op_module = ModuleType('vllm.ir.op')
    op_module.IrOp = IrOp
    ops_module = ModuleType('vllm.ir.ops')
    for (key, value) in ops.items():
        setattr(ops_module, key, value)
    ir_pkg = ModuleType('vllm.ir')
    ir_pkg.__path__ = []
    ir_pkg.op = op_module
    ir_pkg.ops = ops_module
    _install_fake_module(monkeypatch, 'vllm.config.kernel', IrOpPriorityConfig=IrOpPriorityConfig)
    _install_fake_module(monkeypatch, 'vllm.ir', op=op_module, ops=ops_module)
    monkeypatch.setitem(sys.modules, 'vllm.ir.op', op_module)
    monkeypatch.setitem(sys.modules, 'vllm.ir.ops', ops_module)
    return IrOp

def test_vllm_c_registered_provider_true(monkeypatch):
    Good = type('Good', (), {'__init__': lambda self: setattr(self, 'impls', {'vllm_c'})})
    _install_ir_modules(monkeypatch, ops={'rms_norm': Good(), 'fused_add_rms_norm': Good()}, op_class=Good)
    assert PlatformFL._vllm_c_is_registered_provider() is True

def test_vllm_c_registered_provider_missing_impl(monkeypatch):
    Good = type('Good', (), {'__init__': lambda self: setattr(self, 'impls', {'vllm_c'})})
    Bad = type('Bad', (), {'__init__': lambda self: setattr(self, 'impls', {'native'})})
    _install_ir_modules(monkeypatch, ops={'rms_norm': Bad(), 'fused_add_rms_norm': Good()}, op_class=Good)
    assert PlatformFL._vllm_c_is_registered_provider() is False

def test_vllm_c_registered_provider_missing_op(monkeypatch):
    Good = type('Good', (), {'__init__': lambda self: setattr(self, 'impls', {'vllm_c'})})
    _install_ir_modules(monkeypatch, ops={'rms_norm': Good()}, op_class=Good)
    assert PlatformFL._vllm_c_is_registered_provider() is False

def test_vllm_c_registered_provider_wrong_type(monkeypatch):
    Good = type('Good', (), {'__init__': lambda self: setattr(self, 'impls', {'vllm_c'})})

    class NotAnOp:
        impls = {'vllm_c'}
    _install_ir_modules(monkeypatch, ops={'rms_norm': NotAnOp(), 'fused_add_rms_norm': Good()}, op_class=Good)
    assert PlatformFL._vllm_c_is_registered_provider() is False

def test_vllm_c_registered_provider_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'vllm.ir', None)
    assert PlatformFL._vllm_c_is_registered_provider() is False

@pytest.mark.parametrize('vendor,cap,expected', [('nvidia', (9, 0), True), ('nvidia', (8, 0), False), ('amd', (9, 0), False)])
def test_is_arch_support_pdl(vendor, cap, expected):
    fn = SimpleNamespace(current_device=lambda : 0, get_device_capability=lambda d: cap)
    cls = _sub(vendor_name=vendor, torch_device_fn=fn)
    assert cls.is_arch_support_pdl() is expected

def test_is_arch_support_pdl_probe_error():

    def boom(device_id):
        raise RuntimeError('probe')
    fn = SimpleNamespace(current_device=lambda : 0, get_device_capability=boom)
    cls = _sub(vendor_name='nvidia', torch_device_fn=fn)
    assert cls.is_arch_support_pdl() is False

def _pynvml_module(**overrides):
    module = ModuleType('pynvml')
    module.NVML_P2P_CAPS_INDEX_NVLINK = 0
    module.NVML_P2P_STATUS_OK = 0

    class NVMLError(Exception):
        pass
    module.NVMLError = NVMLError
    module.nvmlInit = Mock()
    module.nvmlDeviceGetHandleByIndex = Mock(side_effect=lambda i: f'H{i}')
    module.nvmlDeviceGetP2PStatus = Mock(return_value=0)
    for (key, value) in overrides.items():
        setattr(module, key, value)
    return module

def test_is_fully_connected_true(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pynvml', _pynvml_module())
    assert PlatformFL.is_fully_connected([0, 1]) is True

def test_is_fully_connected_not_ok(monkeypatch):
    module = _pynvml_module()
    module.nvmlDeviceGetP2PStatus = Mock(return_value=1)
    monkeypatch.setitem(sys.modules, 'pynvml', module)
    assert PlatformFL.is_fully_connected([0, 1]) is False

def test_is_fully_connected_nvml_error(monkeypatch):
    module = _pynvml_module()
    module.nvmlDeviceGetP2PStatus = Mock(side_effect=module.NVMLError('x'))
    monkeypatch.setitem(sys.modules, 'pynvml', module)
    assert PlatformFL.is_fully_connected([0, 1]) is False

def test_is_fully_connected_init_error(monkeypatch):
    module = _pynvml_module()
    module.nvmlInit = Mock(side_effect=RuntimeError('no nvml'))
    monkeypatch.setitem(sys.modules, 'pynvml', module)
    assert PlatformFL.is_fully_connected([0, 1]) is False

def test_manual_seed_all_uses_device_fn():
    fn = SimpleNamespace(manual_seed_all=Mock())
    _sub(torch_device_fn=fn).manual_seed_all(7)
    fn.manual_seed_all.assert_called_once_with(7)

def test_manual_seed_all_fallback_sets_each_generator(monkeypatch):
    generators = [Mock(), Mock()]
    fn = SimpleNamespace(device_count=lambda : 2, _get_or_create_default_generator=lambda device_id: generators[device_id])
    manual = Mock()
    monkeypatch.setattr(torch, 'manual_seed', manual)
    _sub(torch_device_fn=fn).manual_seed_all(11)
    manual.assert_called_once_with(11)
    for generator in generators:
        generator.manual_seed.assert_called_once_with(11)

def test_manual_seed_all_fallback_without_generator_api(monkeypatch):
    fn = SimpleNamespace()
    manual = Mock()
    monkeypatch.setattr(torch, 'manual_seed', manual)
    _sub(torch_device_fn=fn).manual_seed_all(3)
    manual.assert_called_once_with(3)

def test_is_integrated_gpu():
    assert PlatformFL.is_integrated_gpu() is False