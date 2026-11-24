"""Coverage tests for worker_symbols_032 / fla relocation / flagtree target info."""
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
import torch
from vllm_fl.patches import flagtree_target_info as fti
from vllm_fl.patches import fla_module_relocation as fla
from vllm_fl.patches import worker_symbols_032 as ws

def _cfg(*, enable_sp=True, tp=2, inductive=False, splitting=(), num=4):
    return (SimpleNamespace(compilation_config=SimpleNamespace(pass_config=SimpleNamespace(enable_sp=enable_sp), use_inductor_graph_partition=inductive, splitting_ops=splitting), parallel_config=SimpleNamespace(tensor_parallel_size=tp)), num)

def test_is_residual_scattered_for_sp():
    (cfg, n) = _cfg(enable_sp=False)
    assert ws._is_residual_scattered_for_sp(cfg, n) is False
    (cfg, n) = _cfg(tp=1)
    assert ws._is_residual_scattered_for_sp(cfg, n) is False
    (cfg, n) = _cfg(splitting=('op',), inductive=False)
    with pytest.raises(AssertionError, match='full-graph'):
        ws._is_residual_scattered_for_sp(cfg, n)
    (cfg, n) = _cfg(inductive=True, num=3)
    with pytest.raises(AssertionError):
        ws._is_residual_scattered_for_sp(cfg, n)
    (cfg, n) = _cfg(inductive=True, num=4)
    assert ws._is_residual_scattered_for_sp(cfg, n) is True
    (cfg, n) = _cfg(splitting=(), num=4)
    assert ws._is_residual_scattered_for_sp(cfg, n) is True

def test_supports_xdrope_and_rotary_helper():

    class Model:

        def get_xdrope_input_positions(self):
            return [1]
    assert ws._supports_xdrope(Model()) is True
    assert ws._supports_xdrope(object()) is False
    assert isinstance(ws._SupportsXDRoPE, type)
    assert isinstance(ws._XDRotaryEmbedding(), object)
    out = np.zeros((2, 6), dtype=np.int64)
    ws._XDRotaryEmbedding.get_next_input_positions_tensor(out, 1, 3, 2)
    assert out[0, 1:3].tolist() == [3, 4]
    assert ws._dtype_size(torch.float32) == 4
    assert ws._dtype_size(torch.float16) == 2

def test_reshape_contiguous_view():
    shape = (2, 3, 4)
    raw = torch.zeros(np.prod(shape) * 4, dtype=torch.uint8)
    spec = SimpleNamespace(dtype=torch.float32, page_size_padded=None)
    view = ws._reshape_attention_kv_cache(raw, spec, shape, (0, 1, 2), num_blocks=2, packing=None)
    assert tuple(view.shape) == shape
    assert view.dtype == torch.float32

def test_reshape_packing_view():
    shape = (2, 3, 4)
    page_bytes = np.prod(shape[1:]) * 4
    block_stride = page_bytes + 8
    raw = torch.zeros(2 * block_stride, dtype=torch.uint8)
    spec = SimpleNamespace(dtype=torch.float32)
    view = ws._reshape_attention_kv_cache(raw, spec, shape, (0, 1, 2), num_blocks=2, packing=(0, block_stride))
    assert tuple(view.shape) == shape

def test_reshape_page_aligned_blocks():
    num_blocks = 4
    shape = (2, num_blocks, 3, 4)
    raw = torch.zeros(np.prod(shape) * 4, dtype=torch.uint8)
    spec = SimpleNamespace(dtype=torch.float32, page_size_padded=None)
    view = ws._reshape_attention_kv_cache(raw, spec, shape, (0, 1, 2, 3), num_blocks=num_blocks, packing=None, page_aligned_blocks=True)
    assert tuple(view.shape) == shape
    with pytest.raises(AssertionError, match='Page-aligned'):
        ws._reshape_attention_kv_cache(raw, spec, shape, (1, 0, 2, 3), num_blocks=num_blocks, packing=None, page_aligned_blocks=True)

def test_routed_experts_carriers():
    lists = ws._RoutedExpertsLists(np.zeros((2, 3), dtype=np.int8), np.zeros((2,), dtype=np.int32))
    assert lists.routing_data.shape == (2, 3)
    tensors = ws._RoutedExpertsTensors(torch.zeros((2, 3)), torch.zeros((2,)))
    assert tensors.to_cpu_nonblocking() is tensors
    lists2 = tensors.tolists()
    assert isinstance(lists2, ws._RoutedExpertsLists)
    assert lists2.routing_data.shape == (2, 3)

def test_set_missing_and_apply(monkeypatch):
    mod = SimpleNamespace(present=1)
    ws._set_missing(mod, 'present', 2)
    ws._set_missing(mod, 'absent', 3)
    assert mod.present == 1 and mod.absent == 3
    layers = ModuleType('vllm.model_executor.layers')
    layers.rotary_embedding = SimpleNamespace()
    models = ModuleType('vllm.model_executor.models')
    models.interfaces = SimpleNamespace()
    v1 = ModuleType('vllm.v1')
    v1.outputs = SimpleNamespace()
    worker = ModuleType('vllm.v1.worker')
    worker.utils = SimpleNamespace()
    gpu = ModuleType('vllm.v1.worker.gpu')
    gpu.attn_utils = SimpleNamespace()
    for (name, mod_) in {'vllm.model_executor.layers': layers, 'vllm.model_executor.models': models, 'vllm.v1': v1, 'vllm.v1.worker': worker, 'vllm.v1.worker.gpu': gpu}.items():
        monkeypatch.setitem(sys.modules, name, mod_)
    ws.apply_worker_symbols_032.cache_clear()
    assert ws.apply_worker_symbols_032() is True
    assert worker.utils.is_residual_scattered_for_sp is ws._is_residual_scattered_for_sp
    assert layers.rotary_embedding.XDRotaryEmbedding is ws._XDRotaryEmbedding
    assert models.interfaces.SupportsXDRoPE is ws._SupportsXDRoPE
    assert models.interfaces.supports_xdrope is ws._supports_xdrope
    assert gpu.attn_utils._reshape_attention_kv_cache is ws._reshape_attention_kv_cache
    assert v1.outputs.RoutedExpertsLists is ws._RoutedExpertsLists
    assert v1.outputs.RoutedExpertsTensors is ws._RoutedExpertsTensors
    assert ws.apply_worker_symbols_032() is True
    ws.apply_worker_symbols_032.cache_clear()

def test_module_exists_swallows_errors(monkeypatch):

    def boom(name):
        raise ValueError('broken parent')
    monkeypatch.setattr(fla.importlib.util, 'find_spec', boom)
    assert fla._module_exists('anything') is False

def test_install_noop_cases(monkeypatch):
    monkeypatch.setattr(fla, '_module_exists', lambda name: True)
    assert fla.install() is False
    monkeypatch.setattr(fla, '_module_exists', lambda name: False)
    assert fla.install() is False

def test_install_aliases_prefix(monkeypatch):
    new_prefix = fla._NEW_PREFIX
    old_prefix = fla._OLD_PREFIX
    new_pkg = ModuleType(new_prefix)
    new_pkg.__path__ = []
    ops_pkg = ModuleType(new_prefix + '.ops')
    ops_pkg.__path__ = ['fake']
    kernel = ModuleType(new_prefix + '.ops.kernel')
    layers = ModuleType('vllm.model_executor.layers')
    present = {new_prefix, new_prefix + '.ops', new_prefix + '.ops.kernel'}
    monkeypatch.setattr(fla, '_module_exists', lambda name: name in present or name in sys.modules)
    monkeypatch.setattr(fla.importlib, 'import_module', lambda name: {new_prefix: new_pkg, new_prefix + '.ops': ops_pkg, new_prefix + '.ops.kernel': kernel, 'vllm.model_executor.layers': layers}[name])
    monkeypatch.setattr(fla.pkgutil, 'iter_modules', lambda path: [SimpleNamespace(name='kernel')])
    added = [old_prefix, old_prefix + '.ops', old_prefix + '.ops.kernel']
    try:
        assert fla.install() is True
        assert sys.modules[old_prefix] is new_pkg
        assert sys.modules[old_prefix + '.ops'] is ops_pkg
        assert sys.modules[old_prefix + '.ops.kernel'] is kernel
        assert layers.fla is new_pkg
        assert fla.install() is False
    finally:
        for name in added:
            sys.modules.pop(name, None)

def test_current_arch():
    assert fti._current_arch(SimpleNamespace(current_target=lambda : SimpleNamespace(arch='gfx942'))) == 'gfx942'

    def raises():
        raise RuntimeError('no target')
    assert fti._current_arch(SimpleNamespace(current_target=raises)) == ''
    assert fti._current_arch(SimpleNamespace(current_target=lambda : None)) == ''
    assert fti._current_arch(SimpleNamespace(current_target=lambda : SimpleNamespace(arch=1))) == ''

def test_make_predicate():
    ti = SimpleNamespace(is_hip=lambda : False)
    pred = fti._make_predicate(ti, 'is_x', ('gfx942',))
    assert pred() is False and pred.__name__ == 'is_x'
    ti = SimpleNamespace(is_hip=lambda : True, current_target=lambda : SimpleNamespace(arch='gfx942'))
    assert fti._make_predicate(ti, 'is_x', ('gfx942',))() is True
    assert fti._make_predicate(ti, 'is_x', ('gfx90a',))() is False
    wrapped = SimpleNamespace(is_hip=lambda : True, current_target=lambda : SimpleNamespace(arch='gfx942'), constexpr_function=lambda f: 'wrapped')
    assert fti._make_predicate(wrapped, 'is_x', ('gfx942',)) == 'wrapped'

    def bad(_fn):
        raise RuntimeError('no decorator')
    raiser = SimpleNamespace(is_hip=lambda : True, current_target=lambda : SimpleNamespace(arch='gfx942'), constexpr_function=bad)
    assert callable(fti._make_predicate(raiser, 'is_x', ('gfx942',)))

def test_apply_flagtree_missing_triton(monkeypatch):
    fti.apply_flagtree_target_info.cache_clear()
    monkeypatch.setitem(sys.modules, 'triton.language.target_info', None)
    assert fti.apply_flagtree_target_info() is False
    fti.apply_flagtree_target_info.cache_clear()

def _fake_target_info(**extra):
    ti = ModuleType('triton.language.target_info')
    ti.is_hip = lambda : True
    ti.current_target = lambda : SimpleNamespace(arch='gfx942')
    for (key, value) in extra.items():
        setattr(ti, key, value)
    return ti

def test_apply_flagtree_noop_when_present(monkeypatch):
    fti.apply_flagtree_target_info.cache_clear()
    existing = {name: lambda : True for name in fti._ARCH_PREDICATES}
    fake = _fake_target_info(**existing)
    monkeypatch.setitem(sys.modules, 'triton.language.target_info', fake)
    assert fti.apply_flagtree_target_info() is True
    fti.apply_flagtree_target_info.cache_clear()