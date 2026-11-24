import sys
from types import ModuleType, SimpleNamespace
import pytest
from vllm_fl.ops import custom_ops

def _stub_vendor_patch(monkeypatch, prefix, func_name, recorder):
    package = ModuleType(prefix)
    package.__path__ = []
    patch_module = ModuleType(f'{prefix}.patch')
    setattr(patch_module, func_name, recorder)
    monkeypatch.setitem(sys.modules, prefix, package)
    monkeypatch.setitem(sys.modules, f'{prefix}.patch', patch_module)

def _patch_utils(monkeypatch, **values):
    import vllm_fl.utils as fl_utils
    defaults = {'is_oot_enabled': lambda : True, 'get_oot_blacklist': lambda : [], 'get_oot_whitelist': lambda : None, 'use_flaggems_op': lambda op_name: True}
    defaults.update(values)
    for (name, value) in defaults.items():
        monkeypatch.setattr(fl_utils, name, value)

def _patch_platform(monkeypatch, device_type='musa', vendor_name='mthreads'):
    import vllm.platforms as platforms
    monkeypatch.setattr(platforms, 'current_platform', SimpleNamespace(device_type=device_type, vendor_name=vendor_name))

def _patch_platform_name(monkeypatch, name='musa'):
    import vllm_fl.dispatch.config.utils as cfg_utils
    monkeypatch.setattr(cfg_utils, 'get_platform_name', lambda : name)

class _FakeCustom:
    pass
_BASES = {}

def _register_recorders(monkeypatch):
    custom_calls = []
    pluggable_calls = []

    class DummyCustom:
        pass

    class DummyPluggable:
        pass
    DummyCustom.register_oot = staticmethod(lambda **kwargs: custom_calls.append(kwargs))
    DummyPluggable.register_oot = staticmethod(lambda **kwargs: pluggable_calls.append(kwargs))
    monkeypatch.setattr(custom_ops, 'CustomOp', DummyCustom)
    monkeypatch.setattr(custom_ops, 'PluggableLayer', DummyPluggable)
    _BASES['custom'] = DummyCustom
    _BASES['pluggable'] = DummyPluggable
    return (custom_calls, pluggable_calls)

def test_oot_ops_mapping_is_complete():
    assert set(custom_ops.OOT_OPS) == {'silu_and_mul', 'gelu_and_mul', 'rms_norm', 'rotary_embedding'}
    assert custom_ops.OOT_OPS['silu_and_mul'] == (custom_ops.SiluAndMulFL, 'SiluAndMul')
    assert custom_ops.OOT_OPS['gelu_and_mul'] == (custom_ops.GeluAndMulFL, 'GeluAndMul')
    assert custom_ops.OOT_OPS['rms_norm'] == (custom_ops.RMSNormFL, 'RMSNorm')
    assert custom_ops.OOT_OPS['rotary_embedding'] == (custom_ops.RotaryEmbeddingFL, 'RotaryEmbedding')

def test_register_oot_ops_disabled_patches_oracle(monkeypatch):
    _patch_utils(monkeypatch, is_oot_enabled=lambda : False)
    oracle_calls = []
    monkeypatch.setattr(custom_ops, '_patch_unquantized_moe_oracle', lambda : oracle_calls.append(True))
    custom_ops.register_oot_ops()
    assert oracle_calls == [True]

def _basic_register_setup(monkeypatch, oot_ops, **utils):
    monkeypatch.setattr(custom_ops, 'OOT_OPS', oot_ops)
    _patch_utils(monkeypatch, **utils)
    _patch_platform(monkeypatch)
    _patch_platform_name(monkeypatch)
    oracle_calls = []
    factory_calls = []
    monkeypatch.setattr(custom_ops, '_patch_unquantized_moe_oracle', lambda : oracle_calls.append(True))
    monkeypatch.setattr(custom_ops, '_patch_fused_moe_factory', lambda : factory_calls.append(True))
    (custom_calls, pluggable_calls) = _register_recorders(monkeypatch)
    return (oracle_calls, factory_calls, custom_calls, pluggable_calls)

def test_register_oot_ops_registers_both_base_classes(monkeypatch):
    (oracle_calls, factory_calls, custom_calls, pluggable_calls) = _basic_register_setup(monkeypatch, {'a': (_FakeCustom, 'A')})
    FakePluggable = type('FakePluggable', (_BASES['pluggable'],), {})
    monkeypatch.setattr(custom_ops, 'OOT_OPS', {'a': (_FakeCustom, 'A'), 'b': (FakePluggable, 'B')})
    custom_ops.register_oot_ops()
    assert custom_calls == [{'_decorated_op_cls': _FakeCustom, 'name': 'A'}]
    assert pluggable_calls == [{'_decorated_layer_cls': FakePluggable, 'name': 'B'}]
    assert oracle_calls == [True]
    assert factory_calls == [True]

def test_register_oot_ops_env_whitelist_wins_over_argument(monkeypatch):
    oot = {'a': (_FakeCustom, 'A'), 'b': (_FakeCustom, 'B')}
    (_, _, custom_calls, _) = _basic_register_setup(monkeypatch, oot, get_oot_whitelist=lambda : ['a'])
    custom_ops.register_oot_ops(whitelist=['b'])
    assert [call['name'] for call in custom_calls] == ['A']

def test_register_oot_ops_uses_argument_whitelist(monkeypatch):
    oot = {'a': (_FakeCustom, 'A'), 'b': (_FakeCustom, 'B')}
    (_, _, custom_calls, _) = _basic_register_setup(monkeypatch, oot)
    custom_ops.register_oot_ops(whitelist=['b'])
    assert [call['name'] for call in custom_calls] == ['B']

def test_register_oot_ops_ignores_unknown_names(monkeypatch):
    oot = {'a': (_FakeCustom, 'A')}
    (_, _, custom_calls, _) = _basic_register_setup(monkeypatch, oot, get_oot_whitelist=lambda : ['missing', 'a'])
    custom_ops.register_oot_ops()
    assert [call['name'] for call in custom_calls] == ['A']

def test_register_oot_ops_blacklist_filters_and_skips_factory(monkeypatch):
    oot = {'a': (_FakeCustom, 'A'), 'b': (_FakeCustom, 'B')}
    (oracle_calls, factory_calls, custom_calls, _) = _basic_register_setup(monkeypatch, oot, get_oot_blacklist=lambda : ['a', 'fused_moe'])
    custom_ops.register_oot_ops()
    assert [call['name'] for call in custom_calls] == ['B']
    assert oracle_calls == [True]
    assert factory_calls == []

def test_register_oot_ops_gates_unquantized_method(monkeypatch):
    oot = {'unquantized_fused_moe_method': (_FakeCustom, 'UFM'), 'a': (_FakeCustom, 'A')}
    (_, _, custom_calls, _) = _basic_register_setup(monkeypatch, oot, use_flaggems_op=lambda op_name: op_name != 'unquantized_fused_moe_method')
    custom_ops.register_oot_ops()
    assert [call['name'] for call in custom_calls] == ['A']

def test_register_oot_ops_applies_kunlunxin_patches(monkeypatch):
    called = []
    _stub_vendor_patch(monkeypatch, 'vllm_fl.dispatch.backends.vendor.kunlunxin', 'apply_kunlunxin_patches', lambda : called.append(True))
    oot = {'a': (_FakeCustom, 'A')}
    _basic_register_setup(monkeypatch, oot)
    _patch_platform_name(monkeypatch, 'kunlunxin')
    custom_ops.register_oot_ops()
    assert called == [True]

def test_patch_fused_moe_factory_is_idempotent_and_patches_cached_modules(monkeypatch):
    import vllm.model_executor.layers.fused_moe as package
    import vllm.model_executor.layers.fused_moe.layer as layer_module
    original = object()
    replacement = object()
    monkeypatch.setattr(layer_module, 'FusedMoEFactory', original, raising=False)
    monkeypatch.setattr(layer_module, 'FusedMoE', original, raising=False)
    monkeypatch.setattr(package, 'FusedMoEFactory', original, raising=False)
    monkeypatch.setattr(package, 'FusedMoE', original, raising=False)
    monkeypatch.setattr(custom_ops, 'FusedMoEFactoryFL', replacement)
    cached = ModuleType('_fl_cov_cached_model')
    cached.FusedMoEFactory = original
    cached.FusedMoE = original
    lazy = ModuleType('_fl_cov_lazy_model')

    def lazy_getattr(name):
        raise AssertionError(f'factory patch resolved lazy attribute {name}')
    lazy.__getattr__ = lazy_getattr
    monkeypatch.setitem(sys.modules, cached.__name__, cached)
    monkeypatch.setitem(sys.modules, lazy.__name__, lazy)
    custom_ops._patch_fused_moe_factory()
    custom_ops._patch_fused_moe_factory()
    assert layer_module.FusedMoEFactory is replacement
    assert package.FusedMoEFactory is replacement
    assert cached.FusedMoEFactory is replacement
    assert cached.FusedMoE is replacement