"""Extra branch coverage for the dispatch package surface.

Covers ``vllm_fl.dispatch`` (CachedOp / warmup / call_op), the abstract
``ops.VLLMFLBackendBase`` bodies, ``types`` edge branches and the vendor
registration helpers in ``builtin_ops``.
"""
import sys
import types
import pytest
import torch
import vllm_fl.dispatch as dp
from vllm_fl.dispatch import builtin_ops as b
from vllm_fl.dispatch.ops import VLLMFLBackendBase
from vllm_fl.dispatch.types import BackendImplKind, OpImpl, match_token

class _Impl:

    def __init__(self, fn, impl_id='i.0'):
        self.fn = fn
        self.impl_id = impl_id

class _Mgr:

    def __init__(self, impls, epoch=0):
        self._impls = list(impls)
        self.policy_epoch = epoch
        self.resolve_calls = 0
        self.recorded = []
        self.calls = []
        self.failed = []
        self.initialized = False

    def _resolve_impl(self, name):
        idx = min(self.resolve_calls, len(self._impls) - 1)
        self.resolve_calls += 1
        return self._impls[idx]

    def _record_first_use(self, name, impl):
        self.recorded.append((name, impl))

    def call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return ('manager', name)

    def _mark_failed_impl(self, name, impl_id):
        self.failed.append((name, impl_id))

    def ensure_initialized(self):
        self.initialized = True

@pytest.fixture
def cached_env(monkeypatch):
    """Common patches so CachedOp.__call__ takes the fast resolved path."""
    holder = {'mgr': None}

    def _get_mgr():
        return holder['mgr']
    monkeypatch.setattr(dp, 'get_default_manager', _get_mgr)
    monkeypatch.setattr(dp, 'get_policy_epoch', lambda : 0)
    monkeypatch.setattr(dp, '_OP_FAST_PATH_ENABLED', True)
    monkeypatch.setattr(dp, 'is_dump_enabled', lambda : False)
    monkeypatch.setattr(dp, '_can_retry_failed_op', lambda name: True)
    return holder

class TestCachedOpBasics:

    def test_init_registers_in_weakset(self):
        op = dp.CachedOp('op_basic')
        assert op._op_name == 'op_basic'
        assert op._impl is None
        assert op._manager_id == -1
        assert op in dp._CACHED_OPS

    def test_warmup_sets_state(self):
        impl = _Impl(lambda x: x)
        mgr = _Mgr([impl], epoch=3)
        op = dp.CachedOp('op_warm')
        op.warmup(mgr)
        assert op._impl is impl
        assert op._manager_id == id(mgr)
        assert op._manager_epoch == 3
        assert mgr.recorded == [('op_warm', impl)]

    def test_warmup_default_manager(self, monkeypatch):
        impl = _Impl(lambda x: x)
        mgr = _Mgr([impl])
        monkeypatch.setattr(dp, 'get_default_manager', lambda : mgr)
        op = dp.CachedOp('op_warm_default')
        op.warmup()
        assert op._impl is impl

class TestCachedOpCall:

    def test_fast_path_resolves_once(self, cached_env):
        impl = _Impl(lambda x: x * 2)
        mgr = _Mgr([impl])
        cached_env['mgr'] = mgr
        op = dp.CachedOp('double')
        assert op(3) == 6
        assert op(4) == 8
        assert mgr.resolve_calls == 1
        assert mgr.calls == []

    def test_fast_path_disabled_uses_manager(self, cached_env, monkeypatch):
        monkeypatch.setattr(dp, '_OP_FAST_PATH_ENABLED', False)
        mgr = _Mgr([_Impl(lambda x: x)])
        cached_env['mgr'] = mgr
        op = dp.CachedOp('disabled')
        assert op(1) == ('manager', 'disabled')
        assert len(mgr.calls) == 1

    def test_dump_enabled_uses_manager(self, cached_env, monkeypatch):
        monkeypatch.setattr(dp, 'is_dump_enabled', lambda : True)
        mgr = _Mgr([_Impl(lambda x: x)])
        cached_env['mgr'] = mgr
        op = dp.CachedOp('dump')
        assert op(1) == ('manager', 'dump')
        assert len(mgr.calls) == 1

    def test_use_manager_call_shortcut(self, cached_env):
        mgr = _Mgr([_Impl(lambda x: x)])
        cached_env['mgr'] = mgr
        op = dp.CachedOp('shortcut')
        op.warmup(mgr)
        op._use_manager_call = True
        assert op(2) == ('manager', 'shortcut')
        assert len(mgr.calls) == 1

    def test_failure_retries_via_manager(self, cached_env):

        def boom(_x):
            raise RuntimeError('impl failed')
        impl = _Impl(boom, impl_id='bad.impl')
        mgr = _Mgr([impl])
        cached_env['mgr'] = mgr
        op = dp.CachedOp('retry')
        assert op(1) == ('manager', 'retry')
        assert op._impl is None
        assert op._use_manager_call is True
        assert mgr.failed == [('retry', 'bad.impl')]

    def test_failure_without_retry_reraises(self, cached_env, monkeypatch):

        def boom(_x):
            raise RuntimeError('fatal')
        monkeypatch.setattr(dp, '_can_retry_failed_op', lambda name: False)
        mgr = _Mgr([_Impl(boom)])
        cached_env['mgr'] = mgr
        op = dp.CachedOp('fatal')
        with pytest.raises(RuntimeError, match='fatal'):
            op(1)
        assert op._impl is None

class TestCachedOpAmdCompile:

    def test_ready_and_compiling_uses_warmed_impl(self, monkeypatch):
        sm = _Impl(lambda x: x + 1)
        op = dp.CachedOp('amd')
        op._amd_compile_ready = True
        op._impl = sm
        monkeypatch.setattr(torch.compiler, 'is_compiling', lambda : True)
        assert op(1) == 2

    def test_ready_and_compiling_without_impl_raises(self, monkeypatch):
        op = dp.CachedOp('amd_missing')
        op._amd_compile_ready = True
        monkeypatch.setattr(torch.compiler, 'is_compiling', lambda : True)
        with pytest.raises(RuntimeError, match='lost its warmed AMD implementation'):
            op(1)

    def test_ready_not_compiling_falls_through(self, cached_env, monkeypatch):
        sm = _Impl(lambda x: x + 1)
        mgr = _Mgr([sm])
        cached_env['mgr'] = mgr
        monkeypatch.setattr(torch.compiler, 'is_compiling', lambda : False)
        op = dp.CachedOp('amd_eager')
        op._amd_compile_ready = True
        op._impl = sm
        op._manager_id = id(mgr)
        op._manager_epoch = mgr.policy_epoch
        op._policy_epoch = 0
        assert op(2) == 3

class TestWarmupCachedOps:

    def test_warms_all_and_sets_amd_flag(self, monkeypatch):
        impl = _Impl(lambda x: x)
        mgr = _Mgr([impl])
        op1 = dp.CachedOp('w1')
        op2 = dp.CachedOp('w2')
        monkeypatch.setattr(dp, '_CACHED_OPS', {op1, op2})
        monkeypatch.setattr(dp, '_OP_FAST_PATH_ENABLED', True)
        monkeypatch.setattr(dp, 'is_dump_enabled', lambda : False)
        dp.warmup_cached_ops(mgr, amd_compile=True)
        assert mgr.initialized is True
        assert op1._amd_compile_ready is True
        assert op2._amd_compile_ready is True

    def test_runtime_error_is_swallowed(self, monkeypatch):

        class _BadMgr(_Mgr):

            def _resolve_impl(self, name):
                raise RuntimeError('optional op unavailable')
        op = dp.CachedOp('unavailable')
        monkeypatch.setattr(dp, '_CACHED_OPS', {op})
        dp.warmup_cached_ops(_BadMgr([]))
        assert op._impl is None

    def test_non_runtime_error_propagates(self, monkeypatch):

        class _BadMgr(_Mgr):

            def _resolve_impl(self, name):
                raise ValueError('unexpected')
        op = dp.CachedOp('unexpected')
        monkeypatch.setattr(dp, '_CACHED_OPS', {op})
        with pytest.raises(ValueError, match='unexpected'):
            dp.warmup_cached_ops(_BadMgr([]))

    def test_default_manager(self, monkeypatch):
        mgr = _Mgr([_Impl(lambda x: x)])
        monkeypatch.setattr(dp, '_CACHED_OPS', set())
        monkeypatch.setattr(dp, 'get_default_manager', lambda : mgr)
        dp.warmup_cached_ops()
        assert mgr.initialized is True

class TestCallResolveOp:

    def test_call_op_delegates(self, monkeypatch):

        class _CallMgr:

            def call(self, name, *args, **kwargs):
                return (name, args, kwargs)
        monkeypatch.setattr(dp, 'get_default_manager', lambda : _CallMgr())
        assert dp.call_op('mm', 1, kw=2) == ('mm', (1,), {'kw': 2})

    def test_resolve_op_delegates(self, monkeypatch):
        sentinel = object()

        class _ResolveMgr:

            def resolve(self, name):
                return (name, sentinel)
        monkeypatch.setattr(dp, 'get_default_manager', lambda : _ResolveMgr())
        assert dp.resolve_op('mm') == ('mm', sentinel)

class _ConcreteBackend(VLLMFLBackendBase):

    def is_available(self):
        return True

    @property
    def name(self):
        return 'concrete'

    def dynamic_per_token_quant_int8(self, x):
        return ('int8', x)

    def silu_and_mul(self, x):
        return x

    def gelu_and_mul(self, x, approximate='none'):
        return x

    def rms_norm(self, x, residual, weight, epsilon):
        return x

    def rotary_embedding(self, query, key, cos, sin, position_ids, rotary_interleaved=False, inplace=True):
        return (query, key)

    def attention_backend(self, use_mla=False, use_sparse=False):
        return 'path'

class TestBackendBase:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            VLLMFLBackendBase()

    def test_concrete_and_vendor(self):
        obj = _ConcreteBackend()
        assert obj.is_available() is True
        assert obj.name == 'concrete'
        assert obj.vendor is None

    def test_base_pass_bodies(self):
        obj = _ConcreteBackend()
        t = torch.zeros(1)
        assert VLLMFLBackendBase.is_available(obj) is None
        assert VLLMFLBackendBase.name.fget(obj) is None
        assert VLLMFLBackendBase.vendor.fget(obj) is None
        assert VLLMFLBackendBase.dynamic_per_token_quant_int8(obj, t) is None
        assert VLLMFLBackendBase.silu_and_mul(obj, t) is None
        assert VLLMFLBackendBase.gelu_and_mul(obj, t) is None
        assert VLLMFLBackendBase.gelu_and_mul(obj, t, 'tanh') is None
        assert VLLMFLBackendBase.rms_norm(obj, t, None, t, 1e-06) is None
        assert VLLMFLBackendBase.rotary_embedding(obj, t, t, t, t, t) is None
        assert VLLMFLBackendBase.attention_backend(obj) is None

class TestTypesEdges:

    def test_non_callable_availability_attr(self):

        def fn(x):
            return x
        fn._is_available = True
        impl = OpImpl(op_name='op', impl_id='default.flagos', kind=BackendImplKind.DEFAULT, fn=fn)
        assert impl.is_available() is True

    def test_vendor_token_requires_vendor_kind(self):
        impl = OpImpl(op_name='op', impl_id='default.flagos', kind=BackendImplKind.DEFAULT, fn=lambda x: x)
        assert match_token(impl, 'vendor:cuda') is False

    def test_impl_token_and_unknown_prefix(self):
        impl = OpImpl(op_name='op', impl_id='vendor.cuda', kind=BackendImplKind.VENDOR, fn=lambda x: x, vendor='cuda')
        assert match_token(impl, 'impl:vendor.cuda') is True
        assert match_token(impl, 'impl:other') is False
        assert match_token(impl, 'vendorx:cuda') is False

def _patch_platform(monkeypatch, obj):
    import vllm.platforms as vp
    monkeypatch.setattr(vp, 'current_platform', obj)

class TestCurrentVendorBackendDirs:

    def test_non_str_vendor(self, monkeypatch):
        _patch_platform(monkeypatch, types.SimpleNamespace(vendor_name=123))
        assert b._get_current_vendor_backend_dirs({'cuda'}) is None

    def test_empty_vendor(self, monkeypatch):
        _patch_platform(monkeypatch, types.SimpleNamespace(vendor_name=''))
        assert b._get_current_vendor_backend_dirs({'cuda'}) is None

    def test_resolves_via_find(self, monkeypatch):
        _patch_platform(monkeypatch, types.SimpleNamespace(vendor_name='cuda'))
        monkeypatch.setattr(b, '_find_vendor_backend_dir', lambda v, a: 'cuda')
        assert b._get_current_vendor_backend_dirs({'cuda'}) == 'cuda'

    def test_vendor_property_error_wrapped(self, monkeypatch):

        class _Raising:

            @property
            def vendor_name(self):
                raise RuntimeError('boom')
        _patch_platform(monkeypatch, _Raising())
        with pytest.raises(RuntimeError, match='Failed to detect current vendor'):
            b._get_current_vendor_backend_dirs({'cuda'})

class TestRegisterVendorBackends:

    def test_missing_dir_returns(self, monkeypatch):
        monkeypatch.setattr(b.os.path, 'isdir', lambda p: False)
        b._register_vendor_backends(object())

    def test_no_current_vendor_returns(self, monkeypatch):
        monkeypatch.setattr(b.os.path, 'isdir', lambda p: True)
        monkeypatch.setattr(b.os, 'listdir', lambda p: ['cuda'])
        monkeypatch.setattr(b, '_get_current_vendor_backend_dirs', lambda a: None)
        called = []
        monkeypatch.setattr(b.importlib, 'import_module', lambda *a, **k: called.append(a))
        b._register_vendor_backends(object())
        assert called == []

    def test_skips_other_vendors_and_missing_register_file(self, monkeypatch):
        monkeypatch.setattr(b.os.path, 'isdir', lambda p: True)
        monkeypatch.setattr(b.os, 'listdir', lambda p: ['metax', 'cuda'])
        monkeypatch.setattr(b, '_get_current_vendor_backend_dirs', lambda a: 'metax')
        monkeypatch.setattr(b.os.path, 'isfile', lambda p: 'metax' in p)
        imported = []

        def _imp(name, package=None):
            imported.append(name)
            return types.SimpleNamespace(register_builtins=lambda registry: None)
        monkeypatch.setattr(b.importlib, 'import_module', _imp)
        b._register_vendor_backends(object())
        assert imported == ['.backends.vendor.metax.register_ops']

    def test_module_without_register_builtins(self, monkeypatch):
        monkeypatch.setattr(b.os.path, 'isdir', lambda p: True)
        monkeypatch.setattr(b.os, 'listdir', lambda p: ['metax'])
        monkeypatch.setattr(b, '_get_current_vendor_backend_dirs', lambda a: 'metax')
        monkeypatch.setattr(b.os.path, 'isfile', lambda p: True)
        monkeypatch.setattr(b.importlib, 'import_module', lambda *a, **k: types.SimpleNamespace())
        b._register_vendor_backends(object())

    def test_import_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(b.os.path, 'isdir', lambda p: True)
        monkeypatch.setattr(b.os, 'listdir', lambda p: ['metax'])
        monkeypatch.setattr(b, '_get_current_vendor_backend_dirs', lambda a: 'metax')
        monkeypatch.setattr(b.os.path, 'isfile', lambda p: True)

        def _imp(*a, **k):
            raise ImportError('vendor runtime missing')
        monkeypatch.setattr(b.importlib, 'import_module', _imp)
        b._register_vendor_backends(object())

class TestRegisterBuiltins:

    def _inject(self, monkeypatch, fg_fn, ref_fn):
        fg = types.ModuleType('vllm_fl.dispatch.backends.flaggems.register_ops')
        fg.register_builtins = fg_fn
        ref = types.ModuleType('vllm_fl.dispatch.backends.reference.register_ops')
        ref.register_builtins = ref_fn
        monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch.backends.flaggems.register_ops', fg)
        monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch.backends.reference.register_ops', ref)

    def test_success_path_registers_everything(self, monkeypatch):
        calls = []
        self._inject(monkeypatch, lambda registry: calls.append('fg'), lambda registry: calls.append('ref'))
        monkeypatch.setattr(b, '_register_vendor_backends', lambda registry: calls.append('vendor'))
        import vllm_fl.dispatch.discovery as disc
        monkeypatch.setattr(disc, 'discover_plugins', lambda registry: 2)
        b.register_builtins(object())
        assert calls == ['fg', 'ref', 'vendor']

    def test_failures_are_isolated(self, monkeypatch):
        calls = []

        def _boom(registry):
            raise RuntimeError('register failed')
        self._inject(monkeypatch, _boom, _boom)
        monkeypatch.setattr(b, '_register_vendor_backends', lambda registry: None)
        import vllm_fl.dispatch.discovery as disc

        def _plugin_boom(registry):
            raise RuntimeError('plugin discovery failed')
        monkeypatch.setattr(disc, 'discover_plugins', _plugin_boom)
        b.register_builtins(object())

    def test_zero_plugins(self, monkeypatch):
        calls = []
        self._inject(monkeypatch, lambda registry: calls.append('fg'), lambda registry: calls.append('ref'))
        monkeypatch.setattr(b, '_register_vendor_backends', lambda registry: None)
        import vllm_fl.dispatch.discovery as disc
        monkeypatch.setattr(disc, 'discover_plugins', lambda registry: 0)
        b.register_builtins(object())
        assert calls == ['fg', 'ref']