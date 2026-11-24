"""Residual branch coverage for ``vllm_fl.dispatch.manager``.

Complements w9b ``test_mgr_manager.py`` by targeting the remaining
availability-probe, cache-hit, empty per-op-order and pre-failed
fallback decisions.
"""
import os
from types import SimpleNamespace
import pytest
from vllm_fl.dispatch import manager as mgr
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, reset_global_policy, set_global_policy
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

@pytest.fixture(autouse=True)
def _reset_policy():
    reset_global_policy()
    yield
    reset_global_policy()

def _manager(registry=None):
    m = OpManager(registry=registry or OpRegistry())
    m._state.initialized = True
    m._state.init_pid = os.getpid()
    return m

def _impl(op, impl_id, fn, kind=BackendImplKind.DEFAULT, priority=None, vendor=None):
    return OpImpl(op_name=op, impl_id=impl_id, kind=kind, fn=fn, priority=BackendPriority.DEFAULT if priority is None else priority, vendor=vendor)

def _probe_raises():

    def _check():
        raise RuntimeError('availability probe exploded')
    return _check

def test_compute_candidates_all_unavailable_raises():
    reg = OpRegistry()

    def off(x):
        return x
    off._is_available = lambda : False
    reg.register_impl(_impl('op', 'default.off', off))
    m = _manager(reg)
    with pytest.raises(RuntimeError, match='No available implementation') as ei:
        m._compute_candidates('op', SelectionPolicy())
    assert 'default.off' in str(ei.value)

def test_matches_vendor_filters_vendor_without_name():
    m = _manager()
    fake = SimpleNamespace(kind=BackendImplKind.VENDOR, vendor=None)
    assert m._matches_vendor_filters(fake, SelectionPolicy()) is False

def test_get_failed_impls_unknown_op_is_empty():
    m = _manager()
    m._failed_impls['known'] = {'x'}
    assert m.get_failed_impls('missing') == {}

def test_resolve_candidates_second_call_hits_cache():
    reg = OpRegistry()
    reg.register_impl(_impl('op', 'default.impl', lambda : None))
    m = _manager(reg)
    first = m.resolve_candidates('op')
    second = m.resolve_candidates('op')
    assert first is second
    assert len(m._candidates_cache) == 1

def test_empty_per_op_tuple_falls_back_to_default_order():
    reg = OpRegistry()
    reg.register_impl(_impl('op', 'default.impl', lambda : None))
    m = _manager(reg)
    set_global_policy(SelectionPolicy(per_op_order=(('op', ()),)))
    assert m._resolve_impl('op').impl_id == 'default.impl'
    assert [c.impl_id for c in m.resolve_candidates('op')] == ['default.impl']

def test_call_skips_previously_failed_primary():
    reg = OpRegistry()
    calls = []

    def primary():
        calls.append('primary')
        return 'primary'
    reg.register_impl(_impl('op', 'default.primary', primary, priority=200))
    reg.register_impl(_impl('op', 'reference.second', lambda : 'second', kind=BackendImplKind.REFERENCE, priority=100))
    m = _manager(reg)
    m._failed_impls['op'] = {'default.primary'}
    assert m.call('op') == 'second'
    assert calls == []
    assert m._called_ops['op'] == 'reference.second'

def test_clear_failed_impls_allows_primary_retry():
    reg = OpRegistry()
    state = {'calls': 0}

    def flaky():
        state['calls'] += 1
        if state['calls'] == 1:
            raise RuntimeError('first attempt fails')
        return 'recovered'
    reg.register_impl(_impl('op', 'default.flaky', flaky, priority=200))
    reg.register_impl(_impl('op', 'reference.fallback', lambda : 'fallback', kind=BackendImplKind.REFERENCE, priority=100))
    m = _manager(reg)
    assert m.call('op') == 'fallback'
    assert m.get_failed_impls('op') == {'op': {'default.flaky'}}
    m.clear_failed_impls('op')
    assert m.call('op') == 'recovered'
    assert state['calls'] == 2

def test_resolve_impl_debug_branch_with_vendor(monkeypatch):
    monkeypatch.setattr(mgr, '_DISPATCH_DEBUG', True)
    reg = OpRegistry()
    reg.register_impl(_impl('default_op', 'default.impl', lambda : None))
    reg.register_impl(_impl('vendor_op', 'vendor.cuda', lambda : None, kind=BackendImplKind.VENDOR, priority=BackendPriority.VENDOR, vendor='CUDA'))
    m = _manager(reg)
    assert m._resolve_impl('default_op').impl_id == 'default.impl'
    set_global_policy(SelectionPolicy(prefer='vendor'))
    assert m._resolve_impl('vendor_op').impl_id == 'vendor.cuda'