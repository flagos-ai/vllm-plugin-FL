from types import SimpleNamespace
import pytest
import vllm_fl.utils as fl_utils
from vllm_fl.quantization.w8a8 import moe as moe_adapter
from vllm_fl.quantization.w8a8 import moe_experts

def _install(monkeypatch, upstream_selector):
    oracle = SimpleNamespace(Int8MoeBackend=SimpleNamespace(TRITON='triton'), select_int8_moe_backend=upstream_selector)
    scheme = SimpleNamespace(select_int8_moe_backend=upstream_selector, make_int8_moe_quant_config=lambda **kwargs: kwargs)
    modules = {moe_adapter._ORACLE_MODULE: oracle, moe_adapter._SCHEME_MODULE: scheme}
    monkeypatch.setattr(moe_adapter, 'import_module', lambda name: modules[name])
    assert moe_adapter.install_fl_w8a8_moe_selector() is True
    assert oracle.select_int8_moe_backend is scheme.select_int8_moe_backend
    assert getattr(oracle.select_int8_moe_backend, moe_adapter._ADAPTER_MARKER) is True
    return (oracle, scheme)

def _patch_policy(monkeypatch, *, oot=True, cuda=False, enabled=True, flaggems=True):
    import vllm.platforms as platforms
    monkeypatch.setattr(platforms, 'current_platform', SimpleNamespace(is_cuda=lambda : cuda, is_out_of_tree=lambda : oot))
    monkeypatch.setattr(fl_utils, 'is_oot_enabled', lambda : enabled)
    monkeypatch.setattr(fl_utils, 'use_flaggems_op', lambda op_name: flaggems)

def _config(*, lora=False, batched=False):
    return SimpleNamespace(is_lora_enabled=lora, moe_parallel_config=SimpleNamespace(use_batched_activation_format=batched))

def test_install_is_idempotent(monkeypatch):
    (oracle, scheme) = _install(monkeypatch, lambda *a, **k: 'upstream')
    wrapper = oracle.select_int8_moe_backend
    assert moe_adapter.install_fl_w8a8_moe_selector() is True
    assert oracle.select_int8_moe_backend is wrapper
    assert scheme.select_int8_moe_backend is wrapper

def test_selector_prefers_flaggems_on_oot(monkeypatch):
    calls = []

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return 'upstream'
    (oracle, _) = _install(monkeypatch, upstream)
    _patch_policy(monkeypatch, oot=True, cuda=False, enabled=True, flaggems=True)
    (backend, experts) = oracle.select_int8_moe_backend(_config())
    assert backend == 'triton'
    assert experts is moe_experts.FlagGemsW8A8Experts
    assert calls == []

def test_selector_prefers_flaggems_on_cuda(monkeypatch):
    (oracle, _) = _install(monkeypatch, lambda *a, **k: 'upstream')
    _patch_policy(monkeypatch, oot=False, cuda=True, enabled=True, flaggems=True)
    (backend, experts) = oracle.select_int8_moe_backend(_config())
    assert backend == 'triton'
    assert experts is moe_experts.FlagGemsW8A8Experts

def test_selector_rejects_flaggems_lora_and_batched(monkeypatch):
    (oracle, _) = _install(monkeypatch, lambda *a, **k: 'upstream')
    _patch_policy(monkeypatch, oot=True, cuda=False, enabled=True, flaggems=True)
    with pytest.raises(NotImplementedError, match='does not support LoRA'):
        oracle.select_int8_moe_backend(_config(lora=True))
    with pytest.raises(ValueError, match='standard activation'):
        oracle.select_int8_moe_backend(_config(batched=True))

def test_selector_non_canonical_bypasses_policy(monkeypatch):
    (oracle, _) = _install(monkeypatch, lambda *a, **k: 'upstream')
    _patch_policy(monkeypatch, oot=True, cuda=True, enabled=True, flaggems=True)
    monkeypatch.setattr(fl_utils, 'use_flaggems_op', lambda op_name: pytest.fail('noncanonical routing must not consult policy'))
    assert oracle.select_int8_moe_backend(_config(), weight_key='noncanonical', activation_key='noncanonical') == 'upstream'