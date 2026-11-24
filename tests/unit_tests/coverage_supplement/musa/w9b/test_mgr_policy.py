"""Extra coverage for vllm_fl/dispatch/policy.py env/config parsing."""
import os
import tempfile
import pytest
from vllm_fl.dispatch import policy as pol
from vllm_fl.dispatch import config as cfg_mod
from vllm_fl.dispatch.policy import PREFER_DEFAULT, PREFER_REFERENCE, PREFER_VENDOR, PolicyManager, SelectionPolicy, get_policy, get_policy_epoch, policy_context, policy_from_config, policy_from_env, reset_global_policy, set_global_policy, with_allowed_vendors, with_denied_vendors, with_preference

@pytest.fixture(autouse=True)
def _clean_policy(monkeypatch):
    for key in ('VLLM_FL_CONFIG', 'VLLM_FL_PREFER', 'VLLM_FL_STRICT', 'VLLM_FL_DENY_VENDORS', 'VLLM_FL_ALLOW_VENDORS', 'VLLM_FL_PER_OP'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pol, 'get_op_config', lambda : None)
    monkeypatch.setattr(cfg_mod, 'get_config_path', lambda : None)
    reset_global_policy()
    yield
    reset_global_policy()

def _write(text):
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False, encoding='utf-8')
    f.write(text)
    f.close()
    return f.name

def test_policy_dict_and_fingerprint():
    p = SelectionPolicy.from_dict(prefer='VENDOR', strict=True, per_op_order={'b': ['flagos'], 'a': ['vendor', 'reference']}, deny_vendors={'AMD'}, allow_vendors={'CUDA'})
    assert p.prefer == 'vendor'
    assert p.per_op_order == (('a', ('vendor', 'reference')), ('b', ('flagos',)))
    assert p.per_op_order_dict == {'a': ['vendor', 'reference'], 'b': ['flagos']}
    fp = p.fingerprint()
    assert 'allow=CUDA' in fp and 'deny=AMD' in fp and ('per=a=' in fp)
    assert hash(p) == hash(SelectionPolicy.from_dict(prefer='VENDOR', strict=True, per_op_order={'b': ['flagos'], 'a': ['vendor', 'reference']}, deny_vendors={'AMD'}, allow_vendors={'CUDA'}))
    assert SelectionPolicy().fingerprint() == 'prefer=flagos;st=0'

def test_policy_manager_reinit_is_noop():
    mgr = PolicyManager.get_instance()
    epoch = mgr.get_policy_epoch()
    mgr.__init__()
    assert mgr.get_policy_epoch() == epoch

def test_set_global_policy_returns_env_when_unset():
    mgr = PolicyManager.get_instance()
    mgr._global_policy = None
    old = set_global_policy(SelectionPolicy(prefer=PREFER_REFERENCE))
    assert old.prefer == PREFER_DEFAULT
    assert get_policy().prefer == PREFER_REFERENCE

def test_parse_helpers():
    P = PolicyManager
    assert P._parse_csv_set('') == set()
    assert P._parse_csv_set(' a, b ,, c ') == {'a', 'b', 'c'}
    assert P._parse_per_op('') == {}
    assert P._parse_per_op('junk;x=a|b;;y=c') == {'x': ['a', 'b'], 'y': ['c']}
    assert P._parse_per_op('z=') == {}
    assert P._parse_op_config({'silu': 'VENDOR', 'rms': 'reference'}) == {'silu': ['vendor'], 'rms': ['reference']}
    with pytest.raises(ValueError, match='Unsupported backend'):
        P._parse_op_config({'silu': 'bogus'})

def test_policy_from_config_errors():
    with pytest.raises(FileNotFoundError):
        pol.policy_from_config('/definitely/not/here.yaml')
    path = _write('key: [unclosed\n')
    try:
        with pytest.raises(ValueError, match='Failed to load config'):
            pol.policy_from_config(path)
    finally:
        os.unlink(path)

def test_policy_from_config_fields():
    text = '\nprefer: bogus\nstrict: 1\ndeny_vendors: "AMD, Intel"\nallow_vendors:\n  - CUDA\nop_backends:\n  silu:\n    - vendor:cuda\n    - flagos\n  rms: "vendor|flagos"\n  bad: 3\n'
    path = _write(text)
    try:
        p = pol.policy_from_config(path)
    finally:
        os.unlink(path)
    assert p.prefer == PREFER_DEFAULT
    assert p.strict is True
    assert p.deny_vendors == frozenset({'AMD', 'Intel'})
    assert p.allow_vendors == frozenset({'CUDA'})
    assert p.get_per_op_order('silu') == ['vendor:cuda', 'flagos']
    assert p.get_per_op_order('rms') == ['vendor', 'flagos']
    assert p.get_per_op_order('bad') is None

def test_env_config_file_overrides(monkeypatch):
    path = _write('prefer: vendor\nstrict: true\n')
    monkeypatch.setenv('VLLM_FL_CONFIG', path)
    try:
        p = policy_from_env()
    finally:
        os.unlink(path)
    assert p.prefer == PREFER_VENDOR and p.strict is True

def test_env_invalid_values_and_bad_platform_config(monkeypatch):
    monkeypatch.setattr(cfg_mod, 'get_config_path', lambda : '/missing/platform.yaml')
    monkeypatch.setenv('VLLM_FL_PREFER', 'nonsense')
    monkeypatch.setenv('VLLM_FL_STRICT', '2')
    p = policy_from_env()
    assert p.prefer == PREFER_DEFAULT and p.strict is False

def test_env_op_config_precedes_env_per_op(monkeypatch):
    monkeypatch.setattr(pol, 'get_op_config', lambda : {'silu': 'vendor'}, raising=False)
    monkeypatch.setenv('VLLM_FL_PER_OP', 'rms=reference')
    p = policy_from_env()
    assert p.get_per_op_order('silu') == ['vendor']
    assert p.get_per_op_order('rms') is None
    monkeypatch.setattr(pol, 'get_op_config', lambda : {'silu': 'bogus'}, raising=False)
    with pytest.raises(ValueError, match='Unsupported backend'):
        policy_from_env()

def test_context_helpers_merge_state():
    set_global_policy(SelectionPolicy(prefer=PREFER_DEFAULT, deny_vendors=frozenset({'AMD'}), allow_vendors=frozenset({'CUDA'}), per_op_order=(('silu', ('vendor',)),)))
    with with_preference('reference'):
        p = get_policy()
        assert p.prefer == PREFER_REFERENCE
        assert p.deny_vendors == frozenset({'AMD'})
        assert p.allow_vendors == frozenset({'CUDA'})
        assert p.per_op_order_dict == {'silu': ['vendor']}
    with with_allowed_vendors('NVIDIA'):
        assert get_policy().allow_vendors == frozenset({'NVIDIA'})
    with with_denied_vendors('Intel', 'AMD'):
        denied = get_policy().deny_vendors
        assert denied == frozenset({'AMD', 'Intel'})

def test_policy_context_exit_without_enter():
    ctx = policy_context(SelectionPolicy())
    ctx.__exit__(None, None, None)

def test_epoch_helpers():
    before = get_policy_epoch()
    pol.bump_policy_epoch()
    assert get_policy_epoch() == before + 1

def test_platform_config_loader_used_by_env(monkeypatch):
    path = _write('prefer: vendor\ndeny_vendors: AMD\nallow_vendors: AMD\nop_backends:\n  rms: [reference]\n')
    monkeypatch.setattr(cfg_mod, 'get_config_path', lambda : path)
    try:
        p = policy_from_env()
    finally:
        os.unlink(path)
    assert p.deny_vendors == frozenset({'AMD'})
    assert p.allow_vendors == frozenset({'AMD'})
    assert p.get_per_op_order('rms') == ['reference']