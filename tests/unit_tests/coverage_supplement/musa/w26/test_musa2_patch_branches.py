"""CPU controls for the remaining MUSA compatibility-patch branches.

Every test loads the production ``patch.py`` by file and installs only the
fake vendor/vLLM modules the exercised patch touches.  No device, graph or
network access happens here.
"""
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
import pytest

def _repo_root() -> Path:
    import os
    candidates = [Path('/tmp/flsrc')]
    if os.environ.get('VLLM_FL_ROOT'):
        candidates.insert(0, Path(os.environ['VLLM_FL_ROOT']))
    for base in [*candidates, Path.cwd(), *Path(__file__).resolve().parents]:
        if not base.is_dir():
            continue
        if (base / 'vllm_fl/dispatch/backends/vendor/musa/patch.py').is_file():
            return base
    raise RuntimeError('cannot locate vllm_fl source tree')
ROOT = _repo_root()

def _module(name, **attrs):
    module = ModuleType(name)
    for (key, value) in attrs.items():
        setattr(module, key, value)
    return module

def _install(monkeypatch, mapping):
    for (name, obj) in mapping.items():
        monkeypatch.setitem(sys.modules, name, obj)
    for name in mapping:
        parts = name.split('.')
        for index in range(1, len(parts) + 1):
            full = '.'.join(parts[:index])
            if full not in sys.modules:
                monkeypatch.setitem(sys.modules, full, ModuleType(full))
            if index > 1:
                parent = sys.modules['.'.join(parts[:index - 1])]
                monkeypatch.setattr(parent, parts[index - 1], sys.modules[full], raising=False)

@pytest.fixture
def patch_mod():
    source = ROOT / 'vllm_fl/dispatch/backends/vendor/musa/patch.py'
    spec = importlib.util.spec_from_file_location('w26_musa_patch_branches', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def triton_cleanup():
    yield
    for name in [key for key in list(sys.modules) if key.startswith('triton.backends.mtgpu')]:
        sys.modules.pop(name, None)

def test_cuda_device_properties_import_failure_is_logged(patch_mod, monkeypatch, caplog):
    torch_module = _module('torch')
    torch_module.cuda = SimpleNamespace(get_device_capability=lambda : 'original')
    monkeypatch.setitem(sys.modules, 'torch', torch_module)
    monkeypatch.setitem(sys.modules, 'torch_musa', None)
    with caplog.at_level(logging.WARNING):
        patch_mod.patch_cuda_get_device_properties()
    assert 'Failed to patch cuda_get_device_properties' in caplog.text
    assert torch_module.cuda.get_device_capability() == 'original'

def test_cuda_stream_import_failure_is_logged(patch_mod, monkeypatch, caplog):
    torch_cuda = _module('torch.cuda', stream=lambda stream: stream, set_stream=lambda stream: None, current_stream=lambda device=None: device)
    torch_module = _module('torch')
    torch_module.cuda = torch_cuda
    monkeypatch.setitem(sys.modules, 'torch', torch_module)
    monkeypatch.setitem(sys.modules, 'torch.cuda', torch_cuda)
    monkeypatch.setitem(sys.modules, 'torch_musa', None)
    with caplog.at_level(logging.WARNING):
        patch_mod.patch_cuda_stream_for_musa()
    assert 'Failed to patch torch.cuda stream APIs' in caplog.text

def _inductor_modules():

    class TritonCompileResult:

        def __init__(self, kernel):
            self.kernel = kernel
            self.calls = 0

        def make_launcher(self):
            self.calls += 1
            return ('launcher', self.kernel)
    leaf = _module('torch._inductor.runtime.triton_heuristics', TritonCompileResult=TritonCompileResult)
    mapping = {'torch': _module('torch'), 'torch._inductor': _module('torch._inductor'), 'torch._inductor.runtime': _module('torch._inductor.runtime'), 'torch._inductor.runtime.triton_heuristics': leaf}
    return (mapping, leaf)

def test_inductor_metadata_without_fields_is_left_alone(patch_mod, monkeypatch):
    (mapping, leaf) = _inductor_modules()
    _install(monkeypatch, mapping)
    kernel = SimpleNamespace(metadata=SimpleNamespace(num_warps=4))
    patch_mod.patch_inductor_triton_for_musa()
    assert leaf.TritonCompileResult(kernel).make_launcher() == ('launcher', kernel)
    assert not hasattr(kernel.metadata, 'cluster_dims')

def test_inductor_invalid_namedtuple_fields_are_tolerated(patch_mod, monkeypatch):
    (mapping, leaf) = _inductor_modules()
    _install(monkeypatch, mapping)
    metadata = SimpleNamespace(_fields=('',), num_warps=4)
    kernel = SimpleNamespace(metadata=metadata)
    patch_mod.patch_inductor_triton_for_musa()
    assert leaf.TritonCompileResult(kernel).make_launcher() == ('launcher', kernel)
    assert kernel.metadata is metadata

def test_triton_alias_skips_when_mtgpu_already_present(patch_mod, monkeypatch, triton_cleanup):
    backends = _module('triton.backends')
    mtgpu = _module('triton.backends.mtgpu')
    backends.mtgpu = mtgpu
    testing = _module('triton.testing', do_bench=lambda *args: 'bench')
    triton = _module('triton', backends=backends, testing=testing)
    stub_key = 'triton.backends.mtgpu.musa_testing'
    monkeypatch.delitem(sys.modules, stub_key, raising=False)
    _install(monkeypatch, {'triton': triton, 'triton.backends': backends, 'triton.backends.mtgpu': mtgpu, 'triton.testing': testing})
    patch_mod.patch_triton_mtgpu_alias_for_musa()
    assert backends.mtgpu is mtgpu
    stub = sys.modules[stub_key]
    assert stub.do_bench is testing.do_bench
    assert mtgpu.musa_testing is stub

def test_triton_alias_keeps_an_existing_testing_stub(patch_mod, monkeypatch, triton_cleanup):
    backends = _module('triton.backends')
    triton = _module('triton', backends=backends)
    stub_key = 'triton.backends.mtgpu.musa_testing'
    existing = _module(stub_key)
    _install(monkeypatch, {'triton': triton, 'triton.backends': backends, stub_key: existing})
    patch_mod.patch_triton_mtgpu_alias_for_musa()
    assert sys.modules[stub_key] is existing
    assert not hasattr(existing, 'do_bench')

def test_triton_alias_outer_failure_is_logged(patch_mod, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, 'triton.backends', None)
    with caplog.at_level(logging.WARNING):
        patch_mod.patch_triton_mtgpu_alias_for_musa()
    assert 'patch_triton_mtgpu_alias_for_musa failed' in caplog.text

def _device_modules(monkeypatch, device_config_cls, device_fn, patched_flag=False):
    if patched_flag:
        device_config_cls._musa_post_init_patched = True
    torch_module = _module('torch', device=device_fn)
    device_mod = _module('vllm.config.device', DeviceConfig=device_config_cls)
    platforms = _module('vllm.platforms', current_platform=SimpleNamespace(device_type='musa'))
    _install(monkeypatch, {'torch': torch_module, 'torch_musa': _module('torch_musa'), 'vllm': _module('vllm'), 'vllm.config': _module('vllm.config'), 'vllm.config.device': device_mod, 'vllm.platforms': platforms})
    return device_config_cls

def _device_config_class(records):

    class DeviceConfig:

        def __post_init__(self):
            records.append(self.device)
            self.device_type = 'delegated'
    return DeviceConfig

def test_device_config_patch_is_idempotent(patch_mod, monkeypatch):
    records = []
    device_calls = []

    def device_fn(*args):
        device_calls.append(args)
        raise RuntimeError('unexpected device probe')
    cls = _device_config_class(records)
    original = cls.__post_init__
    _device_modules(monkeypatch, cls, device_fn, patched_flag=True)
    patch_mod.patch_device_config_for_musa()
    assert cls.__post_init__ is original
    assert device_calls == []

def test_device_config_raises_when_unindexed_device_is_unsupported(patch_mod, monkeypatch):
    records = []

    def device_fn(*args):
        raise RuntimeError('Device string must not be empty')
    cls = _device_config_class(records)
    _device_modules(monkeypatch, cls, device_fn)
    patch_mod.patch_device_config_for_musa()
    config = cls()
    config.device = 'musa'
    with pytest.raises(RuntimeError, match='torch_musa is too old'):
        config.__post_init__()

def test_device_config_delegates_for_non_musa_devices(patch_mod, monkeypatch):
    records = []

    def device_fn(*args):
        raise RuntimeError('Device string must not be empty')
    cls = _device_config_class(records)
    _device_modules(monkeypatch, cls, device_fn)
    patch_mod.patch_device_config_for_musa()
    config = cls()
    config.device = 'musa:1'
    config.__post_init__()
    assert records == ['musa:1']
    assert config.device_type == 'delegated'

def test_vision_rotary_import_failure_is_logged(patch_mod, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, 'torch', _module('torch'))
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.layers.rotary_embedding.common', None)
    with caplog.at_level(logging.WARNING):
        patch_mod.patch_musa_vision_rotary_layout()
    assert 'patch_musa_vision_rotary_layout failed' in caplog.text

def test_vision_rotary_second_application_is_a_noop(patch_mod, monkeypatch):

    class ApplyRotaryEmb:

        def forward_native(self, x, cos, sin):
            return 'original'
    common = _module('vllm.model_executor.layers.rotary_embedding.common', ApplyRotaryEmb=ApplyRotaryEmb)
    _install(monkeypatch, {'torch': _module('torch'), 'vllm': _module('vllm'), 'vllm.model_executor': _module('vllm.model_executor'), 'vllm.model_executor.layers': _module('vllm.model_executor.layers'), 'vllm.model_executor.layers.rotary_embedding': _module('vllm.model_executor.layers.rotary_embedding'), 'vllm.model_executor.layers.rotary_embedding.common': common})
    patch_mod.patch_musa_vision_rotary_layout()
    patched = ApplyRotaryEmb.forward_native
    assert patched._musa_rotary_layout_patched is True
    patch_mod.patch_musa_vision_rotary_layout()
    assert ApplyRotaryEmb.forward_native is patched

def _flag_gems_modules(monkeypatch, libentry, capturing=True):
    torch_module = _module('torch')
    torch_module.musa = SimpleNamespace(is_current_stream_capturing=MagicMock(return_value=capturing))
    _install(monkeypatch, {'torch': torch_module, 'torch_musa': _module('torch_musa'), 'flag_gems': _module('flag_gems'), 'flag_gems.utils': _module('flag_gems.utils'), 'flag_gems.utils.libentry': libentry})

def test_capture_safe_tuning_without_tuner_class_is_ignored(patch_mod, monkeypatch):
    _flag_gems_modules(monkeypatch, _module('flag_gems.utils.libentry'))
    patch_mod.patch_flag_gems_capture_safe_tuning()

def test_capture_safe_tuning_skips_an_already_patched_tuner(patch_mod, monkeypatch):

    class Tuner:
        _fl_capture_safe = True

        def run(self, *args, **kwargs):
            return 'original'
    original = Tuner.run
    _flag_gems_modules(monkeypatch, _module('flag_gems.utils.libentry', LibTuner=Tuner))
    patch_mod.patch_flag_gems_capture_safe_tuning()
    assert Tuner.run is original

def test_capture_safe_tuning_falls_back_when_cache_lookup_raises(patch_mod, monkeypatch):
    config = SimpleNamespace(all_kwargs=lambda : {'block': 8}, pre_hook=None)

    class BrokenCache:

        def get(self, key):
            raise RuntimeError('cache backend unavailable')

    class Tuner:
        _fl_capture_safe = False

        def __init__(self):
            self.arg_names = ['a']
            self.configs = [config]
            self.cache = BrokenCache()
            self.fn = SimpleNamespace(run=MagicMock(return_value='launched'))
            self.best_config = None
            self.nargs = 'stale'
            self.seen_tuned_metas = {'stale': True}

        def get_key(self, key_args):
            return 'key'

        def run(self, *args, **kwargs):
            raise AssertionError('the original autotuner must not run')
    _flag_gems_modules(monkeypatch, _module('flag_gems.utils.libentry', LibTuner=Tuner))
    patch_mod.patch_flag_gems_capture_safe_tuning()
    tuner = Tuner()
    assert tuner.run(1) == 'launched'
    assert tuner.best_config is config
    assert tuner.nargs is None
    assert tuner.seen_tuned_metas == {}
    tuner.fn.run.assert_called_once_with(1, block=8)
    assert Tuner._fl_capture_safe is True