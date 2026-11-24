"""Extra MUSA compatibility-patch controls (no device, no network)."""
import importlib.util
import sys
import types
from collections import namedtuple
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
        candidate = base / 'vllm_fl/dispatch/backends/vendor/musa/patch.py'
        if candidate.is_file():
            return base
    raise RuntimeError('cannot locate vllm_fl source tree')
ROOT = _repo_root()

def _load(name: str):
    source = ROOT / 'vllm_fl/dispatch/backends/vendor/musa/patch.py'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def triton_cleanup():
    yield
    for name in [key for key in list(sys.modules) if key.startswith('triton.backends.mtgpu')]:
        sys.modules.pop(name, None)

@pytest.fixture
def patch_mod():
    return _load('w6b_musa_patch')

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
                setattr(parent, parts[index - 1], sys.modules[full])

def _module(name, **attrs):
    module = ModuleType(name)
    for (key, value) in attrs.items():
        setattr(module, key, value)
    return module

def test_topk_topp_sampler_rebinds_pytorch_fallback(monkeypatch, patch_mod):
    original = MockFn('cuda')
    fallback = MockFn('pytorch')
    leaf = _module('vllm.v1.sample.ops.topk_topp_sampler', apply_top_k_top_p=original, apply_top_k_top_p_pytorch=fallback)
    _install(monkeypatch, {'vllm.v1.sample.ops.topk_topp_sampler': leaf})
    patch_mod.patch_topk_topp_sampler()
    assert leaf.apply_top_k_top_p is fallback

def test_topk_topp_sampler_logs_and_keeps_binding_on_import_error(monkeypatch, patch_mod, caplog):
    original = MockFn('cuda')
    leaf = _module('vllm.v1.sample.ops.topk_topp_sampler', apply_top_k_top_p=original)
    _install(monkeypatch, {'vllm.v1.sample.ops.topk_topp_sampler': leaf})
    patch_mod.patch_topk_topp_sampler()
    assert leaf.apply_top_k_top_p is original
    assert 'Failed to patch apply_top_k_top_p' in caplog.text

class MockFn:

    def __init__(self, tag):
        self.tag = tag

    def __call__(self, *args, **kwargs):
        return self.tag

def test_cuda_device_properties_patch_binds_tuple_helper(monkeypatch, patch_mod):
    torch = _module('torch')
    torch.cuda = SimpleNamespace(get_device_capability=MockFn('stock'))
    musa = _module('torch_musa', get_device_properties=MagicMock(return_value=SimpleNamespace(name='dev')))
    platform_utils = _module('vllm.utils.platform_utils', cuda_get_device_properties=MockFn('pool'))
    usage_lib = _module('vllm.usage.usage_lib', cuda_get_device_properties=MockFn('pool'))
    _install(monkeypatch, {'torch': torch, 'torch_musa': musa, 'vllm.utils.platform_utils': platform_utils, 'vllm.usage.usage_lib': usage_lib})
    patch_mod.patch_cuda_get_device_properties()
    assert torch.cuda.get_device_capability() == (8, 0)
    names = ('name', 'missing')
    result = platform_utils.cuda_get_device_properties(0, names)
    assert result[0] is musa.get_device_properties.return_value.name
    assert result[1] is None
    assert usage_lib.cuda_get_device_properties is platform_utils.cuda_get_device_properties
    assert torch.cuda._musa_device_props_patched is True
    musa.get_device_properties.reset_mock()
    torch.cuda.get_device_capability = MockFn('stock-again')
    patch_mod.patch_cuda_get_device_properties()
    assert torch.cuda.get_device_capability() == 'stock-again'
    musa.get_device_properties.assert_not_called()

def test_cuda_device_properties_missing_vendor_api_fails_on_use(monkeypatch, patch_mod):
    torch = _module('torch')
    torch.cuda = SimpleNamespace(get_device_capability=MockFn('stock'))
    _install(monkeypatch, {'torch': torch, 'torch_musa': _module('torch_musa'), 'vllm.utils.platform_utils': _module('vllm.utils.platform_utils'), 'vllm.usage.usage_lib': _module('vllm.usage.usage_lib')})
    patch_mod.patch_cuda_get_device_properties()
    assert torch.cuda._musa_device_props_patched is True
    with pytest.raises(AttributeError):
        platform_utils = sys.modules['vllm.utils.platform_utils']
        platform_utils.cuda_get_device_properties(0, ('name',))

def test_cuda_stream_helpers_delegate_to_musa(monkeypatch, patch_mod):
    tls = SimpleNamespace(value=None)
    torch_utils = _module('vllm.utils.torch_utils', aux_stream=lambda : 'cuda-aux', _current_stream_tls=tls)
    holders = {name: _module(name, aux_stream=lambda : 'cuda-aux') for name in ('vllm.model_executor.layers.fused_moe.runner.shared_experts', 'vllm.worker.model_runner_base', 'vllm.model_executor.models.utils', 'vllm.model_executor.layers.fused_moe.runner.moe_runner')}

    class Stream:
        pass
    orig_calls = []

    def orig_stream(value):
        orig_calls.append(('stream', value))
        return 'orig-stream'

    def orig_set(value):
        orig_calls.append(('set', value))

    def orig_current(device=None):
        orig_calls.append(('current', device))
        return 'orig-current'
    torch_musa = _module('torch_musa', Stream=Stream)
    torch = _module('torch')
    torch.cuda = _module('torch.cuda', stream=orig_stream, set_stream=orig_set, current_stream=orig_current)
    torch.musa = _module('torch.musa', stream=lambda s: ('musa-stream', s), set_stream=lambda s: orig_calls.append(('musa-set', s)), current_stream=lambda d: ('musa-current', d))
    mapping = {'torch': torch, 'torch.cuda': torch.cuda, 'torch_musa': torch_musa, 'vllm.utils.torch_utils': torch_utils, **holders}
    _install(monkeypatch, mapping)
    patch_mod.patch_cuda_stream_for_musa()
    assert torch_utils.aux_stream() is None
    for module in holders.values():
        assert module.aux_stream is torch_utils.aux_stream
    from contextlib import nullcontext
    assert type(torch.cuda.stream(None)) is nullcontext
    stream = Stream()
    assert torch.cuda.stream(stream) == ('musa-stream', stream)
    assert torch.cuda.stream('x') == 'orig-stream'
    torch.cuda.set_stream(stream)
    assert tls.value is stream
    assert ('musa-set', stream) in orig_calls
    torch.cuda.set_stream('plain')
    assert ('set', 'plain') in orig_calls
    assert torch.cuda.current_stream(2) == ('musa-current', 2)
    torch.musa.current_stream = MagicMock(side_effect=RuntimeError('no current'))
    assert torch.cuda.current_stream(3) == 'orig-current'
    assert torch.cuda._musa_stream_patched is True
    torch.musa.current_stream = MagicMock(return_value='keep')
    patch_mod.patch_cuda_stream_for_musa()
    assert torch.cuda.current_stream(1) == 'keep'

def _inductor_modules(kernel):

    class TritonCompileResult:

        def __init__(self, value):
            self.kernel = value
            self.calls = 0

        def make_launcher(self):
            self.calls += 1
            return ('launcher', self.kernel.metadata if self.kernel else None)
    leaf = _module('torch._inductor.runtime.triton_heuristics', TritonCompileResult=TritonCompileResult)
    torch = _module('torch')
    return ({'torch': torch, 'torch._inductor.runtime.triton_heuristics': leaf}, leaf)

@pytest.mark.parametrize('with_cluster', [False, True])
def test_inductor_launcher_synthesizes_cluster_dims(monkeypatch, patch_mod, with_cluster):
    fields = ('num_warps', 'cluster_dims') if with_cluster else ('num_warps',)
    Meta = namedtuple('KernelMetadata', fields)
    values = (4, (2, 2, 2)) if with_cluster else (4,)
    kernel = SimpleNamespace(metadata=Meta(*values))
    (mapping, leaf) = _inductor_modules(kernel)
    _install(monkeypatch, mapping)
    patch_mod.patch_inductor_triton_for_musa()
    result = leaf.TritonCompileResult(kernel)
    output = result.make_launcher()
    assert result.calls == 1
    assert output[0] == 'launcher'
    if with_cluster:
        assert kernel.metadata == Meta(*values)
    else:
        assert kernel.metadata.cluster_dims == (1, 1, 1)
        assert kernel.metadata.num_warps == 4
    assert leaf._musa_cluster_dims_patched is True
    if not with_cluster:
        kernel.metadata = Meta(*values)
        leaf.TritonCompileResult(kernel).make_launcher()
        assert kernel.metadata.cluster_dims == (1, 1, 1)

def test_inductor_launcher_without_kernel_metadata_is_passthrough(monkeypatch, patch_mod):
    (mapping, leaf) = _inductor_modules(None)
    _install(monkeypatch, mapping)
    patch_mod.patch_inductor_triton_for_musa()
    assert leaf.TritonCompileResult(None).make_launcher() == ('launcher', None)

def _install_dispatch(monkeypatch, call_op):
    dispatch = _module('vllm_fl.dispatch', call_op=call_op)
    package = _module('vllm_fl')
    package.dispatch = dispatch
    _install(monkeypatch, {'vllm_fl': package, 'vllm_fl.dispatch': dispatch})

def test_moe_topk_softmax_patches_leaf_and_routers(monkeypatch, patch_mod):
    calls = []
    _install_dispatch(monkeypatch, lambda *a: calls.append(a))
    ops = _module('vllm._custom_ops', topk_softmax=MockFn('cuda'))
    router_a = _module('vllm.model_executor.layers.fused_moe.router.fused_topk_router', vllm_topk_softmax=MockFn('a'))
    router_b = _module('vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router', vllm_topk_softmax=MockFn('b'))
    _install(monkeypatch, {'vllm._custom_ops': ops, 'vllm.model_executor.layers.fused_moe.router.fused_topk_router': router_a, 'vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router': router_b})
    patch_mod._moe_bias_warned = False
    patch_mod.patch_moe_topk_softmax_for_musa()
    (weights, ids, indices, gating) = (object(), object(), object(), object())
    for _ in range(2):
        result = ops.topk_softmax(weights, ids, indices, gating, True, object())
        assert result == (weights, ids)
    assert len(calls) == 2
    assert calls[0] == ('topk_softmax', weights, ids, indices, gating, True)
    assert router_a.vllm_topk_softmax is ops.topk_softmax
    assert router_b.vllm_topk_softmax is ops.topk_softmax

def test_moe_topk_softmax_import_failure_keeps_silent(monkeypatch, patch_mod, caplog):
    monkeypatch.delitem(sys.modules, 'vllm_fl.dispatch', raising=False)
    monkeypatch.delitem(sys.modules, 'vllm_fl', raising=False)
    blocker = _module('vllm_fl.dispatch')
    monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch', blocker)
    monkeypatch.setitem(sys.modules, 'vllm_fl', _module('vllm_fl'))
    patch_mod.patch_moe_topk_softmax_for_musa()
    assert 'cannot import' in caplog.text

def test_moe_topk_softmax_reports_when_nothing_patched(monkeypatch, patch_mod, caplog):
    _install_dispatch(monkeypatch, MagicMock())
    ops = _module('vllm._custom_ops')
    setattr(ops, 'topk_softmax', None)
    leaf = ModuleType('vllm._custom_ops')

    def _no_setattr(self, name, value):
        raise RuntimeError('readonly module')

    class _Locked:
        __slots__ = ()
    locked = _Locked()
    routers = {name: _module(name) for name in ('vllm.model_executor.layers.fused_moe.router.fused_topk_router', 'vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router')}
    _install(monkeypatch, {'vllm._custom_ops': locked, **routers})
    patch_mod.patch_moe_topk_softmax_for_musa()
    assert 'no call site patched' in caplog.text

class _FakeTensor:

    def __init__(self, value):
        self.value = value
        self.device = SimpleNamespace(type='musa')

    @property
    def dtype(self):
        return self.value.dtype

    def __getitem__(self, item):
        return _FakeTensor(self.value[item])

    def __mul__(self, other):
        return _FakeTensor(self.value * other.value)

    def __sub__(self, other):
        return _FakeTensor(self.value - other.value)

    def __add__(self, other):
        return _FakeTensor(self.value + other.value)

    def __neg__(self):
        return _FakeTensor(-self.value)

    def float(self):
        return _FakeTensor(self.value.float())

    def to(self, dtype=None):
        return _FakeTensor(self.value.to(dtype) if dtype is not None else self.value)

    def unsqueeze(self, dim):
        return _FakeTensor(self.value.unsqueeze(dim))

    def is_contiguous(self):
        return self.value.is_contiguous()

    def contiguous(self):
        return _FakeTensor(self.value.contiguous())

    def flatten(self, start, end):
        return _FakeTensor(self.value.flatten(start, end))

def _fake_torch_for_rotary(real_torch):

    def chunk(tensor, chunks, dim):
        return [_FakeTensor(x) for x in real_torch.chunk(tensor.value, chunks, dim)]

    def cat(values, dim):
        return _FakeTensor(real_torch.cat([v.value for v in values], dim))

    def stack(values, dim):
        return _FakeTensor(real_torch.stack([v.value for v in values], dim))
    return _module('torch', Tensor=_FakeTensor, chunk=chunk, cat=cat, stack=stack)

@pytest.fixture
def rotary_apply(monkeypatch, patch_mod):
    import torch as real_torch
    original = MockFn('original')
    base = _module('vllm.model_executor.layers.rotary_embedding.common')
    base.ApplyRotaryEmb = type('ApplyRotaryEmb', (), {'forward_native': original})
    _install(monkeypatch, {'torch': _fake_torch_for_rotary(real_torch), 'vllm.model_executor.layers.rotary_embedding.common': base})
    patch_mod.patch_musa_vision_rotary_layout()
    return (base.ApplyRotaryEmb, original, real_torch)

def test_vision_rotary_non_musa_delegates(rotary_apply):
    (apply_cls, original, real_torch) = rotary_apply
    layer = SimpleNamespace(is_neox_style=False, enable_fp32_compute=False)
    cpu = _FakeTensor(real_torch.ones(2))
    cpu.device = SimpleNamespace(type='cuda')
    assert apply_cls.forward_native(layer, cpu, cpu, cpu) == 'original'

def test_vision_rotary_is_idempotent(rotary_apply):
    (apply_cls, _, _) = rotary_apply
    patched = apply_cls.forward_native
    assert patched._musa_rotary_layout_patched is True

def _install_libentry(monkeypatch, tuner_cls, capturing):
    fresh = type('LibTuner', (tuner_cls,), {'_fl_capture_safe': False})
    torch = _module('torch')
    torch.musa = SimpleNamespace(is_current_stream_capturing=MagicMock(return_value=capturing))
    libentry = _module('flag_gems.utils.libentry', LibTuner=fresh)
    _install(monkeypatch, {'torch': torch, 'torch_musa': _module('torch_musa'), 'flag_gems': _module('flag_gems'), 'flag_gems.utils': _module('flag_gems.utils'), 'flag_gems.utils.libentry': libentry})
    return fresh

class _Tuner:
    _fl_capture_safe = False

    def __init__(self, configs=(), cache=None, pre_hook=None):
        self.configs = list(configs)
        self.cache = cache or {}
        self.arg_names = ['a']
        self.seen_tuned_metas = {}
        self.nargs = None
        self.best_config = None
        self.orig_calls = 0
        self.fn = SimpleNamespace(run=MagicMock(return_value='captured'))
        self.pre_hook = pre_hook

    def run(self, *args, **kwargs):
        self.orig_calls += 1
        return 'orig'

    def get_key(self, key_args):
        return 'key'

    def config(self):
        config = SimpleNamespace(all_kwargs=lambda : {'block': 4})
        config.pre_hook = self.pre_hook
        return config

def test_capture_safe_tuning_uses_cached_config(monkeypatch, patch_mod):
    hook = MagicMock()
    fresh = _install_libentry(monkeypatch, _Tuner, capturing=True)
    patch_mod.patch_flag_gems_capture_safe_tuning()
    tuner = fresh(configs=(), cache={}, pre_hook=hook)
    cfg = tuner.config()
    cfg.pre_hook = hook
    tuner.cache['key'] = cfg
    tuner.nargs = None
    result = tuner.run(1)
    assert result == 'captured'
    assert tuner.orig_calls == 0
    assert tuner.best_config is cfg
    assert tuner.nargs is None
    tuner.fn.run.assert_called_once_with(1, block=4)
    hook.assert_called_once()
    assert fresh._fl_capture_safe is True

def test_capture_safe_tuning_falls_back_to_static_config(monkeypatch, patch_mod):
    fresh = _install_libentry(monkeypatch, _Tuner, capturing=True)
    patch_mod.patch_flag_gems_capture_safe_tuning()
    cfg = fresh().config()
    tuner = fresh(configs=[cfg])
    assert tuner.run(2) == 'captured'
    assert tuner.best_config is cfg

def test_capture_safe_tuning_without_any_config_raises(monkeypatch, patch_mod):
    fresh = _install_libentry(monkeypatch, _Tuner, capturing=True)
    patch_mod.patch_flag_gems_capture_safe_tuning()
    tuner = fresh(configs=[])
    with pytest.raises(RuntimeError, match='no cached or static config'):
        tuner.run(3)
    assert tuner.nargs is None

def test_capture_safe_tuning_bypasses_outside_capture(monkeypatch, patch_mod):
    fresh = _install_libentry(monkeypatch, _Tuner, capturing=False)
    patch_mod.patch_flag_gems_capture_safe_tuning()
    tuner = fresh(configs=[])
    assert tuner.run(4) == 'orig'
    assert tuner.orig_calls == 1

def test_flagtune_defaults_disabled_and_respects_overrides(monkeypatch, patch_mod):
    monkeypatch.delenv('USE_FLAGTUNE', raising=False)
    monkeypatch.delenv('VLLM_FL_MUSA_FLAGTUNE', raising=False)
    patch_mod.patch_flag_gems_disable_flagtune()
    import os as _os
    assert _os.environ['USE_FLAGTUNE'] == '0'
    monkeypatch.setenv('USE_FLAGTUNE', '1')
    monkeypatch.setenv('VLLM_FL_MUSA_FLAGTUNE', '1')
    patch_mod.patch_flag_gems_disable_flagtune()
    import os
    assert os.environ['USE_FLAGTUNE'] == '1'
    monkeypatch.setenv('USE_FLAGTUNE', '0')
    monkeypatch.setenv('VLLM_FL_MUSA_FLAGTUNE', '0')
    patch_mod.patch_flag_gems_disable_flagtune()
    assert os.environ['USE_FLAGTUNE'] == '0'

def test_apply_musa_patches_idempotent_and_retries_on_failure(monkeypatch, patch_mod):
    order = []
    names = ['patch_topk_topp_sampler', 'patch_cuda_get_device_properties', 'patch_cuda_stream_for_musa', 'patch_inductor_triton_for_musa', 'patch_moe_topk_softmax_for_musa', 'patch_triton_mtgpu_alias_for_musa', 'patch_device_config_for_musa', 'patch_musa_vision_rotary_layout', 'patch_flag_gems_capture_safe_tuning', 'patch_flag_gems_disable_flagtune']
    for name in names:
        monkeypatch.setattr(patch_mod, name, lambda name=name: order.append(name))
    outcome = {'value': False}
    monkeypatch.setattr(patch_mod, 'patch_accelerator_missing_attrs', lambda : outcome['value'])
    patch_mod.apply_musa_patches()
    assert order == names
    assert patch_mod._patches_applied is False
    patch_mod.apply_musa_patches()
    assert len(order) == 2 * len(names)
    outcome['value'] = True
    patch_mod.apply_musa_patches()
    assert patch_mod._patches_applied is True
    before = len(order)
    patch_mod.apply_musa_patches()
    assert len(order) == before