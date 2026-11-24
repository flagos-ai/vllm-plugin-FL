"""CPU controls for MUSA graph config, operator impls and runtime hooks."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
import pytest
import torch

def _repo_root() -> Path:
    import os
    candidates = [Path('/tmp/flsrc')]
    if os.environ.get('VLLM_FL_ROOT'):
        candidates.insert(0, Path(os.environ['VLLM_FL_ROOT']))
    for base in [*candidates, Path.cwd(), *Path(__file__).resolve().parents]:
        if not base.is_dir():
            continue
        if (base / 'vllm_fl').is_dir():
            return base
    raise RuntimeError('cannot locate vllm_fl source tree')
ROOT = _repo_root()

def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

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
                setattr(parent, parts[index - 1], sys.modules[full])

@pytest.fixture
def graph_config():
    return _load('w6b_graph_config', 'vllm_fl/dispatch/backends/vendor/musa/graph_config.py')

def test_collective_splitting_ops_are_appended_once(graph_config):
    splitting = ['custom::op', graph_config.MUSA_COLLECTIVE_SPLITTING_OPS[0]]
    assert graph_config.append_collective_splitting_ops(splitting) is True
    for op in graph_config.MUSA_COLLECTIVE_SPLITTING_OPS:
        assert op in splitting
    assert splitting.count(graph_config.MUSA_COLLECTIVE_SPLITTING_OPS[0]) == 1
    assert graph_config.append_collective_splitting_ops(splitting) is False

def test_thin_cudagraph_sizes_keep_powers_of_two_and_max(graph_config):
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    kept = graph_config.thin_cudagraph_capture_sizes(sizes)
    assert set([1, 2, 4, 8]).issubset(set(kept))
    assert kept[-1] == 12
    assert kept == sorted(set(kept))
    assert len(kept) <= graph_config.MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES

def test_thin_cudagraph_sizes_even_sampling_without_ladder(graph_config):
    sizes = list(range(3, 21, 2))
    kept = graph_config.thin_cudagraph_capture_sizes(sizes)
    assert kept[-1] == sizes[-1]
    assert 1 <= len(kept) <= graph_config.MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES
    assert all((value in sizes for value in kept))

def test_thin_cudagraph_sizes_caps_a_dense_ladder(graph_config):
    sizes = [2 ** power for power in range(12)]
    kept = graph_config.thin_cudagraph_capture_sizes(sizes)
    assert len(kept) <= graph_config.MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES
    assert kept[-1] == sizes[-1]

def test_thin_cudagraph_sizes_empty(graph_config):
    assert graph_config.thin_cudagraph_capture_sizes([]) == []

def test_limit_cudagraph_capture_sizes_small_and_large(graph_config):
    small = SimpleNamespace(cudagraph_capture_sizes=[1, 2, 3])
    assert graph_config.limit_cudagraph_capture_sizes(small) is None
    assert small.cudagraph_capture_sizes == [1, 2, 3]
    large = SimpleNamespace(cudagraph_capture_sizes=list(range(1, 40)))
    result = graph_config.limit_cudagraph_capture_sizes(large)
    assert result == large.cudagraph_capture_sizes
    assert len(result) <= graph_config.MUSA_MAX_CUDAGRAPH_CAPTURE_SIZES
    assert result[-1] == 39
    missing = SimpleNamespace(cudagraph_capture_sizes=None)
    assert graph_config.limit_cudagraph_capture_sizes(missing) is None

@pytest.fixture
def impls():
    return {'activation': _load('w6b_act', 'vllm_fl/dispatch/backends/vendor/musa/impl/activation.py'), 'normalization': _load('w6b_norm', 'vllm_fl/dispatch/backends/vendor/musa/impl/normalization.py'), 'rotary': _load('w6b_rot', 'vllm_fl/dispatch/backends/vendor/musa/impl/rotary.py')}

def test_silu_and_mul_splits_last_dimension(impls):
    x = torch.randn(2, 3, 8)
    out = impls['activation'].silu_and_mul_musa(None, x)
    expected = torch.nn.functional.silu(x[..., :4]) * x[..., 4:]
    assert torch.allclose(out, expected)

def test_rms_norm_without_residual(impls):
    weight = torch.randn(6)
    obj = SimpleNamespace(weight=weight, variance_epsilon=1e-06)
    x = torch.randn(4, 6)
    out = impls['normalization'].rms_norm_musa(obj, x)
    expected = weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-06))
    assert torch.allclose(out, expected)

def test_rms_norm_with_residual_returns_updated_residual(impls):
    weight = torch.ones(6)
    obj = SimpleNamespace(weight=weight, variance_epsilon=1e-05)
    x = torch.randn(4, 6)
    residual = torch.randn(4, 6)
    (out, new_residual) = impls['normalization'].rms_norm_musa(obj, x, residual)
    summed = x + residual
    assert torch.allclose(new_residual, summed)
    assert torch.allclose(out, summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + 1e-05))

def _expected_rotary(query, key, cos_sel, sin_sel, interleaved):

    def rotate_half(x):
        (first, second) = (x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:])
        return torch.cat((-second, first), -1)

    def rotate_interleaved(x):
        (first, second) = (x[..., ::2], x[..., 1::2])
        return torch.stack((-second, first), -1).flatten(-2)
    rotate = rotate_interleaved if interleaved else rotate_half
    return (query * cos_sel + rotate(query) * sin_sel, key * cos_sel + rotate(key) * sin_sel)

@pytest.mark.parametrize('query_dim', [3, 4])
@pytest.mark.parametrize('interleaved', [False, True])
def test_rotary_matches_reference(impls, query_dim, interleaved):
    head_dim = 8
    shape = (5, 2, head_dim) if query_dim == 3 else (2, 2, 5, head_dim)
    query = torch.randn(*shape)
    key = torch.randn(*shape)
    cos = torch.randn(10, head_dim)
    sin = torch.randn(10, head_dim)
    positions = torch.arange(5) if query_dim == 3 else torch.arange(10).reshape(2, 5)
    (q_out, k_out) = impls['rotary'].rotary_embedding_musa(None, query, key, cos, sin, positions, rotary_interleaved=interleaved)
    cos_sel = cos[positions].unsqueeze(1)
    sin_sel = sin[positions].unsqueeze(1)
    (expected_q, expected_k) = _expected_rotary(query, key, cos_sel, sin_sel, interleaved)
    assert torch.allclose(q_out, expected_q)
    assert torch.allclose(k_out, expected_k)

@pytest.fixture
def musa_backend(monkeypatch):
    calls = {}

    def make_impl(tag):

        def impl(*args, **kwargs):
            calls.setdefault(tag, []).append((args, kwargs))
            return tag
        return impl
    pkg = _module('w6b.musa')
    impl_pkg = _module('w6b.musa.impl')
    _install(monkeypatch, {'w6b.musa': pkg, 'w6b.musa.impl': impl_pkg, 'w6b.musa.impl.activation': _module('w6b.musa.impl.activation', silu_and_mul_musa=make_impl('silu')), 'w6b.musa.impl.normalization': _module('w6b.musa.impl.normalization', rms_norm_musa=make_impl('rms')), 'w6b.musa.impl.rotary': _module('w6b.musa.impl.rotary', rotary_embedding_musa=make_impl('rotary')), 'vllm_fl.dispatch.backends.base': _module('vllm_fl.dispatch.backends.base', Backend=type('Backend', (), {}))})
    fake_torch = _module('torch', musa=SimpleNamespace(is_available=MagicMock(return_value=True)))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    module = _load('w6b.musa.musa', 'vllm_fl/dispatch/backends/vendor/musa/musa.py')
    return (module.MusaBackend(), calls)

def test_musa_backend_metadata_and_availability(musa_backend):
    (backend, _) = musa_backend
    assert backend.name == 'musa'
    assert backend.vendor == 'musa'
    assert backend.is_available() is True

def test_musa_backend_delegates_operator_impls(musa_backend):
    (backend, calls) = musa_backend
    assert backend.silu_and_mul(None, torch.ones(2)) == 'silu'
    assert backend.rms_norm(None, torch.ones(2)) == 'rms'
    assert backend.rotary_embedding(None, 1, 2, 3, 4, 5, rotary_interleaved=True) == 'rotary'
    (args, kwargs) = calls['rotary'][0]
    assert args[0] is None
    assert kwargs == {'rotary_interleaved': True, 'inplace': True}

def test_musa_attention_backend_selects_registry(monkeypatch, musa_backend):
    (backend, _) = musa_backend

    class _Enum:
        TRITON_MLA = SimpleNamespace(get_path=lambda : 'mla-path')
        TRITON_ATTN = SimpleNamespace(get_path=lambda : 'attn-path')
    _install(monkeypatch, {'vllm.v1.attention.backends.registry': _module('vllm.v1.attention.backends.registry', AttentionBackendEnum=_Enum)})
    assert backend.attention_backend(use_mla=True) == 'mla-path'
    assert backend.attention_backend() == 'attn-path'
    with pytest.raises(ValueError, match='use_sparse=True requires use_mla=True'):
        backend.attention_backend(use_sparse=True)

def test_register_builtins_registers_vendor_ops(monkeypatch):

    class OpImpl:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Kind:
        VENDOR = 'vendor'

    class Priority:
        VENDOR = 100
    types_mod = _module('vllm_fl.dispatch.types', OpImpl=OpImpl, BackendImplKind=Kind, BackendPriority=Priority)
    monkeypatch.setitem(sys.modules, 'vllm_fl', _module('vllm_fl'))
    monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch', _module('vllm_fl.dispatch'))
    _install(monkeypatch, {'vllm_fl.dispatch.types': types_mod})

    class Backend:

        def __init__(self):
            self.is_available = MagicMock(return_value=True)

        def silu_and_mul(self, *args):
            return 'silu'

        def rms_norm(self, *args):
            return 'rms'

        def rotary_embedding(self, *args):
            return 'rotary'

        def attention_backend(self, *args):
            return 'attn'
    patch_calls = []
    pkg = _module('w6breg.musa')
    _install(monkeypatch, {'w6breg.musa': pkg, 'w6breg.musa.musa': _module('w6breg.musa.musa', MusaBackend=Backend), 'w6breg.musa.patch': _module('w6breg.musa.patch', apply_musa_patches=lambda : patch_calls.append(1))})
    module = _load('w6breg.musa.register_ops', 'vllm_fl/dispatch/backends/vendor/musa/register_ops.py')
    registry = SimpleNamespace(register_many=MagicMock())
    module.register_builtins(registry)
    assert patch_calls == [1]
    impls = registry.register_many.call_args.args[0]
    assert [impl.op_name for impl in impls] == ['silu_and_mul', 'rms_norm', 'rotary_embedding', 'attention_backend']
    assert all((impl.impl_id == 'vendor.musa' for impl in impls))
    assert all((impl.kind == 'vendor' and impl.priority == 100 for impl in impls))
    bound = impls[0].fn
    assert bound(None, 1) == 'silu'
    assert bound._is_available() is True
    assert bound.__name__ == Backend.silu_and_mul.__name__

@pytest.fixture
def graph_params():
    return _load('w6b_graph_params', 'vllm_fl/compilation/graph_params.py')

@pytest.mark.parametrize('bad', [0, -1, True, None, '8', 1.5])
def test_graph_params_reject_invalid_capture_size(graph_params, bad):
    params = graph_params.GraphParams()
    with pytest.raises(ValueError, match='must be positive'):
        params.prepare_capture(SimpleNamespace(num_tokens=bad))

@pytest.fixture
def runtime_module(monkeypatch):
    platform = SimpleNamespace(torch_device_fn=SimpleNamespace(current_stream=MagicMock(return_value=SimpleNamespace(synchronize=MagicMock())), Stream=MagicMock(return_value='stream')))
    graph_mod = _module('vllm_fl.compilation.graph', weak_ref_tensors=MagicMock(return_value='weak'))
    params_mod = _module('vllm_fl.compilation.graph_params', prepare_graph_params=MagicMock(return_value='params'), weak_ref_workspace=MagicMock(), clear_graph_params=MagicMock())
    _install(monkeypatch, {'vllm_fl': _module('vllm_fl'), 'vllm_fl.compilation': _module('vllm_fl.compilation'), 'vllm_fl.compilation.graph': graph_mod, 'vllm_fl.compilation.graph_params': params_mod, 'vllm.platforms': _module('vllm.platforms', current_platform=platform)})
    module = _load('w6b_graph_runtime', 'vllm_fl/compilation/graph_runtime.py')
    return (module, platform, params_mod)

def test_batch_descriptor_validation(runtime_module):
    (module, _, _) = runtime_module
    controller = module.GraphRuntimeController(vllm_config=object())
    descriptor = SimpleNamespace(num_tokens=16)
    assert controller._batch_descriptor(SimpleNamespace(batch_descriptor=descriptor)) is descriptor
    for bad in (SimpleNamespace(batch_descriptor=None), SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=None)), SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=True))):
        with pytest.raises(RuntimeError, match='integer BatchDescriptor.num_tokens'):
            controller._batch_descriptor(bad)

def test_capture_scope_prepares_and_releases(runtime_module):
    (module, _, params_mod) = runtime_module
    config = object()
    controller = module.GraphRuntimeController(vllm_config=config)
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=4), capturing=False)
    with controller.capture_scope(context):
        assert controller._active is True
        assert context.capturing is True
        params_mod.prepare_graph_params.assert_called_once_with(context.batch_descriptor)
    assert context.capturing is False
    assert controller._active is False
    params_mod.weak_ref_workspace.assert_called_once()
    args = params_mod.weak_ref_workspace.call_args.args
    assert args[0] is context.batch_descriptor
    assert args[1] is not None

def test_capture_scope_cleanup_runs_on_error(runtime_module):
    (module, _, params_mod) = runtime_module
    controller = module.GraphRuntimeController(vllm_config=object())
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=4), capturing=False)
    with pytest.raises(RuntimeError, match='boom'):
        with controller.capture_scope(context):
            raise RuntimeError('boom')
    assert context.capturing is False
    assert controller._active is False
    params_mod.weak_ref_workspace.assert_called_once()

def test_replay_scope_synchronizes_and_updates(runtime_module, monkeypatch):
    (module, platform, _) = runtime_module
    controller = module.GraphRuntimeController(vllm_config=object())
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=5))
    controller._update_attention = MagicMock()
    with controller.replay_scope(context):
        assert controller._active is True
    controller._update_attention.assert_called_once_with(context)
    platform.torch_device_fn.current_stream.return_value.synchronize.assert_called_once_with()
    assert controller._active is False
    controller._update_attention = MagicMock()
    with pytest.raises(ValueError):
        with controller.replay_scope(context):
            raise ValueError('native replay')
    controller._update_attention.assert_not_called()
    assert controller._active is False

def test_update_attention_requires_bindings_and_updates_once(runtime_module):
    (module, platform, _) = runtime_module
    controller = module.GraphRuntimeController(vllm_config='config')
    with pytest.raises(RuntimeError, match='must be bound'):
        controller._update_attention(SimpleNamespace())
    controller.attention_impls = ()
    controller._update_attention(SimpleNamespace())
    platform.torch_device_fn.Stream.assert_not_called()
    impl = SimpleNamespace(update_graph_params=MagicMock())
    controller.attention_impls = (impl,)
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=6))
    controller._update_attention(context)
    controller._update_attention(context)
    platform.torch_device_fn.Stream.assert_called_once_with()
    assert impl.update_graph_params.call_count == 2
    assert impl.update_graph_params.call_args.args == ('stream', context, context.batch_descriptor, 'config')

def test_bind_attention_groups_filters_and_validates(runtime_module, monkeypatch):
    (module, _, _) = runtime_module

    def _base_factory():
        return 'base-impl'

    class AttentionBackend:
        get_impl_cls = staticmethod(_base_factory)
    base_factory = _base_factory

    def good_factory():
        return SimpleNamespace(update_graph_params=MagicMock())

    def bad_factory():
        raise ValueError('cannot build')
    GoodBackend = type('GoodBackend', (), {'get_impl_cls': staticmethod(good_factory)})
    BadBackend = type('BadBackend', (), {'get_impl_cls': staticmethod(bad_factory)})
    Incomplete = type('Incomplete', (), {'get_impl_cls': staticmethod(lambda : SimpleNamespace())})
    Plain = type('Plain', (), {'get_impl_cls': base_factory})
    _install(monkeypatch, {'vllm.v1.attention.backend': _module('vllm.v1.attention.backend', AttentionBackend=AttentionBackend)})
    controller = module.GraphRuntimeController(vllm_config=object())
    good = GoodBackend()
    groups = [[SimpleNamespace(backend=Plain), SimpleNamespace(backend=GoodBackend)], [SimpleNamespace(backend=GoodBackend), SimpleNamespace(backend=GoodBackend)]]
    controller.bind_attention_groups(groups)
    assert len(controller.attention_impls) == 1
    assert hasattr(controller.attention_impls[0], 'update_graph_params')
    assert controller.attention_impls[0] is not good
    for factory in (BadBackend, Incomplete):
        with pytest.raises(RuntimeError):
            controller.bind_attention_groups([[SimpleNamespace(backend=factory)]])

def test_clear_decoder_graphs_clears_wrapper_and_params(runtime_module):
    (module, _, params_mod) = runtime_module
    wrapper = MagicMock()
    controller = module.GraphRuntimeController(vllm_config=object(), full_graph_wrapper_type=wrapper)
    controller.clear_decoder_graphs()
    wrapper.clear_all_graphs.assert_called_once_with()
    params_mod.clear_graph_params.assert_called_once_with()
    quiet = module.GraphRuntimeController(vllm_config=object())
    quiet.clear_decoder_graphs()
    assert params_mod.clear_graph_params.call_count == 2