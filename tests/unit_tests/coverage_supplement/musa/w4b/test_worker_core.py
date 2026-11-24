"""CPU contracts for vllm_fl.worker.worker.

The worker class body executes from source with explicit doubles for vLLM
internals (worker base, distributed groups, profilers, allocators). No real
accelerator, process group, model or weight is touched.
"""
import ast
import contextlib
import gc
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
try:
    import regex as _re
except ImportError:
    import re as _re
SOURCE_REL = 'vllm_fl/worker/worker.py'

def _source_root():
    candidates = []
    env = os.environ.get('VLLM_FL_SOURCE_ROOT')
    if env:
        candidates.append(Path(env))
    candidates.extend([Path.cwd(), Path('/tmp/flsrc'), Path(__file__).resolve().parents[2]])
    for root in candidates:
        if (root / SOURCE_REL).is_file():
            return root
    pytest.skip('vllm_fl sources are required', allow_module_level=True)
ROOT = _source_root()
_TREE = ast.parse((ROOT / SOURCE_REL).read_text(encoding='utf-8'))
_FUTURE = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
NAMES = {'_CUDA_TO_MUSA_FOR_V2', '_num_workspace_lanes', '_install_v2_runner_compat', '_install_v2_dispatch_warmup', '_normalize_process_group_backend', '_is_hybrid_kv_model', '_apply_worker_vendor_patches', 'AsyncIntermediateTensors', 'WorkerFL', 'init_worker_distributed_environment'}

def _node_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None

def _exec_module(namespace):
    selected = [n for n in _TREE.body if _node_name(n) in NAMES]
    assert {_node_name(n) for n in selected} == NAMES
    module = ast.fix_missing_locations(ast.Module(body=[_FUTURE, *selected], type_ignores=[]))
    exec(compile(module, SOURCE_REL, 'exec'), namespace)

class _Missing:

    def __call__(self, *args, **kwargs):
        return _Missing()

    def __getattr__(self, name):
        return _Missing()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

class _NS(dict):

    def __missing__(self, key):
        value = _Missing()
        self[key] = value
        return value

class _Logger:

    def __init__(self):
        self.messages = []

    def _log(self, level, msg, *args):
        self.messages.append((level, msg % args if args else msg))

    def info(self, msg, *args):
        self._log('info', msg, *args)

    def info_once(self, msg, *args):
        self._log('info', msg, *args)

    def warning(self, msg, *args):
        self._log('warning', msg, *args)

    def error(self, msg, *args):
        self._log('error', msg, *args)

    def debug(self, msg, *args):
        self._log('debug', msg, *args)

def _identity_decorator(*args, **kwargs):

    def decorate(fn):
        return fn
    return decorate

class _WorkerBase:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_supported_kv_cache_layouts(self):
        return ['default']

class _IntermediateTensors:

    def __init__(self, tensors):
        object.__setattr__(self, 'tensors', tensors)

class _MarkA:
    pass

class _MarkB:
    pass

class _FakeElastic:

    def __init__(self, worker):
        self.worker = worker
        self.calls = []

    def execute(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return ('elastic', method)

def _platform(**over):
    base = dict(device_type='cuda', vendor_name=None, dist_backend='nccl', is_rocm=lambda : False, is_cuda_alike=lambda : True, is_cuda=lambda : True, is_xpu=lambda : False, is_cpu=lambda : False, is_sleep_mode_available=lambda : True, logical_device_id_to_visible_device_id=lambda i: i, set_device=lambda d: None, check_if_supports_dtype=lambda d: None)
    base.update(over)
    return SimpleNamespace(**base)

@pytest.fixture
def env(monkeypatch):
    ns = _NS()
    ns.update(torch=torch, nn=torch.nn, os=os, re=_re, time=time, gc=gc, logger=_Logger(), current_platform=_platform(), envs=SimpleNamespace(VLLM_FLOAT32_MATMUL_PRECISION='highest', VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=True), nullcontext=contextlib.nullcontext, contextmanager=contextlib.contextmanager, AbstractContextManager=contextlib.AbstractContextManager, Handle=object, IntermediateTensors=_IntermediateTensors, WorkerBase=_WorkerBase, CUDAGraphMode=SimpleNamespace(NONE='none'), CompilationMode=SimpleNamespace(VLLM_COMPILE='compile', NONE='none'), GiB_bytes=1 << 30, format_gib=lambda b: f'{b / float(1 << 30):.2f}', request_memory=lambda snapshot, cache: 0, set_current_vllm_config=lambda cfg: contextlib.nullcontext(), get_pp_group=lambda : _Missing(), get_tp_group=lambda : _Missing(), ModelRunnerOutput=_MarkA, AsyncModelRunnerOutput=_MarkB, DraftTokenIds=object, WeightTransferEngine=object, ElasticEPScalingExecutor=_FakeElastic, NoneType=type(None), TYPE_CHECKING=False, instrument=_identity_decorator, managed_inference_mode=_identity_decorator, with_gpu_sync_check=lambda fn: fn)
    _exec_module(ns)
    return ns

def make_worker(env, **attrs):
    worker = object.__new__(env['WorkerFL'])
    for (name, value) in attrs.items():
        setattr(worker, name, value)
    return worker

def _recorder(*args, **kwargs):
    return None

def test_num_workspace_lanes(env):
    dspark = SimpleNamespace(use_dspark=lambda : True)
    plain = SimpleNamespace(use_dspark=lambda : False)
    cfg = SimpleNamespace(speculative_config=dspark)
    assert env['_num_workspace_lanes'](cfg, True) == 2
    assert env['_num_workspace_lanes'](cfg, False) == 1
    assert env['_num_workspace_lanes'](SimpleNamespace(speculative_config=plain), True) == 1
    assert env['_num_workspace_lanes'](SimpleNamespace(speculative_config=None), True) == 1

def test_is_hybrid_kv_model_mamba_mode(env):
    cfg = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode='align'))
    assert env['_is_hybrid_kv_model'](cfg) is True

def test_is_hybrid_kv_model_flag(env):
    cfg = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode='none'), model_config=SimpleNamespace(is_hybrid=True))
    assert env['_is_hybrid_kv_model'](cfg) is True

def test_is_hybrid_kv_model_falls_back_to_layer_types(env):

    class _Raises:

        @property
        def is_hybrid(self):
            raise ValueError('not ready')
        hf_text_config = SimpleNamespace(layer_types=['full_attention', 'linear'])
    cfg = SimpleNamespace(cache_config=None, model_config=_Raises())
    assert env['_is_hybrid_kv_model'](cfg) is True

def test_install_v2_runner_compat_leaves_cuda_when_available(env, monkeypatch):
    calls = []
    env['_install_v2_dispatch_warmup'] = lambda : calls.append('warmup')
    env['torch'] = SimpleNamespace(musa=SimpleNamespace(Stream=object), cuda=SimpleNamespace(is_available=lambda : True))
    env['_install_v2_runner_compat']()
    assert calls == []

def _fake_model_runner_module(monkeypatch, holder):

    class _GPUModelRunner:

        def _dummy_run(self, *args, **kwargs):
            holder.append(('orig', args, kwargs))
            return 'dummy'
    module = SimpleNamespace(GPUModelRunner=_GPUModelRunner)
    monkeypatch.setitem(sys.modules, 'vllm.v1.worker.gpu.model_runner', module)
    dispatch = SimpleNamespace(warmup_cached_ops=lambda : holder.append(('warmup',)))
    monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch', dispatch)
    return _GPUModelRunner

def test_dispatch_warmup_is_idempotent(env, monkeypatch):
    holder = []
    cls = _fake_model_runner_module(monkeypatch, holder)
    env['_install_v2_dispatch_warmup']()
    wrapper = cls._dummy_run
    env['_install_v2_dispatch_warmup']()
    assert cls._dummy_run is wrapper

def test_apply_vendor_patches_skips_other_vendors(env, monkeypatch):
    env['current_platform'] = _platform(vendor_name='amd')
    env['_apply_worker_vendor_patches']()
    assert env['logger'].messages == []

def _allocator(usage=0, log=None):

    class _Alloc:

        def __init__(self):
            self.pools = []

        def get_current_usage(self):
            return usage

        def use_memory_pool(self, tag):
            self.pools.append(tag)
            return contextlib.nullcontext(f'pool:{tag}')
    return _Alloc()

def test_memory_pool_context_npu(env):
    env['current_platform'] = _platform(device_type='npu')
    worker = make_worker(env, vllm_config=SimpleNamespace(model_config=SimpleNamespace()))
    with worker._maybe_get_memory_pool_context('weights') as value:
        assert value is None

def test_memory_pool_context_cuda_without_cumem(env):
    env['current_platform'] = _platform()
    cfg = SimpleNamespace(model_config=SimpleNamespace(enable_cumem_allocator=False))
    worker = make_worker(env, vllm_config=cfg)
    with worker._maybe_get_memory_pool_context('weights') as value:
        assert value is None

def test_memory_pool_context_xpu_without_sleep(env):
    env['current_platform'] = _platform(is_cuda_alike=lambda : False, is_xpu=lambda : True)
    cfg = SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=False))
    worker = make_worker(env, vllm_config=cfg)
    with worker._maybe_get_memory_pool_context('weights') as value:
        assert value is None

def test_memory_pool_context_cpu(env):
    env['current_platform'] = _platform(is_cuda_alike=lambda : False, is_xpu=lambda : False, is_cpu=lambda : True)
    worker = make_worker(env, vllm_config=SimpleNamespace(model_config=SimpleNamespace()))
    with worker._maybe_get_memory_pool_context('weights') as value:
        assert value is None

def test_memory_pool_context_unavailable_sleep(env):
    env['current_platform'] = _platform(is_cuda_alike=lambda : False, is_xpu=lambda : False, is_sleep_mode_available=lambda : False)
    cfg = SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=False))
    worker = make_worker(env, vllm_config=cfg)
    with worker._maybe_get_memory_pool_context('weights') as value:
        assert value is None

def test_memory_pool_context_uses_allocator(env):
    allocator = _allocator()
    env['get_mem_allocator_instance'] = lambda : allocator
    env['current_platform'] = _platform()
    cfg = SimpleNamespace(model_config=SimpleNamespace(enable_cumem_allocator=True))
    worker = make_worker(env, vllm_config=cfg)
    with worker._maybe_get_memory_pool_context('weights') as value:
        assert value == 'pool:weights'
    with worker._maybe_get_memory_pool_context('kv_cache') as value:
        assert value == 'pool:kv_cache'

def test_memory_pool_context_rejects_second_weights_pool(env):
    env['get_mem_allocator_instance'] = lambda : _allocator(usage=7)
    env['current_platform'] = _platform()
    cfg = SimpleNamespace(model_config=SimpleNamespace(enable_cumem_allocator=True))
    worker = make_worker(env, vllm_config=cfg)
    with pytest.raises(AssertionError):
        worker._maybe_get_memory_pool_context('weights')
    with worker._maybe_get_memory_pool_context('kv_cache'):
        pass

def test_scoped_allocator_max_split_noop_outside_cuda(env):
    env['current_platform'] = _platform(is_cuda=lambda : False)
    worker = make_worker(env)
    with worker._scoped_allocator_max_split(20):
        pass

def test_scoped_allocator_max_split_restores_configured_value(env, monkeypatch):
    recorded = []
    env['torch'] = SimpleNamespace(_C=SimpleNamespace(_accelerator_setAllocatorSettings=lambda value: recorded.append(value)))
    env['current_platform'] = _platform(is_cuda=lambda : True)
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:50')
    worker = make_worker(env)
    with worker._scoped_allocator_max_split(20):
        pass
    assert recorded == ['max_split_size_mb:20', 'max_split_size_mb:50']

def _sleep_env(env, info_sequence, monotonic=None):
    env['torch'] = SimpleNamespace(accelerator=SimpleNamespace(synchronize=lambda : None, get_memory_info=lambda : next(info_sequence)))
    if monotonic is not None:
        stamps = iter(monotonic)
        env['time'] = SimpleNamespace(monotonic=lambda : next(stamps), sleep=lambda seconds: None)
    return env

def test_sleep_level1_and_backend_suspend(env):
    _sleep_env(env, iter([(100, 200), (100, 200)]))
    backend = SimpleNamespace(suspend=lambda level: None)
    worker = make_worker(env, _sleep_mode_backend=backend)
    worker.sleep()
    assert any(('Sleep mode freed' in m for (_, m) in env['logger'].messages))

def test_sleep_level2_saves_buffers_and_draft(env):
    _sleep_env(env, iter([(100, 200), (100, 200)]))
    model = torch.nn.Module()
    model.register_buffer('w', torch.tensor([1.0, 2.0]))
    draft = torch.nn.Module()
    draft.register_buffer('d', torch.tensor([3.0]))
    model_runner = SimpleNamespace(model=model, get_draft_model=lambda : draft)
    backend = SimpleNamespace(suspend=lambda level: None)
    worker = make_worker(env, _sleep_mode_backend=backend, model_runner=model_runner)
    worker.sleep(level=2)
    assert set(worker._sleep_saved_buffers) == {'w'}
    assert set(worker._sleep_saved_draft_buffers) == {'d'}
    assert torch.equal(worker._sleep_saved_buffers['w'], torch.tensor([1.0, 2.0]))

def test_sleep_raises_when_memory_usage_grows(env):
    _sleep_env(env, iter([(100, 200), (50, 200)]))
    worker = make_worker(env, _sleep_mode_backend=SimpleNamespace(suspend=lambda level: None))
    with pytest.raises(AssertionError):
        worker.sleep()

def test_sleep_polls_on_rocm_until_deadline(env):
    sleeps = []
    _sleep_env(env, iter([(100, 200), (90, 200), (80, 200), (70, 200)]), monotonic=[0.0, 0.0, 1.0, 6.0])
    env['time'].sleep = lambda seconds: sleeps.append(seconds)
    env['current_platform'] = _platform(is_rocm=lambda : True)
    worker = make_worker(env, _sleep_mode_backend=SimpleNamespace(suspend=lambda level: None))
    with pytest.raises(AssertionError):
        worker.sleep()
    assert sleeps == [0.1, 0.1]

def test_get_sleep_mode_backend_creates_once(env, monkeypatch):
    created = []
    factory = SimpleNamespace(create_backend=lambda cfg: created.append(cfg) or 'backend')
    monkeypatch.setitem(sys.modules, 'vllm.device_allocator.sleep_mode_backend', SimpleNamespace(SleepModeBackendFactory=factory))
    cfg = SimpleNamespace(model_config='mc')
    worker = make_worker(env, vllm_config=cfg, _sleep_mode_backend=None)
    assert worker._get_sleep_mode_backend() == 'backend'
    assert worker._get_sleep_mode_backend() == 'backend'
    assert created == ['mc']

def test_wake_up_skips_weights_but_restores_kv_for_selected_tags(env):
    model = torch.nn.Module()
    model.register_buffer('w', torch.zeros(1))
    calls = []
    worker = make_worker(env, _sleep_mode_backend=SimpleNamespace(resume=lambda tags: None), model_runner=SimpleNamespace(model=model, post_kv_cache_wake_up=lambda : calls.append('kv')), _sleep_saved_buffers={'w': torch.tensor([9.0])}, _sleep_saved_draft_buffers={})
    worker.wake_up(tags=['kv_cache'])
    assert torch.equal(model.w, torch.zeros(1))
    assert worker._sleep_saved_buffers == {'w': torch.tensor([9.0])}
    assert calls == ['kv']

def test_wake_up_restores_draft_buffers(env):
    draft = torch.nn.Module()
    draft.register_buffer('d', torch.zeros(1))
    worker = make_worker(env, _sleep_mode_backend=SimpleNamespace(resume=lambda tags: None), model_runner=SimpleNamespace(model=torch.nn.Module(), get_draft_model=lambda : draft), _sleep_saved_buffers={}, _sleep_saved_draft_buffers={'d': torch.tensor([4.0])})
    worker.wake_up(tags=['weights'])
    assert torch.equal(draft.d, torch.tensor([4.0]))
    assert worker._sleep_saved_draft_buffers == {}

def test_checkpoint_hooks_delegate(env):
    calls = []
    env['checkpoint_prepare_distributed_state'] = lambda : calls.append('prepare')
    env['checkpoint_restore_distributed_state'] = lambda : calls.append('restore')
    worker = make_worker(env)
    worker.checkpoint_prepare()
    worker.checkpoint_restore()
    assert calls == ['prepare', 'restore']

def test_update_max_model_len_without_model_runner(env):
    worker = make_worker(env, model_runner=None, model_config=SimpleNamespace(max_model_len=1))
    worker.update_max_model_len(99)
    assert worker.model_config.max_model_len == 99

def test_elastic_ep_execute_forwards(env):
    executor = SimpleNamespace(execute=lambda method, *a, **k: (method, a, k))
    worker = make_worker(env, elastic_ep_executor=executor)
    assert worker.elastic_ep_execute('scale', 1, x=2) == ('scale', (1,), {'x': 2})

def test_kv_connector_metadata_absent_group(env):
    env['has_kv_transfer_group'] = lambda : False
    worker = make_worker(env)
    assert worker.get_kv_connector_handshake_metadata() is None

def test_kv_connector_metadata_no_metadata(env):
    env['has_kv_transfer_group'] = lambda : True
    env['get_kv_transfer_group'] = lambda : SimpleNamespace(get_handshake_metadata=lambda : None)
    worker = make_worker(env)
    assert worker.get_kv_connector_handshake_metadata() is None

def test_kv_connector_metadata_returns_rank_keyed_map(env):
    env['has_kv_transfer_group'] = lambda : True
    env['get_kv_transfer_group'] = lambda : SimpleNamespace(get_handshake_metadata=lambda : 'meta')
    env['get_pp_group'] = lambda : SimpleNamespace(rank_in_group=2)
    env['get_tp_group'] = lambda : SimpleNamespace(rank_in_group=5)
    worker = make_worker(env)
    assert worker.get_kv_connector_handshake_metadata() == {(2, 5): 'meta'}

def test_handle_ft_command(env):
    worker = make_worker(env, worker_sentinel=SimpleNamespace(handle_command=lambda req: ('ok', req)))
    assert worker.handle_ft_command('req') == ('ok', 'req')

def test_handle_ft_command_requires_sentinel(env):
    worker = make_worker(env, worker_sentinel=None)
    with pytest.raises(AssertionError):
        worker.handle_ft_command('req')

def _engine(**over):
    state = {'starts': 0, 'updates': 0, 'finishes': 0, 'resets': 0, 'targets': []}
    base = dict(supports_draft_weight_update=True, parse_init_info=lambda info: ('typed', info), init_transfer_engine=lambda typed: state.__setitem__('init', typed), start_weight_update=lambda : state.__setitem__('starts', state['starts'] + 1), update_weights=lambda info: state.__setitem__('updates', state['updates'] + 1), finish_weight_update=lambda : state.__setitem__('finishes', state['finishes'] + 1), reset_weight_update_target=lambda : state.__setitem__('resets', state['resets'] + 1), set_weight_update_target=lambda model, cfg: state['targets'].append((model, cfg)))
    base.update(over)
    return (SimpleNamespace(**base), state)

def test_init_weight_transfer_engine(env):
    (engine, state) = _engine()
    worker = make_worker(env, weight_transfer_engine=engine)
    worker.init_weight_transfer_engine({'rank': 0})
    assert state['init'] == ('typed', {'rank': 0})

def test_start_and_finish_weight_update(env):
    (engine, state) = _engine()
    model_runner = SimpleNamespace(reset_lora_state=lambda : state.__setitem__('lora', 1))
    worker = make_worker(env, weight_transfer_engine=engine, model_runner=model_runner, _weight_update_active=False, _weight_update_is_draft=False, vllm_config=SimpleNamespace())
    worker.start_weight_update()
    assert state['starts'] == 1
    assert worker._weight_update_active is True
    worker.update_weights({'chunk': 1})
    assert state['updates'] == 1
    worker.finish_weight_update()
    assert state['finishes'] == 1
    assert state['resets'] == 1
    assert state['lora'] == 1
    assert worker._weight_update_active is False

def test_start_draft_weight_update_sets_target(env):
    (engine, state) = _engine()
    draft = object()
    draft_cfg = object()
    model_runner = SimpleNamespace(get_draft_model=lambda : draft)
    worker = make_worker(env, weight_transfer_engine=engine, model_runner=model_runner, speculative_config=SimpleNamespace(draft_model_config=draft_cfg), _weight_update_active=False, _weight_update_is_draft=False, vllm_config=SimpleNamespace())
    worker.start_draft_weight_update()
    assert state['targets'] == [(draft, draft_cfg)]
    assert worker._weight_update_is_draft is True

def test_profile_starts_torch_profiler(env, monkeypatch):
    created = []
    env['TorchProfilerWrapper'] = lambda *a, **k: created.append(('torch', a, k)) or SimpleNamespace(start=lambda : created.append('start'))
    worker = make_worker(env, profiler=None, profiler_config=SimpleNamespace(profiler='torch'), rank=3, local_rank=1)
    monkeypatch.setitem(sys.modules, 'vllm.distributed.utils', SimpleNamespace(get_worker_rank_suffix=lambda global_rank: f'rank{global_rank}'))
    worker.profile(is_start=True, profile_prefix='p')
    assert created[0][0] == 'torch'
    assert 'start' in created

@pytest.mark.parametrize('profiler_type', ['cuda', 'proton'])
def test_profile_starts_other_wrappers(env, monkeypatch, profiler_type):
    calls = []
    env['CudaProfilerWrapper'] = lambda cfg: calls.append('cuda') or SimpleNamespace(start=lambda : calls.append('start'))
    env['ProtonProfilerWrapper'] = lambda cfg, worker_name: calls.append(('proton', worker_name)) or SimpleNamespace(start=lambda : calls.append('start'))
    worker = make_worker(env, profiler=None, profiler_config=SimpleNamespace(profiler=profiler_type), rank=0, local_rank=0)
    monkeypatch.setitem(sys.modules, 'vllm.distributed.utils', SimpleNamespace(get_worker_rank_suffix=lambda global_rank: 'suffix'))
    worker.profile(is_start=True)
    assert 'start' in calls

def test_profile_stop_without_start_warns(env):
    worker = make_worker(env, profiler=None, profiler_config=SimpleNamespace(profiler='torch'))
    worker.profile(is_start=False)
    assert any((level == 'warning' for (level, _) in env['logger'].messages))

def test_profile_stop_resets_proton_wrapper(env):
    calls = []
    profiler = SimpleNamespace(stop=lambda : calls.append('stop'))
    worker = make_worker(env, profiler=profiler, profiler_config=SimpleNamespace(profiler='proton'))
    worker.profile(is_start=False)
    assert calls == ['stop']
    assert worker.profiler is None

def _scheduler_output():
    new_req = SimpleNamespace(req_id='new', num_computed_tokens=0)
    cached = SimpleNamespace(req_ids=['gen'], num_computed_tokens=[10], is_context_phase=lambda req_id: False)
    return SimpleNamespace(scheduled_new_reqs=[new_req], scheduled_cached_reqs=cached, num_scheduled_tokens={'new': 4, 'gen': 1})

def test_annotate_profile_without_profiler(env):
    worker = make_worker(env, profiler=None)
    with worker.annotate_profile(_scheduler_output()) as value:
        assert value is None

def test_annotate_profile_not_running(env):
    profiler = SimpleNamespace(step=lambda : None, is_running=False)
    worker = make_worker(env, profiler=profiler)
    with worker.annotate_profile(_scheduler_output()) as value:
        assert value is None

def test_load_model_runs_memory_pool_and_engine(env, monkeypatch):
    calls = []
    env['set_current_vllm_config'] = lambda cfg: contextlib.nullcontext()
    engine = SimpleNamespace()
    env['WeightTransferEngineFactory'] = SimpleNamespace(create_engine=lambda *a: calls.append(('engine', a)) or engine)
    env['set_torch_threads_for_runtime'] = lambda : calls.append('threads')
    model_runner = SimpleNamespace(load_model=lambda load_dummy_weights: calls.append(('load', load_dummy_weights)), get_model=lambda : 'model')
    worker = make_worker(env, model_runner=model_runner, device=SimpleNamespace(type='npu'), vllm_config=SimpleNamespace(weight_transfer_config='wtc', model_config=SimpleNamespace()))
    with contextlib.nullcontext():
        pass
    worker._maybe_get_memory_pool_context = lambda tag: contextlib.nullcontext()
    worker._scoped_allocator_max_split = lambda max_split_size_mb: contextlib.nullcontext()
    worker.load_model(load_dummy_weights=True)
    assert ('load', True) in calls
    assert ('engine', ('wtc', worker.vllm_config, worker.device, 'model')) in calls
    assert 'threads' in calls
    assert worker.weight_transfer_engine is engine

def test_load_model_without_weight_transfer(env):
    calls = []
    model_runner = SimpleNamespace(load_model=lambda load_dummy_weights: calls.append(load_dummy_weights))
    worker = make_worker(env, model_runner=model_runner, device=SimpleNamespace(type='cuda'), vllm_config=SimpleNamespace(weight_transfer_config=None, model_config=SimpleNamespace()))
    worker._maybe_get_memory_pool_context = lambda tag: contextlib.nullcontext()
    worker._scoped_allocator_max_split = lambda max_split_size_mb: contextlib.nullcontext()
    worker.load_model()
    assert calls == [False]

def test_sample_tokens_delegates(env):
    worker = make_worker(env, model_runner=SimpleNamespace(sample_tokens=lambda g: ('out', g)))
    assert worker.sample_tokens('grammar') == ('out', 'grammar')

def test_execute_model_returns_runner_output(env):
    output = _MarkA()
    calls = []
    worker = make_worker(env, model_runner=SimpleNamespace(execute_model=lambda so, it: output), _pp_send_work=[], use_v2_model_runner=False, vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()), annotate_profile=lambda so: contextlib.nullcontext())
    env['get_pp_group'] = lambda : SimpleNamespace(is_first_rank=True)
    result = worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=3))
    assert result is output

def test_execute_model_v2_pooling_pool_fallback(env):
    pooled = _MarkB()
    pool_calls = []
    model_runner = SimpleNamespace(execute_model=lambda so, it: None, is_pooling_model=True, pool=lambda : pool_calls.append('pool') or pooled)
    worker = make_worker(env, model_runner=model_runner, _pp_send_work=[], use_v2_model_runner=True, vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()), annotate_profile=lambda so: contextlib.nullcontext())
    env['get_pp_group'] = lambda : SimpleNamespace(is_first_rank=True)
    result = worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=3))
    assert result is pooled
    assert pool_calls == ['pool']

def test_execute_model_waits_previous_pp_sends(env):
    waits = []
    handles = [SimpleNamespace(wait=lambda : waits.append(1))]
    worker = make_worker(env, model_runner=SimpleNamespace(execute_model=lambda so, it: _MarkA()), _pp_send_work=handles, use_v2_model_runner=False, vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()), annotate_profile=lambda so: contextlib.nullcontext())
    env['get_pp_group'] = lambda : SimpleNamespace(is_first_rank=True)
    worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=1))
    assert waits == [1]
    assert worker._pp_send_work == []

def test_initialize_from_config_initializes_cache_and_capturer(env):
    calls = []
    model_runner = SimpleNamespace(initialize_kv_cache=lambda cfg: calls.append(('init', cfg)), init_routed_experts_capturer=lambda : calls.append('capturer'), _init_kv_zero_meta=lambda : calls.append('zero'))
    env['ensure_kv_transfer_initialized'] = lambda cfg, kv: calls.append('kvtransfer')
    kv_config = SimpleNamespace(num_blocks=8, needs_kv_cache_zeroing=True)
    worker = make_worker(env, cache_config=SimpleNamespace(num_gpu_blocks=0), vllm_config=SimpleNamespace(aux_output_config=SimpleNamespace(enable_return_routed_experts=True)), model_config=SimpleNamespace(), model_runner=model_runner, use_v2_model_runner=False)
    worker._maybe_get_memory_pool_context = lambda tag: contextlib.nullcontext()
    worker.initialize_from_config(kv_config)
    assert worker.cache_config.num_gpu_blocks == 8
    assert calls == ['kvtransfer', ('init', kv_config), 'capturer', 'zero']