"""CPU contracts for WorkerFL lifecycle: init, memory profiling, warmup.

Only the actual source bodies run; vLLM internals are explicit doubles.
"""
import ast
import contextlib
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
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
NAMES = {'_is_hybrid_kv_model', '_normalize_process_group_backend', 'WorkerFL', 'init_worker_distributed_environment'}

def _node_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
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
        for name in ('parallel_config', 'model_config', 'cache_config', 'scheduler_config', 'compilation_config', 'observability_config'):
            if not hasattr(self, name):
                setattr(self, name, getattr(self.vllm_config, name, None))

class _ElasticExecutor:

    def __init__(self, worker):
        self.worker = worker

def _platform(**over):
    base = dict(device_type='cuda', vendor_name=None, dist_backend='nccl', is_rocm=lambda : False, is_cuda_alike=lambda : False, is_cuda=lambda : False, is_xpu=lambda : False, is_cpu=lambda : False, is_sleep_mode_available=lambda : True)
    base.update(over)
    return SimpleNamespace(**base)

@pytest.fixture
def env(monkeypatch):
    ns = _NS()
    ns.update(torch=torch, nn=torch.nn, os=os, gc=__import__('gc'), time=__import__('time'), logger=_Logger(), current_platform=_platform(), envs=SimpleNamespace(VLLM_FLOAT32_MATMUL_PRECISION='highest', VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=True), nullcontext=contextlib.nullcontext, contextmanager=contextlib.contextmanager, AbstractContextManager=contextlib.AbstractContextManager, WorkerBase=_WorkerBase, CUDAGraphMode=SimpleNamespace(NONE='none'), CompilationMode=SimpleNamespace(VLLM_COMPILE='compile', NONE='none'), GiB_bytes=1 << 30, format_gib=lambda b: f'{b / float(1 << 30):.2f}', set_current_vllm_config=lambda cfg: contextlib.nullcontext(), ElasticEPScalingExecutor=_ElasticExecutor, WorkerSentinel=lambda worker: SimpleNamespace(worker=worker), ModelRunnerOutput=object, AsyncModelRunnerOutput=object, DraftTokenIds=object, WeightTransferEngine=object, NoneType=type(None), instrument=_identity_decorator, managed_inference_mode=_identity_decorator, with_gpu_sync_check=lambda fn: fn)
    _exec_module(ns)
    return ns

def make_worker(env, **attrs):
    worker = object.__new__(env['WorkerFL'])
    for (name, value) in attrs.items():
        setattr(worker, name, value)
    return worker

def _init_config(**over):
    cfg = SimpleNamespace(num_speculative_tokens=1, use_v2_model_runner=True, scheduler_config=SimpleNamespace(async_scheduling=True, scheduler_cls=None), profiler_config='profiler-config', parallel_config=SimpleNamespace(enable_fault_tolerance=False), cache_config=None)
    for (name, value) in over.items():
        setattr(cfg, name, value)
    return cfg

@pytest.fixture
def init_env(env, monkeypatch):
    created = []

    class _Executor:

        def __init__(self, worker):
            created.append(worker)
    monkeypatch.setitem(sys.modules, 'vllm.distributed.elastic_ep.elastic_execute', SimpleNamespace(ElasticEPScalingExecutor=_Executor))
    monkeypatch.setitem(sys.modules, 'vllm_fl.worker.runtime', SimpleNamespace(_initialize_fl_runtime=lambda rank: created.append(('init', rank))))
    env['_initialize_fl_runtime'] = lambda rank: created.append(('init', rank))
    env['ElasticEPScalingExecutor'] = _Executor
    env['created'] = created
    return env

def _profile_result(*, free=100, non_kv=10, consumed=5, peak=3):
    return SimpleNamespace(total_consumed=consumed, transient_peak_headroom=peak, non_kv_cache_memory=non_kv, after_profile=SimpleNamespace(free_memory=free))

def _memory_env(env, profile_result, **over):
    recorded = []

    @contextlib.contextmanager
    def _profiling(*args, **kwargs):
        recorded.append('profiling')
        yield profile_result
    env['memory_profiling'] = _profiling
    env['maybe_apply_startup_plan'] = lambda worker: recorded.append('startup_plan')
    env['reserve_mm_ipc_gpu_memory'] = lambda memory, mm, api: recorded.append(('reserve', memory, mm, api)) or 'kv'
    model_runner = SimpleNamespace(profile_run=lambda : recorded.append('profile_run'), model_memory_usage=10, profile_cudagraph_memory=lambda : 20)
    cache_config = SimpleNamespace(kv_cache_memory_bytes=over.pop('kv_cache_memory_bytes', None), gpu_memory_utilization=0.9)
    worker = make_worker(env, cache_config=cache_config, model_config=SimpleNamespace(multimodal_config='mm'), parallel_config=over.pop('parallel_config', SimpleNamespace(_api_process_count=2)), init_snapshot=SimpleNamespace(free_memory=200, total_memory=400), requested_memory=150, use_v2_model_runner=False, model_runner=model_runner, device=SimpleNamespace(type='cpu'), vllm_config=over.pop('vllm_config', SimpleNamespace()), **over)
    return (worker, recorded)

def test_determine_available_memory_detects_regression(env):
    (worker, _) = _memory_env(env, _profile_result(free=500))
    with pytest.raises(AssertionError, match='Error in memory profiling'):
        worker.determine_available_memory()

def test_determine_available_memory_musa_exhausted_budget(env):
    env['current_platform'] = _platform(device_type='musa', torch_device_fn=SimpleNamespace(mem_get_info=lambda device: (0, 100)))
    (worker, _) = _memory_env(env, _profile_result(non_kv=500))
    worker.determine_available_memory()
    assert any((level == 'error' and 'budget exhausted' in m for (level, m) in env['logger'].messages))

def test_determine_available_memory_txda_honours_explicit_bytes(env):
    env['current_platform'] = _platform(device_type='txda')
    (worker, _) = _memory_env(env, _profile_result(), kv_cache_memory_bytes=99)
    assert worker.determine_available_memory() == 99

class _Range:

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, value):
        return self.start <= value < self.end

def _warmup_env(env, monkeypatch, **over):
    calls = []

    def _dummy_run(*args, **kwargs):
        calls.append(('dummy', args, kwargs))
        return ('hidden', 'last')
    compilation_config = SimpleNamespace(mode='compile', backend='eager', compile_sizes=[4, 8], cudagraph_mode='full', cudagraph_capture_sizes=[8], get_compile_ranges=lambda : [_Range(10, 20)], compilation_time=1.5, encoder_compilation_time=0.5)
    model_runner = SimpleNamespace(_dummy_run=_dummy_run, maybe_remove_all_loras=lambda cfg: calls.append(('loras', cfg)), lora_config='lora', capture_model=lambda : calls.append('capture') or 123, is_pooling_model=False, _dummy_sampler_run=lambda **kwargs: calls.append(('sampler', kwargs)), _dummy_pooler_run=lambda hidden: calls.append(('pooler', hidden)))
    worker = make_worker(env, model_runner=model_runner, model_config=SimpleNamespace(enforce_eager=over.pop('enforce_eager', False), seed=7), vllm_config=SimpleNamespace(compilation_config=compilation_config, kernel_config=SimpleNamespace(enable_jit_warmup=over.pop('jit_warmup', False)), parallel_config=SimpleNamespace()), compilation_config=compilation_config, cache_config=SimpleNamespace(kv_cache_memory_bytes=over.pop('kv_bytes', None), gpu_memory_utilization=0.9), scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=16), device=SimpleNamespace(type=over.pop('device_type', 'cuda')), observability_config=SimpleNamespace(jit_monitor_mode='warn', jit_monitor_verbose=False), init_snapshot=SimpleNamespace(free_memory=1000, total_memory=2000), requested_memory=900, total_consumed=100, peak_activation_memory=50, available_kv_cache_memory_bytes=700, cudagraph_memory_estimate=over.pop('cudagraph_memory_estimate', 0), use_v2_model_runner=over.pop('use_v2', False), calls=calls)
    env['CompilationMode'] = SimpleNamespace(VLLM_COMPILE='compile', NONE='none')
    env['CUDAGraphMode'] = SimpleNamespace(NONE='none')
    env['kernel_warmup'] = over.pop('kernel_warmup', lambda worker: calls.append('kernel_warmup'))
    env['set_random_seed'] = lambda seed: calls.append(('seed', seed))
    env['set_torch_threads_for_runtime'] = lambda : calls.append('threads')
    env['freeze_gc_heap'] = lambda : calls.append('freeze')
    env['maybe_attach_gc_debug_callback'] = lambda : calls.append('gc_debug')
    env['enable_gpu_sync_check'] = lambda : calls.append('sync_check')
    env['activate_jit_monitor'] = lambda **kwargs: calls.append(('jit', kwargs))
    env['maybe_save_startup_plan'] = lambda worker, limit: calls.append(('startup_plan', limit))
    env['warmup_kernels'] = lambda runner, execute, sample: calls.append('warmup_kernels')
    env['get_pp_group'] = lambda : SimpleNamespace(is_last_rank=True)
    env['CompilationTimes'] = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'vllm.utils.jit_monitor', SimpleNamespace(activate=lambda **kwargs: calls.append(('jit', kwargs))))
    return (worker, calls)

def _has(calls, name):
    return any((isinstance(entry, tuple) and entry and (entry[0] == name) for entry in calls))

def _dist_env(env, monkeypatch, **over):
    calls = []
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.determinism.batch_invariant', SimpleNamespace(init_batch_invariance=lambda : calls.append('batch_invariance')))
    env['override_envs_for_eplb'] = lambda parallel, moe_backend: calls.append(('eplb', moe_backend))
    env['set_custom_all_reduce'] = lambda enabled: calls.append(('custom_ar', enabled))
    env['init_distributed_environment'] = lambda *args: calls.append(('dist', args))
    env['ensure_model_parallel_initialized'] = lambda *args: calls.append(('mp', args))
    env['ensure_ec_transfer_initialized'] = lambda cfg: calls.append('ec')
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2, tensor_parallel_size=1, pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1, distributed_timeout_seconds=over.pop('timeout', None), disable_custom_all_reduce=False), kernel_config=SimpleNamespace(moe_backend=over.pop('moe_backend', None)))
    return (cfg, calls)

def test_init_worker_distributed_environment_explicit_method(env, monkeypatch):
    env['current_platform'] = _platform()
    (cfg, calls) = _dist_env(env, monkeypatch)
    env['init_worker_distributed_environment'](cfg, 0, 'tcp://127.0.0.1:1', -1, 'gloo')
    dist_call = next((c for c in calls if c[0] == 'dist'))
    assert dist_call[1] == (2, 0, 'tcp://127.0.0.1:1', -1, 'gloo', None)

def _init_device_env(env, monkeypatch, **over):
    calls = []
    env['torch'] = SimpleNamespace(accelerator=SimpleNamespace(device_count=lambda : 8, empty_cache=lambda : calls.append('empty')), device=torch.device)
    platform = _platform(device_type=over.pop('device_type', 'cuda'), vendor_name=over.pop('vendor_name', None), set_device=lambda device: calls.append(('set_device', device)), check_if_supports_dtype=lambda dtype: calls.append(('dtype', dtype)), logical_device_id_to_visible_device_id=lambda index: index + 1, dist_backend='nccl')
    for (name, value) in over.pop('platform', {}).items():
        setattr(platform, name, value)
    env['current_platform'] = platform
    env['init_worker_distributed_environment'] = lambda *args: calls.append(('dist', args))
    env['init_workspace_manager'] = lambda device, ubatches, lanes: calls.append(('workspace', device, ubatches, lanes))
    env['request_memory'] = lambda snapshot, cache: 123
    env['report_usage_stats'] = lambda cfg: calls.append('usage')
    env['set_random_seed'] = lambda seed: calls.append(('seed', seed))
    env['_num_workspace_lanes'] = lambda cfg, v2: 1
    env['MemorySnapshot'] = lambda device: SimpleNamespace(device=device, free_memory=10)
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(enable_dbo=over.pop('enable_dbo', False)), kernel_config=SimpleNamespace(moe_backend=None))
    worker = make_worker(env, device_config=SimpleNamespace(device_type=over.pop('config_device_type', 'cuda')), parallel_config=SimpleNamespace(assigned_physical_gpu_ids=over.pop('assigned_ids', None), local_world_size=1, distributed_executor_backend='mp', data_parallel_backend=None, nnodes_within_dp=1, data_parallel_rank_local=None, data_parallel_index=0), local_rank=over.pop('local_rank', 0), rank=over.pop('rank', 0), distributed_init_method='tcp://127.0.0.1:1', model_config=SimpleNamespace(dtype='float16', seed=7), cache_config='cache-config', vllm_config=vllm_config, use_v2_model_runner=over.pop('use_v2', False))
    return (worker, calls)

def test_init_device_v1_builds_model_runner(env, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'vllm_fl.worker.model_runner', SimpleNamespace(ModelRunnerFL=lambda cfg, device: ('v1', cfg, device)))
    (worker, calls) = _init_device_env(env, monkeypatch)
    worker.init_device()
    assert worker.device.type == 'cuda'
    assert worker.device.index == 1
    assert ('set_device', worker.device) in calls
    assert ('dtype', 'float16') in calls
    assert ('seed', 7) in calls
    assert 'empty' in calls
    assert ('workspace', worker.device, 1, 1) in calls
    assert worker.model_runner[0] == 'v1'
    assert worker.init_snapshot.free_memory == 10
    assert worker.requested_memory == 123
    assert 'usage' in calls