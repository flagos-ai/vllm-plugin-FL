"""CPU contracts for vllm_fl.worker.nvidia_worker IO-dump wrappers."""
import ast
import contextlib
import os
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
import pytest
SOURCE_REL = 'vllm_fl/worker/nvidia_worker.py'

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
NAMES = {'_NATIVE_RUNNER_IO_METHODS', '_install_native_runner_io_methods', '_call_native_worker_with_managed_inference', 'NvidiaWorkerFL'}

def _node_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
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

class _NS(dict):

    def __missing__(self, key):
        value = _Missing()
        self[key] = value
        return value

class _Logger:

    def __init__(self):
        self.messages = []

    def info(self, msg, *args):
        self.messages.append(msg % args if args else msg)

def _identity_decorator(*args, **kwargs):

    def decorate(fn):
        return fn
    return decorate

def _native_sample(self, grammar_output):
    self.calls.append(('sample', grammar_output))
    return self.next_output

def _native_execute(self, scheduler_output):
    self.calls.append(('execute', scheduler_output))
    return self.next_output

def _wrap(fn):

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper.__wrapped__ = fn
    return wrapper

class _NativeGPUWorker:
    sample_tokens = _wrap(_native_sample)
    execute_model = _wrap(_native_execute)

    def __init__(self, *args, **kwargs):
        self.rank = kwargs.get('rank', 0)
        self.local_rank = kwargs.get('local_rank', 0)
        self.calls = []
        self.next_output = 'native-output'
        self.model_runner = SimpleNamespace()

    def init_device(self):
        self.device = getattr(self, 'device', 'device')
        self.calls = getattr(self, 'calls', [])
        self.calls.append(('init_device',))

@pytest.fixture
def env(monkeypatch):
    namespace = _NS(logger=_Logger(), NativeGPUWorker=_NativeGPUWorker, managed_inference_mode=_identity_decorator, MethodType=MethodType, Any=object)
    _exec_module(namespace)
    plugin_calls = []
    namespace['_initialize_fl_runtime'] = lambda rank: plugin_calls.append(rank)
    namespace['plugin_calls'] = plugin_calls
    namespace['io_dumper'] = SimpleNamespace(init_io_dump_from_env=lambda enforce_eager: plugin_calls.append(('init_io', enforce_eager)), is_dump_enabled=lambda : True, register_io_module_hooks=lambda model: plugin_calls.append(('hooks', model)), advance_io_step=lambda : plugin_calls.append('advance'))
    monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch.io_dumper', namespace['io_dumper'])
    return namespace

def use_io_dumper(env, monkeypatch, **overrides):
    module = SimpleNamespace(init_io_dump_from_env=lambda enforce_eager: env['plugin_calls'].append(('init_io', enforce_eager)), is_dump_enabled=lambda : overrides.get('enabled', True), register_io_module_hooks=lambda model: env['plugin_calls'].append(('hooks', model)), advance_io_step=lambda : env['plugin_calls'].append('advance'))
    monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch.io_dumper', module)
    return module

def make_worker(env, **attrs):
    worker = object.__new__(env['NvidiaWorkerFL'])
    for (name, value) in attrs.items():
        setattr(worker, name, value)
    return worker

class _Runner:
    execute_model = _wrap(_native_sample)
    sample_tokens = _wrap(_native_sample)
    pool = _wrap(_native_sample)

    def __init__(self):
        self.calls = []
        self.next_output = 'native-output'

def test_install_runner_io_methods_skips_missing_methods(env):
    runner = SimpleNamespace(sample_tokens=None)
    env['_install_native_runner_io_methods'](runner)

def test_init_device_when_qsa_patch_not_applicable(env, monkeypatch):
    monkeypatch.setitem(sys.modules, 'vllm_fl.patches.qsa_cache_hint', SimpleNamespace(patch_qsa_cache_hint=lambda model_config, device: False))
    worker = make_worker(env, model_config='mc', device='dev')
    worker.init_device()
    assert env['logger'].messages == []

def test_ensure_io_dump_init_is_idempotent_when_done(env):
    worker = make_worker(env, _fl_io_dump_initialized=True)
    worker._ensure_io_dump_initialized()
    assert env['plugin_calls'] == []