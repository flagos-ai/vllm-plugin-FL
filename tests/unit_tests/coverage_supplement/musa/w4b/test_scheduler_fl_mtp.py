"""CPU contracts for vllm_fl.worker.scheduler_fl (MTP xGrammar batching).

The upstream scheduler is not imported: the plugin class bodies execute from
source with explicit doubles for the base scheduler, xgrammar and torch.
"""
import ast
import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
SOURCE_REL = 'vllm_fl/worker/scheduler_fl.py'

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

def _node_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None
NAMES = {'_MTPStructuredOutputManager', 'AsyncSchedulerFL'}

def _exec_module(namespace):
    selected = [n for n in _TREE.body if _node_name(n) in NAMES]
    assert len(selected) == len(NAMES)
    module = ast.fix_missing_locations(ast.Module(body=[_FUTURE, *selected], type_ignores=[]))
    code = compile(module, SOURCE_REL, 'exec')
    exec(code, namespace)

class _Logger:

    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        pass

class _FakeBaseScheduler:

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

class _BatchMatcher:

    def __init__(self):
        self.calls = []

    def batch_fill_next_token_bitmask(self, matchers, bitmask, indices):
        self.calls.append((list(matchers), list(indices)))
        for (offset, index) in enumerate(indices):
            bitmask[index].fill_(100 + offset)

class _Grammar:

    def __init__(self, name='g', terminated=False, matcher='matcher', accepts=True):
        self.name = name
        self.terminated = terminated
        self.matcher = matcher
        self.accepts = accepts
        self.accepted = []
        self.rollbacks = []

    def is_terminated(self):
        return self.terminated

    def accept_tokens(self, req_id, tokens):
        self.accepted.append((req_id, list(tokens)))
        return self.accepts

    def rollback(self, n):
        self.rollbacks.append(n)

class _BitmaskBackend:

    def __init__(self):
        self.allocated = []

    def allocate_token_bitmask(self, size):
        self.allocated.append(size)
        return torch.zeros((size, 8), dtype=torch.int32)

class _BaseManager:

    def __init__(self, *, num_spec=1, max_seqs=4, backend=True):
        self.vllm_config = SimpleNamespace(num_speculative_tokens=num_spec, scheduler_config=SimpleNamespace(max_num_seqs=max_seqs))
        self.backend = _BitmaskBackend() if backend else None
        self._grammar_bitmask = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)
        self.reasoners = {}
        self.fallback_calls = []

    def grammar_bitmask(self, requests, ids, spec):
        self.fallback_calls.append((requests, list(ids), spec))
        return 'BASE'

    def _get_reasoner(self, request):
        return self.reasoners.get(id(request))

@pytest.fixture
def env(monkeypatch):
    namespace = {'Any': object, 'logger': _Logger(), '_AsyncScheduler': _FakeBaseScheduler}
    _exec_module(namespace)

    def _batch_matcher(_max_threads=None):
        return _BatchMatcher()
    fake_xgrammar = SimpleNamespace(BatchGrammarMatcher=_batch_matcher)
    monkeypatch.setitem(sys.modules, 'xgrammar', fake_xgrammar)
    namespace['xgrammar'] = fake_xgrammar
    return SimpleNamespace(**namespace)

def _requests(grammars, tokens=None):
    tokens = tokens or {}
    requests = {}
    for (req_id, grammar) in grammars.items():
        requests[req_id] = SimpleNamespace(structured_output_request=SimpleNamespace(grammar=grammar))
    return (requests, tokens)

def test_scheduler_leaves_manager_untouched_for_mtp_gt_one(env):
    manager = _BaseManager()
    scheduler = env.AsyncSchedulerFL(vllm_config=SimpleNamespace(num_speculative_tokens=2), structured_output_manager=manager)
    assert scheduler.init_kwargs['structured_output_manager'] is manager

def test_scheduler_leaves_manager_untouched_when_missing(env):
    scheduler = env.AsyncSchedulerFL(vllm_config=SimpleNamespace(num_speculative_tokens=1))
    assert 'structured_output_manager' not in scheduler.init_kwargs