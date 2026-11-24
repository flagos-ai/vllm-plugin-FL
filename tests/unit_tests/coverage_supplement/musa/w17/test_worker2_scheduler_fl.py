# SPDX-License-Identifier: Apache-2.0
"""Real-import contracts for the remaining scheduler_fl xGrammar branches.

``vllm_fl.worker.scheduler_fl`` is imported for real.  The MTP manager is built
with ``object.__new__`` around explicit doubles, and the scheduler wrapping is
checked by patching the real upstream base ``__init__`` (so ``super()`` resolves
to a double without reimplementing the method).
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _ensure_pkg(monkeypatch, name):
    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _install(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _BatchMatcher:
    def __init__(self):
        self.calls = []

    def batch_fill_next_token_bitmask(self, matchers, bitmask, indices):
        self.calls.append((list(matchers), list(indices)))
        for offset, index in enumerate(indices):
            bitmask[index].fill_(200 + offset)


class _Grammar:
    def __init__(self, name="g", terminated=False, matcher="matcher", accepts=True):
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

    def rollback(self, count):
        self.rollbacks.append(count)


class _Backend:
    def __init__(self, max_seqs):
        self.max_seqs = max_seqs
        self.allocated = []

    def allocate_token_bitmask(self, size):
        self.allocated.append(size)
        return torch.zeros((size, 8), dtype=torch.int32)


class _Base:
    def __init__(self, *, num_spec=1, max_seqs=2, backend=True):
        self.vllm_config = SimpleNamespace(
            num_speculative_tokens=num_spec,
            scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
        )
        self.backend = _Backend(max_seqs) if backend else None
        self._grammar_bitmask = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)
        self.reasoners = {}
        self.fallback_calls = []

    def grammar_bitmask(self, requests, ids, tokens):
        self.fallback_calls.append((requests, list(ids), tokens))
        return "BASE"

    def _get_reasoner(self, request):
        return self.reasoners.get(id(request))


def _requests(grammars, tokens=None):
    requests = {}
    for req_id, grammar in grammars.items():
        requests[req_id] = SimpleNamespace(
            structured_output_request=SimpleNamespace(grammar=grammar)
        )
    return requests, tokens or {}


def _manager(module, base, matcher=None):
    obj = object.__new__(module._MTPStructuredOutputManager)
    obj._base = base
    obj._batch_matcher = matcher or _BatchMatcher()
    obj._logged_active = False
    return obj


def test_full_bitmask_is_not_cropped_when_rows_match():
    import vllm_fl.worker.scheduler_fl as s

    base = _Base(max_seqs=2)
    grammar_a, grammar_b = _Grammar("a"), _Grammar("b")
    requests, tokens = _requests({"a": grammar_a, "b": grammar_b})
    tokens = {"a": [7], "b": [8]}
    manager = _manager(s, base)
    result = manager.grammar_bitmask(requests, ["a", "b"], tokens)

    assert base.backend.allocated == [4]
    assert result.shape == (4, 8)
    assert base._grammar_bitmask.shape == (4, 8)
    assert manager._logged_active is True


def test_terminated_grammar_with_spec_token_stays_masked():
    import vllm_fl.worker.scheduler_fl as s

    base = _Base(max_seqs=2)
    done = _Grammar("done", terminated=True, matcher=None)
    live = _Grammar("live")
    requests, tokens = _requests({"done": done, "live": live})
    tokens = {"done": [5]}
    manager = _manager(s, base)
    result = manager.grammar_bitmask(requests, ["done", "live"], tokens)

    # done: spec row 0 + bonus row 1 both filled with the full mask; live bonus row 2
    assert result.shape == (3, 8)
    assert int(result[0, 0]) == -1
    assert int(result[1, 0]) == -1
    assert int(result[2, 0]) == 200
    assert done.accepted == []
    assert done.rollbacks == []


def test_more_than_one_spec_token_is_asserted():
    import vllm_fl.worker.scheduler_fl as s

    base = _Base(max_seqs=2)
    grammar_a, grammar_b = _Grammar("a"), _Grammar("b")
    requests, _ = _requests({"a": grammar_a, "b": grammar_b})
    manager = _manager(s, base)
    with pytest.raises(AssertionError):
        manager.grammar_bitmask(requests, ["a", "b"], {"a": [1, 2]})


@pytest.mark.parametrize(
    "num_spec,manager_present,vllm_present",
    [(1, True, True), (2, True, True), (1, False, True), (1, True, False)],
)
def test_scheduler_wraps_manager_only_for_mtp_one(
    monkeypatch, num_spec, manager_present, vllm_present
):
    import vllm_fl.worker.scheduler_fl as s

    _install(
        monkeypatch,
        "xgrammar",
        BatchGrammarMatcher=lambda max_threads=None: _BatchMatcher(),
    )
    base_cls = s.AsyncSchedulerFL.__mro__[1]
    recorded = {}

    def fake_base_init(self, *args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(base_cls, "__init__", fake_base_init)

    manager = _Base()
    kwargs = {}
    if manager_present:
        kwargs["structured_output_manager"] = manager
    if vllm_present:
        kwargs["vllm_config"] = SimpleNamespace(num_speculative_tokens=num_spec)

    scheduler = s.AsyncSchedulerFL.__new__(s.AsyncSchedulerFL)
    s.AsyncSchedulerFL.__init__(scheduler, **kwargs)

    wrapped = recorded["kwargs"].get("structured_output_manager")
    should_wrap = num_spec == 1 and manager_present and vllm_present
    if should_wrap:
        assert isinstance(wrapped, s._MTPStructuredOutputManager)
        assert wrapped._base is manager
    else:
        if manager_present:
            assert wrapped is manager
        else:
            assert "structured_output_manager" not in recorded["kwargs"]
