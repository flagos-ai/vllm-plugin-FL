# Copyright (c) 2025 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl.worker.scheduler_fl``.

``_MTPStructuredOutputManager`` wraps vLLM's structured-output manager to
batch xGrammar mask filling when MTP=1.  Only the vLLM-facing boundaries
(the base manager, the ``vllm.v1.core.sched.async_scheduler`` import, and
torch tensors) are faked; the xGrammar calls run against the real
``xgrammar`` package where available.

Branch matrix under test:

* early-return to the base manager: MTP != 1, single request, reasoner;
* normal path: bitmask allocation, current/bonus rows, batch fill,
  accept/rollback of draft tokens, slicing to the used rows;
* fallback to the base manager: grammar without a matcher, terminated
  grammars only, missing backend;
* ``AsyncSchedulerFL`` constructor wiring of the wrapped manager.
"""

import importlib
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

try:
    import xgrammar as xgr
except ImportError:  # pragma: no cover - CI image without xgrammar
    xgr = None

xgrammar_required = pytest.mark.skipif(xgr is None, reason="xgrammar not installed")


def _install_vllm_stubs():
    """Register the minimal fake vllm tree scheduler_fl imports."""
    entries = {}

    def put(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        entries[name] = module
        return module

    put("vllm")
    put("vllm.logger", init_logger=logging.getLogger)
    put("vllm.v1")
    put("vllm.v1.core")
    put("vllm.v1.core.sched")

    class _AsyncScheduler:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    put("vllm.v1.core.sched.async_scheduler", AsyncScheduler=_AsyncScheduler)
    return entries


class _GrammarStub:
    """Request-side grammar handle wrapping a real xgrammar matcher.

    Mirrors vLLM's ``StructuredOutputRequest.grammar``: exposes ``matcher``
    when fillable, plus the accept/rollback API the manager drives.
    """

    def __init__(self, matcher=None, terminated=False):
        self.matcher = matcher
        self._terminated = terminated
        self.accepted = []
        self.rollbacks = []

    def is_terminated(self):
        return self._terminated

    def accept_tokens(self, req_id, tokens):
        self.accepted.append((req_id, list(tokens)))
        return True

    def rollback(self, count):
        self.rollbacks.append(count)


class _BaseManager:
    """Fake of vLLM's structured output manager (external boundary)."""

    def __init__(self, num_speculative_tokens=1, max_num_seqs=16):
        self.vllm_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        )
        self.backend = SimpleNamespace(
            allocate_token_bitmask=lambda size: xgr.allocate_token_bitmask(size, _VOCAB)
        )
        self._grammar_bitmask = None
        self._full_mask = _new_full_mask()
        self.base_calls = []
        self.reasoners = {}
        self.custom_attr = "delegated"

    def _get_reasoner(self, request):
        return self.reasoners.get(getattr(request, "req_id", None))

    def grammar_bitmask(self, requests, request_ids, scheduled_tokens):
        self.base_calls.append((tuple(request_ids), dict(scheduled_tokens)))
        return "BASE_RESULT"


def _new_full_mask():
    """A 0-dim all-ones tensor like the base manager's ``_full_mask``."""
    import torch

    return torch.full((), -1, dtype=torch.int32)


_VOCAB = 32


def _make_matcher():
    """Build a real xgrammar matcher over a two-token enum grammar."""
    vocab = ['"a"', '"b"'] + [f"pad{i}" for i in range(_VOCAB - 2)]
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(vocab))
    compiled = compiler.compile_json_schema('{"enum": ["a", "b"]}')
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def _make_requests(grammar_by_id):
    """Wrap grammar stubs in the request shape ``grammar_bitmask`` reads."""
    requests = {}
    for req_id, grammar in grammar_by_id.items():
        request = SimpleNamespace(
            req_id=req_id,
            structured_output_request=SimpleNamespace(grammar=grammar),
        )
        requests[req_id] = request
    return requests


@pytest.fixture
def scheduler_module(fl_sys_modules):
    """Import the real scheduler_fl module against installed vllm stubs."""
    entries = _install_vllm_stubs()
    yield importlib.import_module("vllm_fl.worker.scheduler_fl")
    for name in entries:
        sys.modules.pop(name, None)


class TestFallbackBranches:
    """Every guard clause must delegate to the base manager."""

    # _MTPStructuredOutputManager.__init__ imports xgrammar unconditionally,
    # so even the pure-delegation branches below need the package present.
    @xgrammar_required
    def test_num_speculative_tokens_not_one_delegates(self, scheduler_module):
        """MTP != 1 skips the plugin batching path entirely."""
        base = _BaseManager(num_speculative_tokens=2)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        result = manager.grammar_bitmask({}, ["r1", "r2"], {})
        assert result == "BASE_RESULT"
        assert base.base_calls == [(("r1", "r2"), {})]

    @xgrammar_required
    def test_single_request_delegates(self, scheduler_module):
        """A single structured request has no batching opportunity."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        result = manager.grammar_bitmask({}, ["r1"], {})
        assert result == "BASE_RESULT"
        assert len(base.base_calls) == 1

    @xgrammar_required
    def test_reasoner_request_delegates(self, scheduler_module):
        """Any reasoning-mode request falls back to the base manager."""
        base = _BaseManager(num_speculative_tokens=1)
        base.reasoners["r1"] = object()
        manager = scheduler_module._MTPStructuredOutputManager(base)
        requests = _make_requests(
            {"r1": _GrammarStub(matcher=_make_matcher()), "r2": _GrammarStub()}
        )
        result = manager.grammar_bitmask(requests, ["r1", "r2"], {})
        assert result == "BASE_RESULT"
        assert base.base_calls == [(("r1", "r2"), {})]

    @xgrammar_required
    def test_missing_backend_delegates(self, scheduler_module):
        """No backend and no allocated bitmask falls back to the base."""
        base = _BaseManager(num_speculative_tokens=1)
        base.backend = None
        manager = scheduler_module._MTPStructuredOutputManager(base)
        requests = _make_requests({"r1": _GrammarStub(), "r2": _GrammarStub()})
        result = manager.grammar_bitmask(requests, ["r1", "r2"], {})
        assert result == "BASE_RESULT"
        assert base.base_calls == [(("r1", "r2"), {})]

    @xgrammar_required
    def test_grammar_without_matcher_delegates_without_accepting(
        self, scheduler_module
    ):
        """A grammar lacking a matcher aborts before any accept_tokens."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        g1, g2 = _GrammarStub(), _GrammarStub()
        requests = _make_requests({"r1": g1, "r2": g2})
        result = manager.grammar_bitmask(requests, ["r1", "r2"], {})
        assert result == "BASE_RESULT"
        assert g1.accepted == [] and g2.accepted == []


class TestNormalPath:
    """The plugin-native batch fill path (MTP=1, multiple plain requests)."""

    @xgrammar_required
    def test_bonus_rows_fills_mask_and_slices(self, scheduler_module):
        """No draft tokens: one bonus row per request, sliced from the pool."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        requests = _make_requests(
            {
                "r1": _GrammarStub(matcher=_make_matcher()),
                "r2": _GrammarStub(matcher=_make_matcher()),
            }
        )
        result = manager.grammar_bitmask(requests, ["r1", "r2"], {})
        assert result.__class__.__name__ == "ndarray"
        assert base.base_calls == []
        # Bitmask pool was allocated with max_num_seqs * 2 rows.
        assert base._grammar_bitmask.shape[0] == 32
        # Only the two used rows are returned.
        assert result.shape == (2, 1)

    @xgrammar_required
    def test_current_and_bonus_rows_ordered(self, scheduler_module):
        """A draft token adds a current row before that request's bonus row."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        g1 = _GrammarStub(matcher=_make_matcher())
        g2 = _GrammarStub(matcher=_make_matcher())
        requests = _make_requests({"r1": g1, "r2": g2})
        result = manager.grammar_bitmask(requests, ["r1", "r2"], {"r1": [5]})
        # r1 current + r1 bonus + r2 bonus = 3 rows.
        assert result.shape == (3, 1)
        assert g1.accepted == [("r1", [5])]
        # The accepted draft token is rolled back after the bonus fill.
        assert g1.rollbacks == [1]

    @xgrammar_required
    def test_minus_one_draft_token_is_never_accepted(self, scheduler_module):
        """A -1 draft token means 'no token' and must not be accepted."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        g1 = _GrammarStub(matcher=_make_matcher())
        g2 = _GrammarStub(matcher=_make_matcher())
        requests = _make_requests({"r1": g1, "r2": g2})
        result = manager.grammar_bitmask(
            requests, ["r1", "r2"], {"r1": [-1], "r2": [-1]}
        )
        assert g1.accepted == [] and g2.accepted == []
        assert g1.rollbacks == [] and g2.rollbacks == []
        # Two current rows (empty after -1 handling) still yield bonus rows.
        assert result.shape == (4, 1)

    @xgrammar_required
    def test_terminated_grammar_row_filled_with_full_mask(self, scheduler_module):
        """Terminated grammars get the all-ones full mask row."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        requests = _make_requests(
            {"r1": _GrammarStub(terminated=True), "r2": _GrammarStub(terminated=True)}
        )
        result = manager.grammar_bitmask(requests, ["r1", "r2"], {})
        assert base.base_calls == []
        assert result.shape == (2, 1)
        assert result[0].tolist() == [-1]

    @xgrammar_required
    def test_bitmask_reused_across_calls(self, scheduler_module):
        """The allocated bitmask survives across calls (allocated once)."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        requests = _make_requests(
            {
                "r1": _GrammarStub(matcher=_make_matcher()),
                "r2": _GrammarStub(matcher=_make_matcher()),
            }
        )
        manager.grammar_bitmask(requests, ["r1", "r2"], {})
        first = base._grammar_bitmask
        manager.grammar_bitmask(requests, ["r1", "r2"], {})
        assert base._grammar_bitmask is first

    @xgrammar_required
    def test_real_matcher_bitmask_rows_differ_from_full_mask(self, scheduler_module):
        """Live grammars produce grammar-specific (not all-ones) rows."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        requests = _make_requests(
            {
                "r1": _GrammarStub(matcher=_make_matcher()),
                "r2": _GrammarStub(matcher=_make_matcher()),
            }
        )
        result = manager.grammar_bitmask(requests, ["r1", "r2"], {})
        # Enum grammar allows exactly the two string tokens (bits 0, 1).
        assert result[0].tolist() == [3]
        assert result[1].tolist() == [3]


class TestDelegation:
    """Attribute access proxies through to the wrapped base manager."""

    @xgrammar_required
    def test_getattr_forwards_unknown_names(self, scheduler_module):
        """Unknown attributes resolve against the base manager."""
        base = _BaseManager(num_speculative_tokens=1)
        manager = scheduler_module._MTPStructuredOutputManager(base)
        assert manager.custom_attr == "delegated"
        base.custom_attr = "updated"
        assert manager.custom_attr == "updated"


class TestAsyncSchedulerFL:
    """Constructor wiring of the plugin async scheduler."""

    @pytest.fixture(autouse=True)
    def _stubs(self, fl_sys_modules):
        """Install the vllm stub tree and import the module for each test."""
        self._entries = _install_vllm_stubs()
        importlib.import_module("vllm_fl.worker.scheduler_fl")
        yield
        for name in self._entries:
            sys.modules.pop(name, None)

    def _reload_with_capture(self):
        """Reload scheduler_fl with a base scheduler capturing kwargs."""
        captured = {}

        class _CapAsyncScheduler:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

        sys.modules[
            "vllm.v1.core.sched.async_scheduler"
        ].AsyncScheduler = _CapAsyncScheduler
        module = importlib.reload(sys.modules["vllm_fl.worker.scheduler_fl"])
        return module, captured

    @xgrammar_required
    def test_mtp1_wraps_structured_output_manager(self):
        """num_speculative_tokens=1 replaces the manager with the wrapper."""
        module, captured = self._reload_with_capture()
        config = SimpleNamespace(num_speculative_tokens=1)
        base_manager = _BaseManager()
        module.AsyncSchedulerFL(
            vllm_config=config, structured_output_manager=base_manager
        )
        wrapped = captured["structured_output_manager"]
        assert isinstance(wrapped, module._MTPStructuredOutputManager)
        assert wrapped._base is base_manager

    def test_non_mtp1_passes_manager_through(self):
        """num_speculative_tokens != 1 keeps the original manager."""
        module, captured = self._reload_with_capture()
        config = SimpleNamespace(num_speculative_tokens=2)
        base_manager = _BaseManager()
        module.AsyncSchedulerFL(
            vllm_config=config, structured_output_manager=base_manager
        )
        assert captured["structured_output_manager"] is base_manager

    def test_missing_manager_kwarg_leaves_kwargs_alone(self):
        """Without a manager kwarg the constructor must not invent one."""
        module, captured = self._reload_with_capture()
        config = SimpleNamespace(num_speculative_tokens=1)
        module.AsyncSchedulerFL(vllm_config=config, other=1)
        assert "structured_output_manager" not in captured
        assert captured["other"] == 1

    def test_missing_vllm_config_kwarg_leaves_kwargs_alone(self):
        """Without vllm_config the manager is left untouched."""
        module, captured = self._reload_with_capture()
        base_manager = _BaseManager()
        module.AsyncSchedulerFL(structured_output_manager=base_manager)
        assert captured["structured_output_manager"] is base_manager
