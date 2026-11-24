# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl/worker/scheduler_fl.py``.

CPU-only checks with xgrammar stubs and a fake base structured-output
manager. Covers:

* ``_MTPStructuredOutputManager`` delegation: constructor wiring,
  ``__getattr__`` passthrough, and the fast-path guard conditions
  (MTP != 1, single request, reasoning requests).
* ``grammar_bitmask`` fast path: dual-row layout (current + bonus), batch
  matcher fallback when a grammar has no matcher, rollback of advanced
  grammars when bonus filling fails, and the truncated-bitmask return.
* ``AsyncSchedulerFL.__init__`` wrapping: substitution only for MTP=1 with
  a manager, and passthrough otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.supplementary]

sf = pytest.importorskip("vllm_fl.worker.scheduler_fl")


class FakeGrammar:
    """xgrammar grammar double with per-instance call recording."""

    def __init__(self, name="g", terminated=False, has_matcher=True):
        self.name = name
        self._terminated = terminated
        self._has_matcher = has_matcher
        self.matched = []
        self.accepted = []
        self.rollbacks = []

    def is_terminated(self):
        return self._terminated

    @property
    def matcher(self):
        if not self._has_matcher:
            return None
        return ("matcher", self.name)

    def accept_tokens(self, req_id, tokens):
        self.accepted.append((req_id, tuple(tokens)))
        return True

    def rollback(self, n):
        self.rollbacks.append(n)


class FakeBitmask:
    """Tensor-like bitmask rows with fill_ and slice support."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.fills = []

    def __getitem__(self, index):
        if isinstance(index, int):
            return NS(
                fill_=lambda value, index=index: self.fills.append((index, value))
            )
        assert isinstance(index, slice)
        assert index.step is None
        assert index.start is None or index.start >= 0
        assert index.stop is None or index.stop >= 0
        sub = FakeBitmask(self.rows[index])
        # Keep the fill recorder of the parent so fills stay observable.
        sub.fills = self.fills
        return sub

    @property
    def shape(self):
        return (len(self.rows),)

    def numpy(self):
        return ("bitmask", len(self.rows))


class FakeManager:
    """Base structured-output manager double."""

    def __init__(self, num_speculative_tokens=1, reasoners=(), base_bitmask=None):
        self.vllm_config = NS(
            num_speculative_tokens=num_speculative_tokens,
            scheduler_config=NS(max_num_seqs=16),
        )
        self._reasoners = (
            dict(reasoners) if hasattr(reasoners, "items") else set(reasoners)
        )
        self._grammar_bitmask = base_bitmask
        self.backend = Mock() if base_bitmask is None else object()
        self.base_calls = []
        self._full_mask = "full-mask"

    def _get_reasoner(self, request):
        # Production receives the request object; our tests key the
        # reasoner table by request identity for simplicity.
        return self._reasoners.get(request, None) if self._reasoners else None

    def grammar_bitmask(self, requests, ids, scheduled):
        self.base_calls.append((requests, tuple(ids), dict(scheduled)))
        return "base-result"

    def allocate_token_bitmask(self, n):
        return FakeBitmask([[0] * 4 for _ in range(n)])


@pytest.fixture()
def fake_xgrammar(monkeypatch):
    """Install a recording xgrammar module double."""
    calls = []

    class FakeBatchMatcher:
        def __init__(self, max_threads=1):
            self.max_threads = max_threads

        def batch_fill_next_token_bitmask(self, matchers, bitmask, indices):
            calls.append(("fill", tuple(matchers), bitmask, list(indices)))

    fake_module = NS(BatchGrammarMatcher=FakeBatchMatcher)
    import sys

    monkeypatch.setitem(sys.modules, "xgrammar", fake_module)
    return calls


def _manager(num_spec=1, **kw):
    return FakeManager(num_speculative_tokens=num_spec, **kw)


# ---------------------------------------------------------------------------
# Construction and delegation
# ---------------------------------------------------------------------------


def test_wraps_manager_for_mtp1(fake_xgrammar):
    base = _manager(num_spec=1)
    wrapped = sf._MTPStructuredOutputManager(base)
    assert wrapped._base is base
    # First activation is logged once.
    assert wrapped._logged_active is False


def test_getattr_delegates_to_base(fake_xgrammar):
    base = _manager(num_spec=1)
    base.custom_attribute = "value"
    wrapped = sf._MTPStructuredOutputManager(base)
    assert wrapped.custom_attribute == "value"
    assert wrapped.vllm_config is base.vllm_config


# ---------------------------------------------------------------------------
# Guard conditions falling back to the base implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_spec", [0, 2, 4])
def test_non_mtp1_delegates_to_base(fake_xgrammar, num_spec):
    base = _manager(num_spec=num_spec)
    wrapped = sf._MTPStructuredOutputManager(base)
    out = wrapped.grammar_bitmask(
        {"r0": object()}, ["r0", "r1"], {"r0": [1]}
    )
    assert out == "base-result"
    assert len(base.base_calls) == 1


def test_single_request_delegates_to_base(fake_xgrammar):
    base = _manager(num_spec=1)
    wrapped = sf._MTPStructuredOutputManager(base)
    out = wrapped.grammar_bitmask({"r0": object()}, ["r0"], {"r0": [1]})
    assert out == "base-result"


def test_reasoning_request_delegates_to_base(fake_xgrammar):
    req0, req1 = object(), object()
    base = _manager(num_spec=1, reasoners={req0: "reasoner"})
    wrapped = sf._MTPStructuredOutputManager(base)
    out = wrapped.grammar_bitmask(
        {"r0": req0, "r1": req1}, ["r0", "r1"], {"r0": [1]}
    )
    assert out == "base-result"


def test_missing_grammar_bitmask_allocates_via_backend(fake_xgrammar):
    calls = []
    backend = Mock(
        allocate_token_bitmask=lambda n: calls.append(n)
        or FakeBitmask([[0] * 4 for _ in range(n)])
    )
    base = _manager(num_spec=1)
    base.backend = backend
    base._grammar_bitmask = None
    wrapped = sf._MTPStructuredOutputManager(base)

    # Two structured requests with spec tokens take the fast path and
    # allocate the bitmask once: max_num_seqs(16) * 2 rows.
    req0 = NS(structured_output_request=NS(grammar=FakeGrammar()))
    req1 = NS(structured_output_request=NS(grammar=FakeGrammar()))
    out = wrapped.grammar_bitmask(
        {"r0": req0, "r1": req1}, ["r0", "r1"], {"r0": [3], "r1": [2]}
    )
    assert calls == [32]
    assert isinstance(out, tuple)


def test_missing_bitmask_without_backend_delegates(fake_xgrammar):
    base = _manager(num_spec=1)
    base.backend = None
    base._grammar_bitmask = None
    wrapped = sf._MTPStructuredOutputManager(base)
    req0 = NS(structured_output_request=NS(grammar=FakeGrammar()))
    req1 = NS(structured_output_request=NS(grammar=FakeGrammar()))
    out = wrapped.grammar_bitmask(
        {"r0": req0, "r1": req1}, ["r0", "r1"], {"r0": [3], "r1": [2]}
    )
    assert out == "base-result"


# ---------------------------------------------------------------------------
# Fast path: dual-row layout
# ---------------------------------------------------------------------------


def _fast_manager(**kw):
    base = _manager(num_spec=1, **kw)
    base._grammar_bitmask = FakeBitmask([[0] * 4 for _ in range(8)])
    return base


def _req(grammar):
    return NS(structured_output_request=NS(grammar=grammar))


def test_fast_path_fills_current_and_bonus_rows(fake_xgrammar):
    calls = fake_xgrammar
    base = _fast_manager()
    g0, g1, g2 = FakeGrammar("a"), FakeGrammar("b"), FakeGrammar("c")
    wrapped = sf._MTPStructuredOutputManager(base)

    requests = {
        "r0": _req(g0),
        "r1": _req(g1),
        "r2": _req(g2),
    }
    # r0 and r2 have scheduled spec tokens; r1 does not (bonus only).
    scheduled = {"r0": [7], "r1": [], "r2": [9]}
    out = wrapped.grammar_bitmask(requests, ["r0", "r1", "r2"], scheduled)

    # Row layout: r0 current=0 bonus=1; r1 bonus=2; r2 current=3 bonus=4.
    assert calls[0] == (
        "fill",
        (("matcher", "a"), ("matcher", "c")),
        base._grammar_bitmask,
        [0, 3],
    )
    assert calls[1] == (
        "fill",
        (("matcher", "a"), ("matcher", "b"), ("matcher", "c")),
        base._grammar_bitmask,
        [1, 2, 4],
    )
    # Accepts: r0 accepted token 7, r2 accepted token 9.
    assert g0.accepted == [("r0", (7,))]
    assert g1.accepted == []
    assert g2.accepted == [("r2", (9,))]
    # Both advanced grammars rolled back after the bonus fill.
    assert g0.rollbacks == [1]
    assert g2.rollbacks == [1]
    # Returned bitmask is truncated to the 5 used rows.
    assert out == ("bitmask", 5)
    # Second call reuses the allocated bitmask (no new allocation).
    assert wrapped.grammar_bitmask(requests, ["r0", "r1", "r2"], scheduled) == (
        "bitmask", 5,
    )
    # Activation warning is logged exactly once.
    assert wrapped._logged_active is True


def test_fast_path_terminated_grammar_fills_full_mask(fake_xgrammar):
    calls = fake_xgrammar
    base = _fast_manager()
    live = FakeGrammar("live")
    done = FakeGrammar("done", terminated=True)
    wrapped = sf._MTPStructuredOutputManager(base)

    requests = {"r0": _req(live), "r1": _req(done)}
    scheduled = {"r0": [1], "r1": []}
    wrapped.grammar_bitmask(requests, ["r0", "r1"], scheduled)

    # The terminated grammar contributes no matcher to either fill.
    fill_matchers = [call[1] for call in calls]
    assert (("matcher", "done"),) not in fill_matchers
    assert all("done" not in str(matchers) for matchers in fill_matchers)
    # Terminated grammar never has tokens accepted.
    assert done.accepted == []
    assert done.rollbacks == []


def test_fast_path_without_matcher_falls_back_on_current(fake_xgrammar):
    base = _fast_manager()
    grammar = FakeGrammar("nomatch", has_matcher=False)
    wrapped = sf._MTPStructuredOutputManager(base)
    requests = {"r0": _req(grammar)}
    out = wrapped.grammar_bitmask({"r0": requests["r0"]}, ["r0"], {"r0": [5]})
    # batch_fill without a matcher returns False -> base fallback.
    assert out == "base-result"
    # No tokens were accepted before the fallback.
    assert grammar.accepted == []


def test_fast_path_bonus_failure_rolls_back_and_falls_back(fake_xgrammar):
    calls = fake_xgrammar
    base = _fast_manager()
    # Current fill succeeds; the bonus fill fails because the grammar loses
    # its matcher mid-flight. Model that with a grammar whose matcher is
    # present for current rows but gone for bonus rows.
    class FlakyGrammar(FakeGrammar):
        def __init__(self):
            super().__init__("flaky")
            self._matcher_accesses = 0

        @property
        def matcher(self):
            self._matcher_accesses += 1
            # First access (current-row fill) yields a matcher; the second
            # (bonus-row fill) loses it and forces the fallback.
            if self._matcher_accesses == 1:
                return ("matcher", "flaky")
            return None

    grammar = FlakyGrammar()
    other = FakeGrammar("other")
    wrapped = sf._MTPStructuredOutputManager(base)
    requests = {"r0": _req(grammar), "r1": _req(other)}
    out = wrapped.grammar_bitmask(requests, ["r0", "r1"], {"r0": [4]})
    # Bonus fill failed -> advanced grammar rolled back -> base fallback.
    assert out == "base-result"
    assert grammar.rollbacks == [1]
    # The accept still happened before the failure.
    assert grammar.accepted == [("r0", (4,))]
    # The non-flaky grammar was never accepted (no scheduled tokens).
    assert other.accepted == []


def test_fast_path_deduplicates_bonus_without_spec_tokens(fake_xgrammar):
    calls = fake_xgrammar
    base = _fast_manager()
    g0 = FakeGrammar("a")
    g1 = FakeGrammar("b")
    wrapped = sf._MTPStructuredOutputManager(base)
    requests = {"r0": _req(g0), "r1": _req(g1)}
    # No request has scheduled spec tokens: current fill is empty and only
    # the bonus rows fill.
    wrapped.grammar_bitmask(requests, ["r0", "r1"], {})
    # No current rows exist, so exactly one fill happens: the bonus fill.
    assert len(calls) == 1
    assert calls[0][1] == (("matcher", "a"), ("matcher", "b"))
    assert calls[0][3] == [0, 1]
    # No accepts because no scheduled spec tokens.
    assert g0.accepted == [] and g1.accepted == []


def test_fast_path_sentinel_minus_one_not_accepted(fake_xgrammar):
    base = _fast_manager()
    grammar = FakeGrammar("sentinel")
    wrapped = sf._MTPStructuredOutputManager(base)
    requests = {"r0": _req(grammar)}
    # A -1 draft token means "no draft": the accept is skipped.
    wrapped.grammar_bitmask(requests, ["r0"], {"r0": [-1]})
    assert grammar.accepted == []
    assert grammar.rollbacks == []


def test_fast_path_multiple_tokens_per_request_asserts(fake_xgrammar):
    base = _fast_manager()
    g0, g1 = FakeGrammar("multi"), FakeGrammar("other")
    wrapped = sf._MTPStructuredOutputManager(base)
    requests = {"r0": _req(g0), "r1": _req(g1)}
    with pytest.raises(AssertionError):
        wrapped.grammar_bitmask(requests, ["r0", "r1"], {"r0": [1, 2]})


# ---------------------------------------------------------------------------
# AsyncSchedulerFL wiring
# ---------------------------------------------------------------------------


def _scheduler_cls(monkeypatch, **kwargs):
    """AsyncSchedulerFL with the upstream base ``__init__`` stubbed.

    ``AsyncSchedulerFL`` subclasses the real ``AsyncScheduler`` directly
    (its ``__bases__`` were fixed at class creation), so ``super().__init__``
    is diverted by stubbing the method on the real base class.
    """
    captured = {}

    def fake_init(self, *args, **kw):
        captured["args"] = args
        captured["kwargs"] = kw

    monkeypatch.setattr(
        sf._AsyncScheduler, "__init__", fake_init
    )
    return sf.AsyncSchedulerFL(**kwargs), captured


def test_async_scheduler_wraps_manager_for_mtp1(monkeypatch):
    manager = _manager(num_spec=1)
    kwargs = {"vllm_config": manager.vllm_config, "structured_output_manager": manager}
    _, captured = _scheduler_cls(monkeypatch, **kwargs)
    assert isinstance(
        captured["kwargs"]["structured_output_manager"],
        sf._MTPStructuredOutputManager,
    )


def test_async_scheduler_passthrough_for_other_mtp(monkeypatch):
    manager = _manager(num_spec=3)
    kwargs = {"vllm_config": manager.vllm_config, "structured_output_manager": manager}
    _, captured = _scheduler_cls(monkeypatch, **kwargs)
    assert captured["kwargs"]["structured_output_manager"] is manager


def test_async_scheduler_passthrough_without_manager(monkeypatch):
    cfg = NS(num_speculative_tokens=1)
    _, captured = _scheduler_cls(
        monkeypatch, vllm_config=cfg, structured_output_manager=None
    )
    assert captured["kwargs"]["structured_output_manager"] is None


def test_async_scheduler_passthrough_without_vllm_config(monkeypatch):
    _, captured = _scheduler_cls(
        monkeypatch, structured_output_manager=_manager()
    )
    assert "vllm_config" not in captured["kwargs"]
