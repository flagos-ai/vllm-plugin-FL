# Copyright 2025 BAAI. All rights reserved.
"""PR-#809-only increments for ``vllm_fl/worker/scheduler_fl.py``.

Coverage boundary (post-refactor): the common behaviour of
``_MTPStructuredOutputManager`` (delegation guards, fast-path dual-row
fills, terminated-grammar masks, bitmask reuse, -1 sentinel handling,
``AsyncSchedulerFL`` wiring) is owned by PR #799's
``tests/unit_tests/coverage/worker/test_scheduler_fl.py`` (real
xgrammar matchers as the oracle). This file keeps only the #809-specific
increments that #799 does not carry:

* bonus-fill failure rollback: when the bonus ``batch_fill`` loses a
  matcher mid-flight, the advanced grammars are rolled back and the call
  falls back to the base manager;
* bonus-only deduplication: with no scheduled spec tokens anywhere,
  exactly one (bonus) fill happens;
* multiple scheduled tokens per request assert (production guard).

CPU-only checks with xgrammar stubs and a fake base structured-output
manager (same doubles as before the reduction).
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


def _fast_manager(**kw):
    base = FakeManager(num_speculative_tokens=1, **kw)
    base._grammar_bitmask = FakeBitmask([[0] * 4 for _ in range(8)])
    return base


def _req(grammar):
    return NS(structured_output_request=NS(grammar=grammar))


# ---------------------------------------------------------------------------
# #809-only increment: bonus-fill failure rollback
# ---------------------------------------------------------------------------


def test_fast_path_bonus_failure_rolls_back_and_falls_back(fake_xgrammar):
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


# ---------------------------------------------------------------------------
# #809-only increment: bonus-only deduplication
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# #809-only increment: production guard on scheduled token count
# ---------------------------------------------------------------------------


def test_fast_path_multiple_tokens_per_request_asserts(fake_xgrammar):
    base = _fast_manager()
    g0, g1 = FakeGrammar("multi"), FakeGrammar("other")
    wrapped = sf._MTPStructuredOutputManager(base)
    requests = {"r0": _req(g0), "r1": _req(g1)}
    with pytest.raises(AssertionError):
        wrapped.grammar_bitmask(requests, ["r0", "r1"], {"r0": [1, 2]})
