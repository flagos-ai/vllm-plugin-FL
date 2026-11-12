# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coverage tests for vllm_fl/worker/scheduler_fl.py (0% in baseline).

``_MTPStructuredOutputManager`` layers batched xGrammar mask filling on a
base vLLM StructuredOutputManager when MTP=1 and several structured
requests are scheduled. The base manager, its backend, and the native
BatchGrammarMatcher are vllm engine collaborators; per CONTRACT R2 they
are mocked as inputs while the plugin orchestration under test — the
delegation conditions, row bookkeeping, accept/rollback pairing, and the
sliced numpy result — is asserted on recorded values.

``AsyncSchedulerFL`` wraps the manager at construction time only when the
vllm config requests MTP=1; the parent AsyncScheduler constructor is a
vllm engine object and is replaced by a recorder (the same module-symbol
boundary the dev tests patch).
"""

from . import _imports  # noqa: F401  (installs local import shims first)

from types import SimpleNamespace

import numpy
import torch

import vllm_fl.worker.scheduler_fl as scheduler_module
from vllm_fl.worker.scheduler_fl import AsyncSchedulerFL, _MTPStructuredOutputManager


class _Grammar:
    """Fake xgrammar Grammar: records accepts/rollbacks, optional matcher."""

    def __init__(self, terminated=False, has_matcher=True, accepts=True):
        self.terminated = terminated
        self.matcher = object() if has_matcher else None
        self.accepts = accepts
        self.accept_calls = []
        self.rollback_calls = []

    def is_terminated(self):
        return self.terminated

    def accept_tokens(self, req_id, tokens):
        self.accept_calls.append((req_id, list(tokens)))
        return self.accepts

    def rollback(self, count):
        self.rollback_calls.append(count)


class _RecorderMatcher:
    """Fake BatchGrammarMatcher recording (matchers, indices) per call."""

    def __init__(self):
        self.calls = []

    def batch_fill_next_token_bitmask(self, matchers, bitmask, indices):
        self.calls.append((list(matchers), list(indices)))


class _FakeBackend:
    """Fake structured-output backend allocating all-zero bitmasks."""

    def __init__(self):
        self.allocated = []

    def allocate_token_bitmask(self, count):
        self.allocated.append(count)
        return torch.zeros((count, 4), dtype=torch.int32)


class _FakeRequest:
    def __init__(self, grammar):
        self.structured_output_request = SimpleNamespace(grammar=grammar)


class _BaseManager:
    """Fake base StructuredOutputManager with delegation recording."""

    def __init__(self, num_speculative_tokens=1):
        self.vllm_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            scheduler_config=SimpleNamespace(max_num_seqs=4),
        )
        self._grammar_bitmask = None
        self.backend = _FakeBackend()
        self._full_mask = torch.tensor(-1, dtype=torch.int32)
        self._reasoners = {}
        self.delegations = []

    def _get_reasoner(self, request):
        return self._reasoners.get(id(request))

    def grammar_bitmask(self, requests, ids, tokens):
        self.delegations.append((list(ids), tokens))
        return "DELEGATED"


def _make_manager(base):
    """Build a _MTPStructuredOutputManager with a recorder batch matcher."""
    manager = _MTPStructuredOutputManager.__new__(_MTPStructuredOutputManager)
    manager._base = base
    manager._batch_matcher = _RecorderMatcher()
    manager._logged_active = False
    return manager


def _call(base, r1_tokens=(5,), g1=None, g2=None):
    """Run grammar_bitmask over two scheduled structured requests.

    Returns (manager, grammar1, grammar2, result); grammar1 carries the
    r1_tokens spec-decode tokens, grammar2 the (empty) r2 ones.
    """
    manager = _make_manager(base)
    g1 = g1 or _Grammar()
    g2 = g2 or _Grammar()
    requests = {"r1": _FakeRequest(g1), "r2": _FakeRequest(g2)}
    result = manager.grammar_bitmask(
        requests, ["r1", "r2"], {"r1": list(r1_tokens), "r2": []}
    )
    return manager, g1, g2, result


def test_proxy_forwards_attribute_access_to_base_manager():
    """__getattr__ resolves unknown names against the wrapped manager."""
    manager = _make_manager(SimpleNamespace(foo=42, bar=lambda x: x + 1))
    assert manager.foo == 42
    assert manager.bar(5) == 6


def test_proxy_raises_for_attributes_missing_on_base():
    """__getattr__ lets missing attributes raise AttributeError."""
    manager = _make_manager(SimpleNamespace())
    raised = False
    try:
        manager.no_such_attribute
    except AttributeError:
        raised = True
    assert raised, "proxy did not raise AttributeError for a missing name"


def test_grammar_bitmask_delegates_when_num_speculative_tokens_not_one():
    """Base configs with MTP != 1 skip the plugin path entirely."""
    base = _BaseManager(num_speculative_tokens=2)
    manager, _, _, result = _call(base)
    assert result == "DELEGATED"
    assert base.delegations == [(["r1", "r2"], {"r1": [5], "r2": []})]
    assert manager._batch_matcher.calls == []
    assert manager._logged_active is False


def test_grammar_bitmask_delegates_when_single_structured_request():
    """A single scheduled structured request takes the delegation branch."""
    base = _BaseManager()
    manager = _make_manager(base)
    requests = {"r1": _FakeRequest(_Grammar())}
    result = manager.grammar_bitmask(requests, ["r1"], {"r1": [5]})
    assert result == "DELEGATED"
    assert base.delegations == [(["r1"], {"r1": [5]})]
    assert manager._batch_matcher.calls == []
    assert manager._logged_active is False


def test_grammar_bitmask_delegates_when_reasoner_present():
    """Any request with a reasoner forces the delegation branch."""
    base = _BaseManager()
    manager = _make_manager(base)
    requests = {"r1": _FakeRequest(_Grammar()), "r2": _FakeRequest(_Grammar())}
    base._reasoners[id(requests["r2"])] = "reasoner-object"
    result = manager.grammar_bitmask(requests, ["r1", "r2"], {"r1": [5], "r2": []})
    assert result == "DELEGATED"
    assert base.delegations == [(["r1", "r2"], {"r1": [5], "r2": []})]
    assert manager._logged_active is False


def test_grammar_bitmask_delegates_when_backend_and_bitmask_absent():
    """No backend and no cached bitmask fall back to the base manager."""
    base = _BaseManager()
    base.backend = None
    manager, _, _, result = _call(base)
    assert result == "DELEGATED"
    # The plugin logged the active path once before noticing the missing backend.
    assert manager._logged_active is True
    assert base._grammar_bitmask is None


def test_grammar_bitmask_allocates_mask_of_twice_max_num_seqs():
    """The cached bitmask is allocated with max_num_seqs * 2 rows."""
    base = _BaseManager()
    _call(base)
    assert base.backend.allocated == [8]
    assert base._grammar_bitmask.shape == (8, 4)


def test_grammar_bitmask_batches_current_rows_then_bonus_rows():
    """Rows are filled twice: pre-accept (current) and post-accept (bonus)."""
    base = _BaseManager()
    manager, _, _, result = _call(base)
    # r1 carries a token: current row 0 and bonus row 1; r2 bonus row 2.
    calls = manager._batch_matcher.calls
    assert [(len(matchers), indices) for matchers, indices in calls] == [
        (1, [0]),
        (2, [1, 2]),
    ]
    assert isinstance(result, numpy.ndarray)
    assert result.shape == (3, 4)
    assert result.dtype == numpy.int32


def test_grammar_bitmask_accepts_token_once_then_rolls_back():
    """Accepted grammars are rewound so the next call re-advances them."""
    base = _BaseManager()
    _, g1, g2, _ = _call(base)
    assert g1.accept_calls == [("r1", [5])]
    assert g1.rollback_calls == [1]
    # A request without a spec-decode token is neither accepted nor rolled back.
    assert g2.accept_calls == []
    assert g2.rollback_calls == []


def test_grammar_bitmask_skips_accept_for_minus_one_token():
    """A -1 (rejected draft) token does not advance the grammar."""
    base = _BaseManager()
    _, g1, _, _ = _call(base, r1_tokens=(-1,))
    assert g1.accept_calls == []
    assert g1.rollback_calls == []


def test_grammar_bitmask_fills_terminated_rows_with_full_mask():
    """Terminated grammars get an all-(-1) row instead of a matcher."""
    base = _BaseManager()
    manager, _, _, result = _call(base, r1_tokens=(3,), g2=_Grammar(terminated=True))
    # Row layout: r1 current 0, r1 bonus 1, r2 (terminated) bonus row 2.
    calls = manager._batch_matcher.calls
    assert [indices for _, indices in calls] == [[0], [1]]
    assert bool((result[2] == -1).all())
    assert bool((result[0] == 0).all())


def test_grammar_bitmask_rejects_more_than_one_token_per_request():
    """The per-request token list is asserted to hold at most one token."""
    manager = _make_manager(_BaseManager())
    requests = {"r1": _FakeRequest(_Grammar()), "r2": _FakeRequest(_Grammar())}
    raised = False
    try:
        manager.grammar_bitmask(requests, ["r1", "r2"], {"r1": [1, 2]})
    except AssertionError:
        raised = True
    assert raised, "multi-token request did not trip the assertion"


def test_grammar_bitmask_rejects_unaccepted_token():
    """A grammar refusing accept_tokens trips the assertion."""
    manager = _make_manager(_BaseManager())
    requests = {
        "r1": _FakeRequest(_Grammar(accepts=False)),
        "r2": _FakeRequest(_Grammar()),
    }
    raised = False
    try:
        manager.grammar_bitmask(requests, ["r1", "r2"], {"r1": [7], "r2": []})
    except AssertionError:
        raised = True
    assert raised, "refused accept_tokens did not trip the assertion"


def test_grammar_bitmask_delegates_when_matcher_missing_in_current_rows():
    """A current-row grammar without .matcher falls back to the base."""
    base = _BaseManager()
    matcherless = _Grammar(has_matcher=False)
    manager, g1, _, result = _call(base, g1=matcherless)
    assert result == "DELEGATED"
    assert base.delegations == [(["r1", "r2"], {"r1": [5], "r2": []})]
    # The matcherless grammar aborted batch_fill before any native call.
    assert manager._batch_matcher.calls == []
    assert g1.accept_calls == []


def test_grammar_bitmask_rolls_back_advances_when_bonus_fill_fails():
    """A failed bonus-row fill rewinds accepted grammars before delegating."""
    base = _BaseManager()
    manager, g1, _, result = _call(
        base, r1_tokens=(4,), g2=_Grammar(has_matcher=False)
    )
    assert result == "DELEGATED"
    # g1 was accepted for its token, then rolled back once the bonus
    # batch_fill (which includes the matcherless grammar) failed.
    assert g1.accept_calls == [("r1", [4])]
    assert g1.rollback_calls == [1]
    # Only the current-row fill reached the matcher before the failure.
    calls = manager._batch_matcher.calls
    assert [indices for _, indices in calls] == [[0]]


def test_grammar_bitmask_logs_the_active_path_exactly_once():
    """The plugin-active warning fires on the first native call only."""
    base = _BaseManager()
    manager, _, _, _ = _call(base)
    assert manager._logged_active is True
    assert _call(base)[0]._logged_active is True
    # The bitmask from the first call stays cached for the second.
    assert base.backend.allocated == [8]


class _InitRecorder:
    """Recorder for the parent AsyncScheduler constructor kwargs.

    The parent AsyncScheduler is a vllm engine object (CONTRACT R2 input
    boundary); replacing only its constructor lets these tests observe the
    exact kwargs AsyncSchedulerFL forwards.
    """

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _InitRecorder.instances.append(self)


def _build_scheduler(**kwargs):
    """Construct AsyncSchedulerFL with the parent __init__ recorded."""
    original = scheduler_module._AsyncScheduler.__init__
    scheduler_module._AsyncScheduler.__init__ = _InitRecorder.__init__
    _InitRecorder.instances = []
    try:
        scheduler = AsyncSchedulerFL(**kwargs)
    finally:
        scheduler_module._AsyncScheduler.__init__ = original
    return scheduler, _InitRecorder.instances[-1]


def test_async_scheduler_wraps_manager_when_mtp_one():
    """MTP=1 rewrites structured_output_manager into the plugin wrapper."""
    manager = SimpleNamespace(marker="raw")
    scheduler, parent = _build_scheduler(
        vllm_config=SimpleNamespace(num_speculative_tokens=1),
        structured_output_manager=manager,
        extra=7,
    )
    wrapped = parent.kwargs["structured_output_manager"]
    assert isinstance(wrapped, _MTPStructuredOutputManager)
    assert wrapped._base is manager
    # Untouched kwargs pass through unchanged.
    assert parent.kwargs["extra"] == 7
    assert scheduler is parent


def test_async_scheduler_keeps_manager_when_mtp_not_one():
    """MTP != 1 leaves the structured_output_manager unwrapped."""
    manager = SimpleNamespace(marker="raw-2")
    _, parent = _build_scheduler(
        vllm_config=SimpleNamespace(num_speculative_tokens=3),
        structured_output_manager=manager,
    )
    assert parent.kwargs["structured_output_manager"] is manager


def test_async_scheduler_keeps_manager_when_manager_missing():
    """A missing structured_output_manager skips the wrap entirely."""
    _, parent = _build_scheduler(
        vllm_config=SimpleNamespace(num_speculative_tokens=1),
        structured_output_manager=None,
    )
    assert parent.kwargs["structured_output_manager"] is None


def test_async_scheduler_keeps_manager_when_config_missing():
    """Without a vllm_config kwarg the wrap is skipped."""
    manager = SimpleNamespace(marker="raw-3")
    _, parent = _build_scheduler(structured_output_manager=manager)
    assert parent.kwargs["structured_output_manager"] is manager
