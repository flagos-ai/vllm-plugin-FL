# SPDX-License-Identifier: Apache-2.0
"""Runner facts that are not part of upstream's padded forward metadata."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from vllm.forward_context import set_forward_context as _set_forward_context


@dataclass(frozen=True)
class ForwardExecutionFacts:
    num_actual_tokens: int | None
    in_profile_run: bool


_execution_facts: ContextVar[ForwardExecutionFacts | None] = ContextVar(
    "fl_forward_execution_facts", default=None
)


def get_forward_execution_facts() -> ForwardExecutionFacts | None:
    return _execution_facts.get()


@contextmanager
def set_forward_context(*args, num_actual_tokens=None, in_profile_run=False, **kwargs):
    # The platform hook runs inside upstream's context entry, before either
    # capture or replay. Keep actual activity separate from public DP padding.
    token = _execution_facts.set(
        ForwardExecutionFacts(num_actual_tokens, in_profile_run)
    )
    try:
        with _set_forward_context(*args, **kwargs):
            yield
    finally:
        _execution_facts.reset(token)
