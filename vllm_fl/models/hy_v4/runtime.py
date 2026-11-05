# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 host execution contracts shared by platform adapters.

The step context, metadata assembly, graph-shape plans and event lease helpers
have no device backend dependency. Platform adapters own resource selection,
patch installation and the actual graph execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Any, NamedTuple

_ABSENT = object()
_LOCK = RLock()
_CURRENT_RUNNER = ContextVar("hy4_runtime_runner", default=None)


def current_runner():
    return _CURRENT_RUNNER.get()


def apply_patch_plan(plan):
    """Publish the fixed method plan or restore each captured native object."""
    with _LOCK:
        changed = []
        try:
            for owner, name, original, replacement in plan:
                changed.append((owner, name, original))
                setattr(owner, name, replacement)
        except BaseException:
            for owner, name, original in reversed(changed):
                if original is _ABSENT:
                    if name in owner.__dict__:
                        delattr(owner, name)
                else:
                    setattr(owner, name, original)
            raise


def bind_common_runtime_metadata(common):
    """Keep immutable metadata facts within their owning worker's active step."""
    from .metadata import get_current_cpu_seq_lens_proof

    runner = current_runner()
    runtime = getattr(runner, "_model_runtime", None)
    if runtime is None or getattr(common, "_model_runtime", None) is not runtime:
        common.cpu_seq_lens_proof = None
        return common
    proof = getattr(common, "cpu_seq_lens_proof", None)
    common.cpu_seq_lens_proof = (
        proof if proof is get_current_cpu_seq_lens_proof() else None
    )
    return common


def make_common_attention_metadata(**kwargs):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    proof = kwargs.pop("cpu_seq_lens_proof", None)
    common = CommonAttentionMetadata(**kwargs)
    common.cpu_seq_lens_proof = proof
    return common


class Hy4ExecuteModelState(NamedTuple):
    scheduler_output: Any
    logits: Any
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    hidden_states: Any
    prompt_logprobs_hidden_states: Any
    sample_hidden_states: Any
    aux_hidden_states: Any
    ec_connector_output: Any
    cudagraph_stats: Any
    slot_mappings: Any


@contextmanager
def runnersynchronize_input_prep(self):
    if self.prepare_inputs_event is None:
        yield
        return
    self.prepare_inputs_event.synchronize()
    # Native event completion owns the lease on the pinned host packet and
    # its asynchronous GPU sources. Release the previous plan only after wait.
    stager = self._fl_async_input_staging
    if stager is not None:
        stager.begin_step()
    try:
        yield
    finally:
        # Retain plan/source owners until the next event wait, including when
        # input preparation raises; an execution error is never retried.
        self.prepare_inputs_event.record()
        if stager is not None:
            stager.end_step()


def decode_plan(common, sizes, *, verified_step):
    """Return (q, padded rows, padded requests), or native eager fallback.

    CPU QSL describes launch shape. Rejected rows remain present and their GPU
    slots/lengths remain authoritative; we never compact or accept tokens here.
    The first proposal after prefill is intentionally excluded.
    """
    if not verified_step:
        return None
    q = common.max_query_len
    b, n = common.num_reqs, common.num_actual_tokens
    if type(q) is not int or q not in (1, 4) or b < 1 or n != b * q:
        return None
    if q == 1 and n > 64:
        return None
    starts = common.query_start_loc_cpu.tolist()
    if len(starts) != b + 1 or starts[0] != 0 or starts[-1] != n:
        return None
    if any(v - u != q for u, v in zip(starts, starts[1:])):
        return None
    prefilling = common.is_prefilling
    if prefilling is not None and any(prefilling.tolist()):
        return None
    if common.causal is not True or common.mm_req_doc_ranges:
        return None
    padded = next((s for s in sizes if s >= n and s % q == 0), None)
    return None if padded is None else (q, padded, padded // q)


def full_draft_plan(common, q4_sizes, q1_sizes):
    """Uniform verified q4 only; capacities never grow implicitly."""
    plan = decode_plan(common, q4_sizes, verified_step=True)
    if plan is None or plan[0] != 4:
        return None
    _, first_rows, first_reqs = plan
    rows = next((s for s in q1_sizes if s >= common.num_reqs), None)
    if rows is None or first_reqs != rows:
        return None
    return first_rows, rows
