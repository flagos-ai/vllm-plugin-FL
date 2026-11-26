# SPDX-License-Identifier: Apache-2.0
"""Phase-coalesced pure-step dynamic batching for MetaX C500.

Separates prefill steps and decode steps so that:
  1. Chunked prefills execute at full token budget without mixed-batch fragmentation
     or tail chunks caused by interleaved decode tokens.
  2. Decode batches execute as pure FULL CUDAGraph steps at maximum concurrency.
  3. Periodic pure decode steps between prefill cohorts prevent starvation and
     stagger multi-wave completions to keep Mean TTFT well within latency budgets.
"""
from __future__ import annotations

import os
from vllm.logger import init_logger

logger = init_logger(__name__)

_SCHED_PATCHED = False


def _populate_resumed_token_ids(sched, out, prev_sched: set[str]) -> None:
    """Ensure V1 model_runner receives output tokens for any cached req absent from prev step."""
    cached = out.scheduled_cached_reqs
    if cached is not None and cached.req_ids:
        for i, rid in enumerate(cached.req_ids):
            if rid not in prev_sched and rid not in cached.all_token_ids:
                num_out = cached.num_output_tokens[i]
                if num_out > 0:
                    req_obj = sched.requests.get(rid)
                    if req_obj is not None:
                        # model_runner only slices resumed_token_ids[-num_output_tokens:].
                        # Sending only output_token_ids avoids copying/pickling 16K prompt tokens.
                        cached.all_token_ids[rid] = req_obj.output_token_ids[-num_out:]


def _count_finished_prefills(sched, out) -> int:
    count = 0
    for req_id in out.num_scheduled_tokens:
        r = sched.requests.get(req_id)
        if r is not None and not r.is_prefill_chunk and r.num_computed_tokens >= r.num_prompt_tokens:
            count += 1
    return count


def _install_c500_scheduler_patch() -> None:
    global _SCHED_PATCHED
    if _SCHED_PATCHED:
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    orig_schedule = Scheduler.schedule

    def _c500_phase_coalesced_schedule(self, throttle_prefills: bool = False):
        prev_sched: set[str] = getattr(self, "_c500_prev_sched_ids", set())
        cohort = int(os.environ.get("FL_C500_PREFILL_COHORT", "2"))

        def _finish(out, count_pf: bool = False, reset_pf: bool = False):
            _populate_resumed_token_ids(self, out, prev_sched)
            self._c500_prev_sched_ids = set(out.num_scheduled_tokens.keys())
            if reset_pf:
                self._c500_pf_completed = 0
            elif count_pf:
                self._c500_pf_completed = getattr(self, "_c500_pf_completed", 0) + _count_finished_prefills(self, out)
            return out

        if cohort <= 0 or throttle_prefills:
            return _finish(orig_schedule(self, throttle_prefills=throttle_prefills))

        if self.connector is not None or self.ec_connector is not None:
            return _finish(orig_schedule(self, throttle_prefills=throttle_prefills))

        running = self.running
        if not running and not self.waiting and not self.skipped_waiting:
            return _finish(orig_schedule(self, throttle_prefills=throttle_prefills), reset_pf=True)

        has_running_prefill = False
        has_running_decode = False
        has_deferred_with_placeholder = False
        for r in running:
            if r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens:
                has_running_prefill = True
            else:
                has_running_decode = True
                if r.request_id not in prev_sched and r.num_output_placeholders > 0:
                    has_deferred_with_placeholder = True

        has_waiting_room = (
            bool(self.waiting or self.skipped_waiting)
            and len(running) < self.max_num_running_reqs
        )

        # Pure decode already (no prefills running or waiting) -> normal fast path.
        if not has_running_prefill and not has_waiting_room:
            return _finish(orig_schedule(self, throttle_prefills=throttle_prefills), reset_pf=True)

        # Pure prefill already (no running decodes to mix with) -> normal fast path.
        if not has_running_decode:
            return _finish(orig_schedule(self, throttle_prefills=throttle_prefills), count_pf=True)

        pf_done = getattr(self, "_c500_pf_completed", 0)

        # Between prefill requests: after completing `cohort` prefill requests, run 1 pure decode step
        # (unless a deferred decode still has an unresolved async placeholder).
        if not has_running_prefill and pf_done >= cohort and not has_deferred_with_placeholder:
            saved_bound = self.prefill_capacity_bound
            self.prefill_capacity_bound = False
            try:
                out = orig_schedule(self, throttle_prefills=True)
            finally:
                self.prefill_capacity_bound = saved_bound
            if out.total_num_scheduled_tokens > 0:
                return _finish(out, reset_pf=True)
            self.current_step -= 1
            return _finish(orig_schedule(self, throttle_prefills=throttle_prefills))

        # Run a PURE PREFILL step by deferring running decodes for this single step.
        step_inside = self.current_step + 1
        defer_until = step_inside + 1
        saved_steps: list[tuple[object, int]] = []
        for r in running:
            if not (r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens):
                saved_steps.append((r, r.next_decode_eligible_step))
                if r.next_decode_eligible_step < defer_until:
                    r.next_decode_eligible_step = defer_until

        out = orig_schedule(self, throttle_prefills=throttle_prefills)
        if out.total_num_scheduled_tokens > 0:
            return _finish(out, count_pf=True)

        # Safety fallback: waiting prefill could not be admitted (e.g., KV cache full).
        self.current_step -= 1
        for r, old_step in saved_steps:
            r.next_decode_eligible_step = old_step
        return _finish(orig_schedule(self, throttle_prefills=throttle_prefills), reset_pf=True)

    Scheduler.schedule = _c500_phase_coalesced_schedule
    _SCHED_PATCHED = True
    logger.info_once("Installed C500 phase-coalesced pure-step scheduler optimization")


try:
    _install_c500_scheduler_patch()
except Exception as exc:
    logger.warning("Failed to install C500 scheduler patch: %s", exc)
