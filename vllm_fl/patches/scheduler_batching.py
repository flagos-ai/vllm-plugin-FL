# Copyright (c) 2026 BAAI. All rights reserved.
"""Domino-Free Exact Decode-Slack Dynamic Batching for vLLM V1 Scheduler.

When PlatformFL elevates ``max_num_batched_tokens`` to ``Q_prefill + max_num_seqs``
(reserving slack for active decodes alongside a power-of-two prefill quantum),
a static budget lets the greedy waiting-queue loop nibble ``max_num_seqs - n_dec``
tokens off the *next* waiting prompt whenever ``n_dec < max_num_seqs``, causing
a partial-chunk domino cascade that breaks ``prefill_is_initial`` across the wave.

This hook dynamically tightens each step's token budget to::

    T_step = min(T_max, n_active_decodes + Q_prefill_step)

so active decodes run at full concurrency while waiting prompts always receive an
exact ``Q_prefill`` quantum with zero tail-sliver nibbling.
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)


def patch_vllm_scheduler_batching() -> bool:
    if os.environ.get("VLLM_FL_AUTO_BATCH_TOKENS", "1") != "1":
        return False
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:  # noqa: BLE001
        logger.debug("vLLM V1 Scheduler unavailable for batching patch: %s", exc)
        return False

    if getattr(Scheduler, "_vllm_fl_domino_free_patched", False):
        return True

    orig_init = Scheduler.__init__
    orig_schedule = Scheduler.schedule

    @functools.wraps(orig_init)
    def _fl_scheduler_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        quantum = 0
        try:
            max_seqs = int(getattr(self, "max_num_running_reqs", 128) or 128)
            expected_slack = ((max_seqs + 63) // 64) * 64
            cand_q = int(self.max_num_scheduled_tokens) - expected_slack
            if cand_q >= 2048 and (cand_q & (cand_q - 1)) == 0:
                quantum = cand_q
        except Exception:  # noqa: BLE001
            quantum = 0
        self._fl_prefill_quantum = quantum
        if quantum > 0:
            logger.info(
                "Scheduler: Domino-Free Decode-Slack dynamic batching active "
                "(prefill_quantum=%d, max_scheduled_tokens=%d).",
                quantum,
                self.max_num_scheduled_tokens,
            )

    @functools.wraps(orig_schedule)
    def _fl_scheduler_schedule(self, *args, **kwargs):
        quantum = getattr(self, "_fl_prefill_quantum", 0)
        if quantum > 0 and (
            self.waiting
            or getattr(self, "skipped_waiting", None)
            or any(r.is_prefill_chunk for r in self.running)
        ):
            orig_max = self.max_num_scheduled_tokens
            next_step = self.current_step + 1
            n_dec = 0
            pf_rem = 0
            for r in self.running:
                if (
                    r.num_output_placeholders > 0
                    and r.num_computed_tokens + 2 - r.num_output_placeholders
                    >= r.num_prompt_tokens + r.max_tokens
                ):
                    continue
                if next_step < getattr(r, "next_decode_eligible_step", 0):
                    continue
                rem = (
                    r.num_tokens_with_spec
                    + r.num_output_placeholders
                    - r.num_computed_tokens
                )
                if rem <= 0:
                    continue
                if r.is_prefill_chunk or rem > 1:
                    pf_rem += rem
                else:
                    n_dec += 1
            pf_budget = pf_rem if (0 < pf_rem < quantum) else quantum
            self.max_num_scheduled_tokens = min(orig_max, max(1, n_dec + pf_budget))
            try:
                return orig_schedule(self, *args, **kwargs)
            finally:
                self.max_num_scheduled_tokens = orig_max
        return orig_schedule(self, *args, **kwargs)

    Scheduler.__init__ = _fl_scheduler_init
    Scheduler.schedule = _fl_scheduler_schedule
    Scheduler._vllm_fl_domino_free_patched = True
    return True
