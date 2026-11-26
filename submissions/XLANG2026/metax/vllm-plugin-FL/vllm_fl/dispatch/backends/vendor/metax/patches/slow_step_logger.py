# SPDX-License-Identifier: Apache-2.0
# MiniCPM-FlagServe diagnostic: log engine phases that exceed a latency budget.
#
# Enabled only when VLLM_FL_SLOW_STEP_MS is set (e.g. "500"). For diagnosing
# periodic multi-second engine stalls; OFF by default, zero effect otherwise.

import os
import time
import logging

_THRESHOLD_MS = float(os.environ.get("VLLM_FL_SLOW_STEP_MS", "0"))
logger = logging.getLogger("vllm_fl.slow_step")


def _batch_info(engine_or_sched):
    try:
        sched = getattr(engine_or_sched, "scheduler", engine_or_sched)
        return f"running={len(sched.running)} waiting={len(sched.waiting)}"
    except Exception:
        return "?"


def _wrap(obj, name, budget_ms):
    fn = getattr(obj, name, None)
    if fn is None:
        return

    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            dt_ms = (time.monotonic() - t0) * 1000
            if dt_ms > budget_ms:
                self = args[0] if args else None
                logger.warning(
                    "[slow-step] %s took %.0f ms (%s)",
                    name, dt_ms, _batch_info(self),
                )

    setattr(obj, name, wrapper)


if _THRESHOLD_MS > 0:
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    step_budget = _THRESHOLD_MS
    phase_budget = _THRESHOLD_MS / 2

    _wrap(EngineCore, "_process_engine_step", step_budget)
    _wrap(Scheduler, "schedule", phase_budget)
    _wrap(Scheduler, "update_from_output", phase_budget)
    _wrap(UniProcExecutor, "execute_model", phase_budget)
    logger.warning(
        "[slow-step] instrumentation on: step>%dms, phase>%.0fms",
        step_budget, phase_budget,
    )
