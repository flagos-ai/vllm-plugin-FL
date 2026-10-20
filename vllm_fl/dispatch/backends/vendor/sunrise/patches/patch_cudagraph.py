# Copyright (c) 2025 BAAI. All rights reserved.
"""PTPU cudagraph integration for FlagCX tensor-parallel collectives."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager, nullcontext

logger = logging.getLogger(__name__)
_patched = False



# Device-wide synchronize after replay is required for TP>1 FlagCX stability.


def patch_ptpu_cudagraph():
    """Hook graph capture/replay and warmup for multi-rank FlagCX on PTPU."""
    global _patched
    if _patched:
        return
    try:
        from vllm.platforms import current_platform

        if current_platform.device_type != "ptpu":
            return
    except Exception:
        return

    _patch_graph_wrapper()
    _patch_model_runner()
    _patch_step_debug()
    _patched = True
    logger.info("Applied PTPU cudagraph patches for FlagCX")


def patch_engine_step_debug():
    """Engine-side per-step wall-clock instrumentation (opt-in).

    Called from both ``vllm_fl:register`` (platform plugin) and
    ``vllm_fl:register_model`` (general plugin). The latter is the only
    one vLLM guarantees to run inside the EngineCore subprocess (see
    ``vllm/plugins/__init__.py:14``), so when we want engine-side timing
    we rely on that path. The function is idempotent: subsequent calls
    are no-ops once the class is patched.

    Enabled by ``VLLM_FL_SUNRISE_DEBUG_STEP=1``. Interval defaults to 100
    cycles; override via ``VLLM_FL_SUNRISE_DEBUG_STEP_INTERVAL``.

    All early-exit branches log at INFO level (with a ``[FL_INIT]``
    marker including the process PID) so the user can confirm in the
    serve log whether the patch landed, was skipped because the env var
    was unset, or hit an import error.
    """
    pid = os.getpid()
    env_val = os.environ.get("VLLM_FL_SUNRISE_DEBUG_STEP", "0")
    if env_val.lower() not in ("1", "true", "yes", "on"):
        # One-line breadcrumb so the user can grep for [FL_INIT] across
        # the full serve log and verify which processes saw the call.
        logger.info(
            "[FL_INIT] patch_engine_step_debug skipped pid=%d "
            "(VLLM_FL_SUNRISE_DEBUG_STEP=%r)",
            pid,
            env_val,
        )
        return

    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception as exc:
        logger.info(
            "[FL_INIT] patch_engine_step_debug failed pid=%d: "
            "could not import EngineCoreProc: %s",
            pid,
            exc,
        )
        return

    if getattr(EngineCoreProc, "_sunrise_engine_debug_patched", False):
        logger.info(
            "[FL_INIT] patch_engine_step_debug already applied pid=%d "
            "(idempotent no-op)",
            pid,
        )
        return

    interval = max(
        1,
        int(
            os.environ.get("VLLM_FL_SUNRISE_DEBUG_STEP_INTERVAL", "100")
        ),
    )

    samples = {
        "cycle_ns": [],
        "step_fn_ns": [],
    }

    def _percentile(sorted_arr, q):
        n = len(sorted_arr)
        if n == 0:
            return 0.0
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return sorted_arr[idx]

    def _stats(arr):
        if not arr:
            return 0.0, 0.0, 0.0, 0.0
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        return (
            sum(arr_sorted) / n / 1000.0,
            _percentile(arr_sorted, 0.5) / 1000.0,
            _percentile(arr_sorted, 0.99) / 1000.0,
            arr_sorted[-1] / 1000.0,
        )

    def _emit() -> None:
        n = len(samples["cycle_ns"])
        if n == 0:
            return
        cyc_m, cyc_p50, cyc_p99, cyc_max = _stats(samples["cycle_ns"])
        stp_m, stp_p50, stp_p99, stp_max = _stats(samples["step_fn_ns"])
        post_m = max(cyc_m - stp_m, 0.0)  # output queue put + post_step
        logger.info(
            "[VLLM_FL_DBG_ENGINE] cycle n=%d | "
            "cycle=%.1fus (p50=%.1f p99=%.1f max=%.1f) | "
            "step_fn=%.1f (p50=%.1f p99=%.1f max=%.1f) | "
            "post(=cycle-step_fn)=%.1f",
            n,
            cyc_m,
            cyc_p50,
            cyc_p99,
            cyc_max,
            stp_m,
            stp_p50,
            stp_p99,
            stp_max,
            post_m,
        )
        for k in samples:
            samples[k].clear()

    _orig_process = EngineCoreProc._process_engine_step

    def _process_engine_step_wrapped(self):
        # Wall-clock around the entire engine cycle as the loop sees it.
        # We also try to time ``self.step_fn()`` alone to separate
        # scheduler / RPC work from output-queue-put + post_step. Because
        # ``step_fn`` is captured as a bound-method attribute on the
        # instance (set in ``EngineCore.__init__``), wrapping the class
        # method here is sufficient to also wrap step_fn -- we just
        # measure step_fn by calling it through the existing instance
        # attribute, accepting that the breakdown is "best-effort" and
        # may slightly over-attribute to step_fn if other ops sneak in
        # between t0 and step_fn() in ``_orig_process``.
        #
        # Implementation note: we wrap ``_process_engine_step`` rather
        # than rebinding ``step_fn`` because ``step_fn`` is set per
        # instance after ``__init__`` runs, and we cannot hook into all
        # past or future instances cleanly. Wrapping the class method
        # is robust.
        t0 = time.perf_counter_ns()
        # Sub-time ``step_fn``: temporarily swap the bound method with
        # a timing wrapper for the duration of this call only. This
        # mutation is per-instance and reverted after the call, so the
        # original ``self.step_fn`` is preserved.
        orig_step_fn = self.step_fn
        step_fn_ns_holder = [0]

        def _timed_step_fn(*a, **kw):
            t_a = time.perf_counter_ns()
            try:
                return orig_step_fn(*a, **kw)
            finally:
                step_fn_ns_holder[0] = time.perf_counter_ns() - t_a

        self.step_fn = _timed_step_fn
        try:
            ret = _orig_process(self)
        finally:
            self.step_fn = orig_step_fn
        t1 = time.perf_counter_ns()
        samples["cycle_ns"].append(t1 - t0)
        samples["step_fn_ns"].append(step_fn_ns_holder[0])
        if len(samples["cycle_ns"]) >= interval:
            _emit()
        return ret

    EngineCoreProc._process_engine_step = _process_engine_step_wrapped
    EngineCoreProc._sunrise_engine_debug_patched = True

    logger.info(
        "[FL_INIT] patch_engine_step_debug installed pid=%d "
        "(interval=%d steps)",
        pid,
        interval,
    )


def _ptpu_stream_ctx():
    from vllm.platforms import current_platform

    from vllm_fl.dispatch.backends.vendor.sunrise.patches.flagcx_stream import (
        get_ptpu_cudagraph_ar_stream,
    )

    stream = get_ptpu_cudagraph_ar_stream()
    if stream is None:
        return nullcontext()
    return current_platform.torch_device_fn.stream(stream)


def _ptpu_cross_rank_sync() -> bool:
    """Device sync + CPU barrier when TP world size > 1."""
    from vllm.platforms import current_platform

    try:
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group is None or tp_group.world_size <= 1:
            return False
        current_platform.torch_device_fn.synchronize()
        tp_group.barrier()
        return True
    except Exception:
        return False


def _bind_flagcx_comm_to_capture_stream() -> None:
    try:
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.platforms import current_platform

        tp_group = get_tp_group()
        if tp_group is None or tp_group.world_size <= 1:
            return
        pfc = getattr(tp_group.device_communicator, "pyflagcx_comm", None)
        if pfc is not None:
            pfc.bind_comm_to_active_capture_stream()
        current_platform.torch_device_fn.synchronize()
        tp_group.barrier()
    except Exception:
        pass


def _patch_graph_wrapper():
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.platforms import current_platform
    from vllm_fl.compilation.graph import GraphWrapper
    from vllm_fl.dispatch.backends.vendor.sunrise.patches.flagcx_stream import (
        sync_capture_stream_before_replay,
        sync_compute_stream_after_replay,
    )

    if getattr(GraphWrapper, "_sunrise_ptpu_cudagraph_patched", False):
        return

    _orig_call = GraphWrapper.__call__

    _pid = os.getpid()
    _env_val = os.environ.get("VLLM_FL_SUNRISE_CUDAGRAPH_DEBUG_SYNC", "0")
    _debug_sync = _env_val.lower() in ("1", "true", "yes", "on")

    if not _debug_sync:
        # Same ``[FL_INIT]`` breadcrumb pattern as engine_step_debug /
        # step_debug so users can grep one prefix to confirm which
        # processes ended up in which branch. The cudagraph wrapper
        # patch is installed in worker processes (via
        # ``apply_sunrise_patches`` -> ``register_oot_ops``); this log
        # lets the user verify that the env var was actually seen on
        # each worker (and disambiguate "env unset" from "patch never
        # ran" from "marker name typo'd").
        logger.info(
            "[FL_INIT] patch_graph_wrapper installed pid=%d mode=production "
            "(VLLM_FL_SUNRISE_CUDAGRAPH_DEBUG_SYNC=%r)",
            _pid,
            _env_val,
        )

        def _call(self, *args, **kwargs):
            if not is_forward_context_available():
                return self.runnable(*args, **kwargs)

            forward_context = get_forward_context()
            graph_runtime_mode = forward_context.cudagraph_runtime_mode
            if (
                graph_runtime_mode == CUDAGraphMode.NONE
                or graph_runtime_mode != self.runtime_mode
            ):
                return self.runnable(*args, **kwargs)

            sync_capture_stream_before_replay()
            with _ptpu_stream_ctx():
                output = _orig_call(self, *args, **kwargs)
                if current_platform.device_type == "ptpu":
                    # Required for TP>1 FlagCX stability.
                    # banner at the top of this module. The device-wide
                    # drain is doing double duty as a FlagCX collective
                    # state-machine pump and CANNOT be skipped (or gated
                    # by an env var) on the replay hot path without
                    # hanging multi-rank runs.
                    current_platform.torch_device_fn.synchronize()
            sync_compute_stream_after_replay()
            return output

        GraphWrapper.__call__ = _call
    else:
        _debug_interval = max(
            1,
            int(
                os.environ.get(
                    "VLLM_FL_SUNRISE_CUDAGRAPH_DEBUG_INTERVAL", "100"
                )
            ),
        )

        # Per-bucket samples in nanoseconds. Two buckets: capture (warmup
        # path, recorded once per (mode, batch_descriptor)) and replay
        # (steady-state, what we actually care about). Stored as plain
        # lists so emit/clear is O(n) once per N steps.
        _stats = {
            "capture": {
                "pre_ns": [],
                "submit_ns": [],
                "sync_ns": [],
                "post_ns": [],
            },
            "replay": {
                "pre_ns": [],
                "submit_ns": [],
                "sync_ns": [],
                "post_ns": [],
            },
        }

        def _rank_label() -> str:
            try:
                from vllm.distributed.parallel_state import get_tp_group

                g = get_tp_group()
                return f"tp_rank={g.rank_in_group}/{g.world_size}"
            except Exception:
                return "tp_rank=?"

        logger.info(
            "[FL_INIT] patch_graph_wrapper installed pid=%d "
            "mode=debug-sync (interval=%d steps). Marker is "
            "[VLLM_FL_DBG_SYNC]; buckets pre/submit/sync/post in us; "
            "capture vs replay reported separately.",
            _pid,
            _debug_interval,
        )

        def _percentile(sorted_arr, q):
            # 0-indexed nearest-rank percentile; arr already sorted.
            n = len(sorted_arr)
            if n == 0:
                return 0.0
            idx = min(n - 1, max(0, int(round(q * (n - 1)))))
            return sorted_arr[idx]

        def _emit(mode: str, bucket: dict) -> None:
            n = len(bucket["sync_ns"])
            if n == 0:
                return

            def s(name):
                arr = sorted(bucket[name])
                mean_us = sum(arr) / n / 1000.0
                return (
                    mean_us,
                    _percentile(arr, 0.5) / 1000.0,
                    _percentile(arr, 0.99) / 1000.0,
                    arr[-1] / 1000.0,
                )

            pre_m, pre_p50, pre_p99, pre_max = s("pre_ns")
            sub_m, sub_p50, sub_p99, sub_max = s("submit_ns")
            sync_m, sync_p50, sync_p99, sync_max = s("sync_ns")
            post_m, post_p50, post_p99, post_max = s("post_ns")
            total_m = pre_m + sub_m + sync_m + post_m
            sync_pct = (sync_m / total_m * 100.0) if total_m > 0 else 0.0

            logger.info(
                "[VLLM_FL_DBG_SYNC] %s mode=%s n=%d total=%.1fus | "
                "pre=%.2f submit=%.2f sync=%.2f (%.1f%%) post=%.2f | "
                "sync p50=%.1f p99=%.1f max=%.1f | "
                "submit p50=%.1f p99=%.1f max=%.1f",
                _rank_label(),
                mode,
                n,
                total_m,
                pre_m,
                sub_m,
                sync_m,
                sync_pct,
                post_m,
                sync_p50,
                sync_p99,
                sync_max,
                sub_p50,
                sub_p99,
                sub_max,
            )

            for k in ("pre_ns", "submit_ns", "sync_ns", "post_ns"):
                bucket[k].clear()

        def _call(self, *args, **kwargs):
            if not is_forward_context_available():
                return self.runnable(*args, **kwargs)

            forward_context = get_forward_context()
            graph_runtime_mode = forward_context.cudagraph_runtime_mode
            if (
                graph_runtime_mode == CUDAGraphMode.NONE
                or graph_runtime_mode != self.runtime_mode
            ):
                return self.runnable(*args, **kwargs)

            # Capture vs replay detection: a missing entry (first time
            # we see this batch_descriptor) or an entry whose ``graph``
            # is still ``None`` means ``_orig_call`` is about to RECORD
            # the cudagraph; any later call with the same descriptor
            # will short-circuit to ``entry.graph.replay()``. Mirrors
            # ``vllm_fl/compilation/graph.py:167-169``.
            bd = forward_context.batch_descriptor
            entry = (
                self.concrete_graph_entries.get(bd)
                if bd is not None
                else None
            )
            mode = (
                "CAPTURE"
                if (entry is None or entry.graph is None)
                else "REPLAY"
            )
            bucket = _stats["capture" if mode == "CAPTURE" else "replay"]

            t0 = time.perf_counter_ns()
            sync_capture_stream_before_replay()
            t1 = time.perf_counter_ns()
            with _ptpu_stream_ctx():
                output = _orig_call(self, *args, **kwargs)
                t2 = time.perf_counter_ns()
                if current_platform.device_type == "ptpu":
                    current_platform.torch_device_fn.synchronize()
                t3 = time.perf_counter_ns()
            sync_compute_stream_after_replay()
            t4 = time.perf_counter_ns()

            bucket["pre_ns"].append(t1 - t0)
            bucket["submit_ns"].append(t2 - t1)
            bucket["sync_ns"].append(t3 - t2)
            bucket["post_ns"].append(t4 - t3)
            if len(bucket["sync_ns"]) >= _debug_interval:
                _emit(mode, bucket)
            return output

        GraphWrapper.__call__ = _call

    GraphWrapper._sunrise_ptpu_cudagraph_patched = True


def _patch_model_runner():
    import vllm_fl.worker.model_runner as model_runner_mod
    from vllm.config import CUDAGraphMode
    from vllm.platforms import current_platform
    from vllm_fl.worker.model_runner import ModelRunnerFL

    if getattr(ModelRunnerFL, "_sunrise_cudagraph_hooks_patched", False):
        return

    _bind_on_graph_capture = {"enabled": False}

    if current_platform.dist_backend == "flagcx":
        _orig_graph_capture = model_runner_mod.graph_capture

        @contextmanager
        def _graph_capture(device):
            with _orig_graph_capture(device) as ctx:
                from vllm_fl.dispatch.backends.vendor.sunrise.patches.flagcx_stream import (
                    set_ptpu_cudagraph_ar_stream,
                )

                set_ptpu_cudagraph_ar_stream(ctx.stream)
                if _bind_on_graph_capture["enabled"]:
                    _bind_flagcx_comm_to_capture_stream()
                yield ctx

        model_runner_mod.graph_capture = _graph_capture

    _orig_capture_model = ModelRunnerFL.capture_model

    def capture_model(self, *args, **kwargs):
        _bind_on_graph_capture["enabled"] = True
        try:
            return _orig_capture_model(self, *args, **kwargs)
        finally:
            _bind_on_graph_capture["enabled"] = False

    ModelRunnerFL.capture_model = capture_model

    def _warmup_and_capture(
        self,
        desc,
        cudagraph_runtime_mode,
        profile_seq_lens=None,
        allow_microbatching=False,
        num_warmups=None,
    ):
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL

        for _ in range(num_warmups):
            _ptpu_cross_rank_sync()
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                force_attention=force_attention,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                profile_seq_lens=profile_seq_lens,
            )
        _ptpu_cross_rank_sync()
        self._dummy_run(
            desc.num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            uniform_decode=desc.uniform,
            allow_microbatching=allow_microbatching,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=desc.num_active_loras,
            is_graph_capturing=True,
            profile_seq_lens=profile_seq_lens,
        )

    ModelRunnerFL._warmup_and_capture = _warmup_and_capture
    ModelRunnerFL._sunrise_cudagraph_hooks_patched = True


def _patch_step_debug():
    """Per-step end-to-end wall-clock breakdown around ``execute_model``.

    Opt-in via ``VLLM_FL_SUNRISE_DEBUG_STEP=1``. Measures execute_model,
    sample_tokens, and the gap between consecutive steps so the user can
    see how much wall time lives OUTSIDE the cudagraph wrapper (where
    ``VLLM_FL_SUNRISE_CUDAGRAPH_DEBUG_SYNC`` measures the GPU forward).
    """
    if not os.environ.get(
        "VLLM_FL_SUNRISE_DEBUG_STEP", "0"
    ).lower() in ("1", "true", "yes", "on"):
        return

    from vllm_fl.worker.model_runner import ModelRunnerFL

    if getattr(ModelRunnerFL, "_sunrise_step_debug_patched", False):
        return

    interval = max(
        1,
        int(
            os.environ.get("VLLM_FL_SUNRISE_DEBUG_STEP_INTERVAL", "100")
        ),
    )

    samples = {
        "execute_model_ns": [],
        "sample_tokens_ns": [],
        "inter_step_gap_ns": [],
    }
    last_st_return_ns = [None]

    def _rank_label() -> str:
        try:
            from vllm.distributed.parallel_state import get_tp_group

            g = get_tp_group()
            return f"tp_rank={g.rank_in_group}/{g.world_size}"
        except Exception:
            return "tp_rank=?"

    def _percentile(sorted_arr, q):
        n = len(sorted_arr)
        if n == 0:
            return 0.0
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return sorted_arr[idx]

    def _stats(arr):
        if not arr:
            return 0.0, 0.0, 0.0, 0.0
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        mean_us = sum(arr_sorted) / n / 1000.0
        return (
            mean_us,
            _percentile(arr_sorted, 0.5) / 1000.0,
            _percentile(arr_sorted, 0.99) / 1000.0,
            arr_sorted[-1] / 1000.0,
        )

    def _emit() -> None:
        n = len(samples["execute_model_ns"])
        if n == 0:
            return
        em_m, em_p50, em_p99, em_max = _stats(samples["execute_model_ns"])
        st_m, st_p50, st_p99, st_max = _stats(samples["sample_tokens_ns"])
        gap_m, gap_p50, gap_p99, gap_max = _stats(
            samples["inter_step_gap_ns"]
        )
        total = em_m + st_m + gap_m
        # Avoid divide-by-zero in the unlikely case of empty samples.
        em_pct = (em_m / total * 100.0) if total > 0 else 0.0
        st_pct = (st_m / total * 100.0) if total > 0 else 0.0
        gap_pct = (gap_m / total * 100.0) if total > 0 else 0.0

        logger.info(
            "[VLLM_FL_DBG_STEP] %s n=%d total/step=%.1fus | "
            "execute_model=%.1f (%.1f%%, p50=%.1f p99=%.1f max=%.1f) | "
            "sample_tokens=%.1f (%.1f%%, p50=%.1f p99=%.1f max=%.1f) | "
            "inter_step_gap=%.1f (%.1f%%, p50=%.1f p99=%.1f max=%.1f)",
            _rank_label(),
            n,
            total,
            em_m,
            em_pct,
            em_p50,
            em_p99,
            em_max,
            st_m,
            st_pct,
            st_p50,
            st_p99,
            st_max,
            gap_m,
            gap_pct,
            gap_p50,
            gap_p99,
            gap_max,
        )
        for k in samples:
            samples[k].clear()

    _orig_execute_model = ModelRunnerFL.execute_model
    _orig_sample_tokens = ModelRunnerFL.sample_tokens

    # One-shot config dump at first ``execute_model`` call. Logs the
    # scheduler-side pipelining config so the user can confirm whether
    # engine-worker overlap is actually active. ``async_scheduling``
    # auto-enables to True unless: pooling model / unsupported spec
    # decode / executor lacks ``supports_async_scheduling`` (see
    # ``vllm/config/vllm.py:770-843``). If False under our TP>1 + PTPU
    # + decode-only setup, the engine serializes worker steps and ~step
    # of engine CPU work compounds per token -- which lines up with the
    # observed gap between worker total and benchmark TPOT.
    _state_logged = [False]

    def _log_config_once(self_runner) -> None:
        if _state_logged[0]:
            return
        _state_logged[0] = True
        try:
            from vllm.distributed.parallel_state import get_tp_group

            if get_tp_group().rank_in_group != 0:
                return  # only rank 0 logs
        except Exception:
            pass
        try:
            logger.info(
                "[VLLM_FL_DBG_STEP] scheduler_config.async_scheduling=%s "
                "model_runner.use_async_scheduling=%s "
                "pp_size=%s tp_size=%s",
                getattr(
                    self_runner.scheduler_config, "async_scheduling", "<unset>"
                ),
                getattr(self_runner, "use_async_scheduling", "<unset>"),
                self_runner.parallel_config.pipeline_parallel_size,
                self_runner.parallel_config.tensor_parallel_size,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "[VLLM_FL_DBG_STEP] could not log async_scheduling state: %s",
                exc,
            )

    def execute_model_wrapped(self, *args, **kwargs):
        _log_config_once(self)
        t0 = time.perf_counter_ns()
        if last_st_return_ns[0] is not None:
            samples["inter_step_gap_ns"].append(t0 - last_st_return_ns[0])
        ret = _orig_execute_model(self, *args, **kwargs)
        t1 = time.perf_counter_ns()
        samples["execute_model_ns"].append(t1 - t0)
        return ret

    def sample_tokens_wrapped(self, *args, **kwargs):
        t0 = time.perf_counter_ns()
        ret = _orig_sample_tokens(self, *args, **kwargs)
        t1 = time.perf_counter_ns()
        samples["sample_tokens_ns"].append(t1 - t0)
        last_st_return_ns[0] = t1
        # Emit when execute_model has accumulated ``interval`` samples;
        # gap may be one short (first step has no predecessor).
        if len(samples["execute_model_ns"]) >= interval:
            _emit()
        return ret

    ModelRunnerFL.execute_model = execute_model_wrapped
    ModelRunnerFL.sample_tokens = sample_tokens_wrapped
    ModelRunnerFL._sunrise_step_debug_patched = True

    logger.info(
        "[VLLM_FL_DBG_STEP] per-step end-to-end instrumentation enabled "
        "(interval=%d steps).",
        interval,
    )
