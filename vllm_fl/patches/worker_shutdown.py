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

"""vLLM 0.20.2 shutdown compatibility for local multiprocessing executors.

Keep EngineCore alive while its own workers are being reclaimed. The upstream
entry point restores default signal handlers *before* cleanup, so a second
SIGINT/SIGTERM can interrupt that cleanup. Its default parent timeout is also
shorter than the worker termination sequence. Neither behavior is chip-specific.
Explicitly reclaim an already-created executor when engine construction fails,
before multiprocessing's automatic daemon-worker joins can hang process exit.

This is a version-scoped backport, not a replacement for device-specific resource
cleanup. Ray, multi-node execution and request-drain timeout semantics are left
unchanged. See vllm-project/vllm#50529 and vllm-project/vllm#52281.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from functools import wraps
from importlib import import_module
from multiprocessing.process import BaseProcess
from types import ModuleType
from typing import Any

from packaging.version import InvalidVersion, Version

from vllm.logger import init_logger

logger = init_logger(__name__)

ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S = 15.0
WORKER_GRACE_TIMEOUT_S = 4.0
WORKER_TERMINATE_TIMEOUT_S = 4.0
WORKER_KILL_TIMEOUT_S = 2.0

_PATCH_MARKER = "_vllm_fl_worker_shutdown_patched"
_REQUEST_TIMEOUT_ATTR = "_vllm_fl_request_shutdown_timeout"
_LOCAL_MULTIPROC_ATTR = "_vllm_fl_local_multiproc_shutdown"
_ORIGINAL_RUN_ATTR = "_vllm_fl_original_run_engine_core"
_EXECUTOR_SHUTDOWN_LOCK_ATTR = "_vllm_fl_executor_shutdown_lock"
_EXECUTOR_SHUTDOWN_ACTIVE_ATTR = "_vllm_fl_executor_shutdown_active"
_EXECUTOR_SHUTDOWN_ERROR_ATTR = "_vllm_fl_executor_shutdown_error"


def get_engine_process_shutdown_timeout(
    request_timeout: float | None,
    process_timeout: float | None,
) -> float | None:
    """Add cleanup time only when requests are being aborted immediately.

    In vLLM 0.20.2 the parent's positive process timeout also enforces the request
    drain deadline. Extending it would change request-drain behavior. Preserve that
    deadline, as well as callers that pass ``None``.
    """
    if request_timeout == 0 and process_timeout == 0:
        return ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S
    return process_timeout


def _join_workers(worker_procs: list[BaseProcess], timeout: float) -> bool:
    """Reap workers with one shared deadline, including already exited children."""
    deadline = time.monotonic() + timeout
    for proc in worker_procs:
        # Initialization can fail before a process has been started.
        if proc.pid is not None:
            proc.join(max(0.0, deadline - time.monotonic()))
    return all(not proc.is_alive() for proc in worker_procs)


def ensure_worker_termination(worker_procs: list[BaseProcess]) -> None:
    """Bounded graceful/TERM/KILL shutdown of precisely these owned processes.

    The 4 + 4 + 2 second budgets are shared across workers, not multiplied by TP
    size. Unlike the upstream helper, the SIGKILL fallback also joins children.
    """
    logger.debug("Worker shutdown: waiting for owned workers to exit")
    if _join_workers(worker_procs, WORKER_GRACE_TIMEOUT_S):
        return

    for proc in worker_procs:
        if proc.is_alive():
            proc.terminate()
    if _join_workers(worker_procs, WORKER_TERMINATE_TIMEOUT_S):
        return

    for proc in worker_procs:
        if proc.is_alive():
            logger.warning("Worker shutdown: sending SIGKILL to owned PID %s", proc.pid)
            proc.kill()
    if not _join_workers(worker_procs, WORKER_KILL_TIMEOUT_S):
        logger.error(
            "Worker shutdown: owned processes remain alive after SIGKILL: %s",
            [proc.pid for proc in worker_procs if proc.is_alive()],
        )


def patch_multiproc_worker_termination(
    multiproc_executor: ModuleType | Any | None = None,
) -> bool:
    """Install the reaper and serialize shutdown in the local EngineCore.

    The daemon worker monitor may enter ``shutdown`` before the main thread.
    Upstream's ``shutting_down`` flag prevents duplicate work, but does not wait
    for that work to finish. The main thread must not exit while the monitor is
    still reclaiming its children.
    """
    if multiproc_executor is None:
        multiproc_executor = import_module("vllm.v1.executor.multiproc_executor")
    executor_cls = multiproc_executor.MultiprocExecutor
    if getattr(executor_cls._ensure_worker_termination, _PATCH_MARKER, False):
        return False
    original_init_executor = executor_cls._init_executor
    original_shutdown = executor_cls.shutdown

    @wraps(original_init_executor)
    def patched_init_executor(self, *args, **kwargs):
        # Initialize before the upstream finalizer is registered or the worker
        # monitor starts; lazy initialization in shutdown would itself race.
        setattr(self, _EXECUTOR_SHUTDOWN_LOCK_ATTR, threading.RLock())
        setattr(self, _EXECUTOR_SHUTDOWN_ACTIVE_ATTR, False)
        setattr(self, _EXECUTOR_SHUTDOWN_ERROR_ATTR, None)
        return original_init_executor(self, *args, **kwargs)

    @wraps(original_shutdown)
    def patched_executor_shutdown(self, *args, **kwargs):
        # The EngineCore parent's process timeout remains the outer bound. A
        # separate lock timeout would let the main thread abandon the daemon
        # cleanup owner, recreating the premature-exit race.
        with getattr(self, _EXECUTOR_SHUTDOWN_LOCK_ATTR):
            if getattr(self, _EXECUTOR_SHUTDOWN_ACTIVE_ATTR):
                # Only the owning thread can reacquire this RLock while active.
                # Permit shutdown callbacks/finalizers to re-enter safely.
                return None
            error = getattr(self, _EXECUTOR_SHUTDOWN_ERROR_ATTR)
            if error is not None:
                raise RuntimeError("Previous worker shutdown failed") from error
            setattr(self, _EXECUTOR_SHUTDOWN_ACTIVE_ATTR, True)
            try:
                return original_shutdown(self, *args, **kwargs)
            except BaseException as exc:
                # Do not let a waiter interpret upstream's shutting_down=True
                # as successful cleanup after the owner failed partway through.
                setattr(self, _EXECUTOR_SHUTDOWN_ERROR_ATTR, exc)
                raise
            finally:
                setattr(self, _EXECUTOR_SHUTDOWN_ACTIVE_ATTR, False)

    executor_cls._init_executor = patched_init_executor
    executor_cls.shutdown = patched_executor_shutdown
    setattr(ensure_worker_termination, _PATCH_MARKER, True)
    executor_cls._ensure_worker_termination = staticmethod(ensure_worker_termination)
    return True


def _is_local_multiproc(vllm_config: Any, executor_class: Any) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return False
    if (
        getattr(parallel_config, "data_parallel_size", 1) != 1
        or getattr(parallel_config, "nnodes_within_dp", 1) != 1
    ):
        return False
    multiproc_executor = import_module("vllm.v1.executor.multiproc_executor")
    return isinstance(executor_class, type) and issubclass(
        executor_class, multiproc_executor.MultiprocExecutor
    )


def _initialize_engine_core(engine_cls: type, *args, **kwargs):
    """Reclaim an executor whose engine fails after creating its workers.

    Called only inside the scoped local-MP child. EngineCore construction can
    fail during KV-cache initialization or graph capture, before assignment in
    run_engine_core and before scheduler/structured-output fields exist. Its
    executor's bound-method weakref finalizer is not a substitute for explicit
    cleanup: multiprocessing may first terminate and indefinitely join workers.
    """
    original_init = engine_cls.__init__

    @wraps(original_init)
    def patched_engine_init(self, *args, **kwargs):
        try:
            original_init(self, *args, **kwargs)
        except BaseException:
            executor = getattr(self, "model_executor", None)
            if executor is not None:
                previous_handlers = {}
                try:
                    for signum in (signal.SIGTERM, signal.SIGINT):
                        previous_handlers[signum] = signal.signal(
                            signum, signal.SIG_IGN
                        )
                    executor.shutdown()
                except BaseException:
                    # Preserve the constructor's original error, including
                    # cancellation, even if cleanup itself also fails.
                    logger.exception("Failed to clean up partially initialized engine")
                finally:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
            raise

    engine_cls.__init__ = patched_engine_init
    try:
        return engine_cls(*args, **kwargs)
    finally:
        # Keep the wrapper local to this construction attempt, including failure.
        engine_cls.__init__ = original_init


def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    """vLLM 0.20.2 entry point with cleanup protected from repeated signals.

    Keep this function at module scope and retain its plugin module/qualified name:
    the multiprocessing ``spawn`` start method must import this entry point rather
    than resolving back to the unpatched upstream method.
    """
    core = import_module("vllm.v1.engine.core")
    if not _is_local_multiproc(kwargs.get("vllm_config"), kwargs.get("executor_class")):
        original_run = getattr(
            core.EngineCoreProc, _ORIGINAL_RUN_ATTR, core.EngineCoreProc.run_engine_core
        )
        return original_run(
            *args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs
        )

    # A spawned child does not inherit its parent's patched class attributes.
    patch_multiproc_worker_termination()
    core.maybe_register_config_serialize_by_value()

    engine_core = None
    signal_callback = None
    try:
        vllm_config = kwargs["vllm_config"]
        parallel_config = vllm_config.parallel_config
        data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
        if data_parallel:
            parallel_config.data_parallel_rank_local = local_dp_rank
            process_title = f"EngineCore_DP{dp_rank}"
        else:
            process_title = "EngineCore"
        core.set_process_title(process_title)
        core.maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
        core.decorate_logs()
        if parallel_config.numa_bind:
            core.numa_utils.log_current_affinity_state(process_title)

        if data_parallel and vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            core.logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        parallel_config.data_parallel_index = dp_rank
        if data_parallel and vllm_config.model_config.is_moe:
            parallel_config.data_parallel_rank = dp_rank
            engine_core = core.DPEngineCoreProc(*args, **kwargs)
        else:
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            engine_core = _initialize_engine_core(
                core.EngineCoreProc, *args, engine_index=dp_rank, **kwargs
            )

        assert engine_core is not None

        def wakeup_engine():
            engine_core.input_queue.put_nowait(
                (core.EngineCoreRequestType.WAKEUP, None)
            )

        signal_callback = core.SignalCallback(wakeup_engine)

        def signal_handler(signum, frame):
            engine_core.shutdown_state = core.EngineShutdownState.REQUESTED
            signal_callback.trigger()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        engine_core.run_busy_loop()
    except SystemExit:
        core.logger.debug("EngineCore exiting.")
        raise
    except Exception:
        if engine_core is None:
            core.logger.exception("EngineCore failed to start.")
        else:
            core.logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise
    finally:
        # Do not restore SIG_DFL before stopping callbacks and reclaiming owned
        # workers. A second signal during this interval otherwise kills their
        # parent before it can complete termination and join.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                if signal_callback is not None:
                    signal_callback.stop()
            finally:
                if engine_core is not None:
                    engine_core.shutdown()
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)


def _supports_vllm_version(version: str) -> bool:
    try:
        parsed = Version(version)
    except InvalidVersion:
        return False
    if parsed.release == (0, 20, 2):
        return True
    # The release MUSA CI image identifies itself as this development build.
    # Its EngineCore entry point, process manager and worker termination helper
    # match v0.20.2. Do not apply this backport to arbitrary later dev builds.
    return (
        parsed.release == (0, 20, 3)
        and parsed.dev == 0
        and parsed.local is not None
        and parsed.local.split(".")[0] == "gbc150f502"
    )


def patch_worker_shutdown(
    engine_utils: ModuleType | Any | None = None,
    engine_core: ModuleType | Any | None = None,
    *,
    vllm_version: str | None = None,
) -> bool:
    """Install the compatibility patch; skip unsupported versions explicitly.

    Module and version arguments allow focused tests without starting an engine.
    The per-instance checks leave non-local and non-multiprocessing executors alone.
    """
    if vllm_version is None:
        vllm_version = import_module("vllm.version").__version__
    if sys.platform != "linux" or not _supports_vllm_version(vllm_version):
        logger.warning(
            "Worker shutdown compatibility patch is not installed for vLLM %s "
            "on %s; it supports Linux vLLM 0.20.2 and the verified MUSA "
            "0.20.3.dev0+gbc150f502 build only",
            vllm_version,
            sys.platform,
        )
        return False
    if engine_utils is None:
        engine_utils = import_module("vllm.v1.engine.utils")
    if engine_core is None:
        engine_core = import_module("vllm.v1.engine.core")

    manager_cls = engine_utils.CoreEngineProcManager
    original_shutdown = manager_cls.shutdown
    if getattr(original_shutdown, _PATCH_MARKER, False):
        return False
    original_init = manager_cls.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config", args[3] if len(args) > 3 else None)
        executor_class = kwargs.get(
            "executor_class", args[6] if len(args) > 6 else None
        )
        setattr(
            self, _REQUEST_TIMEOUT_ATTR, getattr(vllm_config, "shutdown_timeout", None)
        )
        setattr(
            self,
            _LOCAL_MULTIPROC_ATTR,
            _is_local_multiproc(vllm_config, executor_class),
        )
        original_init(self, *args, **kwargs)

    @wraps(original_shutdown)
    def patched_shutdown(self, timeout: float | None = None) -> None:
        process_timeout = timeout
        if getattr(self, _LOCAL_MULTIPROC_ATTR, False):
            process_timeout = get_engine_process_shutdown_timeout(
                getattr(self, _REQUEST_TIMEOUT_ATTR, None), timeout
            )
        if process_timeout != timeout:
            logger.info(
                "EngineCore shutdown: allowing %.1fs to clean up owned worker processes",
                process_timeout,
            )
        original_shutdown(self, timeout=process_timeout)

    setattr(patched_shutdown, _PATCH_MARKER, True)
    manager_cls.__init__ = patched_init
    manager_cls.shutdown = patched_shutdown
    setattr(
        engine_core.EngineCoreProc,
        _ORIGINAL_RUN_ATTR,
        staticmethod(engine_core.EngineCoreProc.run_engine_core),
    )
    engine_core.EngineCoreProc.run_engine_core = staticmethod(run_engine_core)
    return True


__all__ = [
    "ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S",
    "WORKER_GRACE_TIMEOUT_S",
    "WORKER_TERMINATE_TIMEOUT_S",
    "WORKER_KILL_TIMEOUT_S",
    "ensure_worker_termination",
    "get_engine_process_shutdown_timeout",
    "patch_multiproc_worker_termination",
    "patch_worker_shutdown",
    "run_engine_core",
]
