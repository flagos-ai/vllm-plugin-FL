import functools
import logging

import psutil

logger = logging.getLogger(__name__)
WORKER_CLEANUP_TIMEOUT_SECONDS = 30.0


def _snapshot_descendants(processes):
    descendants = {}
    for process in processes:
        if process.pid is None:
            continue
        try:
            for child in psutil.Process(process.pid).children(recursive=True):
                descendants[child.pid] = child
        except psutil.NoSuchProcess:
            continue
    return list(descendants.values())


def _live_processes(processes):
    live = []
    for process in processes:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                live.append(process)
        except psutil.NoSuchProcess:
            continue
    return live


def _wait_for_workers(processes, timeout):
    _, remaining = psutil.wait_procs(_live_processes(processes), timeout=timeout)
    return _live_processes(remaining)


def _cleanup_workers(processes):
    remaining = _wait_for_workers(processes, WORKER_CLEANUP_TIMEOUT_SECONDS)
    for action, timeout in (("terminate", 4.0), ("kill", 5.0)):
        if not remaining:
            return
        logger.warning(
            "Iluvatar shutdown: %s retained workers %s",
            action,
            [process.pid for process in remaining],
        )
        for process in remaining:
            try:
                getattr(process, action)()
            except psutil.NoSuchProcess:
                continue
        remaining = _wait_for_workers(remaining, timeout)
    if remaining:
        raise RuntimeError(
            "Iluvatar workers did not exit: "
            + str([process.pid for process in remaining])
        )


def _with_worker_cleanup(shutdown):
    if getattr(shutdown, "_iluvatar_worker_cleanup", False):
        return shutdown

    @functools.wraps(shutdown)
    def shutdown_with_worker_cleanup(processes, timeout=None):
        descendants = _snapshot_descendants(processes)
        try:
            return shutdown(processes, timeout=timeout)
        finally:
            _cleanup_workers(descendants)

    shutdown_with_worker_cleanup._iluvatar_worker_cleanup = True
    return shutdown_with_worker_cleanup


def install_worker_cleanup():
    from vllm.v1.engine import utils as engine_utils

    engine_utils.shutdown = _with_worker_cleanup(engine_utils.shutdown)
