# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional test fixtures and configuration.

Functional tests validate operator/component correctness (ops, compilation,
distributed). They require GPU but not large model files.

Note: Common fixtures (device, has_accelerator, markers) are inherited
from the root tests/conftest.py. Only functional-specific fixtures belong here.
"""

import os

import pytest

from tests.utils.device_utils import get_visible_device_env_var

# ---------------------------------------------------------------------------
# xdist: assign one physical device per worker (controller side)
# ---------------------------------------------------------------------------


def pytest_configure_node(node):
    """Assign a physical device index to each xdist worker.

    Called on the controller process once per worker before it starts.
    The worker index (0, 1, 2, ...) maps directly to a device index so that
    each worker process sees only one accelerator via the platform's
    VISIBLE_DEVICES env var.
    """
    # node.workerinput is a dict forwarded to the worker's pytest_configure.
    worker_id = node.workerinput.get("workerid", "gw0")
    try:
        index = int(worker_id.replace("gw", ""))
    except ValueError:
        index = 0
    node.workerinput["fl_device_index"] = str(index)


def pytest_configure(config):
    """Worker side: apply the device pinning injected by pytest_configure_node."""
    worker_input = getattr(config, "workerinput", None)
    if worker_input is None:
        return
    index = worker_input.get("fl_device_index")
    if index is None:
        return
    env_var = get_visible_device_env_var()
    os.environ[env_var] = index
    worker_id = worker_input.get("workerid", "?")
    print(f"\n[functional] {worker_id}: {env_var}={index}", flush=True)


# ---------------------------------------------------------------------------
# Crash-safe teardown (Ascend NPU GC workaround)
# ---------------------------------------------------------------------------


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Force clean exit after all plugins finish to avoid NPU GC destructor crash.

    On Ascend ARM64 Python 3.11, torch_npu C++ destructors corrupt memory
    during interpreter shutdown, causing a segfault at a random GC location
    (e.g. _pytest/mark/structures.py) even after all tests have passed.

    trylast=True ensures this hook runs after all other plugins (json-report,
    coverage, etc.) have flushed their output files. os._exit() then bypasses
    Python GC entirely, preventing the NPU destructor crash.

    Secondary fix: also drains any residual inductor SubprocPool whose
    _read_thread would segfault when the subprocess pipe breaks on NPU teardown.
    Primary guard for that is TORCHINDUCTOR_COMPILE_THREADS=1 in ascend.yaml.

    xdist note: worker processes also run this hook. os._exit() is safe here
    because xdist workers communicate results via a socket before sessionfinish,
    so all data has already been sent to the controller by this point.
    """
    import contextlib
    import threading

    # --- drain residual inductor subprocess pool (if any) ---
    try:
        from torch._inductor.compile_worker import subproc_pool as _sp
    except Exception:
        _sp = None

    if _sp is not None:
        pool = None
        for _name in ("_pool", "_global_pool", "pool", "_worker_pool"):
            pool = getattr(_sp, _name, None)
            if pool is not None:
                break

        if pool is not None:
            proc = getattr(pool, "_proc", None)
            if proc is not None:
                with contextlib.suppress(OSError):
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
                try:
                    proc.wait(timeout=3.0)
                except Exception:
                    with contextlib.suppress(Exception):
                        proc.kill()

            read_thread = getattr(pool, "_read_thread", None)
            if isinstance(read_thread, threading.Thread) and read_thread.is_alive():
                read_thread.join(timeout=3.0)

    # Skip os._exit on the xdist controller: it still needs to aggregate results
    # and write reports after all workers finish.
    #
    # xdist.is_xdist_controller() is the stable public API (xdist >= 0.26).
    # Avoid hasattr(config, "_workeroutputs") which is a private attribute that
    # disappeared in xdist 3.x and caused the controller to call os._exit()
    # prematurely, swallowing all failure output from workers.
    try:
        from xdist import is_xdist_controller

        _is_controller = is_xdist_controller(session.config)
    except ImportError:
        # xdist not installed — running serially, never skip os._exit
        _is_controller = False

    if not _is_controller:
        os._exit(int(exitstatus))
