# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional test fixtures and configuration.

Functional tests validate operator/component correctness (ops, compilation,
distributed). They require GPU but not large model files.

Note: Common fixtures (device, has_accelerator, markers) are inherited
from the root tests/conftest.py. Only functional-specific fixtures belong here.
"""

import os
import sys

import pytest


def _is_ascend(config):
    return "ascend" in {
        config.getoption("platform", default=""),
        os.environ.get("FL_TEST_PLATFORM", ""),
        os.environ.get("VLLM_FL_PLATFORM", ""),
    }


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Drain NPU compiler workers and retain the test exit status.

    On Ascend ARM64 Python 3.11, torch_npu C++ destructors corrupt memory
    during interpreter shutdown, causing a segfault at a random GC location
    (e.g. _pytest/mark/structures.py) even after all tests have passed.

    The hard exit runs in pytest_unconfigure, after sessionfinish wrappers have
    reported results. Exiting here kills xdist workers before they can send
    workerfinished, so the controller restarts an otherwise successful worker.

    It also drains any residual inductor SubprocPool whose _read_thread could
    segfault when the subprocess pipe breaks during NPU teardown.
    """
    import contextlib
    import threading

    if not _is_ascend(session.config):
        return
    session.config._fl_ascend_session = session

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


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    # JSON/coverage and xdist's workerfinished message have all been emitted.
    if _is_ascend(config) and hasattr(config, "_fl_ascend_session"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(config._fl_ascend_session.exitstatus))
