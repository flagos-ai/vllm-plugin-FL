# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional test fixtures and configuration.

Functional tests validate operator/component correctness (ops, compilation,
distributed). They require GPU but not large model files.

Note: Common fixtures (device, has_accelerator, markers) are inherited
from the root tests/conftest.py. Only functional-specific fixtures belong here.
"""

import pytest


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Force clean exit after all plugins finish to avoid NPU GC destructor crash.

    On Ascend ARM64 Python 3.11, torch_npu C++ destructors corrupt memory
    during interpreter shutdown, causing a segfault at a random GC location
    (e.g. _pytest/mark/structures.py) even after all tests have passed.

    The outer wrapper resumes after terminal/JUnit/JSON/coverage hooks and
    xdist's workerfinished notification. A regular trylast hook exits inside
    those wrappers and makes every completed worker look crashed.

    It also drains any residual inductor SubprocPool whose _read_thread could
    segfault when the subprocess pipe breaks during NPU teardown.
    """
    result = yield

    import contextlib
    import os
    import sys
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

    # --- bypass Python GC to avoid NPU destructor memory corruption ---
    # Other finish hooks may update the session's status after entry. Keep
    # their nonzero result, then flush before bypassing interpreter teardown.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError):
            stream.flush()
    os._exit(int(session.exitstatus))
    return result
