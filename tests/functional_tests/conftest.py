# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional test fixtures and configuration.

Functional tests validate operator/component correctness (ops, compilation,
distributed). They require GPU but not large model files.

Note: Common fixtures (device, has_accelerator, markers) are inherited
from the root tests/conftest.py. Only functional-specific fixtures belong here.
"""

import pytest


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
    """
    import contextlib
    import os
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
    os._exit(int(exitstatus))
