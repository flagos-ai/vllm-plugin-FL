# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared fixtures for the coverage-focused unit tests.

Merges the isolation machinery the migrated tests rely on (previously
spread across worker/ops/quantization/patches conftests):

* ``fl_sys_modules`` — snapshot/restore of sys.modules entries under
  vllm/vllm_fl/flag_gems (+ torch_npu where relevant) so stub installs
  never leak across files. Autouse: some moved files install module-level
  stubs at collection time.
* ``clean_fl_env`` — clears every VLLM_FL_* env var per test.
* caplog attachment for the five dispatch loggers that set
  propagate=False.
"""

import logging
import os
import sys

import pytest

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems", "torch_npu")

_MISSING = object()


def _pin_real_parallel_state(saved: dict) -> None:
    """Re-inject the real vllm.distributed.parallel_state after any eviction.

    torch.library registers ops once per interpreter; re-importing the
    module re-runs ``direct_register_custom_op`` and raises. Import-system
    reachability of the original object prevents the re-run.
    """
    real = saved.get("vllm.distributed.parallel_state")
    if real is None or real is _MISSING:
        return
    if getattr(real, "__spec__", None) is None:
        return  # fabricated stub — leave it to the test's own restore
    saved["vllm.distributed.parallel_state"] = real


@pytest.fixture(autouse=True)
def fl_sys_modules():
    """Snapshot ``sys.modules`` and restore it exactly after the test.

    The torch dispatcher registers custom ops (e.g. vllm::all_reduce) once
    per interpreter; deleting vllm* modules and letting them re-import would
    re-run the registration and raise. Real vllm modules that were imported
    before the snapshot are therefore never evicted — only modules the test
    ADDED are removed, and a previously-absent entry stays restored-as-absent.
    On hosts where vllm was never imported the fixture behaves as before.
    """

    saved = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS:
            saved[name] = sys.modules.get(name, _MISSING)
    # Entries whose saved module is a REAL (non-fabricated) vllm module and
    # which register dispatcher ops must survive the restore. A module is
    # treated as real when it carries a __spec__ from the site-packages path.
    real_vllm = {
        name
        for name, module in saved.items()
        if module is not _MISSING
        and getattr(module, "__spec__", None) is not None
        and getattr(module.__spec__, "origin", None)
        and "site-packages" in str(module.__spec__.origin)
        and name.split(".")[0] == "vllm"
    }
    # vllm.distributed.parallel_state registers torch dispatcher ops at import
    # (upstream direct_register_custom_op has no idempotence guard). Once its
    # real module object exists in this interpreter, a later re-import of the
    # package re-executes the registration and blows up. Pin it: keep the real
    # object reachable under its name even after a test replaced it, so the
    # import system never re-runs the body.
    _pin_real_parallel_state(saved)
    try:
        yield saved
    finally:
        # Real vllm modules register torch dispatcher ops at import time
        # (parallel_state: all_reduce, offloader/prefetch: wait_prefetch,
        # ...; upstream direct_register_custom_op has no idempotence guard).
        # Once a real module's body has executed in this interpreter, letting
        # the import system re-execute it raises "Tried to register an
        # operator ... multiple times". Modules may have been imported *during*
        # the test (absent from the setup snapshot), so pin every REAL
        # site-packages vllm module currently in sys.modules before the
        # eviction pass — the import system then never re-runs their bodies.
        for name, module in list(sys.modules.items()):
            if name.split(".")[0] != "vllm":
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None) if spec is not None else None
            if origin and "site-packages" in str(origin) and name not in saved:
                saved[name] = module
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in saved
        ]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is _MISSING:
                if name in real_vllm:
                    # the real module may have been re-imported over the stub;
                    # keep whichever instance is loaded now rather than evicting
                    continue
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def clean_fl_env(monkeypatch):
    """Return a helper that clears every ``VLLM_FL_*`` env var for a test."""
    fl_vars = [k for k in os.environ if k.startswith("VLLM_FL_")]
    for key in fl_vars:
        monkeypatch.delenv(key, raising=False)

    def _set(key: str, value: str):
        monkeypatch.setenv(key, value)

    return _set


_LOGGER_NAMES = [
    "vllm_fl.dispatch",
    "vllm_fl.dispatch.io_common",
    "vllm_fl.dispatch.io_dump",
    "vllm_fl.dispatch.io_print",
    "vllm_fl.dispatch.logger_manager",
]


@pytest.fixture(autouse=True)
def _attach_caplog_to_dispatch_loggers(caplog):
    """Attach caplog directly to dispatch loggers with propagate=False."""
    attached = []
    for name in _LOGGER_NAMES:
        lgr = logging.getLogger(name)
        if caplog.handler not in lgr.handlers:
            lgr.addHandler(caplog.handler)
            attached.append(lgr)
    yield
    for lgr in attached:
        lgr.removeHandler(caplog.handler)
