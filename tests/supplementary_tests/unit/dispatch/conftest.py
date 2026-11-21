# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared fixtures for supplementary dispatch unit tests.

The dispatch logger manager configures ``vllm_fl.dispatch`` (and per-module
children such as ``vllm_fl.dispatch.io_dump``) with ``propagate=False`` plus
a private stderr handler as soon as ``logger_manager.get_logger`` is called
(which happens whenever ``builtin_ops`` / ``discovery`` are imported). Records
emitted by ``vllm_fl.dispatch.io_common`` therefore never reach pytest's
root-attached caplog handler.

This conftest attaches the caplog handler directly to the dispatch loggers
for the duration of each test — same approach as the core
``tests/unit_tests/dispatch/conftest.py``.
"""

import logging

import pytest

_LOGGER_NAMES = [
    "vllm_fl.dispatch",
    "vllm_fl.dispatch.io_common",
    "vllm_fl.dispatch.io_dump",
    "vllm_fl.dispatch.io_print",
    "vllm_fl.dispatch.logger_manager",
]


@pytest.fixture(autouse=True)
def _attach_caplog_to_dispatch_loggers(caplog):
    """Attach the caplog handler directly to dispatch loggers.

    Attaching to the loggers themselves (rather than relying on root
    propagation) keeps ``caplog`` assertions working regardless of whether
    ``logger_manager`` has already disabled ``propagate`` on the logger or
    one of its ancestors.
    """
    attached = []
    for name in _LOGGER_NAMES:
        lgr = logging.getLogger(name)
        if caplog.handler not in lgr.handlers:
            lgr.addHandler(caplog.handler)
            attached.append(lgr)
    yield
    for lgr in attached:
        lgr.removeHandler(caplog.handler)
