# Copyright (c) 2026 BAAI. All rights reserved.

"""
Shared fixtures for dispatch unit tests.

Attaches caplog to dispatch loggers that have ``propagate=False``
so that ``caplog``-based assertions work in IO inspector/dumper tests.
"""

import logging

import pytest


@pytest.fixture(autouse=True)
def _attach_caplog_to_dispatch_loggers(caplog):
    """Attach caplog handler to dispatch loggers that have propagate=False."""
    logger_names = [
        "vllm_fl.dispatch.io_dump",
    ]
    loggers = []
    for name in logger_names:
        lgr = logging.getLogger(name)
        if not lgr.propagate:
            lgr.addHandler(caplog.handler)
            loggers.append(lgr)
    yield
    for lgr in loggers:
        lgr.removeHandler(caplog.handler)
