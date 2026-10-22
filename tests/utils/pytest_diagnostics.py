# SPDX-License-Identifier: Apache-2.0
"""Preserve failures and active-test stacks before a CI job times out."""

import faulthandler
import json
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("test diagnostics")
    group.addoption("--test-diagnostics-dir", type=Path)
    group.addoption("--test-stack-timeout", type=float, default=120.0)


def pytest_configure(config):
    directory = config.getoption("test_diagnostics_dir")
    if directory is not None:
        config.pluginmanager.register(_Diagnostics(config, directory))


class _Diagnostics:
    def __init__(self, config, directory):
        self.config = config
        self.worker = getattr(config, "workerinput", {}).get("workerid", "controller")
        self.timeout = config.getoption("test_stack_timeout")
        directory.mkdir(parents=True, exist_ok=True)
        self.events = (directory / f"{self.worker}.jsonl").open("a", buffering=1)
        self.stacks = (directory / f"{self.worker}-stacks.log").open("a", buffering=1)

    def _record(self, **event):
        self.events.write(json.dumps({"time": time.time(), **event}) + "\n")

    def pytest_runtest_logstart(self, nodeid, location):
        self._record(event="start", nodeid=nodeid)

    def pytest_runtest_logreport(self, report):
        event = dict(
            event="report",
            nodeid=report.nodeid,
            phase=report.when,
            outcome=report.outcome,
            worker=getattr(report, "worker_id", self.worker),
        )
        if report.failed:
            event["traceback"] = str(report.longrepr)
            event["captured"] = list(report.sections)
            reporter = self.config.pluginmanager.getplugin("terminalreporter")
            if reporter is not None and self.worker == "controller":
                reporter.write_sep("=", f"FAILED NOW: {report.nodeid} ({report.when})")
                reporter.write_line(str(report.longrepr))
                for title, content in report.sections:
                    reporter.write_sep("-", title)
                    reporter.write_line(content)
                reporter.flush()
        self._record(**event)

    @contextmanager
    def _watch(self, item, phase):
        if self.timeout <= 0:
            yield
            return
        self.stacks.write(f"Watching {item.nodeid} ({phase})\n")
        # The C watchdog can dump a native call holding the GIL. It never
        # interrupts the test; re-arm for teardown after a failing call phase.
        faulthandler.dump_traceback_later(
            self.timeout, repeat=True, file=self.stacks, exit=False
        )
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        with self._watch(item, "setup"):
            yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        with self._watch(item, "call"):
            yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item):
        with self._watch(item, "teardown"):
            yield

    def pytest_unconfigure(self):
        faulthandler.cancel_dump_traceback_later()
        self.events.close()
        self.stacks.close()
