# Copyright (c) 2026 BAAI. All rights reserved.
"""Bound admitted MUSA graph owners without native executable mutation."""

import threading
import weakref
from contextlib import contextmanager


class _MusaGraphAdmissionBudget:
    def __init__(self, cap=1900):
        if cap < 1:
            raise ValueError("MUSA graph admission cap must be positive")
        self.cap = cap
        self._lock = threading.RLock()
        self._graphs = weakref.WeakSet()
        self._pending = 0
        self._capturing = False
        self._replaying = 0
        self._poisoned = False
        self._total = 0
        self._epoch = 0
        self._had_live = False
        self._first = False
        self._rejected = False
        self._cap_announced = False

    def _healthy(self):
        if self._poisoned:
            raise RuntimeError(
                "MUSA graph admission is poisoned after an earlier error"
            )

    def snapshot(self):
        with self._lock:
            return dict(
                cap=self.cap,
                live=len(self._graphs),
                pending=self._pending,
                total_admitted=self._total,
                epoch=self._epoch,
                poisoned=self._poisoned,
            )

    @contextmanager
    def capture(self):
        with self._lock:
            self._healthy()
            if self._capturing or self._replaying:
                raise RuntimeError(
                    "Cannot admit a MUSA capture during capture or replay"
                )
            admitted = len(self._graphs) + self._pending < self.cap
            if admitted:
                self._capturing = True
                self._pending += 1
            try:
                yield admitted
                if admitted and self._pending:
                    raise RuntimeError("MUSA capture admission was not committed")
            except BaseException:
                self._poisoned = True
                raise
            finally:
                if admitted:
                    self._pending = 0
                    self._capturing = False

    def register(self, graph):
        with self._lock:
            if not self._capturing or self._pending != 1:
                raise RuntimeError("MUSA graph has no capture reservation")
            self._graphs.add(graph)
            self._pending = 0
            self._total += 1
            self._had_live = True
            events = []
            if not self._first:
                self._first = True
                events.append("first_native_admission")
            if len(self._graphs) == self.cap and not self._cap_announced:
                self._cap_announced = True
                events.append("cap_reached")
            return events

    def rejected(self):
        with self._lock:
            first = not self._rejected
            self._rejected = True
            return first

    @contextmanager
    def replay(self):
        with self._lock:
            self._healthy()
            if self._capturing:
                raise RuntimeError("Cannot replay a MUSA graph during capture")
            self._replaying += 1
            try:
                yield
            except BaseException:
                self._poisoned = True
                raise
            finally:
                self._replaying -= 1

    @contextmanager
    def eager(self):
        with self._lock:
            self._healthy()
            try:
                yield
            except BaseException:
                self._poisoned = True
                raise

    @contextmanager
    def clearing(self):
        # Clearing never refunds an externally-held native graph or heals errors.
        with self._lock:
            if self._capturing or self._replaying:
                raise RuntimeError("Cannot clear MUSA graphs during capture or replay")
            yield

    def cleared_to_zero(self):
        with self._lock:
            if self._had_live and not self._graphs and not self._pending:
                self._epoch += 1
                self._had_live = False
                self._first = False
                self._rejected = False
                self._cap_announced = False
                return True
            return False


_musa_admission_budget = _MusaGraphAdmissionBudget()
