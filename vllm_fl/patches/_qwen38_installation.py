# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Rollback setup mutations; never catch errors from model execution."""

from inspect import getattr_static

_MISSING = object()


class InstallationTransaction:
    """Restore attributes and registries if setup does not complete."""

    def __init__(self):
        self._undo = []

    def attribute(self, owner, name):
        old = (
            getattr_static(owner, name, _MISSING)
            if isinstance(owner, type)
            else getattr(owner, name, _MISSING)
        )

        def restore():
            if old is _MISSING:
                if hasattr(owner, name):
                    delattr(owner, name)
            else:
                setattr(owner, name, old)

        self._undo.append(restore)

    def mapping(self, value, keys=None):
        if keys is None:
            old = dict(value)

            def restore():
                value.clear()
                value.update(old)
        else:
            old = {key: value.get(key, _MISSING) for key in keys}

            def restore():
                for key, entry in old.items():
                    if entry is _MISSING:
                        value.pop(key, None)
                    else:
                        value[key] = entry

        self._undo.append(restore)

    def sequence_additions(self, value, entries):
        old_counts = {entry: value.count(entry) for entry in entries}

        def restore():
            for entry, count in old_counts.items():
                while value.count(entry) > count:
                    index = len(value) - 1 - value[::-1].index(entry)
                    del value[index]

        self._undo.append(restore)

    def sequence(self, value):
        old = list(value)
        self._undo.append(lambda: value.__setitem__(slice(None), old))

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            failures = []
            for restore in reversed(self._undo):
                try:
                    restore()
                except BaseException as failure:
                    failures.append(failure)
            if failures:
                raise RuntimeError(
                    "Qwen setup rollback failed; the worker must exit"
                ) from error
        self._undo.clear()
        return False


def apply_attributes(changes):
    """Publish a fully prepared plan, restoring even a setter that raises."""
    with InstallationTransaction() as transaction:
        for owner, name, value in changes:
            transaction.attribute(owner, name)
            setattr(owner, name, value)
