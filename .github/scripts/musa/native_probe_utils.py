# Copyright (c) 2026 BAAI. All rights reserved.
"""Small CPU sealing and fatal-unwind guards for owned native CI probes."""

import json
import os
from pathlib import Path


def sealed_json(path, value):
    path = Path(path)
    data = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


class FatalNativeContext:
    def __init__(self, native, fail):
        self.native = native
        self.fail = fail

    def __enter__(self):
        try:
            return self.native.__enter__()
        except BaseException as exc:
            self.fail(exc)

    def __exit__(self, kind, value, tb):
        if kind is not None:
            self.fail(value)
        try:
            return self.native.__exit__(kind, value, tb)
        except BaseException as exc:
            self.fail(exc)


def execution_safe(result):
    return (
        type(result.get("actual_exit")) is int
        and result.get("actual_exit") == 0
        and all(
            result.get(key) is True
            for key in (
                "process_started",
                "direct_process_reaped",
                "cleanup_confirmed",
                "safe_to_continue",
            )
        )
        and type(result.get("remaining_owned_children_count")) is int
        and result.get("remaining_owned_children_count") == 0
        and result.get("deadline_exceeded") is False
        and result.get("deadline_or_signal_exit") is False
    )
