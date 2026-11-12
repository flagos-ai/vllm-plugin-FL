# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared fixtures/helpers for the io_dumper coverage tests.

Keeps both test_cov_io_dumper*.py files under the 400-line limit while
providing the same isolation: every dumper/env reset happens around each
test, and pairing stacks are recreated thread-locally.
"""

import json
import logging

import pytest

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from vllm_fl.dispatch import io_common, io_dumper
from vllm_fl.dispatch.io_common import (
    reset_exec_order,
    reset_rank,
    reset_step,
)
from vllm_fl.dispatch.io_dumper import disable_io_dump

DUMP_ENV_VARS = (
    "VLLM_FL_IO_DUMP", "VLLM_FL_IO_DUMP_OPS", "VLLM_FL_IO_DUMP_MODULES",
    "VLLM_FL_IO_DUMP_LAYERS", "VLLM_FL_IO_DUMP_MAX_CALLS",
    "VLLM_FL_IO_DUMP_STEP_RANGE", "VLLM_FL_IO_DUMP_RANK",
    "VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "VLLM_FL_IO_DUMP_WITH_VALUES",
    "VLLM_FL_IO_DUMP_WITH_METAS", "VLLM_FL_IO_DUMP_WITH_PRINT",
    "VLLM_FL_IO_DUMP_BG_WORKERS",
)


def read_jsonl(path: str) -> dict:
    """Read a JSON Lines file and merge its objects into one dict."""
    data: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.update(json.loads(line))
    return data


class PrintCapture(logging.Handler):
    """Collects records emitted on the dumper's console-print logger."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def text(self):
        return "\n".join(self.messages)


@pytest.fixture(autouse=True)
def clean_dumper_state(monkeypatch):
    """Isolate dumper/env state around every test using these helpers."""
    for var in DUMP_ENV_VARS + ("VLLM_FL_CONFIG", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    disable_io_dump()  # also drains background writes and clears env vars
    reset_step()
    reset_exec_order()
    reset_rank()
    io_dumper._print_pairing = type(io_dumper._print_pairing)()
    io_dumper._dump_pairing = type(io_dumper._dump_pairing)()
    yield
    disable_io_dump()
    reset_step()
    reset_exec_order()
    reset_rank()
    io_dumper._print_pairing = type(io_dumper._print_pairing)()
    io_dumper._dump_pairing = type(io_dumper._dump_pairing)()


@pytest.fixture()
def restore_io_common_globals():
    """Snapshot and restore io_common module-level state around a test."""
    saved_rank = io_common._rank
    saved_registry = list(io_common._TENSOR_STAT_REGISTRY)
    saved_dispatch_cache = dict(io_common._dispatch_table_cache)
    saved_active = io_common._io_active
    yield
    io_common._rank = saved_rank
    io_common._TENSOR_STAT_REGISTRY[:] = saved_registry
    io_common._dispatch_table_cache.clear()
    io_common._dispatch_table_cache.update(saved_dispatch_cache)
    io_common._io_active = saved_active
