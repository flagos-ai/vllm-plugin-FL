# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""Unit tests for buffer_utils_compat."""

import types

import pytest
import torch


class FakeStagedWriteTensor:
    """Minimal stand-in for vllm.v1.worker.gpu.buffer_utils.StagedWriteTensor."""

    def __init__(self, num_rows=4, cols=8, dtype=torch.int32):
        self.gpu = torch.zeros(num_rows, cols, dtype=dtype)
        self.dtype = dtype
        self.device = torch.device("cpu")
        self._staged_write_indices = []
        self._staged_write_starts = []
        self._staged_write_contents = []
        self._staged_write_cu_lens = []

    def stage_write(self, index, start, x):
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def clear_staged_writes(self):
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


class FakeFusedStagedWriter:
    """Minimal stand-in for FusedStagedWriter."""

    def __init__(self):
        self.device = torch.device("cpu")


@pytest.fixture
def fake_buffer_utils(monkeypatch):
    bu = types.ModuleType("vllm.v1.worker.gpu.buffer_utils")
    bu.StagedWriteTensor = FakeStagedWriteTensor
    bu.FusedStagedWriter = FakeFusedStagedWriter
    monkeypatch.setitem(
        __import__("sys").modules, "vllm.v1.worker.gpu.buffer_utils", bu
    )
    return bu


def test_replaces_apply_write(fake_buffer_utils):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.buffer_utils_compat import (
        apply_buffer_utils_compat,
    )
    apply_buffer_utils_compat()
    t = fake_buffer_utils.StagedWriteTensor()
    t.stage_write(1, 2, [10, 20, 30])
    fake_buffer_utils.StagedWriteTensor.apply_write(t)
    assert t.gpu[1, 2:5].tolist() == [10, 20, 30]
    assert len(t._staged_write_indices) == 0


def test_replaces_fused_apply(fake_buffer_utils):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.buffer_utils_compat import (
        apply_buffer_utils_compat,
    )
    apply_buffer_utils_compat()
    t = fake_buffer_utils.StagedWriteTensor()
    t.stage_write(0, 1, [42])
    fw = fake_buffer_utils.FusedStagedWriter()
    fw.apply([t], None, None)
    assert t.gpu[0, 1].item() == 42
    assert len(t._staged_write_indices) == 0


def test_multiple_writes(fake_buffer_utils):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.buffer_utils_compat import (
        apply_buffer_utils_compat,
    )
    apply_buffer_utils_compat()
    t = fake_buffer_utils.StagedWriteTensor()
    t.stage_write(0, 0, [5, 6])
    t.stage_write(2, 3, [7])
    fake_buffer_utils.StagedWriteTensor.apply_write(t)
    assert t.gpu[0, 0:2].tolist() == [5, 6]
    assert t.gpu[2, 3].item() == 7


def test_empty_apply_write(fake_buffer_utils):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.buffer_utils_compat import (
        apply_buffer_utils_compat,
    )
    apply_buffer_utils_compat()
    t = fake_buffer_utils.StagedWriteTensor()
    fake_buffer_utils.StagedWriteTensor.apply_write(t)  # should not raise


def test_idempotent(fake_buffer_utils):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.buffer_utils_compat import (
        _PATCH_FLAG,
        apply_buffer_utils_compat,
    )
    apply_buffer_utils_compat()
    apply_buffer_utils_compat()
    assert getattr(fake_buffer_utils, _PATCH_FLAG, False)
