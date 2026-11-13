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

"""Supplementary coverage tests for ``vllm_fl/dispatch/io_dumper.py`` (5/5).

D3 tranche. Torch-function interception decisions: the
``_should_dump_torch_func`` filter gates and the ``__torch_function__``
bypass/cleanup branches of ``_DumpTorchFuncMode``.  Complex torch work is
avoided by calling the mode method directly with plain-Python functions;
end-to-end torch.matmul coverage lives in ``test_cov_io_dumper.py``.

Local collection note: importing ``vllm_fl`` pulls in ``flag_gems``
which needs triton (no Windows wheel). ``_flaggems_shim`` installs an
inert placeholder only when the real package cannot import.
"""

import os

import pytest
import torch

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from ._dumper_helpers import (  # noqa: F401  (autouse state isolation)
    PrintCapture,
    clean_dumper_state,
)
from vllm_fl.dispatch import io_common, io_dumper
from vllm_fl.dispatch.io_common import (
    acquire_torch_func_tags,
    pop_module_context,
    push_module_context,
    release_torch_func_tags,
    reset_exec_order,
    reset_rank,
)
from vllm_fl.dispatch.io_dumper import (
    disable_io_dump,
    enable_io_dump,
    io_dump_step,
)


class TestShouldDumpTorchFunc:
    """_should_dump_torch_func composable filter gates."""

    def test_torch_funcs_disabled(self, tmp_path):
        """Interception off means every function is rejected."""
        enable_io_dump(dump_dir=str(tmp_path / "d"))
        try:
            assert io_dumper._torch_funcs_enabled is False
            assert io_dumper._should_dump_torch_func("matmul") is False
        finally:
            disable_io_dump()

    def test_step_range_gate(self, tmp_path):
        """Functions pass only while the step is inside the range."""
        enable_io_dump(
            dump_dir=str(tmp_path / "d"),
            with_torch_funcs=True,
            step_range="2-2",
        )
        try:
            assert io_dumper._should_dump_torch_func("matmul") is False
            assert io_dump_step() == 1
            assert io_dumper._should_dump_torch_func("matmul") is False
            assert io_dump_step() == 2
            assert io_dumper._should_dump_torch_func("matmul") is True
        finally:
            disable_io_dump()

    def test_layer_filter_gate(self, tmp_path):
        """An active layer filter rejects calls with no module context."""
        enable_io_dump(
            dump_dir=str(tmp_path / "d"),
            with_torch_funcs=True,
            layers={"0"},
        )
        try:
            assert io_dumper._layer_filter == {"model.layers.0"}
            assert io_dumper._should_dump_torch_func("matmul") is False
        finally:
            disable_io_dump()

    def test_module_filter_gate(self, tmp_path):
        """A module filter matches only inside a pushed module context."""
        enable_io_dump(
            dump_dir=str(tmp_path / "d"),
            with_torch_funcs=True,
            modules={"Linear"},
        )
        try:
            assert io_dumper._match_all is False
            assert io_dumper._should_dump_torch_func("matmul") is False
            push_module_context("Linear")
            try:
                assert io_dumper._should_dump_torch_func("matmul") is True
            finally:
                pop_module_context()
        finally:
            disable_io_dump()

    def test_func_filter_includes_and_excludes(self, tmp_path, monkeypatch):
        """VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS=<names> limits interception."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "matmul")
        enable_io_dump(dump_dir=str(tmp_path / "d"))
        try:
            assert io_dumper._torch_func_filter == {"matmul"}
            assert io_dumper._should_dump_torch_func("matmul") is True
            assert io_dumper._should_dump_torch_func("softmax") is False
        finally:
            disable_io_dump()

    def test_op_filter_gate(self, tmp_path):
        """A specific ops filter also restricts torch function names."""
        enable_io_dump(
            dump_dir=str(tmp_path / "d"),
            with_torch_funcs=True,
            ops={"mm"},
        )
        try:
            assert io_dumper._op_filter == {"mm"}
            assert io_dumper._should_dump_torch_func("matmul") is False
        finally:
            disable_io_dump()


def _matmul(raises=False):
    """Return a plain-Python function whose __name__ is ``matmul``."""

    def matmul(a, b):
        if raises:
            raise RuntimeError("boom")
        return a + b

    return matmul


class TestTorchFuncModeBypassBranches:
    """Direct ``__torch_function__`` calls for pass-through/cleanup paths."""

    def test_compiling_bypasses(self, tmp_path, monkeypatch):
        """torch.compiler.is_compiling() short-circuits interception."""
        enable_io_dump(
            dump_dir=str(tmp_path / "dump"),
            with_torch_funcs=True,
            with_metas=True,
        )
        try:
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            mode = io_dumper._DumpTorchFuncMode()
            assert mode.__torch_function__(_matmul(), (), (1, 2)) == 3
        finally:
            disable_io_dump()
        assert not os.path.isdir(
            os.path.join(str(tmp_path / "dump"), "rank_0000", "step_0000")
        )

    def test_guard_active_bypasses(self, tmp_path):
        """The re-entrancy guard makes the mode pass through."""
        enable_io_dump(
            dump_dir=str(tmp_path / "dump"),
            with_torch_funcs=True,
            with_metas=True,
        )
        try:
            io_dumper.set_guard(True)
            try:
                mode = io_dumper._DumpTorchFuncMode()
                assert mode.__torch_function__(_matmul(), (), (2, 3)) == 5
            finally:
                io_dumper.set_guard(False)
        finally:
            disable_io_dump()

    def test_rank_filter_bypasses(self, tmp_path, monkeypatch):
        """A non-matching rank passes through without dumping."""
        enable_io_dump(
            dump_dir=str(tmp_path / "dump"),
            with_torch_funcs=True,
            with_metas=True,
            ranks={7},
        )
        try:
            monkeypatch.setenv("RANK", "0")
            reset_rank()
            mode = io_dumper._DumpTorchFuncMode()
            assert mode.__torch_function__(_matmul(), (), (4, 5)) == 9
        finally:
            disable_io_dump()

    def test_summary_only_records_seen(self, tmp_path):
        """With metas off, the mode records the op then passes through."""
        enable_io_dump(
            dump_dir=str(tmp_path / "dump"),
            with_torch_funcs=True,
            with_metas=False,
        )
        try:
            mode = io_dumper._DumpTorchFuncMode()
            assert mode.__torch_function__(_matmul(), (), (1, 1)) == 2
            assert "torch.matmul" in io_common._seen_ops
        finally:
            disable_io_dump()

    def test_unmatched_func_passes_through(self, tmp_path, monkeypatch):
        """A func outside the configured filter is not dumped."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "softmax")
        enable_io_dump(
            dump_dir=str(tmp_path / "dump"),
            with_metas=True,
        )
        try:
            mode = io_dumper._DumpTorchFuncMode()
            assert mode.__torch_function__(_matmul(), (), (2, 2)) == 4
        finally:
            disable_io_dump()
        assert not os.path.isdir(
            os.path.join(str(tmp_path / "dump"), "rank_0000", "step_0000")
        )

    def test_over_max_calls_passes_through(self, tmp_path):
        """A per-op call counter at the limit skips dumping."""
        enable_io_dump(
            dump_dir=str(tmp_path / "dump"),
            with_torch_funcs=True,
            with_metas=True,
            max_calls=1,
        )
        try:
            io_dumper._call_counters["torch.matmul"] = 1
            mode = io_dumper._DumpTorchFuncMode()
            assert mode.__torch_function__(_matmul(), (), (3, 3)) == 6
        finally:
            disable_io_dump()
        assert not os.path.isdir(
            os.path.join(str(tmp_path / "dump"), "rank_0000", "step_0000")
        )

    def test_print_mode_logs_combined_block(self, tmp_path):
        """with_print captures inputs and logs one combined I/O block."""
        enable_io_dump(
            dump_dir=str(tmp_path / "dump"),
            with_torch_funcs=True,
            with_metas=True,
            with_print=True,
        )
        cap = PrintCapture()
        io_dumper._print_logger.addHandler(cap)
        try:
            mode = io_dumper._DumpTorchFuncMode()
            assert mode.__torch_function__(_matmul(), (), (2, 4)) == 6
            io_dumper._wait_and_flush()
        finally:
            io_dumper._print_logger.removeHandler(cap)
            disable_io_dump()
        text = cap.text()
        assert "[IO_PRINT][rank=0][step=0]" in text
        assert "INPUTS:" in text and "OUTPUTS:" in text
        assert "arg[0]: 2" in text

    def test_failing_func_cleans_pairing_and_tags(self, tmp_path):
        """A raising function releases pairing/tag state before re-raising."""
        enable_io_dump(
            dump_dir=str(tmp_path / "dump"),
            with_torch_funcs=True,
            with_metas=True,
        )
        try:
            mode = io_dumper._DumpTorchFuncMode()
            with pytest.raises(RuntimeError, match="boom"):
                mode.__torch_function__(_matmul(raises=True), (), (1, 1))
            assert io_dumper._dump_pairing.stack.get("torch.matmul", []) == []
            # Tag depth is back at zero: the next acquisition allocates.
            reset_exec_order()
            _m, _o, order = acquire_torch_func_tags("probe")
            assert order == 1
            release_torch_func_tags()
            reset_exec_order()
        finally:
            disable_io_dump()
