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

"""Supplementary coverage tests for ``vllm_fl/dispatch/io_dumper.py`` (1/2).

D3 tranche of the flagcicd coverage campaign. These tests drive real ops
(``torch.full``, ``@``/matmul) through the enabled dumper's dispatch-mode
and print paths, and assert the produced dump files' contents against
independently computed references (closed-form tensor values, expected
exec_order/call_num/step bookkeeping, rank gating).

Local collection note: importing ``vllm_fl`` pulls in ``flag_gems``
which needs triton (no Windows wheel). ``_flaggems_shim`` installs an
inert placeholder only when the real package cannot import.
"""

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from ._dumper_helpers import (  # noqa: F401  (autouse state isolation)
    PrintCapture,
    clean_dumper_state,
    read_jsonl,
)

import os

import pytest
import torch

from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch.io_dumper import (
    disable_io_dump,
    dump_after,
    dump_before,
    dump_cleanup,
    enable_io_dump,
    io_dump_step,
    pause_dispatch_modes,
    register_io_module_hooks,
    resume_dispatch_modes,
)
from vllm_fl.dispatch.io_common import (
    dispatch_mode_mgr,
    reset_exec_order,
    reset_rank,
)


class TestDispatchModeEndToEnd:
    """Real ops flowing through _DumpDispatchMode on CPU tensors."""

    def test_matmul_dumps_files_with_closed_form_values(self, tmp_path):
        """aten.mm dump files carry the exact operand/result statistics."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True, with_values=True)
        try:
            x = torch.full((2, 2), 3.0)
            result = x @ x  # 3.0 * 3.0 * 2 = 18 per element
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()

        op_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "aten.mm")
        inp = read_jsonl(os.path.join(op_dir, "input.json"))
        out = read_jsonl(os.path.join(op_dir, "output.json"))
        call = inp["call_1"]
        assert call["op_name"] == "aten.mm"
        assert call["call_num"] == 1
        assert call["step"] == 0
        assert call["rank"] == 0
        # Both operands are the same 2x2 tensor of 3s.
        for key in ("arg_0", "arg_1"):
            stats = call["tensors"][key]
            assert stats["shape"] == [2, 2]
            assert stats["min"] == stats["max"] == stats["mean"] == 3.0
            assert stats["std"] == 0.0
        res_stats = out["call_1"]["tensors"]["result"]
        assert res_stats["min"] == res_stats["max"] == res_stats["mean"] == 18.0
        # .pt round-trip holds exact values.
        pt = torch.load(os.path.join(op_dir, "call_1_input.pt"),
                        weights_only=True)
        assert pt["arg_0"].tolist() == [[3.0, 3.0], [3.0, 3.0]]
        pt_out = torch.load(os.path.join(op_dir, "call_1_output.pt"),
                            weights_only=True)
        assert pt_out["result"].tolist() == result.tolist()

    def test_step_advance_moves_to_new_step_dir(self, tmp_path):
        """Each advance_step() lands ops in a fresh step_XXXX directory."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            a = torch.full((2, 2), 2.0)
            a @ a  # aten.mm at step 0
            assert io_dump_step() == 1
            b = torch.full((2, 2), 4.0)
            b @ b  # aten.mm at step 1
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        step0 = os.path.join(dump_dir, "rank_0000", "step_0000", "aten.mm")
        step1 = os.path.join(dump_dir, "rank_0000", "step_0001", "aten.mm")
        assert os.path.isfile(os.path.join(step0, "input.json"))
        assert os.path.isfile(os.path.join(step1, "input.json"))
        # exec_order reset between steps; torch.full consumed order 1 so
        # aten.mm carries order 2 in both steps.
        s0 = read_jsonl(os.path.join(step0, "input.json"))["call_1"]
        s1 = read_jsonl(os.path.join(step1, "input.json"))["call_1"]
        assert s0["exec_order"] == 2
        assert s1["exec_order"] == 2
        # Operand statistics prove which step captured which call.
        assert s0["tensors"]["arg_0"]["min"] == 2.0
        assert s1["tensors"]["arg_0"]["min"] == 4.0

    def test_lazy_step_range_activation(self, tmp_path):
        """step_range "1-1" defers mode entry until step 1 and stops after."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True, step_range="1-1")
        try:
            assert not dispatch_mode_mgr.is_entered("dump")  # deferred
            torch.full((1,), 10.0)  # step 0: out of range, no dump
            assert io_dump_step() == 1
            assert dispatch_mode_mgr.is_entered("dump")  # lazily activated
            a = torch.full((2, 2), 20.0)
            a @ a  # step 1: in range
            assert io_dump_step() == 2
            b = torch.full((2, 2), 30.0)
            b @ b  # step 2: out of range again
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        step0 = os.path.join(dump_dir, "rank_0000", "step_0000")
        step1 = os.path.join(dump_dir, "rank_0000", "step_0001", "aten.mm")
        step2 = os.path.join(dump_dir, "rank_0000", "step_0002")
        assert not os.path.isdir(step0)
        assert not os.path.isdir(step2)
        assert os.path.isfile(os.path.join(step1, "input.json"))
        captured = read_jsonl(os.path.join(step1, "input.json"))["call_1"]
        assert captured["tensors"]["arg_0"]["min"] == 20.0

    def test_failing_op_cleans_pairing_and_keeps_dumping(self, tmp_path):
        """A raising op pops its pairing; the next op still dumps output."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            with pytest.raises(RuntimeError):
                torch.zeros(2, 3) @ torch.zeros(4, 5)  # shape mismatch
            torch.zeros(3, 3) @ torch.zeros(3, 3)
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        mm_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "aten.mm")
        # The failed call left an input.json but no output.json...
        assert os.path.isfile(os.path.join(mm_dir, "input.json"))
        # ...and the successful call appended its own input+output entries.
        entries = read_jsonl(os.path.join(mm_dir, "input.json"))
        assert "call_1" in entries and "call_2" in entries

    def test_pause_and_resume_dispatch_modes(self, tmp_path):
        """pause exits the modes; resume re-enters them while enabled."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            assert dispatch_mode_mgr.is_entered("dump")
            pause_dispatch_modes()
            assert not dispatch_mode_mgr.is_entered("dump")
            torch.full((2, 2), 5.0) @ torch.full((2, 2), 5.0)  # bare
            resume_dispatch_modes()
            assert dispatch_mode_mgr.is_entered("dump")
            torch.full((2, 2), 6.0) @ torch.full((2, 2), 6.0)
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "aten.mm")
        entries = read_jsonl(os.path.join(step_dir, "input.json"))
        # Only the resumed call was dumped; the paused call went bare.
        assert list(entries) == ["call_1"]
        assert entries["call_1"]["tensors"]["arg_0"]["min"] == 6.0

    def test_torch_func_mode_dumps_matmul(self, tmp_path):
        """Opt-in TorchFunctionMode captures bare torch.matmul calls."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_torch_funcs=True,
                       with_metas=True, with_values=True)
        try:
            a = torch.full((2, 2), 2.0)
            out = torch.matmul(a, a)  # torch.matmul, not aten path
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        op_dir = os.path.join(dump_dir, "rank_0000", "step_0000",
                              "torch.matmul")
        entries = read_jsonl(os.path.join(op_dir, "input.json"))
        call = entries["call_1"]
        assert call["op_name"] == "torch.matmul"
        assert call["op_tag"].startswith("[op=")
        out_stats = read_jsonl(os.path.join(op_dir, "output.json"))["call_1"]
        assert out_stats["tensors"]["result"]["min"] == 8.0  # 2*2*2
        pt = torch.load(os.path.join(op_dir, "call_1_output.pt"),
                        weights_only=True)
        assert pt["result"].tolist() == out.tolist()

    def test_torch_func_mode_failure_releases_tags(self, tmp_path):
        """A failing functional op does not corrupt the tag depth counter."""
        from vllm_fl.dispatch.io_common import (
            acquire_torch_func_tags,
            release_torch_func_tags,
        )
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_torch_funcs=True,
                       with_metas=True)
        try:
            with pytest.raises(RuntimeError):
                torch.matmul(torch.zeros(2, 3), torch.zeros(4, 5))
            torch.matmul(torch.eye(2), torch.eye(2))
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        # Tag depth back to zero: outermost acquisition allocates fresh.
        reset_exec_order()
        _m, _o, order = acquire_torch_func_tags("probe")
        assert order == 1
        release_torch_func_tags()
        reset_exec_order()


class TestModuleHooks:
    """register_io_module_hooks drives thread-local module context."""

    def test_hooks_noop_when_disabled(self):
        """Without an enabled dumper the hook registration is a no-op."""
        register_io_module_hooks(torch.nn.Linear(2, 2))
        assert io_dumper._module_hook_handles == []

    def test_hooks_push_context_seen_in_dump_metadata(self, tmp_path):
        """Module forward hooks feed module_tag/path into dump metadata."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            model = torch.nn.Sequential(torch.nn.Linear(4, 4))
            model.add_module("block", torch.nn.Linear(4, 4))
            register_io_module_hooks(model)
            model.block(torch.randn(2, 4))
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        rank_step = os.path.join(dump_dir, "rank_0000", "step_0000")
        # Some aten op under Linear (addmm/mm/t...) carries the module tag.
        tags = []
        for op_name in os.listdir(rank_step):
            meta = read_jsonl(os.path.join(rank_step, op_name, "input.json"))
            for call in meta.values():
                if call.get("module_tag"):
                    tags.append(call["module_tag"])
        assert tags, "expected at least one tagged op"
        assert any(":block" in t for t in tags)
        assert all(t.startswith("Linear") for t in tags)


class TestPrintMode:
    """Console print pairing via dump_before/dump_after."""

    def test_print_pairing_logs_combined_block(self):
        """dump_before + dump_after emit one paired INPUTS+OUTPUTS block."""
        cap = PrintCapture()
        io_dumper._print_logger.addHandler(cap)
        enable_io_dump(with_print=True, ops={"mm"}, with_metas=True)
        try:
            t = torch.eye(3)
            dump_before("mm", (t, t), {})
            dump_after("mm", (t, t), t @ t)
        finally:
            disable_io_dump()
            io_dumper._print_logger.removeHandler(cap)
        text = cap.text()
        assert "[IO_PRINT][rank=0][step=0]" in text
        assert "INPUTS:" in text and "OUTPUTS:" in text
        assert "arg[0]: Tensor(shape=[3, 3]" in text
        assert "min=0" in text and "max=1" in text  # eye(3) statistics
        # Result of eye @ eye is eye: min 0, max 1, mean 1/3.
        assert "mean=0.333333" in text

    def test_print_without_pairing_logs_outputs_only(self):
        """dump_after with no stored pairing logs an OUTPUTS-only line."""
        cap = PrintCapture()
        io_dumper._print_logger.addHandler(cap)
        enable_io_dump(with_print=True, ops={"mm"}, with_metas=True)
        try:
            dump_after("mm", (), torch.zeros(2))
        finally:
            disable_io_dump()
            io_dumper._print_logger.removeHandler(cap)
        text = cap.text()
        assert "OUTPUTS:" in text
        assert "INPUTS" not in text  # no paired input block
        assert "result: Tensor(shape=[2]" in text

    def test_print_only_mode_writes_no_files(self, tmp_path, monkeypatch):
        """with_print without a dump_dir writes nothing under cwd."""
        monkeypatch.chdir(tmp_path)
        enable_io_dump(with_print=True, with_metas=True)
        try:
            t = torch.eye(2)
            dump_before("mm", (t, t), {})
            dump_after("mm", (t, t), t @ t)
        finally:
            disable_io_dump()
        assert io_dumper._dump_dir == ""
        assert list(os.listdir(tmp_path)) == []  # no io_dump/ created

    def test_dump_cleanup_pops_stale_pairings(self):
        """dump_cleanup clears both pairing stacks for a failed op."""
        enable_io_dump(with_print=True, ops={"mm"}, with_metas=True)
        try:
            t = torch.eye(2)
            dump_before("mm", (t, t), {})
            dump_before("mm", (t, t), {})  # second pending pairing
            dump_cleanup("mm")
            dump_after("mm", (t, t), t @ t)  # no pairing left: outputs only
        finally:
            disable_io_dump()
        dump_stack = getattr(io_dumper._dump_pairing, "stack", {})
        print_stack = getattr(io_dumper._print_pairing, "stack", {})
        assert dump_stack.get("mm", []) == []
        assert print_stack.get("mm", []) == []


class TestSanitizePath:
    """_sanitize_path_component traversal protection."""

    def test_separator_and_dotdot_neutralized(self):
        """Path separators, '..' and edge dots are made safe."""
        sanitize = io_dumper._sanitize_path_component
        assert sanitize("aten.mm") == "aten.mm"
        assert sanitize("a/b\\c") == "a_b_c"
        assert sanitize("..") == "__"
        assert sanitize("a..b") == "a__b"
        assert sanitize("  spaced  ") == "spaced"
        assert sanitize(".") == "_unnamed_"
        assert sanitize("") == "_unnamed_"
        assert sanitize("op name") == "op name"


class TestRankFiltering:
    """Rank filter gating of dump_before/dump_after."""

    def test_rank_filter_blocks_and_allows(self, tmp_path, monkeypatch):
        """Ops are dumped only when get_rank() passes the rank filter."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True, ranks={7})
        try:
            monkeypatch.setenv("RANK", "7")
            reset_rank()
            t = torch.eye(2)
            dump_before("mm", (t, t), {})
            dump_after("mm", (t, t), t @ t)
            monkeypatch.setenv("RANK", "0")
            reset_rank()
            dump_before("mm", (t, t), {})
            dump_after("mm", (t, t), t @ t)
            io_dumper._wait_and_flush()
        finally:
            monkeypatch.delenv("RANK", raising=False)
            reset_rank()
            disable_io_dump()
        op_dir = os.path.join(dump_dir, "rank_0007", "step_0000", "mm")
        entries = read_jsonl(os.path.join(op_dir, "input.json"))
        assert list(entries) == ["call_1"]  # only the rank-7 call passed
        # The blocked rank-0 call never produced op files (the rank_0000
        # directory only ever holds the disable-time summary.json).
        blocked = os.path.join(dump_dir, "rank_0000", "step_0000")
        assert not os.path.isdir(blocked)
