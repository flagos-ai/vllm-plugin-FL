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

"""Supplementary coverage tests for ``vllm_fl/dispatch/io_dumper.py`` (4/4).

D3 tranche. Low-level internals: device-event recording (faked accelerator
modules), the background JSON writer's ordering/reorder buffer, handle
close/reset semantics, the input/output queue error paths, and the
FlagGems-vs-ATen classifier's OpManager fallback.

Local collection note: importing ``vllm_fl`` pulls in ``flag_gems``
which needs triton (no Windows wheel). ``_flaggems_shim`` installs an
inert placeholder only when the real package cannot import.
"""

import json
import os
import types

import torch

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from ._dumper_helpers import (  # noqa: F401  (autouse state isolation)
    PrintCapture,
    clean_dumper_state,
    read_jsonl,
)
from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch.io_common import reset_rank
from vllm_fl.dispatch.io_dumper import (
    disable_io_dump,
    dump_after,
    dump_before,
    enable_io_dump,
)


class _FakeTensor:
    """Minimal tensor stand-in exposing only the ``.device.type`` field."""

    def __init__(self, device_type):
        self.device = types.SimpleNamespace(type=device_type)


class TestRecordDeviceEvent:
    """_record_device_event accelerator probing without any real device."""

    def test_cpu_tensors_return_none(self):
        """CPU-only refs never produce an event."""
        refs = {"arg_0": torch.zeros(2), "result": torch.zeros(1)}
        assert io_dumper._record_device_event(refs) is None

    def test_fake_accelerator_event_recorded(self, monkeypatch):
        """A device module exposing Event() yields a recorded event."""
        recorded = []

        class Event:
            def record(self):
                recorded.append(True)

        fake = types.ModuleType("cov_device")
        fake.Event = Event
        monkeypatch.setattr(torch, "cov_device", fake, raising=False)
        ev = io_dumper._record_device_event({"t": _FakeTensor("cov_device")})
        assert isinstance(ev, Event)
        assert recorded == [True]

    def test_device_without_event_support_returns_none(self, monkeypatch):
        """A device module with no Event attribute falls through to None."""
        fake = types.ModuleType("cov_no_event")
        monkeypatch.setattr(torch, "cov_no_event", fake, raising=False)
        assert (
            io_dumper._record_device_event({"t": _FakeTensor("cov_no_event")}) is None
        )

    def test_event_creation_error_returns_none(self, monkeypatch):
        """An Event constructor that raises is swallowed."""

        class Event:
            def __init__(self):
                raise RuntimeError("events unsupported")

        fake = types.ModuleType("cov_bad_event")
        fake.Event = Event
        monkeypatch.setattr(torch, "cov_bad_event", fake, raising=False)
        assert (
            io_dumper._record_device_event({"t": _FakeTensor("cov_bad_event")}) is None
        )


class TestBackgroundWriter:
    """_bg_compute_and_write deterministic behavior and error tolerance."""

    def test_out_of_order_calls_written_in_call_order(self, tmp_path):
        """call_num=2 buffered then 1 flushes 1 followed by 2."""
        path = str(tmp_path / "input.json")
        io_dumper._bg_compute_and_write(
            path, 2, {"op_name": "x", "call_num": 2}, {}, None
        )
        # Nothing is eligible to write yet (next_write is 1).
        assert os.path.isfile(path)
        assert read_jsonl(path) == {}
        io_dumper._bg_compute_and_write(
            path, 1, {"op_name": "x", "call_num": 1}, {}, None
        )
        io_dumper._close_open_files()
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert [next(iter(obj)) for obj in lines] == ["call_1", "call_2"]
        assert lines[0]["call_1"]["op_name"] == "x"
        assert lines[0]["call_1"]["tensors"] == {}

    def test_tensor_stats_failure_is_swallowed(self, tmp_path):
        """An object without tensor attrs is skipped, not raised."""
        path = str(tmp_path / "out.json")
        io_dumper._bg_compute_and_write(
            path, 1, {"op_name": "x"}, {"bad": object()}, None
        )
        io_dumper._close_open_files()
        entry = read_jsonl(path)["call_1"]
        assert entry["tensors"] == {}

    def test_device_event_is_synchronized(self, tmp_path):
        """A supplied event is synchronized before stats are computed."""
        synced = []

        class Event:
            def synchronize(self):
                synced.append(True)

        io_dumper._bg_compute_and_write(
            str(tmp_path / "s.json"), 1, {"op_name": "x"}, {}, Event()
        )
        io_dumper._close_open_files()
        assert synced == [True]

    def test_close_open_files_swallows_oserror(self):
        """flush()/close() OSErrors do not escape handle cleanup."""

        class BadFh:
            def flush(self):
                raise OSError("flush failed")

            def close(self):
                raise OSError("close failed")

        io_dumper._open_file_handles["bad"] = io_dumper._WipHandle(BadFh())
        io_dumper._close_open_files()
        assert io_dumper._open_file_handles == {}


class TestResetState:
    """_reset_state handle/executor teardown."""

    def test_closes_handles_and_shuts_executor(self, tmp_path):
        """Open handles close (errors swallowed); the executor shuts down."""
        # Deliberately left open: _reset_state is responsible for closing it.
        real = open(  # noqa: SIM115
            tmp_path / "h.json", "a", encoding="utf-8"
        )
        io_dumper._open_file_handles["real"] = io_dumper._WipHandle(real)

        class BadFh:
            def flush(self):
                raise OSError("flush failed")

            def close(self):
                raise OSError("close failed")

        io_dumper._open_file_handles["bad"] = io_dumper._WipHandle(BadFh())
        executor = io_dumper._get_executor()
        assert executor._shutdown is False
        io_dumper._reset_state()
        assert real.closed is True
        assert io_dumper._open_file_handles == {}
        assert io_dumper._io_executor is None
        assert executor._shutdown is True


class TestDumpQueueErrors:
    """_dump_input/_dump_output error reporting paths."""

    def test_output_without_pairing_warns_and_writes_nothing(self, tmp_path):
        """_dump_output with no stored pairing logs a warning and skips."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        cap = PrintCapture()
        io_dumper.logger.addHandler(cap)
        try:
            # No tensor passed: nothing else under the dispatch mode runs,
            # so the only output is the skip warning.
            io_dumper._dump_output("no_pair", None)
        finally:
            io_dumper.logger.removeHandler(cap)
            disable_io_dump()
        assert "No pairing info for dump output 'no_pair'" in cap.text()
        assert not os.path.isdir(os.path.join(dump_dir, "rank_0000", "step_0000"))

    def test_malformed_pairing_reports_not_raises(self, tmp_path):
        """A wrong-shaped pairing is caught and logged, not propagated."""
        enable_io_dump(dump_dir=str(tmp_path / "dump"), with_metas=True)
        cap = PrintCapture()
        io_dumper.logger.addHandler(cap)
        try:
            io_dumper._dump_pairing.stack = {"bad": [(1,)]}
            io_dumper._dump_output("bad", torch.zeros(1))
        finally:
            io_dumper.logger.removeHandler(cap)
            disable_io_dump()
        assert "Failed to queue output dump" in cap.text()

    def test_input_directory_failure_reports_not_raises(self, tmp_path):
        """An uncreatable op directory is caught and logged by _dump_input."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        enable_io_dump(dump_dir=str(tmp_path / "dump"), with_metas=True)
        cap = PrintCapture()
        io_dumper.logger.addHandler(cap)
        saved = io_dumper._dump_dir
        try:
            io_dumper._dump_dir = str(blocker)
            io_dumper._dump_input("mm", (torch.zeros(2),), {}, exec_order=1)
        finally:
            io_dumper._dump_dir = saved
            io_dumper.logger.removeHandler(cap)
            disable_io_dump()
        assert "Failed to queue input dump" in cap.text()

    def test_dump_after_rank_filter_returns_early(self, tmp_path, monkeypatch):
        """dump_after short-circuits on a non-matching rank before dumping."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True, ranks={7})
        try:
            monkeypatch.setenv("RANK", "0")
            reset_rank()
            dump_after("mm", (), torch.zeros(1))
        finally:
            disable_io_dump()
        assert not os.path.isdir(os.path.join(dump_dir, "rank_0000", "step_0000"))

    def test_guard_active_short_circuits_before_and_after(self, tmp_path):
        """dump_before/dump_after return immediately when the guard is set."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            io_dumper.set_guard(True)
            try:
                dump_before("mm", (torch.zeros(1),), {})
                dump_after("mm", (), torch.zeros(1))
            finally:
                io_dumper.set_guard(False)
        finally:
            disable_io_dump()
        assert not os.path.isdir(os.path.join(dump_dir, "rank_0000", "step_0000"))


class TestFlaggemsClassifier:
    """_is_flaggems_op dispatch-key and OpManager fallback paths."""

    def test_dispatch_key_marks_flaggems(self):
        """A dispatch key string mentioning FlagGems is definitive."""
        keys = "[(CUDA, FlagGems, False)]"
        assert io_dumper._is_flaggems_op("aten.x", keys) is True
        assert not io_dumper._is_flaggems_op("aten.y", "[(CUDA, Native, True)]")

    def test_manager_fallback_uses_impl_kind(self):
        """The recorded OpManager impl kind decides for dispatch-managed ops."""
        from vllm_fl.dispatch import (
            BackendImplKind,
            BackendPriority,
            OpImpl,
            reset_default_manager,
            reset_global_policy,
        )
        from vllm_fl.dispatch.manager import get_default_manager

        reset_default_manager()
        reset_global_policy()
        try:
            mgr = get_default_manager()
            mgr.registry.register_impl(
                OpImpl(
                    op_name="cov.gem",
                    impl_id="cov.gem.impl",
                    kind=BackendImplKind.DEFAULT,
                    fn=lambda *a, **k: None,
                    priority=BackendPriority.DEFAULT,
                )
            )
            mgr._called_ops["cov.gem"] = "cov.gem.impl"
            assert io_dumper._is_flaggems_op("cov.gem", "") is True

            mgr.registry.register_impl(
                OpImpl(
                    op_name="cov.ref",
                    impl_id="cov.ref.impl",
                    kind=BackendImplKind.REFERENCE,
                    fn=lambda *a, **k: None,
                    priority=BackendPriority.REFERENCE,
                )
            )
            mgr._called_ops["cov.ref"] = "cov.ref.impl"
            assert io_dumper._is_flaggems_op("cov.ref", "") is False
            # Never called: no recorded impl at all.
            assert io_dumper._is_flaggems_op("cov.none", "") is False
        finally:
            reset_default_manager()
            reset_global_policy()

    def test_manager_lookup_exception_returns_false(self, monkeypatch):
        """An exploding manager lookup is swallowed and classified ATen."""
        import vllm_fl.dispatch.manager as manager_mod

        def boom():
            raise RuntimeError("manager unavailable")

        monkeypatch.setattr(manager_mod, "get_default_manager", boom)
        assert io_dumper._is_flaggems_op("cov.x", "") is False
