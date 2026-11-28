# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for ``vllm_fl.dispatch.io_dumper`` pairing/print/config edges.

The core suite (``tests/unit_tests/dispatch/test_io_dumper.py``) exercises the
happy-path file dumping with a fixed ``/tmp`` directory.  This module targets
the gaps that ``tests/unit_tests/dispatch/test_io_dumper.py`` leaves uncovered:

* print-side helpers (``_format_inputs``, ``_log_combined``/``_log_outputs_only``,
  ``_push/_pop_print_pairing``) with hand-computed string oracles;
* the thread-local pairing stacks and ``dump_cleanup``;
* filter gates (``_should_dump``/``_should_dump_torch_func``/``_check_limits``)
  including layer, module, op, rank and step-range variants;
* serialization helpers (``_serialize_value``, ``_build_data``,
  ``_extract_tensor_refs``, ``_sanitize_path_component``, ``_get_op_dir``);
* the background writer machinery (``_WipHandle`` reorder buffer,
  ``_record_device_event`` with fake device modules, ``_wait_and_flush`` on a
  failed future, OSError swallowing in ``_close_open_files``);
* ``enable_io_dump`` config layering (env < yaml < API), print-only mode,
  env-var propagation, and ``_init_from_env`` corner branches;
* step advancement (``_on_step_advance``) with lazy dispatch-mode activation;
* the exception and summary paths of ``dump_before``/``dump_after``;
* ``io_common`` leftovers: distributed ``get_rank``, blank layer specs,
  ``_parse_dispatch_table`` line parsing, stack-based matchers, failing tensor
  stats, list-stat truncation, and YAML parser fallthroughs;
* the remaining ``io_dumper`` branch corners formerly in the supplementary
  ``test_io_dumper_config_layers.py``: ``enable_io_dump`` env-var parsing
  corners, ``dump_after`` early returns, the torch-function summary-only
  pass-through, background-writer corner branches, ``pause_dispatch_modes``
  and the ``_reset_state`` handle close.

All file I/O goes through ``tmp_path`` (never a hard-coded ``/tmp``) so the
tests pass on both Windows and Linux.
"""

import json
import logging
import os
import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch import io_common, io_dumper as d
from vllm_fl.dispatch.types import BackendImplKind


def _drain_module_context():
    """Pop every entry from the thread-local module-context stack."""
    stack = getattr(io_common._module_context, "stack", None)
    while stack:
        stack.pop()


@pytest.fixture(autouse=True)
def _reset_dumper_state(clean_fl_env):
    """Reset every io_dumper/io_common global before and after each test.

    ``clean_fl_env`` (from the dispatch conftest) removes all
    ``VLLM_FL_*`` env vars and restores them afterwards, which also undoes the
    ``VLLM_FL_IO_DUMP*`` propagation done by ``_set_env_vars``.
    """
    saved_stats = list(io_common._TENSOR_STAT_REGISTRY)
    saved_callbacks = list(io_common._step_callbacks)
    io_common.reset_step()
    io_common.reset_exec_order()
    io_common.reset_rank()
    io_common.dispatch_mode_mgr.exit_all()
    io_common.func_mode_mgr.exit_all()
    _drain_module_context()
    d.disable_io_dump()
    yield
    io_common.reset_step()
    io_common.reset_exec_order()
    io_common.reset_rank()
    io_common.dispatch_mode_mgr.exit_all()
    io_common.func_mode_mgr.exit_all()
    _drain_module_context()
    d.disable_io_dump()
    io_common._TENSOR_STAT_REGISTRY[:] = saved_stats
    io_common._step_callbacks.clear()
    io_common._step_callbacks.extend(saved_callbacks)


class TestPrintFormatting:
    """Console-print helpers with independent string oracles."""

    def test_format_inputs_lists_args_and_kwargs(self):
        """Each positional arg gets an ``arg[i]:`` line; kwargs their key."""
        tensor = torch.tensor([1.0, 2.0])
        lines = d._format_inputs((tensor, 3), {"eps": 1e-06})
        # Hand-computed expectations (format_value oracles derived by hand):
        assert lines[0] == (
            "  arg[0]: Tensor(shape=[2], dtype=torch.float32, device=cpu, "
            "min=1, max=2, mean=1.5, std=0.707107)"
        )
        assert lines[1] == "  arg[1]: 3"
        assert lines[2] == "  eps: 1e-06"
        assert len(lines) == 3

    def test_format_inputs_skips_leading_module_arg(self):
        """skip_module_arg drops arg[0] only when it is an nn.Module."""
        module = torch.nn.Linear(2, 2)
        tensor = torch.tensor([1.0, 2.0])
        lines = d._format_inputs((module, tensor, 3), {}, skip_module_arg=True)
        # Indices are preserved (arg[1], arg[2]) — only the module line is gone.
        assert lines == [
            (
                "  arg[1]: Tensor(shape=[2], dtype=torch.float32, device=cpu, "
                "min=1, max=2, mean=1.5, std=0.707107)"
            ),
            "  arg[2]: 3",
        ]
        # Without the flag the module line is present.
        assert d._format_inputs((module, tensor, 3), {})[0] == "  arg[0]: Linear(...)"

    def test_format_inputs_keeps_non_module_first_arg(self):
        """A non-module arg[0] is kept even with skip_module_arg=True."""
        tensor = torch.tensor([3.0])
        lines = d._format_inputs((tensor, 7), {}, skip_module_arg=True)
        assert lines[0].startswith("  arg[0]: Tensor(shape=[1]")
        assert lines[1] == "  arg[1]: 7"

    def test_log_combined_shape(self, caplog):
        """The combined block has the documented header/footer/section lines."""
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_print"):
            d._log_combined(
                "mylabel", ["  arg[0]: x"], 42, op_tag="[op=0,1]", exec_order=3
            )
        message = caplog.records[-1].getMessage()
        lines = message.split("\n")
        # parts[0] is "\n" + the separator, so split() yields a leading "".
        assert lines[0] == ""
        assert lines[1] == "-" * 60
        assert lines[2] == "[IO_PRINT][rank=0][step=0][exec_order=3][op=0,1] mylabel"
        assert lines[3] == "-" * 60
        assert lines[4] == "  INPUTS:"
        assert lines[5] == "    arg[0]: x"
        assert lines[6] == "  OUTPUTS:"
        # format_result already prefixes "  result: "; the block adds two more.
        assert lines[7] == "    result: 42"
        assert lines[8] == "-" * 60

    def test_log_outputs_only_shape(self, caplog):
        """The outputs-only fallback is a single two-line message."""
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_print"):
            d._log_outputs_only("other", (1, 2), op_tag="[op=1,1]", exec_order=9)
        message = caplog.records[-1].getMessage()
        assert message.split("\n") == [
            "[IO_PRINT][rank=0][step=0][exec_order=9][op=1,1] other OUTPUTS:",
            "  result[0]: 1",
            "  result[1]: 2",
        ]

    def test_print_pairing_is_lifo_per_op(self):
        """Print pairing entries stack per op name and pop in LIFO order."""
        d._push_print_pairing("op", "first", 1, ["a"], "[t1]")
        d._push_print_pairing("op", "second", 2, ["b"], "[t2]")
        d._push_print_pairing("other", "other-label", 3, [], "[t3]")
        assert d._pop_print_pairing("op") == ("second", 2, ["b"], "[t2]")
        assert d._pop_print_pairing("op") == ("first", 1, ["a"], "[t1]")
        assert d._pop_print_pairing("op") is None
        assert d._pop_print_pairing("other") == ("other-label", 3, [], "[t3]")
        assert d._pop_print_pairing("never-pushed") is None

    def test_pop_print_pairing_without_stack_returns_none(self):
        """A pristine thread-local yields None instead of raising."""
        old = d._print_pairing
        d._print_pairing = threading.local()
        try:
            assert d._pop_print_pairing("op") is None
        finally:
            d._print_pairing = old


class TestPairingStacks:
    """dump_before/dump_after pairing and cleanup."""

    def test_push_pop_pairing_defaults_and_lifo(self):
        """label defaults to op_name; entries pop LIFO per op."""
        d._push_pairing("op", 1, 5, "dirA")
        d._push_pairing("op", 2, 6, "dirB", label="lbl", module_tag="M", op_tag="T")
        assert d._pop_pairing("op") == (2, 6, "dirB", "lbl", "M", "T")
        assert d._pop_pairing("op") == (1, 5, "dirA", "op", "", "")
        assert d._pop_pairing("op") is None

    def test_pop_pairing_without_stack_returns_none(self):
        """A pristine thread-local yields None instead of raising."""
        old = d._dump_pairing
        d._dump_pairing = threading.local()
        try:
            assert d._pop_pairing("op") is None
        finally:
            d._dump_pairing = old

    def test_dump_cleanup_pops_both_pairings(self, caplog, tmp_path):
        """dump_cleanup removes stale file and print pairings at once."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, with_print=True)
        try:
            tensor = torch.zeros(2)
            d.dump_before("opcl", (tensor,), {})
            # Simulate the op failing: cleanup must empty both stacks.
            d.dump_cleanup("opcl")
            assert d._pop_pairing("opcl") is None
            assert d._pop_print_pairing("opcl") is None
            # dump_after now warns about the missing pairing.
            with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.io_dump"):
                d.dump_after("opcl", (tensor,), tensor)
            assert any(
                "No pairing info for dump output 'opcl'" in r.getMessage()
                for r in caplog.records
            )
        finally:
            d.disable_io_dump()

    def test_dump_output_without_pairing_warns(self, caplog):
        """_dump_output logs a warning and returns when pairing is absent."""
        with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.io_dump"):
            d._dump_output("ghost", torch.zeros(1))
        assert any(
            "No pairing info for dump output 'ghost'" in r.getMessage()
            for r in caplog.records
        )


class TestRankGate:
    """_rank_ok and rank filtering."""

    def test_no_rank_filter_allows_all(self):
        """A None filter (default) lets any rank through."""
        d._rank_filter = None
        assert d._rank_ok() is True

    def test_rank_filter_membership(self, monkeypatch):
        """_rank_ok consults the module-bound get_rank() for membership."""
        d._rank_filter = {0}
        assert d._rank_ok() is True  # default rank is 0
        monkeypatch.setattr(d, "get_rank", lambda: 5)
        assert d._rank_ok() is False
        d._rank_filter = {5, 7}
        assert d._rank_ok() is True
        d._rank_filter = None

    def test_dump_skipped_on_filtered_rank(self, tmp_path):
        """Nothing is dumped when the current rank is filtered out."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, ranks={5})
        try:
            tensor = torch.zeros(2)
            d.dump_before("ranked-op", (tensor,), {}, exec_order=1)
            d.dump_after("ranked-op", (tensor,), tensor)
            d._wait_and_flush()
            assert not os.listdir(tmp_path)
        finally:
            d.disable_io_dump()


class TestShouldDumpGates:
    """_should_dump / _should_dump_torch_func / _check_limits composable gates."""

    def test_layer_filter_blocks_without_context(self):
        """An active layer filter with no module context rejects the call."""
        d._layer_filter = {"model.layers.0"}
        try:
            assert d._should_dump("op", ()) is False
        finally:
            d._layer_filter = set()

    def test_layer_filter_passes_with_matching_context(self):
        """A matching module-context path satisfies the layer gate."""
        d._layer_filter = {"model.layers.0"}
        d._op_filter = {"op"}  # the final gate needs an op or module filter
        io_common.push_module_context("Linear")
        io_common._module_context.stack[-1] = (
            "Linear",
            0,
            1,
            "model.layers.0.self_attn.q_proj",
        )
        try:
            assert d._should_dump("op", ()) is True
        finally:
            io_common.pop_module_context()
            d._layer_filter = set()
            d._op_filter = set()

    def test_module_filter_via_args(self):
        """args[0] nn.Module class name can satisfy the module filter."""
        d._module_filter = {"Linear"}
        linear = torch.nn.Linear(2, 2)
        try:
            assert d._should_dump("op", (linear,)) is True
            assert d._should_dump("op", (torch.zeros(1),)) is False
        finally:
            d._module_filter = set()

    def test_module_filter_via_context(self):
        """Thread-local module context can satisfy the module filter."""
        d._module_filter = {"Conv2d"}
        assert d._should_dump("op", ()) is False
        io_common.push_module_context("Conv2d")
        try:
            assert d._should_dump("op", ()) is True
        finally:
            io_common.pop_module_context()
            d._module_filter = set()

    def test_op_filter_membership(self):
        """The op filter gate requires exact op-name membership."""
        d._op_filter = {"opx"}
        try:
            assert d._should_dump("opx", ()) is True
            assert d._should_dump("opy", ()) is False
        finally:
            d._op_filter = set()

    def test_match_all_short_circuits_filters(self):
        """match_all bypasses module/op filters entirely."""
        d._match_all = True
        d._op_filter = set()
        d._module_filter = {"Nonexistent"}
        assert d._should_dump("anything", ()) is True
        d._match_all = False
        d._module_filter = set()

    def test_no_filters_means_no_dump(self):
        """With no op/module filter configured nothing is dumped."""
        d._op_filter = set()
        d._module_filter = set()
        assert d._should_dump("op", ()) is False

    def test_step_range_and_max_calls_limits(self):
        """_check_limits enforces the half-open step range and per-op cap."""
        d._op_filter = {"opx"}
        d._step_range = (0, 1)
        assert d._should_dump("opx", ()) is True
        io_common._step_counter = 1
        assert d._should_dump("opx", ()) is False  # end-exclusive
        io_common.reset_step()
        d._step_range = None

        d._max_calls = 1
        d._call_counters["opz"] = 1
        assert d._should_dump("opz", ()) is False
        d._call_counters.clear()
        # opz is not in _op_filter; use an op that is, with a fresh counter.
        assert d._should_dump("opx", ()) is True  # counter below the cap
        d._max_calls = 0
        d._op_filter = set()

    def test_should_dump_torch_func_gates(self):
        """Torch-func gate: enablement, filter set, module context, layers."""
        d._torch_funcs_enabled = True
        assert d._should_dump_torch_func("softmax") is True
        assert d._should_dump_torch_func("_private") is False  # dunder-ish

        d._torch_func_filter = {"matmul"}
        assert d._should_dump_torch_func("matmul") is True
        assert d._should_dump_torch_func("softmax") is False
        d._torch_func_filter = set()

        d._module_filter = {"Conv2d"}
        assert d._should_dump_torch_func("softmax") is False
        io_common.push_module_context("Conv2d")
        try:
            assert d._should_dump_torch_func("softmax") is True
        finally:
            io_common.pop_module_context()
            d._module_filter = set()

        d._layer_filter = {"model.layers.0"}
        assert d._should_dump_torch_func("softmax") is False
        d._layer_filter = set()

        d._step_range = (0, 1)
        io_common._step_counter = 9
        assert d._should_dump_torch_func("softmax") is False
        io_common.reset_step()
        d._step_range = None

        d._torch_funcs_enabled = False
        assert d._should_dump_torch_func("softmax") is False


class TestSerializationHelpers:
    """_serialize_value / _build_data / _extract_tensor_refs / path helpers."""

    def test_extract_tensor_refs_key_scheme(self):
        """Only tensors are captured, detached, with the same key scheme."""
        tensor = torch.ones(2)
        refs = d._extract_tensor_refs((tensor, 1), {"k": tensor})
        assert set(refs) == {"arg_0", "kwarg_k"}
        assert all(torch.equal(v, tensor) for v in refs.values())
        refs_out = d._extract_tensor_refs(((tensor, 2),), {}, is_output=True)
        assert set(refs_out) == {"result_0"}
        assert torch.equal(refs_out["result_0"], tensor)
        assert d._extract_tensor_refs((5,), {}, is_output=True) == {}

    def test_sanitize_path_component(self):
        """Separators and '..' are neutralised; blank becomes _unnamed_."""
        assert d._sanitize_path_component("a/b") == "a_b"
        assert d._sanitize_path_component("..") == "__"
        assert d._sanitize_path_component("x..y") == "x__y"
        assert d._sanitize_path_component(".lead") == "lead"
        assert d._sanitize_path_component("trail.") == "trail"
        assert d._sanitize_path_component("  spaced  ") == "spaced"
        assert d._sanitize_path_component("") == "_unnamed_"
        assert d._sanitize_path_component("...") == "__"

    def test_get_op_dir_layout(self):
        """Directory layout is dump_dir/rank_XXXX/step_XXXX/sanitized."""
        d._dump_dir = "DUMP"
        try:
            assert d._get_op_dir("aten.add.Tensor") == os.path.join(
                "DUMP", "rank_0000", "step_0000", "aten.add.Tensor"
            )
            # Only '/' and '..' are cross-platform assertion targets.
            assert d._get_op_dir("weird/../op").endswith(
                os.path.join("rank_0000", "step_0000", "weird____op")
            )
        finally:
            d._dump_dir = ""

    def test_next_call_num_is_per_op_monotonic(self):
        """Counters are independent per op and start at 1."""
        assert d._next_call_num("a") == 1
        assert d._next_call_num("a") == 2
        assert d._next_call_num("b") == 1
        d._call_counters.clear()


class TestBackgroundWriter:
    """Executor submission, device events, reorder buffer, flush failures."""

    def test_submit_bg_noop_when_disabled(self):
        """_submit_bg drops work while the dumper is disabled."""
        d._enabled = False
        calls = []

        def record():
            calls.append(1)

        d._submit_bg(record)
        assert calls == []
        assert d._pending_futures == []

    def test_submit_bg_tracks_future_when_enabled(self):
        """An enabled dumper appends the future for later draining."""
        d._enabled = True
        try:
            d._submit_bg(lambda: 42)
            assert len(d._pending_futures) == 1
        finally:
            d._wait_and_flush()

    def test_record_device_event_fake_module(self, monkeypatch):
        """A non-cpu tensor with a torch.<type>.Event module yields an event."""
        recorded = []

        class FakeEvent:
            def record(self):
                recorded.append("record")

        fake_gpu = types.SimpleNamespace(Event=FakeEvent)

        class FakeTensor:
            device = types.SimpleNamespace(type="fakegpu")

        monkeypatch.setattr(torch, "fakegpu", fake_gpu, raising=False)
        event = d._record_device_event({"x": FakeTensor()})
        assert isinstance(event, FakeEvent)
        assert recorded == ["record"]

    def test_record_device_event_cpu_and_missing_attr(self):
        """CPU tensors and device modules without Event give None."""
        assert d._record_device_event({"x": torch.zeros(1)}) is None

        class NoEventTensor:
            device = types.SimpleNamespace(type="noeventgpu")

        marker = object()
        torch.noeventgpu = marker  # noqa: B010 - deliberate dynamic attr
        try:
            assert d._record_device_event({"x": NoEventTensor()}) is None
        finally:
            delattr(torch, "noeventgpu")

    def test_record_device_event_constructor_failure(self):
        """A raising Event() constructor degrades to None (never propagates)."""

        class BadEvent:
            def __init__(self):
                raise RuntimeError("boom")

        torch.badgpu = types.SimpleNamespace(  # noqa: B010 - dynamic attr
            Event=BadEvent
        )
        try:

            class BadTensor:
                device = types.SimpleNamespace(type="badgpu")

            assert d._record_device_event({"x": BadTensor()}) is None
        finally:
            delattr(torch, "badgpu")

    def test_bg_compute_and_write_reorders_out_of_order_lines(self, tmp_path):
        """Lines are written strictly in call_num order (reorder buffer)."""
        json_path = str(tmp_path / "input.json")
        meta = {"op_name": "op", "exec_order": 0}
        tensor = torch.tensor([1.0, 2.0])
        # call 2 arrives first: it must be buffered, not written.
        d._bg_compute_and_write(json_path, 2, dict(meta), {"arg_0": tensor}, None)
        assert (tmp_path / "input.json").stat().st_size == 0
        # call 1 completes second: both flush, in ascending order.
        d._bg_compute_and_write(json_path, 1, dict(meta), {"arg_0": tensor}, None)
        lines = (tmp_path / "input.json").read_text(encoding="utf-8").splitlines()
        assert [list(json.loads(line))[0] for line in lines] == [
            "call_1",
            "call_2",
        ]
        # Tensor stats land under "tensors".
        first = json.loads(lines[0])["call_1"]
        assert first["tensors"]["arg_0"]["shape"] == [2]
        assert first["tensors"]["arg_0"]["min"] == 1.0
        d._close_open_files()

    def test_bg_compute_and_write_syncs_device_event(self, tmp_path):
        """A non-None device event is synchronized before writing."""
        synced = []

        class FakeSyncEvent:
            def synchronize(self):
                synced.append(True)

        d._bg_compute_and_write(
            str(tmp_path / "e.json"), 1, {"op_name": "x"}, {}, FakeSyncEvent()
        )
        assert synced == [True]
        d._close_open_files()

    def test_bg_save_pt_writes_tensor_file(self, tmp_path):
        """_bg_save_pt serializes inputs to a loadable .pt file."""
        tensor = torch.tensor([1.0, 3.0])
        pt_path = str(tmp_path / "call_1_input.pt")
        d._bg_save_pt(
            (tensor,), {}, is_output=False, pt_path=pt_path, device_event=None
        )
        loaded = torch.load(pt_path, weights_only=True)
        assert torch.equal(loaded["arg_0"], tensor)

    def test_get_or_open_handle_is_cached_per_path(self, tmp_path):
        """The same json path maps to the same _WipHandle instance."""
        path = str(tmp_path / "a.json")
        handle1 = d._get_or_open_handle(path)
        handle2 = d._get_or_open_handle(path)
        assert handle1 is handle2
        assert handle1.next_write == 1
        d._close_open_files()

    def test_wait_and_flush_logs_failed_future(self, caplog):
        """A failed background task logs the exception and drains cleanly."""
        d._enabled = True

        def boom():
            raise ValueError("bg fail")

        d._submit_bg(boom)
        with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.io_dump"):
            d._wait_and_flush()
        assert any(
            "Background IO task failed" in r.getMessage() for r in caplog.records
        )
        assert d._pending_futures == []

    def test_close_open_files_swallows_oserror(self):
        """OSError on flush/close of an open handle is swallowed."""

        class BadFile:
            def flush(self):
                raise OSError("flush fail")

            def close(self):
                raise OSError("close fail")

        d._open_file_handles["bad"] = d._WipHandle(BadFile())
        d._close_open_files()  # must not raise
        assert d._open_file_handles == {}

    def test_reset_state_closes_handles_and_shuts_down(self):
        """_reset_state closes open handles and drops the executor."""

        class TrackingFile:
            closed = []

            def flush(self):
                pass

            def close(self):
                TrackingFile.closed.append(True)

        d._open_file_handles["x"] = d._WipHandle(TrackingFile())
        d._reset_state()
        assert TrackingFile.closed == [True]
        assert d._open_file_handles == {}
        assert d._enabled is False
        assert d._dump_dir == ""


class TestDumpInputOutput:
    """File-dump queueing paths including the exception branch."""

    def test_dump_input_and_output_round_trip(self, tmp_path):
        """A full before/after pair writes input/output JSON + summary."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, with_values=True)
        try:
            tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            d.dump_before(
                "rms_norm",
                (tensor,),
                {"eps": 1e-06},
                exec_order=1,
                op_tag="[op=0,1]",
            )
            d.dump_after("rms_norm", (tensor,), tensor.sum())
            d._wait_and_flush()

            op_dir = tmp_path / "rank_0000" / "step_0000" / "rms_norm"
            assert sorted(p.name for p in op_dir.iterdir()) == [
                "call_1_input.pt",
                "call_1_output.pt",
                "input.json",
                "output.json",
            ]
            entry = json.loads((op_dir / "input.json").read_text(encoding="utf-8"))[
                "call_1"
            ]
            assert entry["op_name"] == "rms_norm"
            assert entry["exec_order"] == 1
            assert entry["call_num"] == 1
            assert entry["step"] == 0 and entry["rank"] == 0
            assert entry["op_tag"] == "[op=0,1]"
            stats = entry["tensors"]["arg_0"]
            assert (
                stats["shape"] == [2, 2] and stats["min"] == 1.0 and stats["max"] == 4.0
            )
            saved = torch.load(op_dir / "call_1_input.pt", weights_only=True)
            assert torch.equal(saved["arg_0"], tensor)
            assert saved["kwarg_eps"] == 1e-06

            out_entry = json.loads(
                (op_dir / "output.json").read_text(encoding="utf-8")
            )["call_1"]
            assert "result" in out_entry["tensors"]
        finally:
            d.disable_io_dump()

        summary = json.loads(
            (tmp_path / "rank_0000" / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["rank"] == 0
        assert "rms_norm" in summary["non_flaggems_ops"]

    def test_dump_input_makedirs_failure_warns(self, tmp_path, caplog):
        """A dump directory under a regular file degrades to a warning."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        d._enabled = True
        d._dump_dir = str(blocker / "rank_0000")  # makedirs under a file fails
        d._with_metas = True
        d._match_all = True
        d._op_filter = set()
        d._module_filter = set()
        try:
            with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.io_dump"):
                d.dump_before("badop", (torch.zeros(2),), {})
            assert any(
                "Failed to queue input dump" in r.getMessage() for r in caplog.records
            )
        finally:
            d._reset_state()

    def test_dump_output_unpack_failure_warns(self, tmp_path, caplog):
        """A malformed pairing tuple makes _dump_output warn, not raise."""
        d._enabled = True
        d._dump_dir = str(tmp_path)
        d._with_metas = True
        d._match_all = True
        d._op_filter = set()
        d._module_filter = set()
        # Deliberately malformed: a 2-tuple where a 6-tuple is expected.
        d._dump_pairing.stack = {"badop": [(1, 2)]}
        try:
            with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.io_dump"):
                d.dump_after("badop", (torch.zeros(2),), torch.zeros(2))
            assert any(
                "Failed to queue output dump" in r.getMessage() for r in caplog.records
            )
        finally:
            d._reset_state()

    def test_summary_only_mode_records_without_files(self, tmp_path):
        """with_metas=False writes only summary.json, no per-op JSON."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=False)
        try:
            (torch.tensor([1.0]) + torch.tensor([1.0])).sum()
        finally:
            d.disable_io_dump()
        rank_dir = tmp_path / "rank_0000"
        assert [p.name for p in rank_dir.iterdir()] == ["summary.json"]
        summary = json.loads((rank_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["flaggems_ops"] == []
        assert summary["non_flaggems_ops"]  # aten ops recorded

    def test_summary_write_oserror_warns(self, tmp_path, caplog):
        """A summary.json target that cannot be written logs a warning."""
        rank_dir = tmp_path / "rank_0000"
        rank_dir.mkdir()
        (rank_dir / "summary.json").mkdir()  # a directory blocks the write
        d._dump_dir = str(tmp_path)
        d._op_summary["opx"] = {"dispatch_keys": "", "call_count": 1}
        try:
            with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.io_dump"):
                d._write_summary()
            assert any(
                "Failed to write summary" in r.getMessage() for r in caplog.records
            )
        finally:
            d._op_summary.clear()
            d._dump_dir = ""

    def test_write_summary_noop_without_dir_or_data(self):
        """No dump dir or empty summary leaves everything untouched."""
        d._dump_dir = ""
        d._op_summary.clear()
        assert d._write_summary() is None
        d._dump_dir = "DUMP"
        assert d._write_summary() is None  # empty _op_summary
        d._dump_dir = ""


class TestSummaryAccumulator:
    """_record_op_summary / _format_dispatch_keys_for_json / _is_flaggems_op."""

    def test_format_dispatch_keys(self):
        """Triples render as (key, impl, is_default) inside brackets."""
        assert (
            d._format_dispatch_keys_for_json(
                [("CUDA", "FlagGems", False), ("CPU", "CPU", True)]
            )
            == "[(CUDA, FlagGems, False), (CPU, CPU, True)]"
        )
        assert d._format_dispatch_keys_for_json([]) == "[]"

    def test_record_op_summary_counts_and_first_keys_stick(self):
        """The first dispatch_keys wins; call_count increments per call."""
        d._record_op_summary("x", [("CUDA", "FlagGems", False)])
        d._record_op_summary("x", None)  # later None keys do not overwrite
        d._record_op_summary("y", None)
        assert d._op_summary == {
            "x": {
                "dispatch_keys": "[(CUDA, FlagGems, False)]",
                "call_count": 2,
            },
            "y": {"dispatch_keys": "", "call_count": 1},
        }
        d._op_summary.clear()

    def test_is_flaggems_op_dispatch_keys_and_import_failure(self, monkeypatch):
        """FlagGems detection via dispatch keys; manager import failure False."""
        assert d._is_flaggems_op("op", "[(CUDA, FlagGems, False)]") is True
        # Replace the manager module with one whose import path raises:
        # the fallback path is exercised inside _is_flaggems_op.
        saved = sys.modules.get("vllm_fl.dispatch.manager")
        fake = types.ModuleType("vllm_fl.dispatch.manager")

        def boom():
            raise RuntimeError("manager unavailable")

        fake.get_default_manager = boom
        monkeypatch.setitem(sys.modules, "vllm_fl.dispatch.manager", fake)
        try:
            assert d._is_flaggems_op("op", "") is False
        finally:
            if saved is not None:
                monkeypatch.setitem(sys.modules, "vllm_fl.dispatch.manager", saved)

    def test_is_flaggems_op_via_manager_snapshot(self):
        """A DEFAULT-kind called impl is classified as a FlagGems op."""

        class FakeManager:
            _called_ops = {"mgmt_op": "flaggems.default"}

            class _registry:
                @staticmethod
                def snapshot():
                    class Snap:
                        impls_by_op = {
                            "mgmt_op": [
                                types.SimpleNamespace(
                                    impl_id="flaggems.default",
                                    kind=BackendImplKind.DEFAULT,
                                )
                            ]
                        }

                    return Snap()

        saved = sys.modules.get("vllm_fl.dispatch.manager")
        module = types.ModuleType("vllm_fl.dispatch.manager")
        module.get_default_manager = lambda: FakeManager()
        sys.modules["vllm_fl.dispatch.manager"] = module
        try:
            assert d._is_flaggems_op("mgmt_op", "") is True
            assert d._is_flaggems_op("unknown_op", "") is False
        finally:
            if saved is not None:
                sys.modules["vllm_fl.dispatch.manager"] = saved
            else:
                sys.modules.pop("vllm_fl.dispatch.manager", None)


class TestEnableIoDump:
    """Programmatic enable: 3-layer merge, print-only mode, env propagation."""

    def test_api_overrides_env(self, tmp_path, monkeypatch):
        """API-layer values win over environment-layer values."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "9")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "envop")
        d.enable_io_dump(
            dump_dir=str(tmp_path), ops={"apiop"}, max_calls=2, with_metas=True
        )
        try:
            assert d._max_calls == 2
            assert d._op_filter == {"apiop"}
            assert d._match_all is False
        finally:
            d.disable_io_dump()

    def test_yaml_overrides_env(self, tmp_path, monkeypatch):
        """YAML values win over env values; API-unset fields keep them."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "io_dump:\n"
            f"  dir: {tmp_path / 'yd'}\n"
            "  ops: [ya]\n"
            "  max_calls: 3\n"
            "  with_metas: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "envop")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "99")
        d.enable_io_dump()
        try:
            assert d._dump_dir == str(tmp_path / "yd")
            assert d._op_filter == {"ya"}
            assert d._max_calls == 3
            assert d._with_metas is True
        finally:
            d.disable_io_dump()

    def test_print_only_mode_skips_dir(self, monkeypatch):
        """with_print=True and no explicit dir yields a print-only config."""
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        d.enable_io_dump(with_print=True, ops={"op_a"}, with_metas=True)
        try:
            assert d._dump_dir == ""
            assert d._print_enabled is True
            assert os.environ["VLLM_FL_IO_DUMP"] == ""
            assert os.environ["VLLM_FL_IO_DUMP_OPS"] == "op_a"
        finally:
            d.disable_io_dump()

    def test_explicit_env_dir_beats_print_only(self, tmp_path, monkeypatch):
        """An env-provided directory keeps file dumping on despite with_print."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        d.enable_io_dump(with_print=True, with_metas=True)
        try:
            assert d._dump_dir == str(tmp_path)
            assert d._print_enabled is True
        finally:
            d.disable_io_dump()

    def test_env_propagation_sets_all_vars(self, tmp_path, monkeypatch):
        """enable_io_dump mirrors the resolved config into env vars."""
        d.enable_io_dump(
            dump_dir=str(tmp_path),
            ops={"b", "a"},
            modules={"M"},
            max_calls=5,
            step_range="1-4",
            with_torch_funcs=True,
            ranks={3, 1},
            with_values=True,
            with_metas=True,
            with_print=True,
        )
        try:
            assert os.environ["VLLM_FL_IO_DUMP"] == str(tmp_path)
            assert os.environ["VLLM_FL_IO_DUMP_OPS"] == "a,b"
            assert os.environ["VLLM_FL_IO_DUMP_MODULES"] == "M"
            assert os.environ["VLLM_FL_IO_DUMP_MAX_CALLS"] == "5"
            assert os.environ["VLLM_FL_IO_DUMP_STEP_RANGE"] == "1-4"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS"] == "1"
            assert os.environ["VLLM_FL_IO_DUMP_RANK"] == "1,3"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_VALUES"] == "1"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_METAS"] == "1"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_PRINT"] == "1"
        finally:
            d.disable_io_dump()
        # disable cleared everything again.
        assert not [k for k in os.environ if k.startswith("VLLM_FL_IO_DUMP")]

    def test_set_env_vars_pops_unset_dimensions(self, monkeypatch):
        """Unset dimensions remove their env var rather than blanking it."""
        for key in (
            "VLLM_FL_IO_DUMP_OPS",
            "VLLM_FL_IO_DUMP_LAYERS",
            "VLLM_FL_IO_DUMP_STEP_RANGE",
            "VLLM_FL_IO_DUMP_WITH_PRINT",
        ):
            monkeypatch.setenv(key, "stale")
        d._set_env_vars(
            "dir",
            None,
            None,
            set(),
            0,
            None,
            False,
            None,
            False,
            False,
            print_on=False,
        )
        for key in (
            "VLLM_FL_IO_DUMP_OPS",
            "VLLM_FL_IO_DUMP_LAYERS",
            "VLLM_FL_IO_DUMP_STEP_RANGE",
            "VLLM_FL_IO_DUMP_WITH_PRINT",
        ):
            assert key not in os.environ
        assert os.environ["VLLM_FL_IO_DUMP"] == "dir"

    def test_disable_resets_state(self, tmp_path):
        """disable_io_dump restores every module global to its default."""
        d.enable_io_dump(dump_dir=str(tmp_path), ops={"x"}, with_metas=True)
        assert d.is_dump_enabled() is True
        d.disable_io_dump()
        assert d.is_dump_enabled() is False
        assert d._dump_dir == ""
        assert d._op_filter == set()
        assert d._match_all is False
        assert d._with_metas is False
        assert d._print_enabled is False
        assert d._op_summary == {}

    def test_enable_with_layers_expands_specs(self, tmp_path):
        """Layer specs expand to full path prefixes."""
        d.enable_io_dump(dump_dir=str(tmp_path), layers={"0-1", "5"}, with_metas=True)
        try:
            assert d._layer_filter == {
                "model.layers.0",
                "model.layers.1",
                "model.layers.5",
            }
            assert d._match_all is True  # no ops/modules -> match all
        finally:
            d.disable_io_dump()


class TestInitFromEnv:
    """_init_from_env corner branches."""

    def test_already_enabled_is_noop(self):
        """When the API path already enabled dumping, env init returns."""
        d._enabled = True
        d._init_from_env()
        assert d._enabled is True

    def test_env_zero_resets_and_returns(self, monkeypatch):
        """VLLM_FL_IO_DUMP=0 explicitly disables and resets the state."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "0")
        d._init_from_env()
        assert d._enabled is False
        assert d._dump_dir == ""

    def test_no_config_resets_and_returns(self, monkeypatch):
        """No env, no yaml, no print request resets the state."""
        for key in list(os.environ):
            if key.startswith("VLLM_FL_IO_DUMP"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
        d._init_from_env()
        assert d._enabled is False

    def test_filter_envs_auto_enable(self, monkeypatch, tmp_path):
        """A step-range env alone auto-enables dumping for all ops."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "0-1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "1")
        monkeypatch.chdir(tmp_path)
        d._init_from_env()
        try:
            assert d._enabled is True
            assert d._match_all is True
            assert d._step_range == (0, 2)
            assert d._dump_dir == os.path.join(str(tmp_path), "io_dump")
        finally:
            d._reset_state()

    def test_makedirs_failure_without_print_disables(self, tmp_path, monkeypatch):
        """An un-creatable dump dir disables dumping unless print is on."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(blocker / "sub"))
        d._init_from_env()
        assert d._enabled is False
        assert d._dump_dir == ""

    def test_makedirs_failure_with_print_keeps_print(self, tmp_path, monkeypatch):
        """With print enabled, a bad dump dir degrades to print-only."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(blocker / "sub"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "1")
        d._init_from_env()
        try:
            assert d._enabled is True
            assert d._dump_dir == ""
            assert d._print_enabled is True
        finally:
            d._reset_state()

    def test_print_only_env_without_dir(self, monkeypatch):
        """VLLM_FL_IO_DUMP_WITH_PRINT=1 with no dir gives print-only mode."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "opx")
        d._init_from_env()
        try:
            assert d._enabled is True
            assert d._dump_dir == ""
            assert d._print_enabled is True
            assert d._op_filter == {"opx"}
        finally:
            d._reset_state()

    def test_yaml_overlays_env(self, tmp_path, monkeypatch):
        """YAML values override env-layer values in _init_from_env too."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "io_dump:\n"
            f"  dir: {tmp_path / 'yd'}\n"
            "  ops: [ya]\n"
            "  max_calls: 3\n"
            "  with_metas: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "envop")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "99")
        d._init_from_env()
        try:
            assert d._op_filter == {"ya"}
            assert d._max_calls == 3
            assert d._with_metas is True
            assert d._dump_dir == str(tmp_path / "yd")
        finally:
            d._reset_state()


class TestStepAdvance:
    """_on_step_advance: counters, summary logging, lazy mode activation."""

    def test_step_advance_logs_summaries(self, tmp_path, caplog):
        """Step summaries land on both the io_dump and io_print loggers."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, with_print=True)
        try:
            io_common._seen_modules.add("M1")
            io_common._seen_ops.add("op1")
            with caplog.at_level(logging.INFO):
                io_common.advance_step()
            dump_msgs = [
                r.getMessage()
                for r in caplog.records
                if r.name == "vllm_fl.dispatch.io_dump"
            ]
            print_msgs = [
                r.getMessage()
                for r in caplog.records
                if r.name == "vllm_fl.dispatch.io_print"
            ]
            assert any(
                "[IO_DUMP][rank=0][step=0] Step summary: "
                "modules=['M1'], ops=['op1']" in m
                for m in dump_msgs
            )
            assert any(
                "[IO_PRINT][rank=0][step=0] Step summary: "
                "modules=['M1'], ops=['op1']" in m
                for m in print_msgs
            )
        finally:
            d.disable_io_dump()

    def test_step_advance_clears_call_counters(self, tmp_path):
        """Per-op call counters reset at each step boundary."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            d._call_counters["someop"] = 4
            io_common.advance_step()
            assert d._call_counters == {}
        finally:
            d.disable_io_dump()

    def test_lazy_activation_within_step_range(self, tmp_path):
        """A deferred step_range enters dispatch modes only in range."""
        d.enable_io_dump(dump_dir=str(tmp_path), step_range="2-3", with_metas=True)
        try:
            assert d._hooks_activated is True
            assert not io_common.dispatch_mode_mgr.is_entered("dump")
            io_common.advance_step()  # completed 0 → next is 1 (out of range)
            assert not io_common.dispatch_mode_mgr.is_entered("dump")
            io_common.advance_step()  # completed 1 → next is 2 (in range)
            assert io_common.dispatch_mode_mgr.is_entered("dump")
            io_common.advance_step()  # completed 2 → next is 3 (out of range)
            # stays entered — only the enter path is lazy, exit is explicit
            assert io_common.dispatch_mode_mgr.is_entered("dump")
        finally:
            d.disable_io_dump()

    def test_advance_io_step_only_when_enabled(self, tmp_path):
        """advance_io_step is a no-op while the dumper is disabled."""
        before = io_common.get_step()
        d.advance_io_step()
        assert io_common.get_step() == before
        d.enable_io_dump(dump_dir=str(tmp_path))
        try:
            d.advance_io_step()
            assert io_common.get_step() == before + 1
        finally:
            d.disable_io_dump()

    def test_io_dump_step_delegates(self, tmp_path):
        """io_dump_step returns the new step number like advance_step."""
        d.enable_io_dump(dump_dir=str(tmp_path))
        try:
            assert d.io_dump_step() == 1
        finally:
            d.disable_io_dump()


class TestModeLifecycle:
    """pause/resume dispatch modes and module hooks."""

    def test_pause_and_resume_round_trip(self, tmp_path):
        """Pause exits the modes; resume re-enters while still enabled."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            assert io_common.dispatch_mode_mgr.is_entered("dump")
            d.pause_dispatch_modes()
            assert not io_common.dispatch_mode_mgr.is_entered("dump")
            d.resume_dispatch_modes()
            assert io_common.dispatch_mode_mgr.is_entered("dump")
        finally:
            d.disable_io_dump()

    def test_resume_after_disable_stays_out(self, tmp_path):
        """resume_dispatch_modes is a no-op once the dumper is disabled."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        d.disable_io_dump()
        d.resume_dispatch_modes()
        assert not io_common.dispatch_mode_mgr.is_entered("dump")

    def test_pause_without_modes_is_noop(self):
        """Pausing when nothing is entered does not raise."""
        d.pause_dispatch_modes()
        assert not io_common.dispatch_mode_mgr.is_entered("dump")

    def test_module_hooks_registered_and_removed(self, tmp_path):
        """register_io_module_hooks installs forward hooks when enabled."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            seen = []

            class Inner(torch.nn.Module):
                def forward(self, x):
                    seen.append(io_common.get_current_module())
                    return x * 2

            class Capturing(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.inner = Inner()

                def forward(self, x):
                    seen.append(io_common.get_current_module())
                    return self.inner(x)

            model = Capturing()
            before = sum(
                len(m._forward_pre_hooks) + len(m._forward_hooks)
                for m in model.modules()
            )
            d.register_io_module_hooks(model)
            assert len(d._module_hook_handles) > 0
            during = sum(
                len(m._forward_pre_hooks) + len(m._forward_hooks)
                for m in model.modules()
            )
            assert during > before
            # Inside forward the thread-local context names the module;
            # post-hooks pop everything afterwards so the stack ends empty.
            out = model(torch.ones(1, 2))
            assert torch.equal(out, torch.ones(1, 2) * 2)  # independent oracle
            assert seen == ["Capturing", "Inner"]
            assert io_common.get_current_module() is None
            d._remove_module_hooks()
            assert d._module_hook_handles == []
            after = sum(
                len(m._forward_pre_hooks) + len(m._forward_hooks)
                for m in model.modules()
            )
            assert after == before
        finally:
            d.disable_io_dump()

    def test_module_hooks_noop_when_disabled(self):
        """register_io_module_hooks is a no-op while dumping is disabled."""
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        before = sum(len(m._forward_pre_hooks) for m in model.modules())
        d.register_io_module_hooks(model)
        assert d._module_hook_handles == []
        assert sum(len(m._forward_pre_hooks) for m in model.modules()) == before

    def test_init_io_dump_from_env_gates(self, monkeypatch, tmp_path):
        """init_io_dump_from_env honours the eager gate and env presence."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "1")
        # Non-eager (graph mode): no-op.
        d.init_io_dump_from_env(eager=False)
        assert d.is_dump_enabled() is False
        d.init_io_dump_from_env(eager=True)
        try:
            assert d.is_dump_enabled() is True
            assert io_common.dispatch_mode_mgr.is_entered("dump")
        finally:
            d.disable_io_dump()

    def test_init_io_dump_from_env_no_vars(self, monkeypatch):
        """Without any VLLM_FL_IO_DUMP* vars nothing happens even in eager."""
        for key in list(os.environ):
            if key.startswith("VLLM_FL_IO_DUMP"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
        d.init_io_dump_from_env(eager=True)
        assert d.is_dump_enabled() is False


class TestDumpBeforeAfterGates:
    """dump_before / dump_after guard, rank, summary-only and print paths."""

    def test_summary_only_records_seen_and_summary(self, tmp_path):
        """with_metas=False still records ops into the summary accumulator."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=False)
        try:
            tensor = torch.zeros(2)
            d.dump_before("summ-op", (tensor,), {})
            d.dump_after("summ-op", (tensor,), tensor)
            assert "summ-op" in d._op_summary
            assert d._op_summary["summ-op"]["call_count"] == 1
            assert d._op_summary["summ-op"]["dispatch_keys"] == ""
            # No per-op directory was created.
            assert not (tmp_path / "rank_0000" / "step_0000" / "summ-op").exists()
        finally:
            d.disable_io_dump()

    def test_dump_after_returns_when_not_with_metas(self, tmp_path, caplog):
        """dump_after without metas does no work at all."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=False)
        try:
            tensor = torch.zeros(2)
            with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_print"):
                d.dump_after("nope-op", (tensor,), tensor)
            assert not any("IO_PRINT" in r.getMessage() for r in caplog.records)
        finally:
            d.disable_io_dump()

    def test_print_pairing_logs_combined_block(self, tmp_path, caplog):
        """with_print=True emits a combined INPUTS+OUTPUTS block."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, with_print=True)
        tensor = torch.tensor([1.0, 2.0])
        try:
            with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_print"):
                d.dump_before("pop", (tensor,), {})
                d.dump_after("pop", (tensor,), tensor + tensor)
            combined = [
                r.getMessage() for r in caplog.records if "INPUTS" in r.getMessage()
            ]
            assert combined, "expected a combined INPUTS/OUTPUTS block"
            message = combined[-1]
            assert "[IO_PRINT][rank=0][step=0]" in message
            assert (
                "  arg[0]: Tensor(shape=[2], dtype=torch.float32, device=cpu, "
                "min=1, max=2, mean=1.5, std=0.707107)" in message
            )
            assert "  OUTPUTS:" in message
            assert (
                "  result: Tensor(shape=[2], dtype=torch.float32, device=cpu, "
                "min=2, max=4, mean=3, std=1.41421)" in message
            )
        finally:
            d.disable_io_dump()

    def test_print_without_pairing_logs_outputs_only(self, tmp_path, caplog):
        """A print-enabled dump_after with no pairing logs outputs only."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, with_print=True)
        tensor = torch.tensor([5.0])
        try:
            with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.io_print"):
                d.dump_after("solo", (tensor,), tensor)
            outs = [
                r.getMessage()
                for r in caplog.records
                if r.getMessage().startswith("[IO_PRINT]")
                and "OUTPUTS:" in r.getMessage()
            ]
            assert outs, "expected an outputs-only message"
            assert "OUTPUTS:" in outs[-1].split("\n")[0]
        finally:
            d.disable_io_dump()

    def test_guard_active_blocks_reentry(self, tmp_path):
        """dump_before/dump_after are no-ops while the guard is held."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            d.set_guard(True)
            tensor = torch.zeros(2)
            d.dump_before("guarded", (tensor,), {})
            d.dump_after("guarded", (tensor,), tensor)
            d.set_guard(False)
            assert "guarded" not in d._op_summary
        finally:
            d.set_guard(False)
            d.disable_io_dump()

    def test_module_tag_from_context_in_metadata(self, tmp_path):
        """Module context from hooks lands in the JSON module_tag field."""

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.inner(x)

        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            model = Wrapper()
            d.register_io_module_hooks(model)
            model(torch.zeros(1, 2))
            d._wait_and_flush()
            op_dir = tmp_path / "rank_0000" / "step_0000"
            op_dirs = [p.name for p in op_dir.iterdir()]
            assert op_dirs, "expected at least one op dir from the forward pass"
            found_tag = False
            for name in op_dirs:
                path = op_dir / name / "input.json"
                if path.exists():
                    # input.json is JSON Lines: one object per call.
                    for line in path.read_text(encoding="utf-8").splitlines():
                        data = json.loads(line)
                        for entry in data.values():
                            if "Linear:inner" in entry.get("module_tag", ""):
                                found_tag = True
            assert found_tag, "expected 'Linear:inner' in a module_tag"
        finally:
            d.disable_io_dump()


class TestDispatchModeInterception:
    """End-to-end TorchDispatchMode / TorchFunctionMode interception."""

    def test_dispatch_mode_dumps_aten_ops(self, tmp_path):
        """Plain tensor math is captured under rank/step directories."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            tensor = torch.tensor([1.0, 2.0])
            result = tensor + tensor
            assert torch.equal(result, tensor * 2)  # independent oracle
            d._wait_and_flush()
            op_dir = tmp_path / "rank_0000" / "step_0000"
            names = {p.name for p in op_dir.iterdir()}
            assert "aten.add.Tensor" in names
            entry = json.loads(
                (op_dir / "aten.add.Tensor" / "input.json").read_text(encoding="utf-8")
            )
            call = entry["call_1"]
            assert call["op_name"] == "aten.add.Tensor"
            assert call["tensors"]["arg_0"]["min"] == 1.0
        finally:
            d.disable_io_dump()

    def test_dispatch_mode_exception_cleans_pairing(self, tmp_path):
        """A failing aten op propagates and leaves no pairing behind."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            with pytest.raises(RuntimeError):
                torch.zeros(2).reshape(4, 3)  # invalid shape
            stack = getattr(d._dump_pairing, "stack", {})
            assert all(not entries for entries in stack.values())
        finally:
            d.disable_io_dump()

    def test_torch_func_mode_intercepts_bare_functions(self, tmp_path):
        """with_torch_funcs=True captures torch.matmul and friends."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, with_torch_funcs=True)
        try:
            left = torch.tensor([[1.0, 2.0]])
            result = torch.matmul(left, left.transpose(0, 1))
            # Independent oracle: 1*1 + 2*2.
            assert result.item() == 5.0
            d._wait_and_flush()
            names = {p.name for p in (tmp_path / "rank_0000" / "step_0000").iterdir()}
            assert "torch.matmul" in names
        finally:
            d.disable_io_dump()

    def test_torch_func_mode_exception_releases_tags(self, tmp_path):
        """A failing torch function propagates and releases func tags."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, with_torch_funcs=True)
        try:
            with pytest.raises(RuntimeError):
                torch.matmul(torch.zeros(2), torch.zeros(3, 4))
            depth = getattr(io_common._tf_tags, "depth", 0)
            assert depth == 0
        finally:
            d.disable_io_dump()

    def test_step_range_gates_dispatch_mode(self, tmp_path):
        """Ops outside the step range are not intercepted."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, step_range="1-2")
        try:
            # Step 0 is outside [1, 3): nothing dumped (activation deferred).
            (torch.tensor([1.0]) + torch.tensor([1.0])).sum()
            d._wait_and_flush()
            assert not (tmp_path / "rank_0000").exists()
        finally:
            d.disable_io_dump()

    def test_layer_filter_gates_dispatch_mode(self, tmp_path):
        """A layer filter without matching context blocks dispatch dumps."""
        d.enable_io_dump(
            dump_dir=str(tmp_path), with_metas=True, layers={"model.layers.99"}
        )
        try:
            (torch.tensor([1.0]) + torch.tensor([1.0])).sum()
            d._wait_and_flush()
            rank_dir = tmp_path / "rank_0000"
            assert not rank_dir.exists() or not any(rank_dir.iterdir())
        finally:
            d.disable_io_dump()


class TestIoCommonLeftovers:
    """io_common gaps not exercised by the dumper integration paths."""

    def test_get_rank_uses_initialized_process_group(self, monkeypatch):
        """An initialized torch.distributed provides the rank (and caches it)."""
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 7)
        io_common.reset_rank()
        try:
            assert io_common.get_rank() == 7
            assert io_common._rank == 7  # cached: definitive source
        finally:
            io_common.reset_rank()

    def test_expand_layer_specs_skips_blank_entries(self):
        """Blank layer specs are skipped rather than crashing."""
        assert io_common.expand_layer_specs({"", "  ", "0-1"}) == {
            "model.layers.0",
            "model.layers.1",
        }

    def test_parse_dispatch_table_line_parsing(self, monkeypatch):
        """Kernel lines parse; fallthrough and marker-less lines are skipped."""
        table = (
            "CPU: registered at /pytorch/build/aten/src/ATen/"
            "RegisterCPU_0.cpp:100 [kernel]\n"
            "CUDA: registered at /site/flag_gems/__init__.py:20 "
            "[default backend kernel]\n"
            "BackendSelect: fallthrough registered at /x.cpp:1 "
            "[backend fallback]\n"
            "XPU: registered somewhere odd [kernel]\n"
            "Meta: registered at /torch/_meta_registrations.py:53 [kernel]\n"
        )

        class FakeOp:
            def name(self):
                return "aten::fake_op"

        monkeypatch.setattr(torch._C, "_dispatch_dump_table", lambda qname: table)
        entries = io_common._parse_dispatch_table(FakeOp())
        assert entries == [
            ("CPU", "CPU", False),
            ("CUDA", "FlagGems", True),
            ("Meta", "Meta", False),
        ]
        # The parsed list is cached under the qualified op name.
        assert io_common._dispatch_table_cache["aten::fake_op"] == entries

    def test_parse_dispatch_table_swallows_errors(self, monkeypatch):
        """A raising _dispatch_dump_table yields an empty (cached) list."""

        def boom(qname):
            raise RuntimeError("dump table unavailable")

        class FakeOp:
            def name(self):
                return "aten::raising_op"

        monkeypatch.setattr(torch._C, "_dispatch_dump_table", boom)
        assert io_common._parse_dispatch_table(FakeOp()) == []
        assert io_common._dispatch_table_cache["aten::raising_op"] == []

    def test_layer_path_matches_from_stack_with_explicit_ctx(self):
        """Prefix and glob matching work on stack-derived context tuples."""
        ctx = [("Linear", "model.layers.0.self_attn.q_proj"), ("", "skipped")]
        assert io_common.layer_path_matches_from_stack({"model.layers.0"}, ctx)
        # fnmatch globs match the whole path; '*' spans dots.
        assert io_common.layer_path_matches_from_stack({"model.layers.*.q_proj"}, ctx)
        assert not io_common.layer_path_matches_from_stack(
            {"model.layers.*.self_attn"}, ctx
        )  # path continues past self_attn (.q_proj)
        assert not io_common.layer_path_matches_from_stack({"model.layers.5"}, ctx)

    def test_layer_path_matches_from_stack_empty_path_skipped(self):
        """Context entries without a path are skipped."""
        assert not io_common.layer_path_matches_from_stack(
            {"anything"}, [("Linear", "")]
        )

    def test_stack_matchers_derive_context_when_none(self):
        """module_ctx=None derives the context from the live call stack."""

        captured = {}

        class Probe(torch.nn.Module):
            def forward(self, x):
                captured["prefix"] = io_common.layer_path_matches_from_stack(
                    {"model.layers"}
                )
                captured["glob"] = io_common.layer_path_matches_from_stack(
                    {"model.layers.*"}
                )
                captured["cls"] = io_common.module_context_matches_from_stack({"Probe"})
                return x

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Probe()])

            def forward(self, x):
                return self.model.layers[0](x)

        root = Root()
        io_common.register_module_paths(root)
        root(torch.zeros(1, 2))
        assert captured == {"prefix": True, "glob": True, "cls": True}

    def test_module_context_matches_from_stack(self):
        """Class-name matching on explicit stack context."""
        ctx = [("Linear", "a.b"), ("Conv2d", "c.d")]
        assert io_common.module_context_matches_from_stack({"Linear"}, ctx)
        assert io_common.module_context_matches_from_stack({"Conv2d"}, ctx)
        assert not io_common.module_context_matches_from_stack({"LSTM"}, ctx)

    def test_tensor_stats_failing_stat_is_skipped(self):
        """A registered stat that raises is dropped, others still computed."""
        tensor = torch.tensor([1.0, 2.0])

        def bad_stat(t):
            raise RuntimeError("stat unavailable")

        io_common._TENSOR_STAT_REGISTRY.append(("bad", bad_stat, False))
        try:
            stats = io_common.tensor_stats(tensor)
        finally:
            io_common._TENSOR_STAT_REGISTRY.pop()
        assert "bad" not in stats
        assert stats["min"] == 1.0
        assert stats["max"] == 2.0

    def test_format_stat_value_list_and_truncation(self):
        """Lists format inline; long lists are truncated with '...'."""
        assert io_common._format_stat_value([1.0, 2.0]) == "[1, 2]"
        assert io_common._format_stat_value(("a",)) == "['a']"
        assert io_common._format_stat_value([1, 2, 3, 4, 5]) == "[1, 2, 3, 4, ...]"
        assert io_common._format_stat_value(5) == "5"
        assert io_common._format_stat_value("x") == "'x'"
        assert io_common._format_stat_value(0.5) == "0.5"

    def test_parse_string_list_other_types(self):
        """Unsupported YAML types degrade to an empty set."""
        assert io_common._parse_string_list(42) == set()
        assert io_common._parse_string_list({"a", "b"}) == set()

    def test_parse_ranks_yaml_other_types(self):
        """Unsupported ranks YAML types degrade to None (all ranks)."""
        assert io_common._parse_ranks_yaml(3.5) is None
        # NOTE: bool is a subclass of int in Python, so True -> {1}.
        assert io_common._parse_ranks_yaml([1, "x", 2]) == {1, 2}


# ---------------------------------------------------------------------------
# Remaining io_dumper branch corners (merged from the former supplementary
# test_io_dumper_config_layers.py — env-var parsing corners, dump_after
# early returns, torch-func summary-only pass-through, background-writer
# corner branches, pause/resume modes and the reset-state handle close).
# ---------------------------------------------------------------------------


class TestEnableIoDumpEnvCorners:
    """``enable_io_dump`` layer-1 env parsing corners."""

    def test_env_dir_one_keeps_default(self, tmp_path, monkeypatch):
        """``VLLM_FL_IO_DUMP=1`` keeps the cwd-based default directory."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        d.enable_io_dump()
        assert d._dump_dir == os.path.join(str(tmp_path), "io_dump")
        assert os.path.isdir(d._dump_dir)

    def test_env_dir_explicit_path(self, tmp_path, monkeypatch):
        """A non-``0``/``1`` env value is used verbatim as dump dir."""
        target = tmp_path / "envdir"
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(target))
        d.enable_io_dump()
        assert d._dump_dir == str(target)

    def test_bad_max_calls_and_bg_workers_kept(self, tmp_path, monkeypatch):
        """Unparseable ints fall back to their defaults (0 and 8)."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "not-a-number")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_BG_WORKERS", "oops")
        d.enable_io_dump()
        assert d._max_calls == 0
        assert d._bg_workers == 8

    def test_bg_workers_clamped_to_one(self, tmp_path, monkeypatch):
        """A zero bg_workers env value is clamped up to 1."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_BG_WORKERS", "0")
        d.enable_io_dump(bg_workers=-5)
        assert d._bg_workers == 1

    def test_env_ops_and_modules_parsed(self, tmp_path, monkeypatch):
        """Comma-separated env filters are trimmed into sets."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", " op.a , op.b ,, ")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MODULES", " Lin1 , Lin2 ")
        d.enable_io_dump()
        assert d._op_filter == {"op.a", "op.b"}
        assert d._module_filter == {"Lin1", "Lin2"}
        assert d._match_all is False

    def test_env_ranks_and_step_range(self, tmp_path, monkeypatch):
        """Rank filter and step range env values are parsed."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_RANK", "0,2")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "1-3")
        d.enable_io_dump()
        assert d._rank_filter == {0, 2}
        assert tuple(d._step_range) == (1, 4)

    def test_env_torch_funcs_filter(self, tmp_path, monkeypatch):
        """``with_torch_funcs`` env accepts an enable plus name filter."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "sin,cos")
        d.enable_io_dump()
        assert d._torch_funcs_enabled is True
        assert d._torch_func_filter == {"sin", "cos"}

    def test_with_values_and_metas_env(self, tmp_path, monkeypatch):
        """``with_values``/``with_metas``/``with_print`` truthy env values."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_VALUES", "true")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "true")
        d.enable_io_dump()
        assert d._with_values is True
        assert d._with_metas is True
        assert d._print_enabled is True

    def test_with_print_false_env(self, tmp_path, monkeypatch):
        """An explicit falsey print env var keeps printing off."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "false")
        d.enable_io_dump()
        assert d._print_enabled is False

    def test_api_empty_dir_becomes_default(self, tmp_path, monkeypatch):
        """An explicit falsy dump_dir falls back to the cwd default dir."""
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        monkeypatch.chdir(tmp_path)
        d.enable_io_dump(dump_dir="", with_print=True)
        assert d._dump_dir == os.path.join(str(tmp_path), "io_dump")
        assert d._print_enabled is True

    def test_layers_env_parsed(self, tmp_path, monkeypatch):
        """Layer specs from env expand into the layer filter."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_LAYERS", "0-1")
        d.enable_io_dump()
        assert d._layer_filter  # non-empty after expansion
        assert "model.layers.0" in d._layer_filter
        assert "model.layers.1" in d._layer_filter

    def test_none_ops_resets_to_match_all(self, tmp_path, monkeypatch):
        """An explicit ops=None API override widens back to match-all."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "op.a")
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        d.enable_io_dump(ops=None)
        assert d._match_all is True
        assert d._op_filter == set()


class TestDumpAfterCorners:
    """``dump_after`` early returns and the no-pairing print path."""

    def test_non_matching_op_returns_early(self, tmp_path, monkeypatch, caplog):
        """An op outside the filter returns before touching pairings."""
        d.enable_io_dump(
            dump_dir=str(tmp_path), ops=["other.op"], with_metas=True, with_print=True
        )
        with caplog.at_level(logging.WARNING, logger=d.logger.name):
            d.dump_after("kept.op", (torch.ones(2),), torch.ones(2))
        assert not os.path.exists(
            os.path.join(d._dump_dir, "rank_0000", "step_0000", "kept.op")
        )
        # The warning belongs to the un-matched early return, not a pairing
        # miss — nothing was pushed, so no message is expected either way.
        assert d._pop_pairing("kept.op") is None

    def test_print_without_pairing_logs_outputs_only(self, tmp_path, caplog):
        """A print-enabled dump_after without pushed pairing logs outputs."""
        d.enable_io_dump(dump_dir="", with_print=True, with_metas=True, ops=["solo.op"])
        with caplog.at_level(logging.INFO, logger=d.logger.name):
            d.dump_after("solo.op", (torch.ones(3),), torch.zeros(3))
        joined = "\n".join(r.message for r in caplog.records)
        assert "OUTPUTS" in joined or "outputs" in joined.lower()


class TestTorchFuncSummaryOnly:
    """``_DumpTorchFuncMode`` summary-only pass-through."""

    def test_summary_only_records_seen(self, tmp_path):
        """with_metas=False records the op in _seen_ops and passes through."""
        mode_cls = getattr(d, "_DumpTorchFuncMode", None)
        if mode_cls is None:  # HAS_TORCH_FUNC_MODE off (no torch.overrides)
            pytest.skip("_DumpTorchFuncMode requires TorchFunctionMode support")
        d.enable_io_dump(
            dump_dir=str(tmp_path), with_torch_funcs=True, with_metas=False, ops=None
        )
        mode = mode_cls()
        tensor = torch.arange(4.0)
        out = mode.__torch_function__(torch.sin, [torch.Tensor], (tensor,), {})
        assert torch.equal(out, torch.sin(tensor))
        assert "torch.sin" in io_common._seen_ops


class TestBackgroundCorners:
    """Background-task corner branches."""

    def test_bg_compute_swallows_broken_stats(self, tmp_path):
        """A tensor whose stats raise is skipped, not fatal."""
        json_path = str(tmp_path / "input.json")
        skeleton = {"op_name": "op"}

        class _BrokenTensor:
            device = SimpleNamespace(type="cpu")

        refs = {"arg_0": _BrokenTensor()}
        d._bg_compute_and_write(json_path, 1, skeleton, refs, None)
        data = json.loads((tmp_path / "input.json").read_text(encoding="utf-8"))
        assert data == {"call_1": {"op_name": "op", "tensors": {}}}

    def test_bg_save_pt_syncs_device_event(self, tmp_path):
        """_bg_save_pt synchronizes the device event before saving."""
        synced = []

        class _Event:
            def synchronize(self):
                synced.append(True)

        pt_path = str(tmp_path / "call_1_input.pt")
        d._bg_save_pt(
            (torch.ones(2),),
            {},
            is_output=False,
            pt_path=pt_path,
            device_event=_Event(),
        )
        assert synced == [True]
        loaded = torch.load(pt_path, weights_only=False)
        assert set(loaded) == {"arg_0"}
        assert torch.equal(loaded["arg_0"], torch.ones(2))


class TestPauseResumeModes:
    """``pause_dispatch_modes`` mode-manager branches."""

    def test_pause_exits_func_mode_only(self, tmp_path):
        """Pausing with only the torch-function mode entered exits it."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_torch_funcs=True, with_metas=True)
        assert d.func_mode_mgr.is_entered("dump")
        d.pause_dispatch_modes()
        assert not d.func_mode_mgr.is_entered("dump")

    def test_resume_reenters_when_enabled(self, tmp_path):
        """Resuming re-enters modes while dumping stays enabled."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_torch_funcs=True, with_metas=True)
        d.pause_dispatch_modes()
        d.resume_dispatch_modes()
        assert d.dispatch_mode_mgr.is_entered("dump")
        assert d.func_mode_mgr.is_entered("dump")

    def test_resume_noop_after_disable(self, tmp_path):
        """Resuming after disable does not re-enter any mode."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_torch_funcs=True, with_metas=True)
        d.pause_dispatch_modes()
        d.disable_io_dump()
        d.resume_dispatch_modes()
        assert not d.dispatch_mode_mgr.is_entered("dump")


class TestResetStateHandleClose:
    """``_reset_state`` OSError-swallowing handle close."""

    def test_close_failure_swallowed(self, tmp_path):
        """A handle whose close() raises OSError is dropped silently."""

        class _BadFH:
            encoding = "utf-8"

            def flush(self):
                pass

            def close(self):
                raise OSError("closed already")

        handle = d._WipHandle(_BadFH())
        d._open_file_handles[str(tmp_path / "x.json")] = handle
        d._reset_state()
        assert d._open_file_handles == {}
