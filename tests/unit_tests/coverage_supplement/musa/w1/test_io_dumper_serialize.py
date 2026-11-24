# Copyright (c) 2026 BAAI. All rights reserved.

"""
High-coverage unit tests for io_dumper serialization / formatting helpers.

Self-contained: only needs CPU torch and tmp_path.  No accelerator, no vllm
runtime objects.  Exercises the pure/utility helpers in
``vllm_fl/dispatch/io_dumper.py``:

    _serialize_value, _build_data, _extract_tensor_refs,
    _format_inputs, _log_combined, _log_outputs_only,
    _push_print_pairing, _pop_print_pairing,
    _sanitize_path_component, _get_op_dir, _next_call_num,
    _format_dispatch_keys_for_json, _record_device_event
"""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch.io_common import reset_exec_order, reset_rank, reset_step


@pytest.fixture(autouse=True)
def _clean_state():
    reset_step()
    reset_exec_order()
    reset_rank()
    io_dumper.disable_io_dump()
    for tls, name in (
        (io_dumper._dump_pairing, "stack"),
        (io_dumper._print_pairing, "stack"),
    ):
        try:
            delattr(tls, name)
        except AttributeError:
            pass
    yield
    io_dumper.disable_io_dump()
    reset_step()
    reset_exec_order()


class TestSerializeValue:
    """_serialize_value moves tensors to CPU and stringifies exotic types."""

    def test_tensor_detached_and_cpu(self):
        t = torch.randn(2, 3, requires_grad=True)
        out = io_dumper._serialize_value(t)
        assert isinstance(out, torch.Tensor)
        assert out.device.type == "cpu"
        assert out.requires_grad is False
        assert out.shape == (2, 3)

    def test_module_rendered_as_string(self):
        m = torch.nn.Linear(3, 4)
        out = io_dumper._serialize_value(m)
        assert isinstance(out, str)
        assert "Linear" in out

    def test_dict_recurses(self):
        d = {"a": torch.zeros(2), "b": 7, "c": {"d": 1.5}}
        out = io_dumper._serialize_value(d)
        assert isinstance(out["a"], torch.Tensor)
        assert out["b"] == 7
        assert out["c"] == {"d": 1.5}

    def test_tuple_preserved(self):
        out = io_dumper._serialize_value((torch.zeros(1), None, 3))
        assert isinstance(out, tuple)
        assert isinstance(out[0], torch.Tensor)
        assert out[1] is None
        assert out[2] == 3

    def test_list_preserved(self):
        out = io_dumper._serialize_value([1, 2, 3])
        assert isinstance(out, list)
        assert out == [1, 2, 3]

    @pytest.mark.parametrize(
        "value",
        [1, 1.25, True, False, "text", None],
    )
    def test_scalars_passthrough(self, value):
        assert io_dumper._serialize_value(value) == value
        assert type(io_dumper._serialize_value(value)) is type(value)

    def test_unknown_object_stringified(self):
        obj = object()
        out = io_dumper._serialize_value(obj)
        assert isinstance(out, str)
        assert "object" in out


class TestBuildData:
    """_build_data chooses arg_N / kwarg_<name> / result[_N] keys."""

    def test_input_args_and_kwargs(self):
        t = torch.zeros(2, 3)
        data = io_dumper._build_data((t, None, 1.5), {"eps": 1e-6, "flag": True})
        assert isinstance(data["arg_0"], torch.Tensor)
        assert data["arg_1"] is None
        assert data["arg_2"] == 1.5
        assert data["kwarg_eps"] == 1e-6
        assert data["kwarg_flag"] is True

    def test_output_single(self):
        t = torch.ones(4)
        data = io_dumper._build_data((t,), {}, is_output=True)
        assert set(data) == {"result"}
        assert isinstance(data["result"], torch.Tensor)

    def test_output_tuple(self):
        data = io_dumper._build_data(
            ((torch.zeros(1), torch.ones(2)),), {}, is_output=True
        )
        assert "result_0" in data
        assert "result_1" in data

    def test_output_empty_args(self):
        data = io_dumper._build_data((), {}, is_output=True)
        assert data == {"result": None}


class TestExtractTensorRefs:
    """_extract_tensor_refs only keeps detached Tensor leaves."""

    def test_input_only_tensors(self):
        t = torch.randn(2, 3, requires_grad=True)
        refs = io_dumper._extract_tensor_refs((t, "nope", 5), {})
        assert set(refs) == {"arg_0"}
        assert refs["arg_0"].requires_grad is False

    def test_input_kwargs(self):
        t = torch.ones(4)
        refs = io_dumper._extract_tensor_refs((), {"w": t, "eps": 1e-3})
        assert set(refs) == {"kwarg_w"}
        assert refs["kwarg_w"].shape == torch.Size([4])

    def test_output_single(self):
        refs = io_dumper._extract_tensor_refs((torch.ones(3, 2),), {}, is_output=True)
        assert set(refs) == {"result"}
        assert refs["result"].shape == torch.Size([3, 2])

    def test_output_tuple_filters_non_tensors(self):
        refs = io_dumper._extract_tensor_refs(
            ((torch.zeros(2), 5, torch.ones(1)),), {}, is_output=True
        )
        assert set(refs) == {"result_0", "result_2"}

    def test_output_non_tensor(self):
        refs = io_dumper._extract_tensor_refs((42,), {}, is_output=True)
        assert refs == {}


class TestFormatInputs:
    """_format_inputs renders args / kwargs lazily for console logging."""

    def test_args_and_kwargs(self):
        lines = io_dumper._format_inputs((torch.zeros(2), 3), {"k": "v"})
        assert lines[0].startswith("  arg[0]:")
        assert "Tensor(" in lines[0]
        assert lines[1] == "  arg[1]: 3"
        assert lines[2] == "  k: str(...)"

    def test_skip_module_arg(self):
        m = torch.nn.Linear(2, 2)
        lines = io_dumper._format_inputs((m, torch.zeros(3)), {}, skip_module_arg=True)
        # module at index 0 skipped, tensor at index 1 kept
        assert len(lines) == 1
        assert lines[0].startswith("  arg[1]:")

    def test_skip_module_arg_non_module_first(self):
        lines = io_dumper._format_inputs((torch.zeros(1),), {}, skip_module_arg=True)
        assert len(lines) == 1
        assert lines[0].startswith("  arg[0]:")


class TestLogHelpers:
    """_log_combined / _log_outputs_only forward to the print logger."""

    def test_log_combined(self, monkeypatch):
        logger = SimpleNamespace(
            calls=[],
            info=lambda msg: logger.calls.append(msg),
        )
        monkeypatch.setattr(io_dumper, "_print_logger", logger)
        io_dumper._log_combined(
            "label", ["  arg[0]: Tensor()"], torch.zeros(2),
            op_tag="[op=0,1]", exec_order=5,
        )
        assert len(logger.calls) == 1
        msg = logger.calls[0]
        assert "INPUTS:" in msg
        assert "OUTPUTS:" in msg
        assert "[op=0,1]" in msg
        assert "exec_order=5" in msg

    def test_log_outputs_only(self, monkeypatch):
        logger = SimpleNamespace(
            calls=[],
            info=lambda msg: logger.calls.append(msg),
        )
        monkeypatch.setattr(io_dumper, "_print_logger", logger)
        io_dumper._log_outputs_only("lbl", (torch.zeros(1), torch.ones(1)), exec_order=2)
        assert len(logger.calls) == 1
        msg = logger.calls[0]
        assert "OUTPUTS:" in msg
        assert "exec_order=2" in msg


class TestPrintPairing:
    """Thread-local print pairing stack push/pop semantics."""

    def test_pop_without_push_returns_none(self):
        assert io_dumper._pop_print_pairing("missing") is None

    def test_push_then_pop_lifo(self):
        io_dumper._push_print_pairing("op", "lbl1", 1, ["a"], "tag1")
        io_dumper._push_print_pairing("op", "lbl2", 2, ["b"], "tag2")
        assert io_dumper._pop_print_pairing("op") == ("lbl2", 2, ["b"], "tag2")
        assert io_dumper._pop_print_pairing("op") == ("lbl1", 1, ["a"], "tag1")
        assert io_dumper._pop_print_pairing("op") is None

    def test_different_ops_independent(self):
        io_dumper._push_print_pairing("a", "la", 1, [], "ta")
        io_dumper._push_print_pairing("b", "lb", 2, [], "tb")
        assert io_dumper._pop_print_pairing("a")[0] == "la"
        assert io_dumper._pop_print_pairing("b")[0] == "lb"


class TestDumpPairing:
    """Thread-local dump pairing stack push/pop semantics."""

    def test_pop_without_push_returns_none(self):
        assert io_dumper._pop_pairing("missing") is None

    def test_push_then_pop_lifo(self):
        io_dumper._push_pairing("op", 1, 10, "/a", label="L1", module_tag="M1", op_tag="T1")
        io_dumper._push_pairing("op", 2, 20, "/b", label="L2", module_tag="M2", op_tag="T2")
        assert io_dumper._pop_pairing("op") == (2, 20, "/b", "L2", "M2", "T2")
        assert io_dumper._pop_pairing("op") == (1, 10, "/a", "L1", "M1", "T1")
        assert io_dumper._pop_pairing("op") is None

    def test_label_defaults_to_op_name(self):
        io_dumper._push_pairing("op", 1, 1, "/x")
        pairing = io_dumper._pop_pairing("op")
        assert pairing[3] == "op"


class TestSanitizePathComponent:
    """_sanitize_path_component prevents path traversal."""

    def test_os_separator_replaced(self):
        out = io_dumper._sanitize_path_component("a" + os.sep + "b")
        assert os.sep not in out
        assert out == "a_b"

    def test_dotdot_collapsed(self):
        out = io_dumper._sanitize_path_component("a..b")
        assert ".." not in out

    def test_strips_dots_and_spaces(self):
        assert io_dumper._sanitize_path_component("  .name.  ") == "name"

    def test_empty_becomes_unnamed(self):
        assert io_dumper._sanitize_path_component(" . ") == "_unnamed_"

    def test_plain_name_unchanged(self):
        assert io_dumper._sanitize_path_component("rms_norm") == "rms_norm"


class TestGetOpDir:
    """_get_op_dir builds dump_dir/rank_XXXX/step_XXXX/op_name."""

    def test_layout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        reset_step()
        op_dir = io_dumper._get_op_dir("mm")
        expected = os.path.join(str(tmp_path), "rank_0000", "step_0000", "mm")
        assert op_dir == expected

    def test_sanitizes_op_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        op_dir = io_dumper._get_op_dir("a/b")
        assert op_dir.endswith(os.path.join("step_0000", "a_b"))


class TestNextCallNum:
    """_next_call_num increments per-op counters monotonically."""

    def test_increments(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_call_counters", {})
        assert io_dumper._next_call_num("op") == 1
        assert io_dumper._next_call_num("op") == 2
        assert io_dumper._next_call_num("other") == 1

    def test_independent_ops(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_call_counters", {"op": 41})
        assert io_dumper._next_call_num("op") == 42


class TestFormatDispatchKeys:
    """_format_dispatch_keys_for_json renders (key, impl, is_default)."""

    def test_multiple(self):
        out = io_dumper._format_dispatch_keys_for_json(
            [("CUDA", "FlagGems", False), ("CPU", "CPU", True)]
        )
        assert out == "[(CUDA, FlagGems, False), (CPU, CPU, True)]"

    def test_single(self):
        assert io_dumper._format_dispatch_keys_for_json([("Meta", "Meta", False)]) == (
            "[(Meta, Meta, False)]"
        )

    def test_empty(self):
        assert io_dumper._format_dispatch_keys_for_json([]) == "[]"


class _FakeTensorRef:
    """Minimal object exposing the ``.device.type`` attribute."""

    def __init__(self, dev_type):
        self.device = SimpleNamespace(type=dev_type)


class TestRecordDeviceEvent:
    """_record_device_event returns None for CPU / unsupported devices."""

    def test_empty_dict(self):
        assert io_dumper._record_device_event({}) is None

    def test_cpu_tensor_returns_none(self):
        assert io_dumper._record_device_event({"a": torch.zeros(2)}) is None

    def test_non_cpu_with_event(self, monkeypatch):
        recorded = {"n": 0}

        class _Event:
            def record(self):
                recorded["n"] += 1

            def synchronize(self):
                pass

        fake_mod = SimpleNamespace(Event=_Event)
        monkeypatch.setattr(torch, "fakedev", fake_mod, raising=False)
        ref = _FakeTensorRef("fakedev")
        ev = io_dumper._record_device_event({"a": ref})
        assert isinstance(ev, _Event)
        assert recorded["n"] == 1

    def test_non_cpu_event_raises_is_swallowed(self, monkeypatch):
        class _BadEvent:
            def __init__(self):
                raise RuntimeError("no accelerator")

        fake_mod = SimpleNamespace(Event=_BadEvent)
        monkeypatch.setattr(torch, "fakedev", fake_mod, raising=False)
        ref = _FakeTensorRef("fakedev")
        assert io_dumper._record_device_event({"a": ref}) is None

    def test_non_cpu_no_event_attr_returns_none(self, monkeypatch):
        fake_mod = SimpleNamespace()  # no Event attribute
        monkeypatch.setattr(torch, "fakedev2", fake_mod, raising=False)
        ref = _FakeTensorRef("fakedev2")
        assert io_dumper._record_device_event({"a": ref}) is None
