# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.dispatch.io_dumper``.

``tests/unit_tests/dispatch/test_io_dumper.py`` covers the core file dump
lifecycle, env-var enablement, filter sets and summary writing. What the
baseline still misses — verified against the coverage JSON — is:

* the print helpers: ``_format_inputs`` (module-arg skip), the
  combined/outputs-only logging functions, and the print pairing stack;
* ``_on_step_advance``'s lazy dispatch-mode re-entry when a ``step_range``
  defers activation, plus its step-summary printing path;
* the ``_should_dump`` layer-filter reject arm and the
  ``_should_dump_torch_func`` step/module gates;
* ``_serialize_value`` dict values and ``_sanitize_path_component``
  traversal stripping;
* ``_record_device_event`` with a fake non-cpu device module;
* ``_bg_compute_and_write`` reorder buffering (out-of-order call numbers)
  and a stats call that raises;
* ``_dump_input``/``_dump_output`` exception swallowing (a dump dir under
  an unwritable parent), the missing-pairing early return, and
  ``dump_after``'s outputs-only fallback;
* the full ``enable_io_dump`` three-layer merge (env < yaml < api) with
  per-key yaml overlays, print-only dir suppression, ``bg_workers``
  clamping and env propagation via ``_set_env_vars``/``_clear_env_vars``;
* ``_init_from_env`` yaml overlay, the filter-env auto-enable, the
  undumpable-directory reset, and the rank-filter application;
* the summary-only arms of both TorchDispatchMode and TorchFunctionMode;
* ``_is_flaggems_op``'s manager fallback branch;
* ``_write_summary``'s OSError branch, ``_WipHandle`` close failure, and
  the FlagGems classification via injected dispatch keys.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch import io_common
from vllm_fl.dispatch import io_dumper as d

pytestmark = [pytest.mark.supplementary]


def _drain_module_context():
    stack = getattr(io_common._module_context, "stack", None)
    while stack:
        stack.pop()


def _drain_pairing_stacks():
    """Clear both thread-local pairing stacks (dump and print)."""
    for tls in (d._dump_pairing, d._print_pairing):
        stack = getattr(tls, "stack", None)
        if stack:
            stack.clear()


@pytest.fixture(autouse=True)
def _reset_dumper_state(clean_fl_env):
    """Reset io_dumper/io_common globals around each test."""
    saved_stats = list(io_common._TENSOR_STAT_REGISTRY)
    saved_callbacks = list(io_common._step_callbacks)
    io_common.reset_step()
    io_common.reset_exec_order()
    io_common.reset_rank()
    io_common.dispatch_mode_mgr.exit_all()
    io_common.func_mode_mgr.exit_all()
    _drain_module_context()
    _drain_pairing_stacks()
    d.disable_io_dump()
    yield
    io_common.reset_step()
    io_common.reset_exec_order()
    io_common.reset_rank()
    io_common.dispatch_mode_mgr.exit_all()
    io_common.func_mode_mgr.exit_all()
    _drain_module_context()
    _drain_pairing_stacks()
    d.disable_io_dump()
    io_common._TENSOR_STAT_REGISTRY[:] = saved_stats
    io_common._step_callbacks.clear()
    io_common._step_callbacks.extend(saved_callbacks)


def _write_yaml(tmp_path, io_cfg) -> str:
    lines = ["io_dump:"]
    for key, value in io_cfg.items():
        lines.append(f"  {key}: {json.dumps(value)}")
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _read_jsonl(path) -> dict:
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.update(json.loads(line))
    return data


class TestPrintHelpers:
    """Console-printing helpers left uncovered by the core suite."""

    def test_format_inputs_skips_module_arg_and_formats_kwargs(self):
        module = torch.nn.Linear(2, 2)
        lines = d._format_inputs(
            (module, torch.ones(2)), {"epsilon": 1e-6}, skip_module_arg=True
        )
        assert lines[0].startswith("  arg[1]: Tensor(")
        assert lines[1] == "  epsilon: 1e-06"
        # Without the skip the module is rendered as its repr
        lines2 = d._format_inputs((module,), {}, skip_module_arg=False)
        assert "arg[0]" in lines2[0]

    def test_log_combined_writes_structured_block(self, caplog):
        with caplog.at_level(logging.INFO, logger=d._print_logger.name):
            d._log_combined(
                "my_label", ["  arg[0]: x"], torch.ones(2),
                op_tag="[op=1,1]", exec_order=3,
            )
        joined = "\n".join(r.message for r in caplog.records)
        assert "[IO_PRINT][rank=0][step=0][exec_order=3][op=1,1] my_label" in joined
        assert "INPUTS:" in joined and "OUTPUTS:" in joined

    def test_log_outputs_only(self, caplog):
        with caplog.at_level(logging.INFO, logger=d._print_logger.name):
            d._log_outputs_only("solo", torch.zeros(3), op_tag="[op=2,1]")
        joined = "\n".join(r.message for r in caplog.records)
        assert "solo OUTPUTS:" in joined
        assert "result:" in joined

    def test_print_pairing_stack(self):
        assert d._pop_print_pairing("nope") is None
        d._push_print_pairing("op_a", "label", 1, ["x"], "[op=t]")
        entry = d._pop_print_pairing("op_a")
        assert entry == ("label", 1, ["x"], "[op=t]")
        assert d._pop_print_pairing("op_a") is None  # stack now empty


class TestStepAdvance:
    """``_on_step_advance`` re-entry and logging branches."""

    def test_lazy_dispatch_mode_entry(self, tmp_path):
        """step_range deferring activation re-enters at the boundary."""
        d.enable_io_dump(
            dump_dir=str(tmp_path), with_metas=True, step_range="1-2"
        )
        assert not io_common.dispatch_mode_mgr.is_entered("dump")
        # Completing step 0 brings next_step to 1, inside [1, 3).
        d._on_step_advance(0, set(), set())
        assert io_common.dispatch_mode_mgr.is_entered("dump")

    def test_step_summary_prints_when_ops_seen(self, tmp_path, caplog):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, with_print=True)
        with caplog.at_level(logging.INFO, logger=d.logger.name):
            d._on_step_advance(4, {"Lin"}, {"rms_norm"})
        joined = "\n".join(r.message for r in caplog.records)
        assert "Step summary" in joined
        assert "modules=['Lin']" in joined
        assert "ops=['rms_norm']" in joined


class TestShouldDumpGates:
    def test_layer_filter_rejects_without_context(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True,
                         layers={"model.layers.9"})
        assert not d._should_dump("any_op", ())

    def test_should_dump_torch_func_step_gate(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True,
                         with_torch_funcs=True, step_range="5-9")
        assert not d._should_dump_torch_func("matmul")  # step 0 < 5

    def test_should_dump_torch_func_module_gate(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True,
                         with_torch_funcs=True, modules={"Missing"})
        assert not d._should_dump_torch_func("matmul")


class TestSerializationHelpers:
    def test_serialize_value_dict(self):
        out = d._serialize_value({"a": torch.zeros(1), "b": None})
        assert isinstance(out["a"], torch.Tensor)
        assert out["b"] is None

    def test_sanitize_path_component(self):
        # os.sep is "/" on POSIX; ".." collapses and edge dots/space strip.
        assert d._sanitize_path_component("a/b/c..d. ") == "a_b_c__d"
        # Strips to nothing -> the _unnamed_ placeholder.
        assert d._sanitize_path_component(" . ") == "_unnamed_"

    def test_record_device_event_with_fake_device(self, monkeypatch):
        recorded = []

        class _Event:
            def record(self):
                recorded.append("recorded")

        fake_device_mod = SimpleNamespace(Event=lambda: _Event())
        monkeypatch.setattr(torch, "fakexu", fake_device_mod, raising=False)

        class _FakeTensor:
            device = SimpleNamespace(type="fakexu")

        refs = {"arg_0": _FakeTensor()}
        event = d._record_device_event(refs)
        assert event is not None
        assert recorded == ["recorded"]

    def test_record_device_event_cpu_only(self):
        event = d._record_device_event({"arg_0": torch.zeros(1)})
        assert event is None


class TestBackgroundWriter:
    def test_out_of_order_calls_are_reordered(self, tmp_path):
        """The reorder buffer writes lines in call_num order."""
        json_path = str(tmp_path / "input.json")
        handle = d._get_or_open_handle(json_path)
        # Simulate tasks completing out of order: 2 then 1.
        d._bg_compute_and_write(json_path, 2, {"op_name": "x"}, {}, None)
        d._bg_compute_and_write(json_path, 1, {"op_name": "x"}, {}, None)
        lines = (tmp_path / "input.json").read_text(encoding="utf-8").splitlines()
        assert [json.loads(l) for l in lines] == [
            {"call_1": {"op_name": "x", "tensors": {}}},
            {"call_2": {"op_name": "x", "tensors": {}}},
        ]
        assert handle.next_write == 3
        d._close_open_files()

    def test_bg_compute_swallows_broken_stats(self, tmp_path):
        class _BrokenTensor:
            device = SimpleNamespace(type="cpu")

        d._bg_compute_and_write(
            str(tmp_path / "input.json"), 1, {"op": "o"}, {"arg_0": _BrokenTensor()},
            None,
        )
        data = _read_jsonl(tmp_path / "input.json")
        assert data == {"call_1": {"op": "o", "tensors": {}}}

    def test_bg_save_pt_syncs_device_event(self, tmp_path):
        synced = []

        class _Event:
            def synchronize(self):
                synced.append(True)

        d._bg_save_pt(
            (torch.ones(2),), {}, is_output=False,
            pt_path=str(tmp_path / "c1.pt"), device_event=_Event(),
        )
        assert synced == [True]
        loaded = torch.load(tmp_path / "c1.pt", weights_only=False)
        assert torch.equal(loaded["arg_0"], torch.ones(2))

    def test_close_open_files_swallows_oserror(self):
        class _BadFH:
            encoding = "utf-8"

            def flush(self):
                pass

            def close(self):
                raise OSError("already closed")

        d._open_file_handles["/nowhere/x.json"] = d._WipHandle(_BadFH())
        d._close_open_files()  # must not raise
        assert d._open_file_handles == {}

    def test_reset_state_closes_bad_handle(self):
        class _BadFH:
            encoding = "utf-8"

            def close(self):
                raise OSError("nope")

        d._open_file_handles["/nowhere/y.json"] = d._WipHandle(_BadFH())
        d._reset_state()
        assert d._open_file_handles == {}


class TestDumpInputOutputEdges:
    def test_dump_input_swallows_unwritable_dir(self, tmp_path, monkeypatch, caplog):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        # Break the dump after enable so _get_op_dir raises on makedirs.
        monkeypatch.setattr(d, "_dump_dir", os.path.join(str(tmp_path), "x") * 80)
        with caplog.at_level(logging.WARNING, logger=d.logger.name):
            d._dump_input("op_q", (torch.zeros(1),), {}, exec_order=1)
        assert any("Failed to queue input dump" in r.message for r in caplog.records)

    def test_dump_output_swallows_internal_errors(self, tmp_path, monkeypatch, caplog):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        monkeypatch.setattr(d, "_pop_pairing", lambda name: (_ for _ in ()).throw(
            RuntimeError("pairing stack broken")
        ))
        with caplog.at_level(logging.WARNING, logger=d.logger.name):
            d._dump_output("op_q", torch.zeros(1))
        assert any("Failed to queue output dump" in r.message for r in caplog.records)

    def test_dump_output_missing_pairing_warns(self, tmp_path, caplog):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        # The enable call itself ran under the dispatch mode and may have
        # recorded other ops; only assert on our probe op's directory.
        op_dir = tmp_path / "rank_0000" / "step_0000" / "never_dumped"
        with caplog.at_level(logging.WARNING, logger=d.logger.name):
            d._dump_output("never_dumped", torch.zeros(1))
        assert any("No pairing info" in r.message for r in caplog.records)
        assert not op_dir.exists()

    def test_dump_after_without_pairing_logs_outputs_only(self, tmp_path, caplog):
        """with_print + no pushed pairing logs the outputs-only fallback."""
        d.enable_io_dump(
            dump_dir="", with_print=True, with_metas=True, ops=["solo_op"]
        )
        with caplog.at_level(logging.INFO, logger=d._print_logger.name):
            d.dump_after("solo_op", (torch.ones(3),), torch.zeros(3))
        joined = "\n".join(r.message for r in caplog.records)
        assert "OUTPUTS" in joined

    def test_dump_before_guard_and_rank_gates(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        d.set_guard(True)
        try:
            # Guard active: neither dump nor pairing happens.
            d.dump_before("guarded_op", (torch.zeros(1),), {})
        finally:
            d.set_guard(False)
        d._rank_filter = {7}  # rank 0 not allowed
        try:
            d.dump_before("rank_blocked_op", (torch.zeros(1),), {})
        finally:
            d._rank_filter = None
        assert not (tmp_path / "rank_0000").exists()

    def test_summary_only_before_records_seen_ops(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=False)
        d.dump_before("sumop", (torch.zeros(1),), {})
        d.dump_after("sumop", (torch.zeros(1),), torch.zeros(1))
        d._wait_and_flush()
        # no per-op directory, but the op summary accumulated
        assert not (tmp_path / "rank_0000" / "step_0000" / "sumop").exists()
        assert "sumop" in d._op_summary


class TestEnableIoDumpLayers:
    """The three-layer (env < yaml < api) merge in ``enable_io_dump``."""

    def test_env_values_parsed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", " op.a , op.b ,, ")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MODULES", " Lin1 , Lin2 ")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "not-a-number")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_BG_WORKERS", "oops")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_RANK", "0,2")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "1-3")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "sin,cos")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_VALUES", "true")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "true")
        d.enable_io_dump()
        assert d._op_filter == {"op.a", "op.b"}
        assert d._module_filter == {"Lin1", "Lin2"}
        assert d._max_calls == 0  # unparseable int kept default
        assert d._bg_workers == 8  # unparseable int kept default
        assert d._rank_filter == {0, 2}
        assert tuple(d._step_range) == (1, 4)
        assert d._torch_funcs_enabled is True
        assert d._torch_func_filter == {"sin", "cos"}
        assert d._with_values is True and d._with_metas is True
        assert d._print_enabled is True

    def test_env_with_print_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "false")
        d.enable_io_dump()
        assert d._print_enabled is False

    def test_env_dir_one_keeps_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        d.enable_io_dump()
        assert d._dump_dir == os.path.join(str(tmp_path), "io_dump")

    def test_yaml_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "env"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "env.op")
        monkeypatch.setenv(
            "VLLM_FL_CONFIG",
            _write_yaml(
                tmp_path,
                {
                    "dir": str(tmp_path / "yaml"),
                    "ops": ["yaml.op"],
                    "modules": ["Yam"],
                    "layers": ["model.layers.3"],
                    "max_calls": 4,
                    "step_range": [0, 2],
                    "with_torch_funcs": ["y1"],
                    "ranks": [3],
                    "with_values": True,
                    "with_metas": True,
                    "with_print": True,
                },
            ),
        )
        d.enable_io_dump()
        assert d._dump_dir == str(tmp_path / "yaml")
        assert d._op_filter == {"yaml.op"}
        assert d._module_filter == {"Yam"}
        assert "model.layers.3" in d._layer_filter
        assert d._max_calls == 4
        assert tuple(d._step_range) == (0, 3)
        assert d._torch_funcs_enabled is True and d._torch_func_filter == {"y1"}
        assert d._rank_filter == {3}
        assert d._with_values is True and d._with_metas is True
        assert d._print_enabled is True

    def test_yaml_falsy_keys_keep_env_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "env"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "env.op")
        monkeypatch.setenv(
            "VLLM_FL_CONFIG",
            _write_yaml(
                tmp_path,
                {
                    "dir": "",
                    "ops": [],
                    "modules": [],
                    "layers": [],
                    "max_calls": 0,
                    "with_values": False,
                    "with_metas": False,
                    "with_print": True,
                },
            ),
        )
        d.enable_io_dump()
        assert d._dump_dir == str(tmp_path / "env")
        assert d._op_filter == {"env.op"}
        assert d._layer_filter == set()
        assert d._with_values is False and d._with_metas is False
        assert d._print_enabled is True

    def test_api_overrides_all_layers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "env"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "env.op")
        monkeypatch.setenv(
            "VLLM_FL_CONFIG", _write_yaml(tmp_path, {"ops": ["yaml.op"]})
        )
        d.enable_io_dump(
            dump_dir=str(tmp_path / "api"),
            ops={"api.op"},
            modules={"ApiMod"},
            layers={"model.layers.7"},
            max_calls=6,
            step_range="2-5",
            with_torch_funcs=False,
            ranks={0},
            with_values=True,
            with_metas=True,
            with_print=True,
            bg_workers=0,
        )
        assert d._dump_dir == str(tmp_path / "api")
        assert d._op_filter == {"api.op"}
        assert d._module_filter == {"ApiMod"}
        assert d._layer_filter == {"model.layers.7"}
        assert d._max_calls == 6
        assert tuple(d._step_range) == (2, 6)
        assert d._torch_funcs_enabled is False
        assert d._rank_filter == {0}
        assert d._with_values is True and d._with_metas is True
        assert d._print_enabled is True
        assert d._bg_workers == 1  # clamped up to 1

    def test_api_none_widens_to_match_all(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "env.op")
        d.enable_io_dump(ops=None)
        assert d._match_all is True
        assert d._op_filter == set()

    def test_print_only_suppresses_default_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        d.enable_io_dump(with_print=True, with_metas=True)
        assert d._dump_dir == ""
        assert d._print_enabled is True

    def test_layers_string_becomes_set(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), layers="model.layers.0")
        assert d._layer_filter == {"model.layers.0"}

    def test_env_propagation_and_clear(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        d.enable_io_dump(
            dump_dir=str(tmp_path),
            ops={"op_z"},
            modules={"Mod"},
            layers={"model.layers.1"},
            max_calls=5,
            step_range="0-3",
            with_torch_funcs=True,
            ranks={1, 2},
            with_values=True,
            with_metas=True,
            with_print=True,
        )
        try:
            assert os.environ["VLLM_FL_IO_DUMP"] == str(tmp_path)
            assert os.environ["VLLM_FL_IO_DUMP_OPS"] == "op_z"
            assert os.environ["VLLM_FL_IO_DUMP_MODULES"] == "Mod"
            assert os.environ["VLLM_FL_IO_DUMP_LAYERS"] == "model.layers.1"
            assert os.environ["VLLM_FL_IO_DUMP_MAX_CALLS"] == "5"
            assert os.environ["VLLM_FL_IO_DUMP_STEP_RANGE"] == "0-3"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS"] == "1"
            assert os.environ["VLLM_FL_IO_DUMP_RANK"] == "1,2"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_VALUES"] == "1"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_METAS"] == "1"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_PRINT"] == "1"
        finally:
            d._clear_env_vars()
        for key in list(os.environ):
            assert not (key.startswith("VLLM_FL_IO_DUMP")
                       and os.environ[key] == str(tmp_path))


class TestInitFromEnv:
    """``_init_from_env`` overlay and edge branches."""

    def test_yaml_overlay(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "env"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "env.op")
        monkeypatch.setenv(
            "VLLM_FL_CONFIG",
            _write_yaml(
                tmp_path,
                {
                    "dir": str(tmp_path / "yaml"),
                    "ops": ["yaml.op"],
                    "modules": ["Yam"],
                    "layers": ["model.layers.3"],
                    "max_calls": 4,
                    "step_range": [0, 2],
                    "with_torch_funcs": ["y1"],
                    "ranks": [3],
                    "with_values": True,
                    "with_metas": True,
                    "with_print": True,
                },
            ),
        )
        d._init_from_env()
        assert d._enabled is True
        assert d._dump_dir == str(tmp_path / "yaml")
        assert d._op_filter == {"yaml.op"}
        assert d._module_filter == {"Yam"}
        assert "model.layers.3" in d._layer_filter
        assert d._max_calls == 4
        assert tuple(d._step_range) == (0, 3)
        assert d._torch_funcs_enabled is True and d._torch_func_filter == {"y1"}
        assert d._rank_filter == {3}
        assert d._with_values is True and d._with_metas is True
        assert d._print_enabled is True

    def test_filter_envs_auto_enable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        monkeypatch.setenv("VLLM_FL_IO_DUMP_LAYERS", "0-1")
        d._init_from_env()
        assert d._enabled is True
        assert d._dump_dir  # default cwd-based dir

    def test_step_range_env_alone_auto_enables(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        monkeypatch.delenv("VLLM_FL_IO_DUMP_LAYERS", raising=False)
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "0-1")
        d._init_from_env()
        assert d._enabled is True

    def test_zero_dir_disables(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "0")
        d._init_from_env()
        assert d._enabled is False

    def test_no_config_resets_state(self, monkeypatch):
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        monkeypatch.delenv("VLLM_FL_IO_DUMP_STEP_RANGE", raising=False)
        monkeypatch.delenv("VLLM_FL_IO_DUMP_LAYERS", raising=False)
        d._init_from_env()
        assert d._enabled is False

    def test_already_enabled_short_circuits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "other"))
        d._init_from_env()
        assert d._dump_dir == str(tmp_path)  # unchanged

    def test_print_only_without_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "1")
        d._init_from_env()
        assert d._enabled is True
        assert d._dump_dir == ""
        assert d._print_enabled is True

    def test_undumpable_dir_resets(self, monkeypatch, tmp_path):
        # A file path as the dump dir makes makedirs fail with OSError.
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(blocker))
        d._init_from_env()
        assert d._enabled is False


class TestSummaryClassification:
    def _summary(self, tmp_path):
        return json.loads(
            (tmp_path / "rank_0000" / "summary.json").read_text(encoding="utf-8")
        )

    def test_flaggems_vs_non_flaggems(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path))
        with d._lock:
            d._op_summary["aten.mm"] = {
                "dispatch_keys": "[(CUDA, FlagGems, False)]", "call_count": 2,
            }
            d._op_summary["aten.add"] = {
                "dispatch_keys": "[(CPU, CPU, False)]", "call_count": 1,
            }
        d.disable_io_dump()
        summary = self._summary(tmp_path)
        assert summary["flaggems_ops"] == ["aten.mm"]
        assert summary["non_flaggems_ops"] == ["aten.add"]

    def test_summary_oserror_swallowed(self, tmp_path, monkeypatch, caplog):
        d.enable_io_dump(dump_dir=str(tmp_path))
        d.dump_before("op_w", (torch.zeros(1),), {})
        monkeypatch.setattr(
            d.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("busy"))
        )
        with caplog.at_level(logging.WARNING, logger=d.logger.name):
            d._write_summary()
        assert any("Failed to write summary" in r.message for r in caplog.records)

    def test_is_flaggems_op_manager_fallback(self, tmp_path):
        """The manager-registry fallback for dispatch-managed ops."""
        from vllm_fl.dispatch import get_default_manager

        mgr = get_default_manager()
        d.enable_io_dump(dump_dir=str(tmp_path))
        try:
            # No impl recorded for a made-up op: fallback answers False.
            assert d._is_flaggems_op("never_registered_op", "") is False
            # A real registered op resolves through the registry snapshot.
            registered = mgr.registry.snapshot().impls_by_op
            if registered:
                some_op = next(iter(registered))
                result = d._is_flaggems_op(some_op, "")
                assert isinstance(result, bool)
        finally:
            d.disable_io_dump()

    def test_no_summary_without_dir(self, tmp_path):
        d.enable_io_dump(dump_dir="")
        d.dump_before("op_x", (torch.zeros(1),), {})
        d.disable_io_dump()
        assert not (tmp_path / "rank_0000").exists()


class TestModeLifecycleHelpers:
    def test_pause_resume_roundtrip(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True,
                         with_torch_funcs=True)
        assert io_common.dispatch_mode_mgr.is_entered("dump")
        assert io_common.func_mode_mgr.is_entered("dump")
        d.pause_dispatch_modes()
        assert not io_common.dispatch_mode_mgr.is_entered("dump")
        assert not io_common.func_mode_mgr.is_entered("dump")
        d.resume_dispatch_modes()
        assert io_common.dispatch_mode_mgr.is_entered("dump")
        assert io_common.func_mode_mgr.is_entered("dump")

    def test_resume_after_disable_is_noop(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        d.pause_dispatch_modes()
        d.disable_io_dump()
        d.resume_dispatch_modes()
        assert not io_common.dispatch_mode_mgr.is_entered("dump")

    def test_exit_dispatch_modes_requests_both(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True,
                         with_torch_funcs=True)
        d._exit_dispatch_modes()
        assert not io_common.dispatch_mode_mgr.is_entered("dump")
        assert not io_common.func_mode_mgr.is_entered("dump")


class TestDispatchModeInterception:
    """TorchDispatchMode arms beyond the core suite's happy path."""

    def test_summary_only_dispatch_mode(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=False)
        try:
            tensor = torch.tensor([1.0, 2.0])
            result = tensor + tensor
            assert torch.equal(result, tensor * 2)  # independent oracle
            d._wait_and_flush()
            # Only the rank dir with summary content; no per-op dirs.
            rank_dir = tmp_path / "rank_0000"
            assert not (rank_dir / "step_0000").exists()
        finally:
            d.disable_io_dump()

    def test_dispatch_mode_step_range_gate(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, step_range="2-4")
        try:
            (torch.tensor([1.0]) + torch.tensor([1.0])).sum()
            d._wait_and_flush()
            assert not (tmp_path / "rank_0000").exists()
        finally:
            d.disable_io_dump()

    def test_dispatch_mode_dumps_aten_op(self, tmp_path):
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            tensor = torch.tensor([1.0, 2.0])
            result = tensor + tensor
            assert torch.equal(result, tensor * 2)
            d._wait_and_flush()
            names = {p.name for p in (tmp_path / "rank_0000" / "step_0000").iterdir()}
            assert "aten.add.Tensor" in names
        finally:
            d.disable_io_dump()

    def test_dispatch_mode_exception_cleans_pairing(self, tmp_path):
        # Restrict to a matching op so only our failing op is dumped.
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True, ops={"reshape"})
        try:
            bad = torch.zeros(2)
            with pytest.raises(RuntimeError):
                bad.reshape(4, 3)  # invalid shape, dumped before the failure
            stack = getattr(d._dump_pairing, "stack", {})
            # reshape either passed the filter (and was cleaned) or never
            # matched; either way no entries may remain.
            assert all(not entries for entries in stack.values())
        finally:
            d.disable_io_dump()


class TestTorchFuncModeInterception:
    def test_summary_only_records_seen(self, tmp_path):
        io_common._seen_ops.discard("torch.sin")
        d.enable_io_dump(
            dump_dir=str(tmp_path), with_metas=False, with_torch_funcs=True
        )
        try:
            out = torch.sin(torch.arange(4.0))
            assert torch.equal(out, torch.sin(torch.arange(4.0)))
            assert "torch.sin" in io_common._seen_ops
        finally:
            d.disable_io_dump()

    def test_torch_func_mode_dumps_and_releases_tags(self, tmp_path):
        d.enable_io_dump(
            dump_dir=str(tmp_path), with_metas=True, with_torch_funcs=True
        )
        try:
            left = torch.tensor([[1.0, 2.0]])
            result = torch.matmul(left, left.transpose(0, 1))
            assert result.item() == 5.0  # 1*1 + 2*2
            d._wait_and_flush()
            names = {p.name for p in (tmp_path / "rank_0000" / "step_0000").iterdir()}
            assert "torch.matmul" in names
            with pytest.raises(RuntimeError):
                torch.matmul(torch.zeros(2), torch.zeros(3, 4))
            assert getattr(io_common._tf_tags, "depth", 0) == 0
        finally:
            d.disable_io_dump()


class TestInitIoDumpFromEnvGates:
    def test_not_eager_skips(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        d.init_io_dump_from_env(eager=False)
        assert d._enabled is False

    def test_no_config_skips(self, monkeypatch):
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
        d.init_io_dump_from_env(eager=True)
        assert d._enabled is False

    def test_config_only_activates(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VLLM_FL_IO_DUMP", raising=False)
        monkeypatch.setenv("VLLM_FL_CONFIG", _write_yaml(tmp_path, {"dir": str(tmp_path)}))
        d.init_io_dump_from_env(eager=True)
        assert d._enabled is True
        assert d._hooks_activated is True
