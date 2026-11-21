# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for remaining ``vllm_fl.dispatch.io_dumper`` branches.

``test_io_dumper_pairing.py`` already covers the print helpers, pairing stacks,
filter gates, serialization helpers, background-writer machinery, the main
``enable_io_dump``/``_init_from_env`` layering scenarios and the dispatch-mode
interception.  This file targets what is still uncovered:

* ``enable_io_dump`` env-var parsing corners (bad ints, ``"0"``/``"1"``
  special dirs, per-variable YAML keys, ``bg_workers`` clamping, print-only
  override of an explicit empty dir);
* ``_init_from_env`` YAML overlays and the print-only-without-dir branch;
* ``dump_after``'s non-matching op early return and its print path without a
  pushed pairing;
* ``_DumpTorchFuncMode`` summary-only pass-through (``with_metas=False``);
* ``_bg_compute_and_write`` swallowing a broken tensor-stats call;
* ``_bg_save_pt`` syncing a device event;
* ``pause_dispatch_modes`` when the torch-function mode is entered;
* ``_reset_state`` closing a handle whose ``close()`` raises ``OSError``;
* ``advance_io_step`` / ``init_io_dump_from_env`` gating.
"""

import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch import io_common, io_dumper as d

pytestmark = [pytest.mark.supplementary]


def _drain_module_context():
    """Pop every entry from the thread-local module-context stack."""
    stack = getattr(io_common._module_context, "stack", None)
    while stack:
        stack.pop()


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


def _write_yaml(tmp_path, io_cfg):
    """Write a VLLM_FL_CONFIG yaml carrying the given io_dump mapping."""
    lines = ["io_dump:"]
    for key, value in io_cfg.items():
        lines.append(f"  {key}: {json.dumps(value)}")
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


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


class TestEnableIoDumpYaml:
    """``enable_io_dump`` layer-2 YAML keys."""

    def test_yaml_full_config(self, tmp_path, monkeypatch):
        """Every yaml io_dump key overrides the env layer."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "env"))
        monkeypatch.setenv(
            "VLLM_FL_CONFIG",
            _write_yaml(
                tmp_path,
                {
                    "dir": str(tmp_path / "yaml"),
                    "ops": ["yaml.op"],
                    "modules": ["Yaml"],
                    "layers": ["model.layers.2"],
                    "max_calls": 7,
                    "step_range": [2, 5],
                    "with_torch_funcs": ["t1", "t2"],
                    "ranks": [1, 4],
                    "with_values": True,
                    "with_metas": True,
                    "with_print": False,
                },
            ),
        )
        d.enable_io_dump()
        assert d._dump_dir == str(tmp_path / "yaml")
        assert d._op_filter == {"yaml.op"}
        assert d._module_filter == {"Yaml"}
        assert "model.layers.2" in d._layer_filter
        assert d._max_calls == 7
        assert tuple(d._step_range) == (2, 6)
        assert d._torch_funcs_enabled is True
        assert d._torch_func_filter == {"t1", "t2"}
        assert d._rank_filter == {1, 4}
        assert d._with_values is True
        assert d._with_metas is True
        assert d._print_enabled is False

    def test_yaml_falsy_keys_ignored(self, tmp_path, monkeypatch):
        """Falsy dir/ops/modules/layers/max_calls yaml keys keep env values."""
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
        assert d._module_filter == set()
        assert d._layer_filter == set()
        assert d._max_calls == 0
        assert d._with_values is False
        assert d._with_metas is False
        assert d._print_enabled is True


class TestInitFromEnvYaml:
    """``_init_from_env`` yaml overlay and print-only corners."""

    def test_yaml_overlays_env(self, tmp_path, monkeypatch):
        """Yaml values replace env values in the env-only init path."""
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
        assert d._torch_funcs_enabled is True
        assert d._torch_func_filter == {"y1"}
        assert d._rank_filter == {3}
        assert d._with_values is True
        assert d._with_metas is True
        assert d._print_enabled is True

    def test_env_layer_values_in_env_init(self, tmp_path, monkeypatch):
        """Layer-1 env parsing inside _init_from_env (ops/ranks/step)."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", " e.op ")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MODULES", " EMod ")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_RANK", "5")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "0-1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "9")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "e1,e2")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_VALUES", "1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "true")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "0")
        d._init_from_env()
        assert d._op_filter == {"e.op"}
        assert d._module_filter == {"EMod"}
        assert d._rank_filter == {5}
        assert tuple(d._step_range) == (0, 2)  # "0-1" inclusive -> half-open
        assert d._max_calls == 9
        assert d._torch_funcs_enabled is True
        assert d._torch_func_filter == {"e1", "e2"}
        assert d._with_values is True
        assert d._with_metas is True
        assert d._print_enabled is False

    def test_env_bad_values_in_env_init(self, tmp_path, monkeypatch):
        """Bad ints and empty filters fall back to defaults in _init_from_env."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "zzz")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "false")
        d._init_from_env()
        assert d._max_calls == 0
        assert d._print_enabled is False


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


class TestStepGating:
    """``advance_io_step`` and ``init_io_dump_from_env`` gates."""

    def test_advance_noop_when_disabled(self):
        """Advancing the step while disabled leaves the counter alone."""
        before = io_common.get_step()
        d.advance_io_step()
        assert io_common.get_step() == before

    def test_advance_when_enabled(self, tmp_path):
        """Advancing while enabled bumps the shared step counter."""
        d.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        before = io_common.get_step()
        d.advance_io_step()
        assert io_common.get_step() == before + 1

    def test_init_from_env_skipped_when_not_eager(self, monkeypatch, tmp_path):
        """A non-eager call leaves the dumper disabled."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        d.init_io_dump_from_env(eager=False)
        assert d._enabled is False

    def test_init_from_env_skipped_without_config(self, monkeypatch):
        """No VLLM_FL_IO_DUMP* env and no config leaves the dumper off."""
        d.init_io_dump_from_env(eager=True)
        assert d._enabled is False

    def test_init_from_env_activates_when_configured(self, monkeypatch, tmp_path):
        """An eager call with dump env vars initializes and activates hooks."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "1")
        d.init_io_dump_from_env(eager=True)
        assert d._enabled is True
        assert d._hooks_activated is True
