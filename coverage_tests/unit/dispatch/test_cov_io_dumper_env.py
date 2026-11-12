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

"""Supplementary coverage tests for ``vllm_fl/dispatch/io_dumper.py`` (3/3).

D3 tranche. The ``enable_io_dump`` env/API resolution matrix (including
invalid values) and the ``_init_from_env`` parsing/lifecycle matrix, plus
the disabled/passthrough contract of ``dump_before``/``dump_after``.

Local collection note: importing ``vllm_fl`` pulls in ``flag_gems``
which needs triton (no Windows wheel). ``_flaggems_shim`` installs an
inert placeholder only when the real package cannot import.
"""

import os

import torch

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from ._dumper_helpers import (  # noqa: F401  (autouse state isolation)
    clean_dumper_state,
)
from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch.io_dumper import (
    disable_io_dump,
    dump_after,
    dump_before,
    enable_io_dump,
)


class TestEnableEnvMatrix:
    """enable_io_dump layer-1 env parsing and layer-3 API overrides."""

    def test_env_dump_dir_one_uses_cwd_default(self, tmp_path, monkeypatch):
        """VLLM_FL_IO_DUMP=1 keeps the cwd-relative io_dump default."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        enable_io_dump()
        try:
            assert io_dumper._enabled is True
            expected = os.path.join(str(tmp_path), "io_dump")
            assert io_dumper._dump_dir == expected
            assert os.path.isdir(expected)
        finally:
            disable_io_dump()

    def test_invalid_numeric_env_values_fall_back(self, tmp_path, monkeypatch):
        """Non-numeric max_calls/bg_workers keep defaults; print 0 is off."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "many")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_BG_WORKERS", "lots")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "0")
        enable_io_dump()
        try:
            assert io_dumper._max_calls == 0
            assert io_dumper._bg_workers == 8
            assert io_dumper._print_enabled is False
        finally:
            disable_io_dump()

    def test_env_filters_parsed(self, tmp_path, monkeypatch):
        """Ops/modules/ranks env values are trimmed, split and validated."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "d"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", " mm , add ,,")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MODULES", "Linear, RMSNormFL")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_RANK", "0, 2, bogus")
        enable_io_dump()
        try:
            assert io_dumper._match_all is False
            assert io_dumper._op_filter == {"mm", "add"}
            assert io_dumper._module_filter == {"Linear", "RMSNormFL"}
            assert io_dumper._rank_filter == {0, 2}
        finally:
            disable_io_dump()

    def test_yaml_option_matrix(self, tmp_path, monkeypatch):
        """Every YAML io_dump key is honored (YAML beats env)."""
        yaml_dir = str(tmp_path / "yaml")
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "io_dump:\n"
            f"  dir: {yaml_dir.replace(os.sep, '/')}\n"
            "  ops: [mm]\n"
            "  modules: [Linear]\n"
            '  layers: ["0"]\n'
            "  max_calls: 4\n"
            '  step_range: "2-4"\n'
            "  with_torch_funcs: [matmul]\n"
            "  ranks: [0, 1]\n"
            "  with_values: true\n"
            "  with_metas: true\n"
            "  with_print: true\n",
            encoding="utf-8",
        )
        # A non-empty env dir keeps print-mode from dropping the dump dir;
        # the YAML dir then overrides it.
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "env"))
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        enable_io_dump()
        try:
            assert os.path.normpath(io_dumper._dump_dir) == os.path.normpath(
                yaml_dir
            )
            assert io_dumper._op_filter == {"mm"}
            assert io_dumper._module_filter == {"Linear"}
            assert io_dumper._layer_filter == {"model.layers.0"}
            assert io_dumper._max_calls == 4
            assert io_dumper._step_range == (2, 5)
            assert io_dumper._torch_funcs_enabled is True
            assert io_dumper._torch_func_filter == {"matmul"}
            assert io_dumper._rank_filter == {0, 1}
            assert io_dumper._with_values is True
            assert io_dumper._with_metas is True
            assert io_dumper._print_enabled is True
        finally:
            disable_io_dump()

    def test_api_none_dir_and_remaining_overrides(self, tmp_path, monkeypatch):
        """A falsy API dump_dir resets to cwd default; other API wins."""
        monkeypatch.chdir(tmp_path)
        enable_io_dump(
            dump_dir=None,
            with_torch_funcs=True,
            with_print=True,
            bg_workers=2,
        )
        try:
            assert io_dumper._dump_dir == os.path.join(str(tmp_path), "io_dump")
            assert io_dumper._torch_funcs_enabled is True
            assert io_dumper._print_enabled is True
            assert io_dumper._bg_workers == 2
        finally:
            disable_io_dump()

    def test_empty_ops_filter_dumps_nothing(self, tmp_path):
        """An explicitly empty ops set is a filter that matches nothing."""
        enable_io_dump(dump_dir=str(tmp_path / "d"), ops=set())
        try:
            assert io_dumper._match_all is False
            assert io_dumper._should_dump("aten.mm", ()) is False
        finally:
            disable_io_dump()

    def test_remaining_filter_envs(self, tmp_path, monkeypatch):
        """Layers/step-range/values/metas/print env vars all resolve."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "d"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_LAYERS", "0")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "1-2")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_VALUES", "1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "1")
        enable_io_dump()
        try:
            assert io_dumper._layer_filter == {"model.layers.0"}
            assert io_dumper._step_range == (1, 3)
            assert io_dumper._with_values is True
            assert io_dumper._with_metas is True
            assert io_dumper._print_enabled is True
        finally:
            disable_io_dump()

    def test_api_layers_string_is_wrapped_and_expanded(self, tmp_path):
        """A bare-string API ``layers`` value expands like a one-item set."""
        enable_io_dump(dump_dir=str(tmp_path / "d"), layers="0")
        try:
            assert io_dumper._layer_filter == {"model.layers.0"}
        finally:
            disable_io_dump()


class TestDisabledPassthrough:
    """dump_before/dump_after no-op contracts."""

    def test_dump_noop_when_disabled(self, tmp_path, monkeypatch):
        """With no config, both entry points record nothing and write none."""
        monkeypatch.chdir(tmp_path)
        t = torch.eye(2)
        dump_before("mm", (t, t), {})
        dump_after("mm", (t, t), t @ t)
        assert io_dumper._call_counters == {}
        assert io_dumper._op_summary == {}
        assert list(tmp_path.iterdir()) == []

    def test_dump_after_summary_only_is_noop(self, tmp_path):
        """with_metas=False: dump_before records summary, dump_after returns."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=False)
        try:
            t = torch.eye(2)
            dump_before("mm", (t, t), {})
            assert io_dumper._op_summary["mm"]["call_count"] == 1
            dump_after("mm", (t, t), t @ t)
            assert io_dumper._op_summary["mm"]["call_count"] == 1
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        # No per-op metadata directory was ever created.
        assert not os.path.isdir(os.path.join(dump_dir, "rank_0000", "step_0000"))


class TestInitFromEnvMatrix:
    """_init_from_env parsing, auto-enable and failure paths."""

    def test_env_dir_one_enables_with_default_dir(self, tmp_path, monkeypatch):
        """VLLM_FL_IO_DUMP=1 auto-enables dumping for all ops."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        io_dumper._init_from_env()
        try:
            assert io_dumper._enabled is True
            assert io_dumper._match_all is True
            expected = os.path.join(str(tmp_path), "io_dump")
            assert io_dumper._dump_dir == expected
            assert os.path.isdir(expected)
        finally:
            disable_io_dump()

    def test_step_range_only_auto_enables(self, tmp_path, monkeypatch):
        """A bare step-range filter auto-enables with the default dir."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "1-2")
        io_dumper._init_from_env()
        try:
            assert io_dumper._enabled is True
            assert io_dumper._step_range == (1, 3)
            assert io_dumper._dump_dir == os.path.join(str(tmp_path), "io_dump")
        finally:
            disable_io_dump()

    def test_print_only_skips_file_io(self, tmp_path, monkeypatch):
        """WITH_PRINT alone enables console logging without creating files."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "1")
        io_dumper._init_from_env()
        try:
            assert io_dumper._enabled is True
            assert io_dumper._print_enabled is True
            assert io_dumper._dump_dir == ""
            assert list(tmp_path.iterdir()) == []
        finally:
            disable_io_dump()

    def test_invalid_values_ignored(self, tmp_path, monkeypatch):
        """Bad max_calls/step_range are ignored; print 'false' stays off."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "abc")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "bad-range")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "false")
        io_dumper._init_from_env()
        try:
            assert io_dumper._enabled is True
            assert io_dumper._max_calls == 0
            assert io_dumper._step_range is None
            assert io_dumper._print_enabled is False
        finally:
            disable_io_dump()

    def test_env_modules_torch_funcs_rank_values(self, tmp_path, monkeypatch):
        """Modules/torch-funcs/ranks/values env vars resolve in _init_from_env."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "d"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MODULES", "Linear")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "matmul")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_RANK", "0,1")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_VALUES", "1")
        io_dumper._init_from_env()
        try:
            assert io_dumper._match_all is False
            assert io_dumper._module_filter == {"Linear"}
            assert io_dumper._torch_funcs_enabled is True
            assert io_dumper._torch_func_filter == {"matmul"}
            assert io_dumper._rank_filter == {0, 1}
            assert io_dumper._with_values is True
        finally:
            disable_io_dump()

    def test_yaml_full_matrix(self, tmp_path, monkeypatch):
        """The YAML section drives every resolved attribute in _init_from_env."""
        yaml_dir = str(tmp_path / "yaml")
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "io_dump:\n"
            f"  dir: {yaml_dir.replace(os.sep, '/')}\n"
            "  ops: [mm]\n"
            "  modules: [Linear]\n"
            '  layers: ["0"]\n'
            "  max_calls: 4\n"
            '  step_range: "2-4"\n'
            "  with_torch_funcs: [matmul]\n"
            "  ranks: [0, 1]\n"
            "  with_values: true\n"
            "  with_metas: true\n"
            "  with_print: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        io_dumper._init_from_env()
        try:
            assert io_dumper._enabled is True
            assert os.path.normpath(io_dumper._dump_dir) == os.path.normpath(
                yaml_dir
            )
            assert io_dumper._op_filter == {"mm"}
            assert io_dumper._module_filter == {"Linear"}
            assert io_dumper._layer_filter == {"model.layers.0"}
            assert io_dumper._max_calls == 4
            assert io_dumper._step_range == (2, 5)
            assert io_dumper._torch_funcs_enabled is True
            assert io_dumper._torch_func_filter == {"matmul"}
            assert io_dumper._rank_filter == {0, 1}
            assert io_dumper._with_values is True
            assert io_dumper._with_metas is True
            assert io_dumper._print_enabled is True
        finally:
            disable_io_dump()

    def test_dir_creation_failure_disables(self, tmp_path, monkeypatch):
        """An uncreatable dump dir resets state when printing is off."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(blocker / "sub"))
        io_dumper._init_from_env()
        assert io_dumper._enabled is False
        assert io_dumper._dump_dir == ""

    def test_dir_creation_failure_with_print_keeps_console(
        self, tmp_path, monkeypatch
    ):
        """With print enabled, a dir failure degrades to print-only mode."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(blocker / "sub"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "1")
        io_dumper._init_from_env()
        try:
            assert io_dumper._enabled is True
            assert io_dumper._dump_dir == ""
            assert io_dumper._print_enabled is True
        finally:
            disable_io_dump()

    def test_noop_when_already_enabled(self, tmp_path, monkeypatch):
        """The programmatic API wins: _init_from_env leaves live state alone."""
        api_dir = str(tmp_path / "api")
        enable_io_dump(dump_dir=api_dir, with_metas=True)
        try:
            monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "env"))
            io_dumper._init_from_env()
            assert io_dumper._dump_dir == api_dir
            assert io_dumper._with_metas is True
            assert io_dumper._enabled is True
        finally:
            disable_io_dump()
