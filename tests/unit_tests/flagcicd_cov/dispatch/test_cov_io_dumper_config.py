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

"""Supplementary coverage tests for ``vllm_fl/dispatch/io_dumper.py`` (2/2).

D3 tranche. Config resolution and lifecycle: env/yaml config merge in
``_init_from_env``/``init_io_dump_from_env``, the state export of
``enable_io_dump``/``disable_io_dump``, per-op call limits, summary.json
classification, and the OpManager.call routing through the dump hooks.

Local collection note: importing ``vllm_fl`` pulls in ``flag_gems``
which needs triton (no Windows wheel). ``_flaggems_shim`` installs an
inert placeholder only when the real package cannot import.
"""

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from ._dumper_helpers import (  # noqa: F401  (autouse state isolation)
    DUMP_ENV_VARS,
    clean_dumper_state,
    read_jsonl,
)

import json
import os

import torch

from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch.io_dumper import (
    advance_io_step,
    disable_io_dump,
    dump_after,
    dump_before,
    enable_io_dump,
    init_io_dump_from_env,
    is_dump_enabled,
    io_dump_step,
)
from vllm_fl.dispatch.io_common import (
    dispatch_mode_mgr,
    get_exec_order,
    get_step,
    reset_rank,
    reset_step,
)


class TestManagerCallPath:
    """OpManager.call routes through dump_before/dump_after."""

    def test_manager_call_dumps_real_op_execution(self, tmp_path):
        """An op invoked via mgr.call is dumped with the manager's order."""
        from vllm_fl.dispatch import (
            BackendImplKind,
            BackendPriority,
            OpImpl,
            reset_default_manager,
            reset_global_policy,
        )
        from vllm_fl.dispatch.io_common import next_exec_order  # noqa: F401
        from vllm_fl.dispatch.manager import get_default_manager

        reset_default_manager()
        reset_global_policy()
        try:
            mgr = get_default_manager()
            mgr._state.initialized = True
            mgr._state.init_pid = os.getpid()
            mgr.registry.register_impl(OpImpl(
                op_name="test_cov.matmul",
                impl_id="covtest.matmul",
                kind=BackendImplKind.REFERENCE,
                fn=lambda a, b: a @ b,
                priority=BackendPriority.REFERENCE,
            ))

            dump_dir = str(tmp_path / "dump")
            x = torch.full((2, 2), 5.0)  # built before enabling the mode
            enable_io_dump(dump_dir=dump_dir, with_metas=True)
            try:
                out = mgr.call("test_cov.matmul", x, x)
                io_dumper._wait_and_flush()
            finally:
                disable_io_dump()
            assert out.tolist() == [[50.0, 50.0], [50.0, 50.0]]
            op_dir = os.path.join(dump_dir, "rank_0000", "step_0000",
                                  "test_cov.matmul")
            inp = read_jsonl(os.path.join(op_dir, "input.json"))["call_1"]
            out_stats = read_jsonl(os.path.join(op_dir,
                                                "output.json"))["call_1"]
            # _call_with_hooks allocated the first order of this step for
            # the manager-routed call; it is also the first op type seen
            # this step, so its tag carries type index 0...
            assert inp["exec_order"] == 1
            assert inp["op_tag"] == "[op=0,1]"
            assert inp["tensors"]["arg_0"]["mean"] == 5.0
            assert out_stats["tensors"]["result"]["mean"] == 50.0
            # ...while the aten.mm fired inside the impl got order 2 and
            # type index 1.
            mm_dir = os.path.join(dump_dir, "rank_0000", "step_0000",
                                  "aten.mm")
            mm = read_jsonl(os.path.join(mm_dir, "input.json"))["call_1"]
            assert mm["exec_order"] == 2
            assert mm["op_tag"] == "[op=1,1]"
            # The manager recorded the op as routed through its impl.
            assert mgr._called_ops["test_cov.matmul"] == "covtest.matmul"
        finally:
            reset_default_manager()
            reset_global_policy()

    def test_exec_order_and_op_tag_shared_between_before_after(self, tmp_path):
        """dump_after reuses the exec_order captured at dump_before time."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            x = torch.full((3,), 2.0)
            dump_before("add", (x, x), {}, exec_order=7, op_tag="[op=9,9]")
            result = x + x
            dump_after("add", (x, x), result)
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        op_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "add")
        inp = read_jsonl(os.path.join(op_dir, "input.json"))["call_1"]
        out = read_jsonl(os.path.join(op_dir, "output.json"))["call_1"]
        assert inp["exec_order"] == 7 and inp["op_tag"] == "[op=9,9]"
        assert out["exec_order"] == 7 and out["op_tag"] == "[op=9,9]"
        assert out["tensors"]["result"]["max"] == 4.0


class TestEnvInit:
    """_init_from_env / init_io_dump_from_env entry points."""

    def test_env_var_dump_dir_activates(self, tmp_path, monkeypatch):
        """VLLM_FL_IO_DUMP=<dir> alone enables dumping to that dir."""
        dump_dir = str(tmp_path / "envdump")
        monkeypatch.setenv("VLLM_FL_IO_DUMP", dump_dir)
        monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "1")
        # _init_from_env only resolves config; init_io_dump_from_env is the
        # model_runner entry point that also activates the hooks.
        init_io_dump_from_env(eager=True)
        assert io_dumper._enabled is True
        assert io_dumper._dump_dir == dump_dir
        assert io_dumper._with_metas is True
        assert dispatch_mode_mgr.is_entered("dump")
        try:
            torch.full((2, 2), 3.0) @ torch.full((2, 2), 3.0)
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        op_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "aten.mm")
        assert os.path.isfile(os.path.join(op_dir, "input.json"))

    def test_env_disable_zero_resets(self, monkeypatch):
        """VLLM_FL_IO_DUMP=0 leaves the dumper disabled and reset."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "0")
        io_dumper._init_from_env()
        assert io_dumper._enabled is False
        assert io_dumper._dump_dir == ""

    def test_layer_env_auto_enables_with_default_dir(self, tmp_path,
                                                    monkeypatch):
        """A bare layer filter auto-enables with the cwd io_dump default."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VLLM_FL_IO_DUMP_LAYERS", "0-1")
        io_dumper._init_from_env()
        assert io_dumper._enabled is True
        assert io_dumper._layer_filter == {"model.layers.0", "model.layers.1"}
        assert io_dumper._dump_dir == os.path.join(str(tmp_path), "io_dump")
        # The default dump directory is created eagerly.
        assert os.path.isdir(os.path.join(str(tmp_path), "io_dump"))
        disable_io_dump()

    def test_no_env_and_no_config_stays_disabled(self, tmp_path, monkeypatch):
        """No env vars at all: _init_from_env resets and stays off."""
        monkeypatch.chdir(tmp_path)
        io_dumper._init_from_env()
        assert io_dumper._enabled is False
        assert not os.listdir(tmp_path)  # no dump dir created

    def test_yaml_config_overrides_env(self, tmp_path, monkeypatch):
        """YAML dir beats the env dir; YAML ops beat env ops."""
        env_dir = str(tmp_path / "env")
        yaml_dir = str(tmp_path / "yaml")
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "io_dump:\n"
            "  dir: %s\n"
            "  ops: [aten.mm]\n"
            "  max_calls: 3\n"
            "  step_range: \"0-4\"\n" % yaml_dir.replace("\\", "/"),
            encoding="utf-8",
        )
        monkeypatch.setenv("VLLM_FL_IO_DUMP", env_dir)
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "aten.add")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        io_dumper._init_from_env()
        try:
            assert os.path.normpath(io_dumper._dump_dir) == \
                os.path.normpath(yaml_dir)
            assert io_dumper._op_filter == {"aten.mm"}
            assert io_dumper._max_calls == 3
            assert io_dumper._step_range == (0, 5)
            assert io_dumper._match_all is False
        finally:
            disable_io_dump()

    def test_init_io_dump_from_env_eager_gates(self, tmp_path, monkeypatch):
        """init_io_dump_from_env needs eager=True plus an env marker."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "d"))
        init_io_dump_from_env(eager=False)
        assert is_dump_enabled() is False
        init_io_dump_from_env(eager=True)
        assert is_dump_enabled() is True
        assert dispatch_mode_mgr.is_entered("dump")
        disable_io_dump()
        # No env markers at all: stays off even when eager.
        init_io_dump_from_env(eager=True)
        assert is_dump_enabled() is False

    def test_advance_io_step_noop_when_disabled(self):
        """advance_io_step does not move the counter while disabled."""
        reset_step()
        advance_io_step()
        assert get_step() == 0


class TestEnableDisableLifecycle:
    """enable_io_dump config resolution and state propagation."""

    def test_enable_propagates_resolved_config_to_env(self, tmp_path):
        """enable_io_dump exports the resolved config as env vars."""
        dump_dir = str(tmp_path / "prop")
        enable_io_dump(dump_dir=dump_dir, ops={"aten.mm"}, modules={"Linear"},
                       layers={"0"}, max_calls=5, step_range="1-3",
                       ranks={0, 2}, with_values=True, with_metas=True)
        try:
            assert os.environ["VLLM_FL_IO_DUMP"] == dump_dir
            assert os.environ["VLLM_FL_IO_DUMP_OPS"] == "aten.mm"
            assert os.environ["VLLM_FL_IO_DUMP_MODULES"] == "Linear"
            assert os.environ["VLLM_FL_IO_DUMP_LAYERS"] == "model.layers.0"
            assert os.environ["VLLM_FL_IO_DUMP_MAX_CALLS"] == "5"
            # Half-open (1, 4) is re-exported as the inclusive "1-3".
            assert os.environ["VLLM_FL_IO_DUMP_STEP_RANGE"] == "1-3"
            assert os.environ["VLLM_FL_IO_DUMP_RANK"] == "0,2"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_VALUES"] == "1"
            assert os.environ["VLLM_FL_IO_DUMP_WITH_METAS"] == "1"
            assert "VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS" not in os.environ
        finally:
            disable_io_dump()

    def test_disable_clears_env_and_state(self, tmp_path):
        """disable_io_dump scrubs env vars and resets module state."""
        enable_io_dump(dump_dir=str(tmp_path / "d"), with_metas=True,
                       ops={"aten.mm"})
        assert io_dumper._op_summary == {}
        disable_io_dump()
        for var in DUMP_ENV_VARS:
            assert var not in os.environ
        assert io_dumper._enabled is False
        assert io_dumper._dump_dir == ""
        assert io_dumper._op_filter == set()
        assert io_dumper._rank_filter is None
        assert io_dumper._op_summary == {}

    def test_max_calls_stops_per_op_dumping(self, tmp_path):
        """max_calls=1 dumps only the first call of each op."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True,
                       ops={"mm"}, max_calls=1)
        try:
            a = torch.full((2, 2), 2.0)
            a @ a  # call 1: dumped
            a @ a  # call 2: over the limit
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        op_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "aten.mm")
        entries = read_jsonl(os.path.join(op_dir, "input.json"))
        assert list(entries) == ["call_1"]

    def test_bg_workers_env_and_api(self, tmp_path, monkeypatch):
        """bg_workers resolves from env then API, clamped to >= 1."""
        monkeypatch.setenv("VLLM_FL_IO_DUMP_BG_WORKERS", "3")
        enable_io_dump(dump_dir=str(tmp_path / "d"))
        try:
            assert io_dumper._bg_workers == 3
        finally:
            disable_io_dump()
        enable_io_dump(dump_dir=str(tmp_path / "d"), bg_workers=0)
        try:
            assert io_dumper._bg_workers == 1  # max(1, 0)
        finally:
            disable_io_dump()


class TestSummaryJson:
    """summary.json classification written on disable/step advance."""

    def test_summary_classifies_flaggems_vs_aten(self, tmp_path):
        """Ops with a FlagGems dispatch key land in flaggems_ops."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            # Directly record one op with a FlagGems dispatch key string and
            # one plain ATen op through the real dispatch mode.
            io_dumper._record_op_summary(
                "aten.fake_gems", [("CUDA", "FlagGems", False)])
            torch.full((1,), 1.0)
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        summary_path = os.path.join(dump_dir, "rank_0000", "summary.json")
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["rank"] == 0
        assert "aten.fake_gems" in summary["flaggems_ops"]
        assert "aten.full" in summary["non_flaggems_ops"]

    def test_summary_counts_calls_per_op(self, tmp_path):
        """call_count accumulates per op across calls."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            for _ in range(3):
                torch.full((1,), 1.0)
            io_dumper._wait_and_flush()
        finally:
            disable_io_dump()
        assert io_dumper._op_summary == {}  # cleared by disable
        # But the step summary callback saw the op three times:
        # verified via the dumped input.json having three calls.
        op_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "aten.full")
        entries = read_jsonl(os.path.join(op_dir, "input.json"))
        assert sorted(entries) == ["call_1", "call_2", "call_3"]

    def test_summary_write_failure_is_survivable(self, tmp_path):
        """A blocked summary.json.tmp logs the OSError and does not raise."""
        dump_dir = str(tmp_path / "dump")
        enable_io_dump(dump_dir=dump_dir, with_metas=True)
        try:
            torch.full((2, 2), 1.0) @ torch.full((2, 2), 1.0)
            io_dumper._wait_and_flush()
            # Block the temp-file write: a directory where summary.json.tmp
            # is opened for writing makes open() raise OSError, which
            # _write_summary catches and logs (io_dumper.py:1298-1299).
            rank_dir = os.path.join(dump_dir, "rank_0000")
            os.mkdir(os.path.join(rank_dir, "summary.json.tmp"))
            io_dump_step()  # _write_summary -> OSError caught + logged
            assert not os.path.isfile(os.path.join(rank_dir, "summary.json"))
        finally:
            # vllm_fl/dispatch/io_dumper.py:1269 — os.makedirs(rank_dir) sits
            # outside the try/except OSError that starts at line 1288, so a
            # rank_0000 *file* would crash disable_io_dump with
            # FileExistsError (SOURCE BUG: the "Failed to write summary"
            # handler cannot see it). This test blocks only summary.json.tmp,
            # which stays inside the guarded region.
            disable_io_dump()
        assert get_step() == 1  # step advance completed despite the failure
