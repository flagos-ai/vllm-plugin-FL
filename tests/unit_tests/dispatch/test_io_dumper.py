# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tests for IO Dumper module.
"""

import json
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
import torch

from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch.io_common import (
    HAS_TORCH_DISPATCH_MODE,
    HAS_TORCH_FUNC_MODE,
    advance_step,
    expand_layer_specs,
    get_exec_order,
    get_step,
    layer_path_matches,
    list_model_layers,
    next_exec_order,
    parse_io_config_from_yaml,
    parse_torch_funcs_config,
    pop_module_context,
    push_module_context,
    register_module_paths,
    reset_exec_order,
    reset_rank,
    reset_step,
)
from vllm_fl.dispatch.io_dumper import (
    _build_data,
    _extract_tensor_refs,
    _serialize_value,
    _should_dump,
    _should_dump_torch_func,
    advance_io_step,
    disable_io_dump,
    dump_after,
    dump_before,
    dump_cleanup,
    enable_io_dump,
    init_io_dump_from_env,
    io_dump_step,
    is_dump_enabled,
    register_io_module_hooks,
)


@pytest.fixture
def dump_dir():
    """Create a temporary directory for dump files."""
    d = tempfile.mkdtemp(prefix="vllm_fl_dump_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _read_jsonl(path: str) -> dict:
    """Read a JSON Lines file (one JSON object per line) and merge into a dict."""
    data: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.update(json.loads(line))
    return data


class TestSerializeValue:
    """Test _serialize_value conversion."""

    def test_tensor_to_cpu(self):
        t = torch.zeros(2, 3, dtype=torch.float32)
        result = _serialize_value(t)
        assert isinstance(result, torch.Tensor)
        assert result.device == torch.device("cpu")
        assert result.shape == (2, 3)

    def test_none(self):
        assert _serialize_value(None) is None

    def test_scalar(self):
        assert _serialize_value(42) == 42
        assert _serialize_value(1.5) == 1.5

    def test_bool(self):
        assert _serialize_value(True) is True

    def test_module_to_string(self):
        m = torch.nn.Linear(10, 10)
        result = _serialize_value(m)
        assert isinstance(result, str)
        assert "Linear" in result

    def test_tuple(self):
        t = torch.zeros(2)
        result = _serialize_value((t, None, 42))
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], torch.Tensor)
        assert result[1] is None
        assert result[2] == 42

    def test_list(self):
        result = _serialize_value([1, 2, 3])
        assert isinstance(result, list)
        assert result == [1, 2, 3]


class TestBuildDicts:
    """Test _extract_tensor_refs and _build_data key naming scheme."""

    def test_input_meta_args(self):
        t = torch.zeros(2, 3)
        refs = _extract_tensor_refs((t, None, 1.0), {})
        assert "arg_0" in refs
        assert refs["arg_0"].shape == torch.Size([2, 3])
        # Non-tensors are not included
        assert "arg_1" not in refs
        assert "arg_2" not in refs

    def test_input_meta_kwargs(self):
        t = torch.ones(4)
        refs = _extract_tensor_refs((), {"weight": t, "epsilon": 1e-6})
        assert "kwarg_weight" in refs
        assert refs["kwarg_weight"].shape == torch.Size([4])
        assert "kwarg_epsilon" not in refs

    def test_input_data_args(self):
        t = torch.zeros(2, 3)
        d = _build_data((t, None, 1.0), {})
        assert "arg_0" in d
        assert isinstance(d["arg_0"], torch.Tensor)
        assert d["arg_1"] is None
        assert d["arg_2"] == 1.0

    def test_input_data_kwargs(self):
        d = _build_data((), {"epsilon": 1e-6, "inplace": True})
        assert d["kwarg_epsilon"] == 1e-6
        assert d["kwarg_inplace"] is True

    def test_output_meta_single(self):
        t = torch.ones(4, 5)
        refs = _extract_tensor_refs((t,), {}, is_output=True)
        assert "result" in refs
        assert refs["result"].shape == torch.Size([4, 5])

    def test_output_meta_tuple(self):
        t1 = torch.zeros(2)
        t2 = torch.ones(3)
        refs = _extract_tensor_refs(((t1, t2),), {}, is_output=True)
        assert "result_0" in refs
        assert "result_1" in refs

    def test_output_data_single(self):
        t = torch.ones(4, 5)
        d = _build_data((t,), {}, is_output=True)
        assert "result" in d
        assert isinstance(d["result"], torch.Tensor)

    def test_output_data_tuple(self):
        t1 = torch.zeros(2)
        t2 = torch.ones(3)
        d = _build_data(((t1, t2),), {}, is_output=True)
        assert "result_0" in d
        assert "result_1" in d


class TestShouldDump:
    """Test _should_dump filtering logic."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_disabled_returns_false(self):
        assert not _should_dump("rms_norm", ())

    def test_dump_all(self, dump_dir):
        enable_io_dump(dump_dir)
        assert _should_dump("rms_norm", ())
        assert _should_dump("silu_and_mul", ())

    def test_op_filter(self, dump_dir):
        enable_io_dump(dump_dir, ops={"rms_norm"})
        assert _should_dump("rms_norm", ())
        assert not _should_dump("silu_and_mul", ())

    def test_module_filter(self, dump_dir):
        enable_io_dump(dump_dir, modules={"Linear"})
        m = torch.nn.Linear(10, 10)
        assert _should_dump("any_op", (m,))
        assert not _should_dump("any_op", (torch.zeros(2),))

    def test_max_calls_limit(self, dump_dir):
        enable_io_dump(dump_dir, max_calls=2)
        # Simulate call counts
        io_dumper._call_counters["test_op"] = 2
        assert not _should_dump("test_op", ())
        # Different op should still work
        assert _should_dump("other_op", ())

    def test_step_range(self, dump_dir):
        from vllm_fl.dispatch import io_common

        enable_io_dump(dump_dir, step_range="5-10")
        io_common._step_counter = 3
        assert not _should_dump("test_op", ())
        io_common._step_counter = 5
        assert _should_dump("test_op", ())
        io_common._step_counter = 10
        assert _should_dump("test_op", ())
        io_common._step_counter = 11
        assert not _should_dump("test_op", ())


class TestDumpBeforeAfter:
    """Test dump_before and dump_after file creation."""

    def setup_method(self):
        reset_step()
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()
        reset_step()

    def test_dump_creates_input_file(self, dump_dir):
        enable_io_dump(dump_dir, with_values=True, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2, 3)
        dump_before("test_op", (t,), {"epsilon": 1e-6})
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isdir(step_dir)
        # Per-call PT file
        assert os.path.isfile(os.path.join(step_dir, "call_1_input.pt"))

        data = torch.load(os.path.join(step_dir, "call_1_input.pt"), weights_only=False)
        assert "arg_0" in data
        assert isinstance(data["arg_0"], torch.Tensor)
        assert data["kwarg_epsilon"] == 1e-6
        # .pt files should NOT contain __meta__ (moved to JSON)
        assert "__meta__" not in data
        # Check merged JSON meta file
        assert os.path.isfile(os.path.join(step_dir, "input.json"))
        all_meta = _read_jsonl(os.path.join(step_dir, "input.json"))
        meta = all_meta["call_1"]
        assert "test_op" in meta["op_name"]
        assert meta["exec_order"] >= 1

    def test_dump_creates_output_file(self, dump_dir):
        enable_io_dump(dump_dir, with_values=True, with_metas=True)
        reset_exec_order()
        t_in = torch.zeros(2, 3)
        # Call dump_before first to set call counter
        dump_before("test_op", (t_in,), {})

        t_out = torch.ones(2, 3)
        dump_after("test_op", (t_in,), t_out)
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isfile(os.path.join(step_dir, "call_1_output.pt"))

        data = torch.load(
            os.path.join(step_dir, "call_1_output.pt"), weights_only=False
        )
        assert "result" in data
        # .pt files should NOT contain __meta__ (moved to JSON)
        assert "__meta__" not in data
        # Check merged JSON meta file
        assert os.path.isfile(os.path.join(step_dir, "output.json"))
        all_meta = _read_jsonl(os.path.join(step_dir, "output.json"))
        meta = all_meta["call_1"]
        assert "op_name" in meta

    def test_dump_tuple_output(self, dump_dir):
        enable_io_dump(dump_dir, with_values=True, with_metas=True)
        reset_exec_order()
        t_in = torch.zeros(2)
        dump_before("test_op", (t_in,), {})

        t1 = torch.zeros(2)
        t2 = torch.ones(3)
        dump_after("test_op", (t_in,), (t1, t2))
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isfile(os.path.join(step_dir, "call_1_output.pt"))
        data = torch.load(
            os.path.join(step_dir, "call_1_output.pt"), weights_only=False
        )
        assert "result_0" in data
        assert "result_1" in data

    def test_dump_skips_when_filtered(self, dump_dir):
        enable_io_dump(dump_dir, ops={"other_op"})
        t = torch.zeros(2, 3)
        dump_before("test_op", (t,), {})

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert not os.path.exists(step_dir)


class TestIoDumpStep:
    """Test io_dump_step step counter management."""

    def setup_method(self):
        reset_step()
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_step_increments(self, dump_dir):
        enable_io_dump(dump_dir)
        assert get_step() == 0
        new_step = io_dump_step()
        assert new_step == 1
        assert get_step() == 1

    def test_step_resets_call_counters(self, dump_dir):
        enable_io_dump(dump_dir)
        io_dumper._call_counters["test_op"] = 5
        io_dump_step()
        assert io_dumper._call_counters == {}

    def test_dump_files_in_different_steps(self, dump_dir):
        enable_io_dump(dump_dir, with_values=True, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)

        # Step 0
        dump_before("test_op", (t,), {})
        io_dumper._wait_and_flush()
        step0_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isdir(step0_dir)
        assert os.path.isfile(os.path.join(step0_dir, "call_1_input.pt"))

        # Step 1
        io_dump_step()
        dump_before("test_op", (t,), {})
        io_dumper._wait_and_flush()
        step1_dir = os.path.join(dump_dir, "rank_0000", "step_0001", "test_op")
        assert os.path.isdir(step1_dir)
        assert os.path.isfile(os.path.join(step1_dir, "call_1_input.pt"))


class TestProgrammaticAPI:
    """Test enable/disable programmatic API."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_enable_disable(self, dump_dir):
        assert not is_dump_enabled()
        enable_io_dump(dump_dir)
        assert is_dump_enabled()
        disable_io_dump()
        assert not is_dump_enabled()

    def test_enable_with_filters(self, dump_dir):
        enable_io_dump(
            dump_dir,
            ops={"rms_norm"},
            modules={"Linear"},
            max_calls=10,
            step_range="0-100",
        )
        assert is_dump_enabled()
        assert io_dumper._op_filter == {"rms_norm"}
        assert io_dumper._module_filter == {"Linear"}
        assert io_dumper._max_calls == 10
        assert io_dumper._step_range == (0, 101)

    def test_disable_resets_everything(self, dump_dir):
        enable_io_dump(dump_dir, ops={"rms_norm"}, max_calls=5)
        io_dump_step()
        disable_io_dump()

        assert not is_dump_enabled()
        assert io_dumper._dump_dir == ""
        assert io_dumper._op_filter == set()
        assert io_dumper._max_calls == 0
        # Step counter is shared; disabling the dumper
        # should NOT reset it (only explicit reset_step() should).
        assert get_step() >= 1


class TestEnvVarInit:
    """Test initialization from environment variables."""

    def teardown_method(self):
        disable_io_dump()

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump"},
        clear=False,
    )
    def test_env_basic(self):
        io_dumper.init_io_dump_from_env(True)
        assert is_dump_enabled()
        assert io_dumper._dump_dir == "/tmp/test_dump"
        assert io_dumper._match_all

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_OPS": "rms_norm,silu_and_mul",
        },
        clear=False,
    )
    def test_env_ops(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._op_filter == {"rms_norm", "silu_and_mul"}
        assert not io_dumper._match_all

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_MODULES": "RMSNormFL",
        },
        clear=False,
    )
    def test_env_modules(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._module_filter == {"RMSNormFL"}

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_MAX_CALLS": "10",
        },
        clear=False,
    )
    def test_env_max_calls(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._max_calls == 10

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_STEP_RANGE": "5-14",
        },
        clear=False,
    )
    def test_env_step_range(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._step_range == (5, 15)

    @patch.dict(os.environ, {}, clear=False)
    def test_env_unset(self):
        os.environ.pop("VLLM_FL_IO_DUMP", None)
        os.environ.pop("VLLM_FL_IO_DUMP_OPS", None)
        os.environ.pop("VLLM_FL_IO_DUMP_MODULES", None)
        os.environ.pop("VLLM_FL_IO_DUMP_MAX_CALLS", None)
        os.environ.pop("VLLM_FL_IO_DUMP_STEP_RANGE", None)
        io_dumper.init_io_dump_from_env(True)
        assert not is_dump_enabled()


class TestParseTorchFuncsConfig:
    """Test parse_torch_funcs_config parsing logic."""

    def test_empty_string(self):
        enabled, funcs = parse_torch_funcs_config("")
        assert not enabled
        assert funcs == set()

    def test_zero_disables(self):
        enabled, funcs = parse_torch_funcs_config("0")
        assert not enabled
        assert funcs == set()

    def test_one_enables_all(self):
        enabled, funcs = parse_torch_funcs_config("1")
        assert enabled
        assert funcs == set()

    def test_specific_funcs(self):
        enabled, funcs = parse_torch_funcs_config("matmul,softmax")
        assert enabled
        assert funcs == {"matmul", "softmax"}


class TestShouldDumpTorchFunc:
    """Test _should_dump_torch_func filtering logic."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_disabled_returns_false(self):
        assert not _should_dump_torch_func("matmul")

    def test_enabled_all(self, dump_dir):
        enable_io_dump(dump_dir, with_torch_funcs=True)
        assert _should_dump_torch_func("matmul")
        assert _should_dump_torch_func("softmax")

    def test_skips_dunder(self, dump_dir):
        enable_io_dump(dump_dir, with_torch_funcs=True)
        assert not _should_dump_torch_func("__add__")

    def test_skips_trivial_ops(self, dump_dir):
        enable_io_dump(dump_dir, with_torch_funcs=True)
        assert not _should_dump_torch_func("size")
        assert not _should_dump_torch_func("dim")

    def test_op_filter_match(self, dump_dir):
        enable_io_dump(dump_dir, ops={"matmul"}, with_torch_funcs=True)
        assert _should_dump_torch_func("matmul")
        assert not _should_dump_torch_func("softmax")


class TestTorchDispatchMode:
    """Test TorchDispatchMode for ATen-level op interception in dumper."""

    def setup_method(self):
        reset_step()
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    @pytest.mark.skipif(
        not HAS_TORCH_DISPATCH_MODE,
        reason="TorchDispatchMode not available in this PyTorch version",
    )
    def test_dispatch_mode_auto_activated(self, dump_dir):
        enable_io_dump(dump_dir)
        from vllm_fl.dispatch.io_common import dispatch_mode_mgr

        assert dispatch_mode_mgr.is_entered("dump")

    @pytest.mark.skipif(
        not HAS_TORCH_DISPATCH_MODE,
        reason="TorchDispatchMode not available in this PyTorch version",
    )
    def test_dispatch_mode_creates_files(self, dump_dir):
        """TorchDispatchMode should dump files when dump is enabled."""
        enable_io_dump(dump_dir, modules={"Linear"}, with_values=True, with_metas=True)
        model = torch.nn.Linear(4, 3)

        x = torch.randn(2, 4)
        push_module_context("Linear", model)
        try:
            model(x)
        finally:
            pop_module_context()

        # Step stays at 0 (step is advanced by model_runner.execute_model)
        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000")
        assert os.path.isdir(step_dir)
        found_files = False
        for root, dirs, files in os.walk(step_dir):
            for f in files:
                if f.endswith(".pt") or f.endswith(".json"):
                    found_files = True
        assert found_files

    @pytest.mark.skipif(
        not HAS_TORCH_DISPATCH_MODE,
        reason="TorchDispatchMode not available in this PyTorch version",
    )
    def test_dispatch_mode_deactivated_on_disable(self, dump_dir):
        enable_io_dump(dump_dir, modules={"Linear"})
        from vllm_fl.dispatch.io_common import dispatch_mode_mgr

        assert dispatch_mode_mgr.is_entered("dump")
        disable_io_dump()

        assert not dispatch_mode_mgr.is_entered("dump")

    @pytest.mark.skipif(
        not HAS_TORCH_DISPATCH_MODE,
        reason="TorchDispatchMode not available in this PyTorch version",
    )
    def test_dispatch_mode_activated_in_match_all(self, dump_dir):
        """In match-all mode, TorchDispatchMode is activated."""
        enable_io_dump(dump_dir)
        from vllm_fl.dispatch.io_common import dispatch_mode_mgr

        assert dispatch_mode_mgr.is_entered("dump")


class TestDumpTorchFunctionMode:
    """Test TorchFunctionMode for bare torch functional ops dumping."""

    def setup_method(self):
        reset_step()
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_activated(self, dump_dir):
        enable_io_dump(dump_dir, with_torch_funcs=True)
        from vllm_fl.dispatch.io_common import func_mode_mgr

        assert func_mode_mgr.is_entered("dump")

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_deactivated_on_disable(self, dump_dir):
        enable_io_dump(dump_dir, with_torch_funcs=True)
        disable_io_dump()
        from vllm_fl.dispatch.io_common import func_mode_mgr

        assert not func_mode_mgr.is_entered("dump")

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_dumps_matmul(self, dump_dir):
        enable_io_dump(
            dump_dir,
            ops={"matmul"},
            with_torch_funcs=True,
            with_values=True,
            with_metas=True,
        )

        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        torch.matmul(a, b)

        # Check that files were created under torch.matmul directory
        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000")
        assert os.path.isdir(step_dir)
        found_pt = False
        for root, dirs, files in os.walk(step_dir):
            for f in files:
                if f.endswith(".pt"):
                    found_pt = True
        assert found_pt

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_not_activated_by_default(self, dump_dir):
        enable_io_dump(dump_dir)  # torch_funcs=False by default
        from vllm_fl.dispatch.io_common import func_mode_mgr

        assert not func_mode_mgr.is_entered("dump")

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_torch_func_mode_not_activated_when_disabled(self, dump_dir):
        enable_io_dump(dump_dir, with_torch_funcs=False)
        from vllm_fl.dispatch.io_common import func_mode_mgr

        assert not func_mode_mgr.is_entered("dump")

    @pytest.mark.skipif(
        not HAS_TORCH_FUNC_MODE,
        reason="TorchFunctionMode not available in this PyTorch version",
    )
    def test_no_infinite_recursion(self, dump_dir):
        """Ensure re-entrancy guard prevents infinite recursion."""
        enable_io_dump(dump_dir, with_torch_funcs=True)
        a = torch.randn(3, 3)
        b = torch.randn(3, 3)
        result = torch.matmul(a, b)
        assert result.shape == (3, 3)


class TestEnvVarTorchFuncs:
    """Test torch funcs env var initialization for dumper."""

    def teardown_method(self):
        disable_io_dump()

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS": "1",
        },
        clear=False,
    )
    def test_env_torch_funcs_all(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._torch_funcs_enabled
        assert io_dumper._torch_func_filter == set()

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS": "matmul,softmax",
        },
        clear=False,
    )
    def test_env_torch_funcs_specific(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._torch_funcs_enabled
        assert io_dumper._torch_func_filter == {"matmul", "softmax"}

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump"},
        clear=False,
    )
    def test_env_torch_funcs_unset_match_all(self):
        os.environ.pop("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", None)
        os.environ.pop("VLLM_FL_IO_DUMP_OPS", None)
        os.environ.pop("VLLM_FL_IO_DUMP_MODULES", None)
        io_dumper.init_io_dump_from_env(True)
        # torch_funcs defaults to False (TorchDispatchMode handles ops)
        assert not io_dumper._torch_funcs_enabled

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump", "VLLM_FL_IO_DUMP_OPS": "rms_norm"},
        clear=False,
    )
    def test_env_torch_funcs_unset_filtered(self):
        os.environ.pop("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", None)
        io_dumper.init_io_dump_from_env(True)
        # torch_funcs defaults to False when specific ops are filtered
        assert not io_dumper._torch_funcs_enabled


class TestExecOrder:
    """Test execution order tracking in dump files."""

    def setup_method(self):
        reset_step()
        disable_io_dump()
        reset_exec_order()

    def teardown_method(self):
        disable_io_dump()
        reset_exec_order()

    def test_exec_order_increments(self):
        o1 = next_exec_order()
        o2 = next_exec_order()
        assert o2 == o1 + 1

    def test_exec_order_resets(self):
        next_exec_order()
        next_exec_order()
        reset_exec_order()
        assert get_exec_order() == 0

    def test_dump_files_contain_exec_order(self, dump_dir):
        enable_io_dump(dump_dir, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2, 3)
        dump_before("op_a", (t,), {})
        dump_before("op_b", (t,), {})
        io_dumper._wait_and_flush()

        # Check merged JSON files contain exec_order in metadata
        for op in ["op_a", "op_b"]:
            op_dir = os.path.join(dump_dir, "rank_0000", "step_0000", op)
            assert os.path.isfile(os.path.join(op_dir, "input.json"))
            all_meta = _read_jsonl(os.path.join(op_dir, "input.json"))
            meta = all_meta["call_1"]
            assert "exec_order" in meta

    def test_dump_metadata_has_exec_order(self, dump_dir):
        enable_io_dump(dump_dir, with_torch_funcs=False, with_metas=True)
        t = torch.zeros(2)
        reset_exec_order()
        dump_before("test_op", (t,), {})
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isfile(os.path.join(step_dir, "input.json"))
        all_meta = _read_jsonl(os.path.join(step_dir, "input.json"))
        meta = all_meta["call_1"]
        assert meta["exec_order"] == 1

    def test_io_dump_step_resets_exec_order(self, dump_dir):
        enable_io_dump(dump_dir, with_torch_funcs=False, with_metas=True)
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})
        assert get_exec_order() >= 1
        io_dump_step()
        assert get_exec_order() == 0


class TestYamlConfig:
    """Test YAML config parsing for IO dump."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_parse_io_config_dump_section(self):
        cfg_content = """
io_dump:
  dir: /tmp/yaml_dump
  ops:
    - rms_norm
    - silu_and_mul
  modules:
    - Linear
  max_calls: 50
  step_range: "2-10"
  with_torch_funcs: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            dump_cfg = result["io_dump"]
            assert dump_cfg["dir"] == "/tmp/yaml_dump"
            assert dump_cfg["ops"] == {"rms_norm", "silu_and_mul"}
            assert dump_cfg["modules"] == {"Linear"}
            assert dump_cfg["max_calls"] == 50
            assert dump_cfg["step_range"] == (2, 11)
            tf_enabled, tf_filter = dump_cfg["with_torch_funcs"]
            assert tf_enabled is True
            assert tf_filter == set()
        finally:
            os.unlink(cfg_path)

    def test_parse_io_config_missing_file(self):
        result = parse_io_config_from_yaml("/nonexistent/path.yaml")
        assert result == {}

    def test_parse_io_config_torch_funcs_list(self):
        cfg_content = """
io_dump:
  dir: /tmp/yaml_dump
  with_torch_funcs:
    - matmul
    - softmax
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            dump_cfg = result["io_dump"]
            tf_enabled, tf_filter = dump_cfg["with_torch_funcs"]
            assert tf_enabled is True
            assert tf_filter == {"matmul", "softmax"}
        finally:
            os.unlink(cfg_path)

    @patch.dict(os.environ, {}, clear=False)
    def test_init_from_yaml_config(self, dump_dir):
        cfg_content = f"""
io_dump:
  dir: {dump_dir}
  ops:
    - rms_norm
  max_calls: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            os.environ.pop("VLLM_FL_IO_DUMP", None)
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": cfg_path}, clear=False):
                io_dumper.init_io_dump_from_env(True)
            assert is_dump_enabled()
            assert io_dumper._dump_dir == dump_dir
            assert io_dumper._op_filter == {"rms_norm"}
            assert io_dumper._max_calls == 10
        finally:
            os.unlink(cfg_path)


class TestDumpCleanup:
    """Test dump_cleanup for stale pairing removal."""

    def setup_method(self):
        reset_step()
        disable_io_dump()
        reset_exec_order()

    def teardown_method(self):
        disable_io_dump()
        reset_exec_order()

    def test_cleanup_pops_stale_pairing(self, dump_dir):
        enable_io_dump(dump_dir, with_values=True, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)

        # dump_before pushes pairing
        dump_before("test_op", (t,), {})

        # Simulate fn() raising — cleanup should pop the stale entry
        dump_cleanup("test_op")

        # dump_after should now find no pairing and log a warning (not crash)
        dump_after("test_op", (t,), torch.ones(2))
        io_dumper._wait_and_flush()

        # Only the input file should exist (no output file)
        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isfile(os.path.join(step_dir, "call_1_input.pt"))
        assert not os.path.isfile(os.path.join(step_dir, "call_1_output.pt"))
        # output.json should not exist (dump_after found no pairing)
        assert not os.path.isfile(os.path.join(step_dir, "output.json"))

    def test_cleanup_noop_when_no_pairing(self):
        # Should not raise even with nothing to clean up
        dump_cleanup("nonexistent_op")


class TestExecOrderParam:
    """Test that pre-allocated exec_order flows through dump files."""

    def setup_method(self):
        reset_step()
        disable_io_dump()
        reset_exec_order()

    def teardown_method(self):
        disable_io_dump()
        reset_exec_order()

    def test_dump_before_uses_given_order(self, dump_dir):
        enable_io_dump(dump_dir, with_values=True, with_metas=True)
        t = torch.zeros(2)

        dump_before("test_op", (t,), {}, exec_order=99)
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isfile(os.path.join(step_dir, "call_1_input.pt"))

        all_meta = _read_jsonl(os.path.join(step_dir, "input.json"))
        meta = all_meta["call_1"]
        assert meta["exec_order"] == 99

    def test_dump_before_none_order_allocates_internally(self, dump_dir):
        enable_io_dump(dump_dir, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)

        dump_before("test_op", (t,), {})  # exec_order=None
        assert get_exec_order() >= 1


class TestRankInDumper:
    """Test rank directory layout and metadata in dump files."""

    def setup_method(self):
        reset_step()
        disable_io_dump()
        reset_exec_order()
        reset_rank()
        io_dumper._rank_filter = None

    def teardown_method(self):
        disable_io_dump()
        reset_exec_order()
        reset_rank()
        io_dumper._rank_filter = None
        os.environ.pop("RANK", None)

    def test_dump_creates_rank_directory(self, dump_dir):
        reset_rank()
        enable_io_dump(dump_dir, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})

        rank_dir = os.path.join(dump_dir, "rank_0000")
        assert os.path.isdir(rank_dir)

    @patch.dict(os.environ, {"RANK": "3"}, clear=False)
    def test_dump_rank_nonzero(self, dump_dir):
        reset_rank()
        enable_io_dump(dump_dir, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})

        rank_dir = os.path.join(dump_dir, "rank_0003")
        assert os.path.isdir(rank_dir)
        step_dir = os.path.join(rank_dir, "step_0000", "test_op")
        assert os.path.isdir(step_dir)

    def test_dump_metadata_has_rank(self, dump_dir):
        reset_rank()
        enable_io_dump(dump_dir, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        all_meta = _read_jsonl(os.path.join(step_dir, "input.json"))
        meta = all_meta["call_1"]
        assert meta["rank"] == 0

    def test_output_metadata_has_rank(self, dump_dir):
        reset_rank()
        enable_io_dump(dump_dir, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})
        dump_after("test_op", (t,), torch.ones(2))
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        all_meta = _read_jsonl(os.path.join(step_dir, "output.json"))
        meta = all_meta["call_1"]
        assert meta["rank"] == 0


class TestRankFilterInDumper:
    """Test rank filtering prevents dumping on non-matching ranks."""

    def setup_method(self):
        reset_step()
        disable_io_dump()
        reset_exec_order()
        reset_rank()
        io_dumper._rank_filter = None

    def teardown_method(self):
        disable_io_dump()
        reset_exec_order()
        reset_rank()
        io_dumper._rank_filter = None
        os.environ.pop("RANK", None)

    def test_rank_filter_blocks_dump(self, dump_dir):
        reset_rank()  # rank 0
        enable_io_dump(dump_dir, ranks={1, 2})
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})

        # No files should be created
        rank_dir = os.path.join(dump_dir, "rank_0000")
        assert not os.path.exists(rank_dir)

    def test_rank_filter_allows_matching(self, dump_dir):
        reset_rank()  # rank 0
        enable_io_dump(dump_dir, ranks={0}, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})

        rank_dir = os.path.join(dump_dir, "rank_0000")
        assert os.path.isdir(rank_dir)

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump", "VLLM_FL_IO_DUMP_RANK": "0"},
        clear=False,
    )
    def test_env_rank_filter(self):
        reset_rank()
        io_dumper.init_io_dump_from_env(True)
        assert is_dump_enabled()
        assert io_dumper._rank_ok()

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump", "VLLM_FL_IO_DUMP_RANK": "3"},
        clear=False,
    )
    def test_env_rank_filter_blocks(self):
        reset_rank()
        io_dumper.init_io_dump_from_env(True)
        assert is_dump_enabled()
        assert not io_dumper._rank_ok()


class TestYamlRanksConfigDumper:
    """Test YAML ranks field parsing for dumper."""

    def setup_method(self):
        disable_io_dump()
        reset_rank()
        io_dumper._rank_filter = None

    def teardown_method(self):
        disable_io_dump()
        reset_rank()
        io_dumper._rank_filter = None

    def test_yaml_dump_ranks_list(self):
        cfg_content = """
io_dump:
  dir: /tmp/yaml_dump
  ranks: [0, 1]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            result = parse_io_config_from_yaml(cfg_path)
            assert result["io_dump"]["ranks"] == {0, 1}
        finally:
            os.unlink(cfg_path)

    @patch.dict(os.environ, {}, clear=False)
    def test_init_from_yaml_with_ranks(self, dump_dir):
        reset_rank()
        cfg_content = f"""
io_dump:
  dir: {dump_dir}
  ranks: [0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg_content)
            cfg_path = f.name

        try:
            os.environ.pop("VLLM_FL_IO_DUMP", None)
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": cfg_path}, clear=False):
                io_dumper.init_io_dump_from_env(True)
            assert is_dump_enabled()
            assert io_dumper._rank_ok()
        finally:
            os.unlink(cfg_path)
            io_dumper._rank_filter = None


class TestStepSummary:
    """Test that advance_step logs a summary of seen modules and ops."""

    def test_step_advance_logs_summary(self, dump_dir, caplog):
        """After dump_before/dump_after, advance_step should log a summary."""
        enable_io_dump(dump_dir)
        try:
            t = torch.zeros(2)
            dump_before("test_op", (t,), {})
            dump_after("test_op", (t,), t)
            caplog.clear()
            with caplog.at_level("INFO"):
                advance_step()
            summary_msgs = [
                r.message for r in caplog.records if "Step summary" in r.message
            ]
            assert len(summary_msgs) == 1
            assert "test_op" in summary_msgs[0]
        finally:
            disable_io_dump()
            reset_step()

    def test_no_summary_when_no_ops(self, dump_dir, caplog):
        """If no ops were dumped, no summary should be logged."""
        enable_io_dump(dump_dir)
        try:
            caplog.clear()
            with caplog.at_level("INFO"):
                advance_step()
            summary_msgs = [
                r.message for r in caplog.records if "Step summary" in r.message
            ]
            assert len(summary_msgs) == 0
        finally:
            disable_io_dump()
            reset_step()


class TestWithMetas:
    """Test with_metas / with_values dump mode flags."""

    def setup_method(self):
        reset_step()
        disable_io_dump()
        reset_exec_order()

    def teardown_method(self):
        disable_io_dump()
        reset_exec_order()

    def test_with_metas_creates_json_only(self, dump_dir):
        """with_values=False (default) creates .json but not .pt files."""
        enable_io_dump(dump_dir, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2, 3)
        dump_before("test_op", (t,), {})
        dump_after("test_op", (t,), t)
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isdir(step_dir)
        files = os.listdir(step_dir)
        pt_files = [f for f in files if f.endswith(".pt")]
        json_files = [f for f in files if f.endswith(".json")]
        assert len(pt_files) == 0
        assert "input.json" in json_files
        assert "output.json" in json_files

    def test_with_metas_json_has_tensor_stats(self, dump_dir):
        """JSON (without .pt files) should still have tensor stats via 'tensors' key."""
        enable_io_dump(dump_dir, with_metas=True)
        reset_exec_order()
        t = torch.randn(4, 5)
        dump_before("test_op", (t,), {})
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        all_meta = _read_jsonl(os.path.join(step_dir, "input.json"))
        meta = all_meta["call_1"]
        assert "tensors" in meta
        assert "arg_0" in meta["tensors"]
        assert meta["tensors"]["arg_0"]["shape"] == [4, 5]

    def test_with_values_creates_pt(self, dump_dir):
        """with_values=True should create .pt files."""
        enable_io_dump(dump_dir, with_values=True, with_metas=True)
        reset_exec_order()
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})
        io_dumper._wait_and_flush()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        assert os.path.isfile(os.path.join(step_dir, "call_1_input.pt"))

    def test_with_metas_state(self, dump_dir):
        enable_io_dump(dump_dir)
        assert io_dumper._with_values is False

    def test_with_values_resets(self, dump_dir):
        enable_io_dump(dump_dir, with_values=True)
        disable_io_dump()
        assert io_dumper._with_values is False

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump"},
        clear=False,
    )
    def test_env_with_values_default_false(self):
        os.environ.pop("VLLM_FL_IO_DUMP_WITH_VALUES", None)
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._with_values is False

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump"},
        clear=False,
    )
    def test_env_with_values_unset(self):
        os.environ.pop("VLLM_FL_IO_DUMP_WITH_VALUES", None)
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._with_values is False  # default is False

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump", "VLLM_FL_IO_DUMP_WITH_VALUES": "1"},
        clear=False,
    )
    def test_env_with_values_enabled(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._with_values is True

    def test_default_with_values_is_false(self, dump_dir):
        """Default enable_io_dump() should have with_values=False."""
        enable_io_dump(dump_dir)
        assert io_dumper._with_values is False


class TestLayerFilter:
    """Test layer path filtering for IO dumper."""

    def setup_method(self):
        reset_step()
        disable_io_dump()
        reset_exec_order()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        register_module_paths(self.model)

    def teardown_method(self):
        disable_io_dump()
        reset_exec_order()

    def test_layer_filter_state(self, dump_dir):
        enable_io_dump(dump_dir, layers={"model.layers.0"})
        assert io_dumper._layer_filter == {"model.layers.0"}

    def test_layer_filter_resets(self, dump_dir):
        enable_io_dump(dump_dir, layers={"model.layers.0"})
        disable_io_dump()
        assert io_dumper._layer_filter == set()

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_LAYERS": "model.layers.0",
        },
        clear=False,
    )
    def test_env_layer_filter(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._layer_filter == {"model.layers.0"}

    @patch.dict(
        os.environ,
        {
            "VLLM_FL_IO_DUMP": "/tmp/test_dump",
            "VLLM_FL_IO_DUMP_LAYERS": "model.layers.0,model.layers.1",
        },
        clear=False,
    )
    def test_env_dump_layer_filter(self):
        io_dumper.init_io_dump_from_env(True)
        assert io_dumper._layer_filter == {"model.layers.0", "model.layers.1"}

    @pytest.mark.skipif(
        not HAS_TORCH_DISPATCH_MODE,
        reason="TorchDispatchMode not available in this PyTorch version",
    )
    def test_layer_filter_activates_dispatch_mode(self, dump_dir):
        """Layer filter needs TorchDispatchMode for stack-based path tracking."""
        enable_io_dump(dump_dir, ops={"rms_norm"}, layers={"model.layers.0"})
        from vllm_fl.dispatch.io_common import dispatch_mode_mgr

        assert dispatch_mode_mgr.is_entered("dump")


class TestComposableFilters:
    """Test AND-based composable filter logic for dumper."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_op_and_module_filter(self, dump_dir):
        """When both op and module filter are set, both must match (AND)."""
        enable_io_dump(dump_dir, ops={"rms_norm"}, modules={"Linear"})
        m = torch.nn.Linear(10, 10)
        assert _should_dump("rms_norm", (m,))
        assert not _should_dump("silu_and_mul", (m,))
        assert not _should_dump("rms_norm", (torch.zeros(2),))

    def test_op_filter_only(self, dump_dir):
        enable_io_dump(dump_dir, ops={"rms_norm"})
        assert _should_dump("rms_norm", ())
        assert not _should_dump("silu_and_mul", ())

    def test_module_filter_only(self, dump_dir):
        enable_io_dump(dump_dir, modules={"Linear"})
        m = torch.nn.Linear(10, 10)
        assert _should_dump("any_op", (m,))
        assert not _should_dump("any_op", (torch.zeros(2),))


class TestExpandLayerSpecs:
    """Test expand_layer_specs() shorthand expansion."""

    def test_integer_shorthand(self):
        result = expand_layer_specs({"0", "1", "2"})
        assert result == {"model.layers.0", "model.layers.1", "model.layers.2"}

    def test_range_shorthand(self):
        result = expand_layer_specs({"0-3"})
        assert result == {
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
        }

    def test_full_path_kept(self):
        result = expand_layer_specs({"model.layers.0.self_attn"})
        assert result == {"model.layers.0.self_attn"}

    def test_glob_pattern_kept(self):
        result = expand_layer_specs({"model.layers.*.self_attn"})
        assert result == {"model.layers.*.self_attn"}

    def test_mixed_specs(self):
        result = expand_layer_specs({"0", "2-4", "model.layers.*.mlp"})
        assert "model.layers.0" in result
        assert "model.layers.2" in result
        assert "model.layers.3" in result
        assert "model.layers.4" in result
        assert "model.layers.*.mlp" in result

    def test_custom_prefix(self):
        result = expand_layer_specs({"0", "1"}, prefix="encoder.layers.")
        assert result == {"encoder.layers.0", "encoder.layers.1"}

    def test_empty_specs(self):
        result = expand_layer_specs(set())
        assert result == set()

    def test_whitespace_stripped(self):
        result = expand_layer_specs({" 0 ", " model.layers.1 "})
        assert "model.layers.0" in result
        assert "model.layers.1" in result


class TestGlobLayerMatching:
    """Test glob/wildcard layer path matching."""

    def setup_method(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        register_module_paths(self.model)

    def test_glob_matches(self):
        """Wildcard '*' should match any segment."""
        # Path for model[0] is "0" (Sequential names children 0, 1, 2)
        push_module_context("Linear", self.model[0])
        try:
            # Pattern "?" matches single char "0"
            assert layer_path_matches({"?"})
        finally:
            pop_module_context()

    def test_glob_no_match(self):
        push_module_context("Linear", self.model[0])
        try:
            assert not layer_path_matches({"model.layers.*"})
        finally:
            pop_module_context()

    def test_expand_then_match(self, dump_dir):
        """Integration: expand integer spec then match."""
        # model children are named "0", "1", "2" by Sequential
        specs = expand_layer_specs({"0"}, prefix="")
        # expand_layer_specs("0", prefix="") => {"0"}
        push_module_context("Linear", self.model[0])
        try:
            assert layer_path_matches(specs)
        finally:
            pop_module_context()

    def test_layer_shorthand_in_enable_dump(self, dump_dir):
        """Integer shorthand should be expanded in enable_io_dump."""
        enable_io_dump(dump_dir, layers={"0", "1-2"})
        assert "model.layers.0" in io_dumper._layer_filter
        assert "model.layers.1" in io_dumper._layer_filter
        assert "model.layers.2" in io_dumper._layer_filter
        disable_io_dump()

    @patch.dict(
        os.environ,
        {"VLLM_FL_IO_DUMP": "/tmp/test_dump", "VLLM_FL_IO_DUMP_LAYERS": "0,1-3"},
        clear=False,
    )
    def test_env_layer_expansion(self):
        """Env var layer specs should be expanded."""
        io_dumper.init_io_dump_from_env(True)
        assert "model.layers.0" in io_dumper._layer_filter
        assert "model.layers.1" in io_dumper._layer_filter
        assert "model.layers.2" in io_dumper._layer_filter
        assert "model.layers.3" in io_dumper._layer_filter
        disable_io_dump()


class TestListModelLayers:
    """Test list_model_layers() utility."""

    def test_list_layers(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
        )
        paths = list_model_layers(model)
        assert len(paths) == 2
        assert any("Linear" in p for p in paths)
        assert any("ReLU" in p for p in paths)

    def test_list_layers_max_depth(self):
        model = torch.nn.Sequential(
            torch.nn.Sequential(
                torch.nn.Linear(4, 3),
            ),
        )
        # depth 1 should show "0" (inner Sequential) but not "0.0" (Linear)
        paths = list_model_layers(model, max_depth=1)
        assert len(paths) == 1
        assert any("Sequential" in p for p in paths)

    def test_list_layers_empty_model(self):
        model = torch.nn.Module()
        paths = list_model_layers(model)
        assert paths == []


class TestTensorStats:
    """Test tensor_stats() and register_tensor_stat() extensibility."""

    def test_basic_stats(self):
        from vllm_fl.dispatch.io_common import tensor_stats

        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        s = tensor_stats(t)
        assert s["shape"] == [4]
        assert "torch.float" in s["dtype"]
        assert s["min"] == 1.0
        assert s["max"] == 4.0
        assert abs(s["mean"] - 2.5) < 1e-5

    def test_integer_tensor_skips_float_only(self):
        from vllm_fl.dispatch.io_common import tensor_stats

        t = torch.tensor([1, 2, 3])
        s = tensor_stats(t)
        assert s["shape"] == [3]
        assert s["min"] == 1
        assert s["max"] == 3
        # mean and std are float_only, skipped for int
        assert "mean" not in s
        assert "std" not in s

    def test_empty_tensor(self):
        from vllm_fl.dispatch.io_common import tensor_stats

        t = torch.zeros(0)
        s = tensor_stats(t)
        assert s["shape"] == [0]
        # No stats for empty tensor
        assert "min" not in s

    def test_register_custom_stat(self):
        from vllm_fl.dispatch.io_common import (
            _TENSOR_STAT_REGISTRY,
            register_tensor_stat,
            tensor_stats,
        )

        original_len = len(_TENSOR_STAT_REGISTRY)
        try:
            register_tensor_stat("l2_norm", lambda t: t.norm(2).item())
            t = torch.tensor([3.0, 4.0])
            s = tensor_stats(t)
            assert "l2_norm" in s
            assert abs(s["l2_norm"] - 5.0) < 1e-5
        finally:
            # Clean up: remove the custom stat
            while len(_TENSOR_STAT_REGISTRY) > original_len:
                _TENSOR_STAT_REGISTRY.pop()

    def test_register_non_float_stat(self):
        from vllm_fl.dispatch.io_common import (
            _TENSOR_STAT_REGISTRY,
            register_tensor_stat,
            tensor_stats,
        )

        original_len = len(_TENSOR_STAT_REGISTRY)
        try:
            register_tensor_stat(
                "num_zeros",
                lambda t: int((t == 0).sum().item()),
                float_only=False,
            )
            t = torch.tensor([0, 1, 0, 2])
            s = tensor_stats(t)
            assert s["num_zeros"] == 2
        finally:
            while len(_TENSOR_STAT_REGISTRY) > original_len:
                _TENSOR_STAT_REGISTRY.pop()


class TestSummaryJson:
    """Test summary.json generation on disable."""

    def setup_method(self):
        disable_io_dump()

    def teardown_method(self):
        disable_io_dump()

    def test_summary_written_on_disable(self, dump_dir):
        """disable_io_dump() should write summary.json under rank dir."""
        enable_io_dump(dump_dir=dump_dir)
        t = torch.zeros(2, 3)
        dump_before("test_op", (t,), {})
        dump_after("test_op", (t,), t)
        disable_io_dump()

        summary_path = os.path.join(dump_dir, "rank_0000", "summary.json")
        assert os.path.isfile(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert "flaggems_ops" in summary
        assert "non_flaggems_ops" in summary
        # test_op has no dispatch_keys → non-FlagGems
        assert "test_op" in summary["non_flaggems_ops"]

    def test_summary_empty_when_no_ops(self, dump_dir):
        """No summary.json when no ops were dumped."""
        enable_io_dump(dump_dir=dump_dir)
        disable_io_dump()

        summary_path = os.path.join(dump_dir, "rank_0000", "summary.json")
        assert not os.path.exists(summary_path)

    def test_summary_flaggems_classification(self, dump_dir):
        """Ops with FlagGems dispatch keys go into flaggems_ops."""
        from vllm_fl.dispatch.io_dumper import _lock, _op_summary

        enable_io_dump(dump_dir=dump_dir)
        # Manually inject a FlagGems op entry
        with _lock:
            _op_summary["aten.mm"] = {
                "dispatch_keys": "[(CUDA, FlagGems, False)]",
                "call_count": 5,
            }
            _op_summary["aten.add"] = {
                "dispatch_keys": "[(CPU, CPU, False)]",
                "call_count": 3,
            }
        disable_io_dump()

        summary_path = os.path.join(dump_dir, "rank_0000", "summary.json")
        assert os.path.isfile(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert "aten.mm" in summary["flaggems_ops"]
        assert "aten.add" in summary["non_flaggems_ops"]
        assert "aten.mm" not in summary["non_flaggems_ops"]

    def test_summary_multiple_ops(self, dump_dir):
        """Multiple ops are collected across calls."""
        enable_io_dump(dump_dir=dump_dir)
        t = torch.zeros(2)
        dump_before("op_a", (t,), {})
        dump_after("op_a", (t,), t)
        dump_before("op_b", (t,), {})
        dump_after("op_b", (t,), t)
        dump_before("op_a", (t,), {})
        dump_after("op_a", (t,), t)
        disable_io_dump()

        summary_path = os.path.join(dump_dir, "rank_0000", "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        non_fg = summary["non_flaggems_ops"]
        assert "op_a" in non_fg
        assert "op_b" in non_fg


class TestSummaryOnly:
    """Tests for summary (no per-op JSON) mode."""

    def setup_method(self):
        reset_exec_order()
        reset_step()
        reset_rank()

    def teardown_method(self):
        disable_io_dump()

    def test_summary_only_writes_summary(self, dump_dir):
        """Default mode (with_metas=False) still writes summary.json."""
        enable_io_dump(dump_dir=dump_dir)
        t = torch.zeros(2)
        dump_before("test_op", (t,), {})
        dump_after("test_op", (t,), t)
        disable_io_dump()

        summary_path = os.path.join(dump_dir, "rank_0000", "summary.json")
        assert os.path.isfile(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert "test_op" in summary["non_flaggems_ops"]

    def test_summary_only_no_per_op_dirs(self, dump_dir):
        """with_metas=False (default) does NOT create per-op directories."""
        enable_io_dump(dump_dir=dump_dir)
        # Run ops through dispatch mode
        t = torch.zeros(2)
        _ = t + t  # triggers dispatch mode
        disable_io_dump()

        rank_dir = os.path.join(dump_dir, "rank_0000")
        if os.path.isdir(rank_dir):
            # Only summary.json should exist, no step_XXXX dirs
            contents = os.listdir(rank_dir)
            step_dirs = [d for d in contents if d.startswith("step_")]
            assert step_dirs == [], f"Expected no step dirs, found: {step_dirs}"

    def test_summary_only_env_var(self, dump_dir):
        """Without VLLM_FL_IO_DUMP_WITH_METAS, with_metas defaults to False."""
        with patch.dict(
            os.environ,
            {
                "VLLM_FL_IO_DUMP": dump_dir,
            },
        ):
            from vllm_fl.dispatch.io_dumper import _init_from_env

            _init_from_env()
            from vllm_fl.dispatch import io_dumper

            assert io_dumper._with_metas is False
        disable_io_dump()

    def test_summary_only_dump_env_var(self, dump_dir):
        """VLLM_FL_IO_DUMP_WITH_METAS=1 enables per-op JSON dumps."""
        with patch.dict(
            os.environ,
            {
                "VLLM_FL_IO_DUMP": dump_dir,
                "VLLM_FL_IO_DUMP_WITH_METAS": "1",
            },
        ):
            from vllm_fl.dispatch.io_dumper import _init_from_env

            _init_from_env()
            from vllm_fl.dispatch import io_dumper

            assert io_dumper._with_metas is True
        disable_io_dump()


class TestJsonMergedWrite:
    """Tests for merged JSON write (multiple calls to same op)."""

    def setup_method(self):
        reset_exec_order()
        reset_step()
        reset_rank()

    def teardown_method(self):
        disable_io_dump()

    def test_buffer_produces_correct_merged_output(self, dump_dir):
        """Multiple buffered writes produce correct merged JSON."""
        enable_io_dump(dump_dir=dump_dir, with_values=True, with_metas=True)
        t = torch.zeros(2)
        # Multiple calls to the same op
        dump_before("test_op", (t,), {})
        dump_after("test_op", (t,), t)
        dump_before("test_op", (t,), {})
        dump_after("test_op", (t,), t)
        disable_io_dump()

        step_dir = os.path.join(dump_dir, "rank_0000", "step_0000", "test_op")
        input_json = os.path.join(step_dir, "input.json")
        assert os.path.isfile(input_json)
        data = _read_jsonl(input_json)
        assert "call_1" in data
        assert "call_2" in data


class TestInitIoDumpFromEnv:
    """Test init_io_dump_from_env() public entry point."""

    def teardown_method(self):
        disable_io_dump()

    @patch.dict(os.environ, {"VLLM_FL_IO_DUMP": "/tmp/test_init"}, clear=False)
    def test_eager_true_enables_dump(self):
        init_io_dump_from_env(True)
        assert is_dump_enabled()

    @patch.dict(os.environ, {"VLLM_FL_IO_DUMP": "/tmp/test_init"}, clear=False)
    def test_eager_false_skips_dump(self):
        """Graph mode: IO dumping silently skipped."""
        init_io_dump_from_env(False)
        assert not is_dump_enabled()

    def test_no_env_vars_no_op(self):
        for k in list(os.environ):
            if k.startswith("VLLM_FL_IO_DUMP") or k == "VLLM_FL_CONFIG":
                os.environ.pop(k)
        init_io_dump_from_env(True)
        assert not is_dump_enabled()

    @patch.dict(os.environ, {"VLLM_FL_IO_DUMP": "/tmp/test_init"}, clear=False)
    def test_idempotent(self):
        """Calling twice should not raise or reset state."""
        init_io_dump_from_env(True)
        assert is_dump_enabled()
        init_io_dump_from_env(True)
        assert is_dump_enabled()


class TestAdvanceIoStep:
    """Test advance_io_step() public helper."""

    def teardown_method(self):
        disable_io_dump()

    def test_no_op_when_disabled(self):
        step_before = get_step()
        advance_io_step()
        assert get_step() == step_before

    def test_advances_when_enabled(self, dump_dir):
        enable_io_dump(dump_dir)
        step_before = get_step()
        advance_io_step()
        assert get_step() == step_before + 1


class TestRegisterIoModuleHooks:
    """Test register_io_module_hooks() public helper."""

    def teardown_method(self):
        disable_io_dump()

    def test_no_op_when_disabled(self):
        """Should not install hooks when IO is not enabled."""
        model = torch.nn.Linear(4, 4)
        hooks_before = len(model._forward_pre_hooks) + len(model._forward_hooks)
        register_io_module_hooks(model)
        hooks_after = len(model._forward_pre_hooks) + len(model._forward_hooks)
        assert hooks_before == hooks_after

    def test_installs_hooks_when_enabled(self, dump_dir):
        """Should install forward hooks on all modules when IO is enabled."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        enable_io_dump(dump_dir)
        register_io_module_hooks(model)
        total_hooks = sum(
            len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules()
        )
        assert total_hooks > 0
