# Copyright (c) 2026 BAAI. All rights reserved.

"""
High-coverage unit tests for io_dumper filtering / gating helpers.

Covers:
    _rank_ok, _check_limits, _should_dump, _should_dump_torch_func,
    dump_cleanup, is_dump_enabled, advance_io_step, io_dump_step
"""

import pytest
import torch

from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch.io_common import (
    pop_module_context,
    push_module_context,
    reset_exec_order,
    reset_rank,
    reset_step,
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    reset_step()
    reset_exec_order()
    reset_rank()
    io_dumper.disable_io_dump()
    monkeypatch.setattr(io_dumper, "_step_range", None)
    monkeypatch.setattr(io_dumper, "_max_calls", 0)
    monkeypatch.setattr(io_dumper, "_match_all", False)
    monkeypatch.setattr(io_dumper, "_op_filter", set())
    monkeypatch.setattr(io_dumper, "_module_filter", set())
    monkeypatch.setattr(io_dumper, "_layer_filter", set())
    monkeypatch.setattr(io_dumper, "_rank_filter", None)
    monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", False)
    monkeypatch.setattr(io_dumper, "_torch_func_filter", set())
    monkeypatch.setattr(io_dumper, "_call_counters", {})
    monkeypatch.setattr(io_dumper, "_enabled", False)
    for tls in (io_dumper._dump_pairing, io_dumper._print_pairing):
        try:
            delattr(tls, "stack")
        except AttributeError:
            pass
    yield
    io_dumper.disable_io_dump()
    reset_step()
    reset_exec_order()


class TestRankOk:
    """_rank_ok honours the per-subsystem rank filter."""

    def test_all_ranks_when_filter_none(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_rank_filter", None)
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 7)
        assert io_dumper._rank_ok() is True

    def test_match(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_rank_filter", {0, 2})
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 2)
        assert io_dumper._rank_ok() is True

    def test_mismatch(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_rank_filter", {0, 2})
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 1)
        assert io_dumper._rank_ok() is False


class TestCheckLimits:
    """_check_limits gates on step_range and max_calls."""

    def test_no_limits_passes(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", None)
        monkeypatch.setattr(io_dumper, "_max_calls", 0)
        assert io_dumper._check_limits("op") is True

    def test_step_below_range(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", (5, 8))
        monkeypatch.setattr(io_dumper, "get_step", lambda: 4)
        assert io_dumper._check_limits("op") is False

    def test_step_in_range(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", (5, 8))
        monkeypatch.setattr(io_dumper, "get_step", lambda: 7)
        assert io_dumper._check_limits("op") is True

    def test_step_at_upper_bound_excluded(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", (5, 8))
        monkeypatch.setattr(io_dumper, "get_step", lambda: 8)
        assert io_dumper._check_limits("op") is False

    def test_max_calls_reached(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_max_calls", 2)
        monkeypatch.setattr(io_dumper, "_call_counters", {"op": 2})
        assert io_dumper._check_limits("op") is False

    def test_max_calls_not_reached(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_max_calls", 2)
        monkeypatch.setattr(io_dumper, "_call_counters", {"op": 1})
        assert io_dumper._check_limits("op") is True

    def test_max_calls_for_other_op(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_max_calls", 1)
        monkeypatch.setattr(io_dumper, "_call_counters", {"other": 5})
        assert io_dumper._check_limits("op") is True


class TestShouldDump:
    """_should_dump composes all filter dimensions with AND logic."""

    def test_limits_fail_short_circuits(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", (10, 20))
        monkeypatch.setattr(io_dumper, "get_step", lambda: 0)
        assert io_dumper._should_dump("op", ()) is False

    def test_layer_filter_mismatch(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", True)
        monkeypatch.setattr(io_dumper, "_layer_filter", {"model.layers.0"})
        monkeypatch.setattr(io_dumper, "layer_path_matches", lambda flt: False)
        assert io_dumper._should_dump("op", ()) is False

    def test_layer_filter_match_continues(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", True)
        monkeypatch.setattr(io_dumper, "_layer_filter", {"model.layers.0"})
        monkeypatch.setattr(io_dumper, "layer_path_matches", lambda flt: True)
        assert io_dumper._should_dump("op", ()) is True

    def test_match_all_returns_true(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", True)
        assert io_dumper._should_dump("anything", ()) is True

    def test_module_filter_via_arg_module(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
        monkeypatch.setattr(io_dumper, "get_module_class_name", lambda args: "Linear")
        monkeypatch.setattr(io_dumper, "module_context_matches", lambda s: False)
        assert io_dumper._should_dump("op", ()) is True

    def test_module_filter_via_context(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
        monkeypatch.setattr(io_dumper, "get_module_class_name", lambda args: None)
        monkeypatch.setattr(io_dumper, "module_context_matches", lambda s: True)
        assert io_dumper._should_dump("op", ()) is True

    def test_module_filter_no_match(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
        monkeypatch.setattr(io_dumper, "get_module_class_name", lambda args: None)
        monkeypatch.setattr(io_dumper, "module_context_matches", lambda s: False)
        assert io_dumper._should_dump("op", ()) is False

    def test_op_filter_mismatch(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_op_filter", {"rms_norm"})
        assert io_dumper._should_dump("silu", ()) is False

    def test_op_filter_match(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_op_filter", {"rms_norm"})
        assert io_dumper._should_dump("rms_norm", ()) is True

    def test_no_filter_no_match_all_returns_false(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_op_filter", set())
        monkeypatch.setattr(io_dumper, "_module_filter", set())
        assert io_dumper._should_dump("op", ()) is False

    def test_real_module_context_match(self, monkeypatch, tmp_path):
        # Integration with the real io_common module-context stack.
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
        push_module_context("Linear", torch.nn.Linear(2, 2))
        try:
            assert io_dumper._should_dump("op", ()) is True
        finally:
            pop_module_context()


class TestShouldDumpTorchFunc:
    """_should_dump_torch_func gates bare torch functions."""

    def test_step_out_of_range(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", (5, 6))
        monkeypatch.setattr(io_dumper, "get_step", lambda: 0)
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        assert io_dumper._should_dump_torch_func("matmul") is False

    def test_step_in_range(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", (5, 6))
        monkeypatch.setattr(io_dumper, "get_step", lambda: 5)
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_match_all", True)
        assert io_dumper._should_dump_torch_func("matmul") is True

    def test_layer_filter_mismatch(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_layer_filter", {"model.layers.0"})
        monkeypatch.setattr(io_dumper, "layer_path_matches", lambda flt: False)
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        assert io_dumper._should_dump_torch_func("matmul") is False

    def test_disabled_torch_funcs(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", False)
        assert io_dumper._should_dump_torch_func("matmul") is False

    def test_skips_dunder(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_match_all", True)
        assert io_dumper._should_dump_torch_func("__add__") is False

    def test_skips_trivial(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_match_all", True)
        assert io_dumper._should_dump_torch_func("size") is False

    def test_specific_filter_match(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_torch_func_filter", {"matmul"})
        assert io_dumper._should_dump_torch_func("matmul") is True
        assert io_dumper._should_dump_torch_func("softmax") is False

    def test_module_filter_requires_context(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
        monkeypatch.setattr(io_dumper, "module_context_matches", lambda s: False)
        assert io_dumper._should_dump_torch_func("matmul") is False

    def test_module_filter_context_match(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
        monkeypatch.setattr(io_dumper, "module_context_matches", lambda s: True)
        assert io_dumper._should_dump_torch_func("matmul") is True


class TestDumpCleanup:
    """dump_cleanup pops stale file + print pairings."""

    def test_pops_both(self):
        io_dumper._push_pairing("op", 1, 1, "/x", label="L", module_tag="M", op_tag="T")
        io_dumper._push_print_pairing("op", "L", 1, ["a"], "T")
        io_dumper.dump_cleanup("op")
        assert io_dumper._pop_pairing("op") is None
        assert io_dumper._pop_print_pairing("op") is None

    def test_noop_when_empty(self):
        # Must not raise when nothing was pushed.
        io_dumper.dump_cleanup("never_pushed")


class TestEnableFlagHelpers:
    """is_dump_enabled / advance_io_step / io_dump_step."""

    def test_is_dump_enabled_false(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", False)
        assert io_dumper.is_dump_enabled() is False

    def test_is_dump_enabled_true(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", True)
        assert io_dumper.is_dump_enabled() is True

    def test_advance_io_step_noop_when_disabled(self, monkeypatch):
        calls = []
        monkeypatch.setattr(io_dumper, "_enabled", False)
        monkeypatch.setattr(io_dumper, "advance_step", lambda: calls.append(1))
        io_dumper.advance_io_step()
        assert calls == []

    def test_advance_io_step_calls_advance_when_enabled(self, monkeypatch):
        calls = []
        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "advance_step", lambda: calls.append(1) or 42)
        io_dumper.advance_io_step()
        assert calls == [1]

    def test_io_dump_step_returns_new_step(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "advance_step", lambda: 7)
        assert io_dumper.io_dump_step() == 7
