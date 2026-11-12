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

"""Supplementary coverage tests for ``vllm_fl/dispatch/io_common.py`` (1/2).

D3 tranche of the flagcicd coverage campaign. Every test asserts an
independently computed behavior value (expected set/dict/string built
by hand, not a copy of the implementation output). This file covers the
mode manager, rank/step/layer parsing, stack-derived module context,
tensor statistics registry, and value formatting.

Local collection note: importing ``vllm_fl`` pulls in ``flag_gems``
which needs triton (no Windows wheel). ``_flaggems_shim`` installs an
inert placeholder only when the real package cannot import.
"""

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from ._dumper_helpers import restore_io_common_globals  # noqa: F401

import threading

import pytest
import torch

from vllm_fl.dispatch import io_common
from vllm_fl.dispatch.io_common import (
    ModeManager,
    expand_layer_specs,
    format_result,
    format_value,
    get_current_module_counter,
    get_module_context_from_stack,
    get_rank,
    layer_path_matches_from_stack,
    list_model_layers,
    make_label,
    make_module_tag_from_ctx,
    module_context_matches_from_stack,
    parse_layers_env,
    parse_rank_filter,
    parse_step_range,
    parse_step_range_env,
    reset_rank,
)

pytestmark = pytest.mark.usefixtures("restore_io_common_globals")


class _Toy(torch.nn.Module):
    """Root module for stack-based context tests; calls its inner child."""

    def __init__(self):
        super().__init__()
        self.inner = _Inner()

    def forward(self, x):
        return self.inner(x)


class _Inner(torch.nn.Module):
    """Nested module for stack-based context tests."""

    def forward(self, x):
        return x


class _StackProbeModel(_Toy):
    """Captures the stack-derived module context from inside forward."""

    def __init__(self):
        super().__init__()
        self.captured = []

    def run_probe(self):
        # Build the capturing child first so the id->path registry covers it.
        outer_self = self

        class Inner(_Inner):
            def forward(self_, x_):
                outer_self.captured.extend(get_module_context_from_stack())
                return x_

        self.inner = Inner()
        io_common.register_module_paths(self)
        self(torch.zeros(1))


class TestModeManager:
    """LIFO-safe mode manager behavior."""

    def test_exit_all_unwinds_in_reverse_lifo_order(self):
        """exit_all pops every entered mode, innermost first, and clears pending."""
        mgr = ModeManager()
        exits = []

        class _Mode:
            def __init__(self, tag):
                self.tag = tag

            def __enter__(self):
                entered.append(self.tag)

            def __exit__(self, *exc):
                exits.append(self.tag)

        entered = []
        outer, inner = _Mode("outer"), _Mode("inner")
        mgr.enter("a", outer)
        mgr.enter("b", inner)
        assert entered == ["outer", "inner"]
        # Request exit for the *outer* (not top) mode: stays pending.
        mgr.request_exit("a")
        assert mgr.is_entered("a") and mgr.is_entered("b")
        mgr.exit_all()
        assert exits == ["inner", "outer"]
        assert not mgr.is_entered("a") and not mgr.is_entered("b")
        assert mgr._pending == set()

    def test_request_exit_drains_pending_stack_top(self):
        """A pending non-top exit completes once the mode above it exits."""
        mgr = ModeManager()

        class _Mode:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                exits.append(1)

        exits = []
        m1, m2 = _Mode(), _Mode()
        mgr.enter("low", m1)
        mgr.enter("high", m2)
        mgr.request_exit("low")   # not top: stays entered
        assert mgr.is_entered("low")
        mgr.request_exit("high")  # top: exits, then drains pending "low"
        assert exits == [1, 1]
        assert not mgr.is_entered("low") and not mgr.is_entered("high")

    def test_enter_is_idempotent_per_name(self):
        """Entering the same name twice keeps a single stack entry."""
        mgr = ModeManager()

        class _Mode:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return None

        mode = _Mode()
        mgr.enter("dump", mode)
        mgr.enter("dump", mode)
        assert mgr._stack == [("dump", mode)]


class TestGetRank:
    """Rank resolution order (env vars, caching, invalid values)."""

    def test_rank_env_var_cached_after_first_read(self, monkeypatch):
        """RANK env var is parsed once and cached; later env changes are ignored."""
        monkeypatch.setenv("RANK", "3")
        reset_rank()
        assert get_rank() == 3
        monkeypatch.setenv("RANK", "7")
        assert get_rank() == 3  # cached
        reset_rank()

    def test_local_rank_used_when_rank_unset(self, monkeypatch):
        """LOCAL_RANK is consulted only when RANK is absent/invalid."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "5")
        reset_rank()
        try:
            assert get_rank() == 5
        finally:
            monkeypatch.delenv("LOCAL_RANK", raising=False)
            reset_rank()

    def test_invalid_rank_env_falls_back_to_zero_uncached(self, monkeypatch):
        """A non-integer RANK yields 0 and is not cached (fallback never is)."""
        monkeypatch.setenv("RANK", "notanint")
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        reset_rank()
        assert get_rank() == 0
        monkeypatch.setenv("RANK", "9")
        assert get_rank() == 9  # previous fallback was not cached
        reset_rank()

    def test_rank_env_precedence_over_local_rank(self, monkeypatch):
        """RANK wins over LOCAL_RANK when both are set."""
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("LOCAL_RANK", "6")
        reset_rank()
        try:
            assert get_rank() == 2
        finally:
            monkeypatch.delenv("RANK", raising=False)
            monkeypatch.delenv("LOCAL_RANK", raising=False)
            reset_rank()


class TestParseRankFilter:
    """Rank filter string parsing."""

    def test_all_and_empty_yield_none(self):
        """"all" and "" both mean every rank (None)."""
        assert parse_rank_filter("all") is None
        assert parse_rank_filter("  ") is None
        assert parse_rank_filter("ALL") is None

    def test_comma_separated_ranks(self):
        """Comma-separated integers parse into a set, junk tokens dropped."""
        assert parse_rank_filter("0,2,4") == {0, 2, 4}
        assert parse_rank_filter(" 1 , x , 3 ") == {1, 3}

    def test_no_valid_rank_returns_none(self):
        """A string with no valid integers returns None (all ranks)."""
        assert parse_rank_filter("a,b") is None


class TestStepRange:
    """Step range string parsing (inclusive → half-open)."""

    def test_dash_range_is_inclusive(self):
        """"0-2" covers steps 0,1,2 → half-open (0, 3)."""
        assert parse_step_range("0-2") == (0, 3)
        assert parse_step_range("5-15") == (5, 16)

    def test_bare_integer_is_single_step(self):
        """"5" covers only step 5 → (5, 6)."""
        assert parse_step_range("5") == (5, 6)

    def test_invalid_values_return_none(self):
        """None, empty, non-string and unparseable values yield None."""
        assert parse_step_range(None) is None
        assert parse_step_range("") is None
        assert parse_step_range(7) is None
        assert parse_step_range("abc") is None
        assert parse_step_range("x-y") is None

    def test_env_var_resolution_first_non_empty_wins(self, monkeypatch):
        """The first env var with a valid value is used; empty ones skipped."""
        monkeypatch.setenv("COV_EMPTY", "")
        monkeypatch.setenv("COV_GOOD", "2-3")
        assert parse_step_range_env("COV_EMPTY", "COV_GOOD") == (2, 4)
        monkeypatch.setenv("COV_EMPTY", "junk")
        assert parse_step_range_env("COV_EMPTY", "COV_GOOD") == (2, 4)
        assert parse_step_range_env("COV_MISSING_VAR") is None


class TestLayerSpecs:
    """Layer spec expansion and env parsing."""

    def test_integer_range_glob_and_full_path(self):
        """Integer → prefixed path, range expands, glob/full paths kept."""
        assert expand_layer_specs({"2"}) == {"model.layers.2"}
        assert expand_layer_specs({"0-2"}) == {
            "model.layers.0", "model.layers.1", "model.layers.2"}
        assert expand_layer_specs({"model.layers.*.self_attn"}) == {
            "model.layers.*.self_attn"}
        assert expand_layer_specs({"decoder.h.0.mlp"}) == {"decoder.h.0.mlp"}
        assert expand_layer_specs({"5"}, prefix="layers.") == {"layers.5"}
        assert expand_layer_specs({"", "  "}) == set()

    def test_parse_layers_env_expands_first_non_empty(self, monkeypatch):
        """First non-empty env var is expanded into full layer paths."""
        monkeypatch.setenv("COV_LAYERS", "1-2, model.layers.9.self_attn")
        assert parse_layers_env("COV_LAYERS") == {
            "model.layers.1", "model.layers.2", "model.layers.9.self_attn"}
        monkeypatch.setenv("COV_LAYERS", "")
        assert parse_layers_env("COV_LAYERS") == set()
        assert parse_layers_env("COV_MISSING_LAYERS") == set()

    def test_list_model_layers_depth_and_format(self):
        """list_model_layers returns "name (Class)" entries cut at max_depth."""
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        model.add_module("block", _Inner())
        model.block.add_module("inner", _Toy())
        # Depth 1: direct children only; depth 2 includes grandchildren.
        d1 = list_model_layers(model, max_depth=1)
        d2 = list_model_layers(model, max_depth=2)
        assert d1 == ["0 (Linear)", "1 (ReLU)", "block (_Inner)"]
        assert "block.inner (_Toy)" in d2
        assert all(" (" in entry and entry.endswith(")") for entry in d1)
        assert list_model_layers(torch.nn.Linear(2, 2)) == []


class TestStackContext:
    """Stack-derived module context (TorchDispatchMode path)."""

    def test_context_from_stack_innermost_first(self):
        """Stack walk reports enclosing modules innermost-first with paths."""
        io_common._module_path_map.clear()
        probe_model = _StackProbeModel()
        probe_model.run_probe()
        ctx = probe_model.captured
        assert ctx[0] == ("Inner", "inner")  # class name from the Inner subclass
        assert ctx[1] == ("_StackProbeModel", "")  # root: unregistered path
        io_common._module_path_map.clear()

    def test_layer_and_module_matching_from_explicit_ctx(self):
        """layer_path matching supports prefix, exact, and glob semantics."""
        ctx = [("Linear", "model.layers.0.self_attn.q_proj")]
        assert layer_path_matches_from_stack({"model.layers.0"}, ctx) is True
        assert layer_path_matches_from_stack(
            {"model.layers.*.self_attn.q_proj"}, ctx) is True
        assert layer_path_matches_from_stack({"model.layers.1"}, ctx) is False
        assert layer_path_matches_from_stack({"model.layers.0x"}, ctx) is False
        assert module_context_matches_from_stack({"Linear"}, ctx) is True
        assert module_context_matches_from_stack({"ReLU"}, ctx) is False
        # Entries without a path never layer-match but still class-match.
        empty = [("Linear", "")]
        assert layer_path_matches_from_stack({"anything"}, empty) is False
        assert module_context_matches_from_stack({"Linear"}, empty) is True


class TestFormatting:
    """format_value / format_result / label builders."""

    def test_format_value_tensor_includes_stat_values(self):
        """Tensor formatting inlines shape, dtype, device and stat values."""
        s = format_value(torch.tensor([1.5, 2.5]))
        assert s == ("Tensor(shape=[2], dtype=torch.float32, device=cpu, "
                     "min=1.5, max=2.5, mean=2, std=0.707107)")

    def test_format_value_scalars_and_containers(self):
        """Non-tensor values render compactly; long lists collapse to len."""
        assert format_value(None) == "None"
        assert format_value(True) == "True"
        assert format_value(42) == "42"
        assert format_value([1, 2, 3]) == "list([1, 2, 3])"
        assert format_value([1, 2, 3, 4, 5]) == "list(len=5)"
        assert format_value((1, 2)) == "tuple([1, 2])"
        assert format_value("abc") == "str(...)"
        assert format_value({"a": 1}) == "dict(...)"

    def test_format_result_single_and_tuple(self):
        """format_result indents single values and numbered tuple members."""
        out = format_result(7)
        assert out == "  result: 7"
        two = format_result((1, torch.tensor([2.0])))
        assert two.splitlines()[0] == "  result[0]: 1"
        assert "result[1]: Tensor(shape=[1]" in two.splitlines()[1]

    def test_make_label_and_module_tag(self):
        """Label assembles op, module, layer, dispatch keys."""
        label = make_label(
            "mm", module_name="Linear", layer_path="model.layers.0",
            dispatch_keys=[("CPU", "CPU", False), ("CUDA", "FlagGems", True)])
        assert label == ("mm (module=Linear, layer=model.layers.0, "
                          "dispatch_keys=[(CPU, CPU), (CUDA, FlagGems)])")
        # Omitted optional parts disappear from the label.
        assert make_label("add", module_name="", layer_path="") == "add (module=)"
        assert make_module_tag_from_ctx("Linear", "a.b.c", for_json=True) == \
            "Linear:a.b.c"
        assert make_module_tag_from_ctx("Linear", "a.b.c") == "[Linear:a.b.c]"
        assert make_module_tag_from_ctx("Linear", "") == "[Linear]"
        assert make_module_tag_from_ctx("", "path") == ""
        # Module tag falls back to the hook-context stack counter.
        io_common.push_module_context("Linear")
        try:
            assert io_common.make_module_tag() == "[module=0,1]"
            assert get_current_module_counter() == (0, 1)
        finally:
            io_common.pop_module_context()


    def test_thread_local_context_stack(self):
        """Each thread sees its own module-context stack."""
        io_common.push_module_context("Outer")
        holder = {}

        def _other():
            io_common.push_module_context("OtherThread")
            holder["inner"] = io_common.get_current_module()
            io_common.pop_module_context()
            holder["after_pop"] = io_common.get_current_module()

        t = threading.Thread(target=_other)
        t.start()
        t.join()
        assert holder == {"inner": "OtherThread", "after_pop": None}
        assert io_common.get_current_module() == "Outer"
        io_common.pop_module_context()
        assert io_common.get_current_module() is None
